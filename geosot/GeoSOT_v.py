import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

class geoSOT(nn.Module):
    def __init__(self, feature_dim, img_size=384, patch_size=32, num_heads=4, iters=3, epsilon=0.05):
        super(geoSOT, self).__init__()
        self.iters = iters
        self.epsilon = epsilon
        self.feature_dim = feature_dim

        self.grid_size = img_size // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, feature_dim, self.grid_size, self.grid_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.norm_input = nn.LayerNorm(feature_dim)

        self.context_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads, 
            dim_feedforward=feature_dim*2, 
            dropout=0.1, 
            batch_first=True,
            norm_first=True
        )
        
        self.dustbin_cost = nn.Parameter(torch.tensor(2.0))

    def _get_pos_embed(self, H_tgt, W_tgt):
        if H_tgt == self.pos_embed.shape[2] and W_tgt == self.pos_embed.shape[3]:
            return self.pos_embed.flatten(2).transpose(1, 2)
        pos_embed = F.interpolate(self.pos_embed, size=(H_tgt, W_tgt), mode='bicubic', align_corners=False)
        return pos_embed.flatten(2).transpose(1, 2)

    def forward(self, feat_g, feat_s):
        B, C, Hg, Wg = feat_g.shape
        _, _, Hs, Ws = feat_s.shape
        Ng, Ns = Hg * Wg, Hs * Ws
        
        # 1. Flatten & Norm
        feat_g_flat = self.norm_input(feat_g.flatten(2).transpose(1, 2))
        feat_s_flat = self.norm_input(feat_s.flatten(2).transpose(1, 2))

        # 2. Pos Embed
        pos_g = self._get_pos_embed(Hg, Wg)
        pos_s = self._get_pos_embed(Hs, Ws)
        
        # 3. Context
        feat_g_ctx = self.context_layer(feat_g_flat + pos_g)
        feat_s_ctx = self.context_layer(feat_s_flat + pos_s)

        # 4. Cost Matrix
        feat_g_norm = F.normalize(feat_g_ctx, dim=-1)
        feat_s_norm = F.normalize(feat_s_ctx, dim=-1)
        sim_matrix = torch.bmm(feat_g_norm, feat_s_norm.transpose(1, 2))
        cost_matrix = 1.0 - sim_matrix 

        # 5. Dustbin Expansion
        bin_val = F.softplus(self.dustbin_cost)
        
        bin_col = bin_val.view(1, 1, 1).expand(B, Ng, 1)
        cost_with_col = torch.cat([cost_matrix, bin_col], dim=2)

        bin_row_vals = bin_val.view(1, 1, 1).expand(B, 1, Ns)
        zero_corner = torch.zeros(B, 1, 1, device=feat_g.device, dtype=cost_matrix.dtype)
        bin_row = torch.cat([bin_row_vals, zero_corner], dim=2)

        expanded_cost = torch.cat([cost_with_col, bin_row], dim=1)

        # 6. FP32 Sinkhorn (Prevent numerical collapse)
        orig_dtype = expanded_cost.dtype
        expanded_cost_fp32 = expanded_cost.float()
        
        log_mu = torch.zeros(B, Ng + 1, device=feat_g.device, dtype=torch.float32)
        log_nu = torch.zeros(B, Ns + 1, device=feat_g.device, dtype=torch.float32)
        log_mu[:, Ng] = torch.log(torch.tensor(float(Ns), device=feat_g.device))
        log_nu[:, Ns] = torch.log(torch.tensor(float(Ng), device=feat_g.device))

        log_K = -expanded_cost_fp32 / self.epsilon
        log_T = log_sinkhorn_iterations(log_K, log_mu, log_nu, self.iters)
        T = torch.exp(log_T)
        
        T = T.to(orig_dtype)
        T_core = T[:, :Ng, :Ns]
        
        # 7. Alignment
        feat_g_aligned = torch.bmm(T_core, feat_s_flat) 
        feat_s_aligned = torch.bmm(T_core.transpose(1, 2), feat_g_flat)

        return feat_g_aligned, feat_s_aligned, T_core, feat_g_ctx, feat_s_ctx

class GeoSOT(nn.Module):
    def __init__(self, model_path: str, img_size: int = 384):
        super(GeoSOT, self).__init__()
        self.img_size = img_size
        self.model_name = 'convnext_base.fb_in22k_ft_in1k_384'
        
        self.model_rgb = timm.create_model(self.model_name, pretrained=True)
        self.model_phase = timm.create_model(self.model_name, pretrained=True)
        
        feature_dim = self.model_rgb.num_features 
        
        self.model_rgb.head.fc = nn.Identity()
        self.model_phase.head.fc = nn.Identity()
        self.default_cfg = self.model_rgb.default_cfg
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.fusion_dim = feature_dim * 2 
        self.scot = geoSOT(
            feature_dim=self.fusion_dim, 
            img_size=self.img_size, 
            patch_size=32, 
            num_heads=4, 
            iters=3
        )
        
        self.reduced_dim = 512
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(self.fusion_dim, self.reduced_dim, kernel_size=1),
            nn.GroupNorm(32, self.reduced_dim),
            nn.ReLU(inplace=True)
        )
        
        self.spatial_grid_size = 4 
        self.spatial_pool = nn.AdaptiveAvgPool2d((self.spatial_grid_size, self.spatial_grid_size))

        regression_input_dim = 4 * (self.spatial_grid_size ** 2) * self.reduced_dim
        
        self.reg_bottleneck = nn.Sequential(
            nn.Linear(regression_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.mean_head = nn.Linear(1024, 2)
        self.cholesky_head = nn.Linear(1024, 3)
        
        with torch.no_grad():
            self.cholesky_head.bias[0] = 1.0
            self.cholesky_head.bias[2] = 1.0
            nn.init.normal_(self.cholesky_head.weight, mean=0, std=0.001)
            nn.init.normal_(self.mean_head.weight, mean=0, std=0.001)
            nn.init.constant_(self.mean_head.bias, 0)

    def get_config(self):
        return {"mean": self.default_cfg['mean'], "std": self.default_cfg['std'],
                "input_size": self.default_cfg['input_size']}

    def set_grad_checkpointing(self, enable=True):
        self.model_rgb.set_grad_checkpointing(enable)
        self.model_phase.set_grad_checkpointing(enable)

    def _extract_features(self, model, x):
        feat = model.forward_features(x)
        return feat

    def forward(self, img1_rgb, img1_phase, img2_rgb=None, img2_phase=None):
        
        raw_rgb_feat_q = self._extract_features(self.model_rgb, img1_rgb)
        raw_phase_feat_q = self._extract_features(self.model_phase, img1_phase)
        
        pool_rgb_q = self.model_rgb.forward_head(raw_rgb_feat_q)
        pool_phase_q = self.model_phase.forward_head(raw_phase_feat_q)
        
        if img2_rgb is not None:
            # Reference
            raw_rgb_feat_ref = self._extract_features(self.model_rgb, img2_rgb)
            raw_phase_feat_ref = self._extract_features(self.model_phase, img2_phase)
            
            pool_rgb_ref = self.model_rgb.forward_head(raw_rgb_feat_ref)
            pool_phase_ref = self.model_phase.forward_head(raw_phase_feat_ref)
            
            # GeoSOT Forward
            feat_q_full = torch.cat([raw_rgb_feat_q, raw_phase_feat_q], dim=1)
            feat_ref_full = torch.cat([raw_rgb_feat_ref, raw_phase_feat_ref], dim=1)
            
            B, C2, Hq, Wq = feat_q_full.shape
            _, _, Hr, Wr = feat_ref_full.shape

            g_aligned, s_aligned, _, g_ctx, s_ctx = self.scot(feat_q_full, feat_ref_full)
            
            # Process Features
            def process_features(x_flat, H, W):
                x_2d = x_flat.transpose(1, 2).reshape(B, C2, H, W).contiguous()
                x_reduced = self.channel_reducer(x_2d) 
                x_pooled = self.spatial_pool(x_reduced) 
                return x_pooled.flatten(1)

            g_aligned_flat = process_features(g_aligned, Hq, Wq)
            s_aligned_flat = process_features(s_aligned, Hr, Wr)
            g_ctx_flat = process_features(g_ctx, Hq, Wq)
            s_ctx_flat = process_features(s_ctx, Hr, Wr)

            # Regression
            reg_input = torch.cat([g_ctx_flat, s_ctx_flat, g_aligned_flat, s_aligned_flat], dim=1)
            
            reg_base_feat = self.reg_bottleneck(reg_input)
            pred_mean = self.mean_head(reg_base_feat)
            cholesky_elements = self.cholesky_head(reg_base_feat)

            # , pool_phase_q, pool_phase_ref, pred_mean.float(), cholesky_elements.float()
            return pool_rgb_q, pool_rgb_ref, pool_phase_q, pool_phase_ref, pred_mean.float(), cholesky_elements.float()
        
        else:
            return pool_rgb_q, pool_phase_q
