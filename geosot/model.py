import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TimmModel(nn.Module):

    def __init__(self,
                 model_name,
                 pretrained=True,
                 img_size=384):

        super(TimmModel, self).__init__()

        self.img_size = img_size

        # 1. 创建主干网络
        if "vit" in model_name:
            self.model_rgb = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model_rgb = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.model_phase = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.model_phase.load_state_dict(self.model_rgb.state_dict())

        # 2. 动态获取特征维度，用于构建融合层
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            feature_dim = self.model_rgb(dummy_input).shape[-1]

        # 3. 构建投影融合层 (唯一的融合方式)
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),  # BatchNorm 有助于稳定训练
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        # 4. 温度参数
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self, ):
        data_config = timm.data.resolve_model_data_config(self.model_rgb)
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model_rgb.set_grad_checkpointing(enable)
        self.model_phase.set_grad_checkpointing(enable)

    def forward(self, img1_rgb, img2_rgb=None, img1_phase=None, img2_phase=None):
        if img2_rgb is not None:
            rgb_features1 = self.model_rgb(img1_rgb)
            rgb_features2 = self.model_rgb(img2_rgb)
            phase_features1 = self.model_phase(img1_phase)
            phase_features2 = self.model_phase(img2_phase)

            fused1 = self.fusion_layer(torch.cat([rgb_features1, phase_features1], dim=1))
            fused2 = self.fusion_layer(torch.cat([rgb_features2, phase_features2], dim=1))

            combined_features1 = F.normalize(fused1, p=2, dim=1)
            combined_features2 = F.normalize(fused2, p=2, dim=1)

            return combined_features1, combined_features2
        else:
            rgb_features = self.model_rgb(img1_rgb)
            phase_features = self.model_phase(img1_phase)

            fused = self.fusion_layer(torch.cat([rgb_features, phase_features], dim=1))

            combined_features = F.normalize(fused, p=2, dim=1)

            return combined_features