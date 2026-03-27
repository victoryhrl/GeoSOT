import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.amp import autocast
import torch.nn.functional as F

def gaussian_nll_loss(pred_mean, cholesky_elements, offset_gt):
    pred_mean = pred_mean.float()
    cholesky_elements = cholesky_elements.float()
    offset_gt = offset_gt.float()

    B = pred_mean.size(0)
    L = torch.zeros(B, 2, 2, device=pred_mean.device)

    raw_diag_0 = F.softplus(cholesky_elements[:, 0])
    raw_diag_1 = F.softplus(cholesky_elements[:, 2])

    L_diag_0 = torch.clamp(raw_diag_0, min=1e-6, max=15.0)
    L_diag_1 = torch.clamp(raw_diag_1, min=1e-6, max=15.0)

    L[:, 0, 0] = L_diag_0
    L[:, 1, 0] = cholesky_elements[:, 1]
    L[:, 1, 1] = L_diag_1

    diff = (offset_gt - pred_mean).unsqueeze(-1)  # Shape: [B, 2, 1]
    
    x = torch.linalg.solve_triangular(L, diff, upper=False)
    mahalanobis_dist = torch.sum(x * x, dim=(1, 2))

    # Calculate the log-determinant term: log(|Σ|)
    # log(|Σ|) = log(|L*L^T|) = 2 * log(|L|) = 2 * sum(log(diag(L)))
    log_det_sigma = 2 * (torch.log(L_diag_0) + torch.log(L_diag_1))

    # The Gaussian NLL loss is 0.5 * (Mahalanobis distance + log-determinant)
    loss_per_sample = 0.5 * (mahalanobis_dist + log_det_sigma)

    # Return the mean loss over the batch
    return loss_per_sample.mean()

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, current_lambda_reg=None):
    model.train()
    losses = AverageMeter()
    time.sleep(0.1)

    accumulation_steps = getattr(train_config, 'gradient_accumulation_steps', 1)
    optimizer.zero_grad()

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    for i, (query, reference, ids, query_phase, reference_phase, offset_gt_scaled_px) in enumerate(bar):

        with autocast(device_type=train_config.device, dtype=torch.float16, enabled=scaler is not None):
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)
            query_phase = query_phase.to(train_config.device)
            reference_phase = reference_phase.to(train_config.device)
            offset_gt_scaled_px = offset_gt_scaled_px.to(train_config.device)

            #, phase_feat_q, phase_feat_ref, pred_mean, cholesky_elements 
            rgb_feat_q, rgb_feat_ref, phase_feat_q, phase_feat_ref, pred_mean, cholesky_elements = model(
                img1_rgb=query,
                img1_phase=query_phase,
                img2_rgb=reference,
                img2_phase=reference_phase
            )

            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                logit_scale = model.module.logit_scale.exp()
            else:
                logit_scale = model.logit_scale.exp()

            loss_cross_rgb = loss_function(rgb_feat_q, rgb_feat_ref, logit_scale)
            loss_cross_phase = loss_function(phase_feat_q, phase_feat_ref, logit_scale)
            loss_retrieval = loss_cross_rgb + loss_cross_phase

            loss_reg_uncertainty = gaussian_nll_loss(pred_mean, cholesky_elements, offset_gt_scaled_px)

            loss_reg = current_lambda_reg * loss_reg_uncertainty

            loss = loss_retrieval + loss_reg
            losses.update(loss.item())
            loss = loss / accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if scaler:
                if train_config.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if train_config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad)
                optimizer.step()

            if scheduler:
                scheduler.step()

            optimizer.zero_grad()

        if train_config.verbose:
            monitor = {
                "loss": f"{loss.item() * accumulation_steps:.4f}",
                "loss_avg": f"{losses.avg:.4f}",
                "L_ret": f"{loss_retrieval.item():.4f}",
                "L_reg": f"{loss_reg.item():.4f}",
                "lr": f"{optimizer.param_groups[-1]['lr']:.6f}"
            }
            bar.set_postfix(ordered_dict=monitor)

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    model.eval()
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    rgb_features_list = []
    phase_features_list = []
    ids_list = []
    with torch.no_grad():
        for img_rgb, ids, img_phase in bar:
            ids_list.append(ids)

            with autocast(device_type=train_config.device, dtype=torch.float16, enabled=True):
                img_rgb = img_rgb.to(train_config.device)
                img_phase = img_phase.to(train_config.device)

                rgb_feature, phase_feature = model(img1_rgb=img_rgb, img1_phase=img_phase)

                if train_config.normalize_features:
                    rgb_feature = F.normalize(rgb_feature, dim=-1)
                    phase_feature = F.normalize(phase_feature, dim=-1)

            rgb_features_list.append(rgb_feature.to(torch.float32))
            phase_features_list.append(phase_feature.to(torch.float32))

        rgb_features = torch.cat(rgb_features_list, dim=0)
        phase_features = torch.cat(phase_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

    if train_config.verbose:
        bar.close()

    return (rgb_features, phase_features), ids_list
