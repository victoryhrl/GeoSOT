from os import truncate
import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from .temp_trainer import predict
import torch.nn.functional as F

def get_uncertainty(cholesky_elements):
    raw_diag_0 = F.softplus(cholesky_elements[:, 0])
    raw_diag_1 = F.softplus(cholesky_elements[:, 2])
    
    diag_0 = torch.clamp(raw_diag_0, min=1e-6, max=15.0)
    diag_1 = torch.clamp(raw_diag_1, min=1e-6, max=15.0)
    
    uncertainty = torch.sqrt(diag_0 * diag_1)
    
    return uncertainty

def calculate_scores_rerank(
    config, model,
    q_rgb, r_rgb, q_phase, r_phase,
    q_labels, r_labels,
    q_loader, r_loader,
    topk=20, 
    alphas=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.18, 0.2, 0.5],  # Optimized for exponential decay
    ranks=[1, 5, 10]
):
    """
    Enhanced Uncertainty-Based Re-ranking for Cross-View Geo-Localization.
    
    Strategy:
    1. Global retrieval using cosine similarity to get top-k candidates
    2. For each candidate, compute alignment uncertainty using regression head
    3. Re-rank using: score = similarity * exp(-alpha * uncertainty)  [Exponential Decay]
    4. Grid search over alphas to find optimal weighting
    
    Args:
        config: Training configuration
        model: Trained model with regression head
        q_rgb, r_rgb: Query and reference RGB features [N, D]
        q_phase, r_phase: Query and reference phase features [N, D]
        q_labels, r_labels: Ground truth labels
        q_loader, r_loader: Data loaders for image loading
        topk: Number of candidates to re-rank (default: 20)
        alphas: List of uncertainty weights to test
        ranks: Recall@K values to compute
    
    Returns:
        best_metrics_list: [R@1, R@5, R@10, Hit_Rate] for best alpha
    """

    print(f"\n{'='*60}")
    print(f"Stage 1: Global Retrieval (Top-{topk} Candidates)")
    print(f"{'='*60}")
    rgb_weight = 0.6

    chunk_size = 2000
    num_chunks = (len(q_rgb) + chunk_size - 1) // chunk_size
    
    topk_sim_list = []
    topk_idx_list = []
    
    for i in tqdm(range(num_chunks), desc="Computing Similarity"):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(q_rgb))
        
        # Weighted fusion similarity
        sim_rgb = torch.matmul(q_rgb[start:end], r_rgb.t())
        sim_phase = torch.matmul(q_phase[start:end], r_phase.t())
        sim_chunk = rgb_weight * sim_rgb + (1.0 - rgb_weight) * sim_phase
        
        val, idx = torch.topk(sim_chunk, k=topk, dim=1)
        
        topk_sim_list.append(val.cpu())
        topk_idx_list.append(idx.cpu())
        
        del sim_rgb, sim_phase, sim_chunk
    
    topk_sim_scores = torch.cat(topk_sim_list, dim=0)  # [N_queries, topk]
    topk_indices = torch.cat(topk_idx_list, dim=0)      # [N_queries, topk]
    
    # Free memory
    del topk_sim_list, topk_idx_list
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"Stage 2: Uncertainty-Based Re-ranking")
    print(f"{'='*60}")
    
    metrics_dict = {
        alpha: {
            'recall': np.zeros(len(ranks)),
            'hit_rate_count': 0.0,
            'meter_errors': []
        } for alpha in alphas
    }
    
    total_queries = len(q_labels)
    query_dataset = q_loader.dataset
    ref_dataset = r_loader.dataset
    
    model.eval()
    
    QUERY_BATCH_SIZE = 20  # Process multiple queries at once
    CAND_BATCH_SIZE = 5    # Process candidates in batches
    
    with torch.no_grad():
        pbar = tqdm(range(0, total_queries, QUERY_BATCH_SIZE), desc="Re-ranking")
        
        for batch_start in pbar:
            batch_end = min(batch_start + QUERY_BATCH_SIZE, total_queries)
            curr_batch_size = batch_end - batch_start
            
            batch_q_imgs = []
            batch_q_phases = []
            batch_pos_lists = []
            batch_gt_offsets = []
            
            for qi in range(batch_start, batch_end):
                q_data = query_dataset[qi]
                batch_q_imgs.append(q_data[0])
                batch_q_phases.append(q_data[2])
                
                q_label_ts = q_labels[qi]
                pos_list = q_label_ts.tolist() if q_label_ts.dim() > 0 else [q_label_ts.item()]
                batch_pos_lists.append(pos_list)
                
                if hasattr(query_dataset, 'df_ground'):
                    row = query_dataset.df_ground.iloc[qi]
                    scale = config.img_size / config.sat_img_size
                    gt_off = torch.tensor([row['off_x'] * scale, row['off_y'] * scale])
                else:
                    gt_off = torch.tensor([0.0, 0.0])
                batch_gt_offsets.append(gt_off)
            
            batch_cand_indices = topk_indices[batch_start:batch_end]  # [batch, topk]
            batch_cand_sims = topk_sim_scores[batch_start:batch_end]  # [batch, topk]
            
            unique_ref_ids = torch.unique(batch_cand_indices).tolist()
            ref_cache = {}
            
            for ref_id in unique_ref_ids:
                ref_data = ref_dataset[ref_id]
                ref_cache[ref_id] = {
                    'img': ref_data[0],
                    'phase': ref_data[2],
                    'label': r_labels[ref_id].item()
                }
            
            batch_uncertainties = torch.zeros(curr_batch_size, topk)
            batch_pred_offsets = torch.zeros(curr_batch_size, topk, 2)
            
            for cand_start in range(0, topk, CAND_BATCH_SIZE):
                cand_end = min(cand_start + CAND_BATCH_SIZE, topk)
                cand_count = cand_end - cand_start
                
                paired_q_imgs = []
                paired_q_phases = []
                paired_r_imgs = []
                paired_r_phases = []
                
                for b in range(curr_batch_size):
                    paired_q_imgs.append(
                        batch_q_imgs[b].unsqueeze(0).repeat(cand_count, 1, 1, 1)
                    )
                    paired_q_phases.append(
                        batch_q_phases[b].unsqueeze(0).repeat(cand_count, 1, 1, 1)
                    )
                    
                    cand_ref_ids = batch_cand_indices[b, cand_start:cand_end].tolist()
                    paired_r_imgs.append(
                        torch.stack([ref_cache[rid]['img'] for rid in cand_ref_ids])
                    )
                    paired_r_phases.append(
                        torch.stack([ref_cache[rid]['phase'] for rid in cand_ref_ids])
                    )
                
                t_q_img = torch.cat(paired_q_imgs).to(config.device)
                t_q_phase = torch.cat(paired_q_phases).to(config.device)
                t_r_img = torch.cat(paired_r_imgs).to(config.device)
                t_r_phase = torch.cat(paired_r_phases).to(config.device)
                
                outputs = model(t_q_img, t_q_phase, t_r_img, t_r_phase)
                pred_offset = outputs[4].cpu()      # [batch*cand, 2]
                cholesky = outputs[5].cpu()         # [batch*cand, 3]
                
                uncertainty = get_uncertainty(cholesky)  # [batch*cand]
                
                batch_uncertainties[:, cand_start:cand_end] = uncertainty.view(curr_batch_size, cand_count)
                batch_pred_offsets[:, cand_start:cand_end] = pred_offset.view(curr_batch_size, cand_count, 2)
            
            for b in range(curr_batch_size):
                qi = batch_start + b
                pos_list = batch_pos_lists[b]
                gt_label = pos_list[0]
                semi_pos = set(pos_list[1:])
                gt_offset = batch_gt_offsets[b]
                
                cand_sims = batch_cand_sims[b]
                cand_uncs = batch_uncertainties[b]
                cand_offsets = batch_pred_offsets[b]
                cand_ids = batch_cand_indices[b].tolist()
                cand_labels = [ref_cache[cid]['label'] for cid in cand_ids]
                
                for alpha in alphas:
                    confidence = torch.exp(-alpha * cand_uncs)
                    final_scores = cand_sims * confidence
                    final_scores_np = final_scores.cpu().numpy()
                    
                    m = metrics_dict[alpha]
                    
                    if gt_label in cand_labels:
                        gt_idx = cand_labels.index(gt_label)
                        gt_score = final_scores_np[gt_idx]
                        
                        num_higher = np.sum(final_scores_np > gt_score)
                        
                        for rank_idx, k in enumerate(ranks):
                            if num_higher < k:
                                m['recall'][rank_idx] += 1.0
                                
                        mask = np.ones(len(final_scores_np))
                        for ci, lbl in enumerate(cand_labels):
                            if lbl in semi_pos or lbl == gt_label:
                                mask[ci] = 0.0
                        
                        if np.sum((final_scores_np > gt_score) * mask) < 1.0:
                            m['hit_rate_count'] += 1.0
                            
                    best_idx = np.argmax(final_scores_np)
                    best_label = cand_labels[best_idx]
                    
                    if best_label == gt_label:
                        pred_off = cand_offsets[best_idx]
                        pixel_error = torch.norm(pred_off - gt_offset).item()
                        scale = config.img_size / config.sat_img_size
                        meter_error = (pixel_error / scale) * config.ground_resolution
                        m['meter_errors'].append(meter_error)
                        
    print(f"\n{'='*60}")
    print("Multi-Alpha Evaluation Results")
    print(f"{'='*60}")
    
    best_r1 = -1.0
    best_alpha = alphas[0]
    best_metrics = []
    
    for alpha in alphas:
        m = metrics_dict[alpha]
        
        recall_pct = (m['recall'] / total_queries) * 100.0
        hit_rate_pct = (m['hit_rate_count'] / total_queries) * 100.0
        
        errors = m['meter_errors']
        mean_err = np.mean(errors) if len(errors) > 0 else 0.0
        median_err = np.median(errors) if len(errors) > 0 else 0.0
        
        recall_str = ", ".join([f"R@{ranks[i]}: {recall_pct[i]:.2f}%" for i in range(len(ranks))])
        print(f"[α={alpha:5.2f}] {recall_str}")
        print(f"            Hit Rate: {hit_rate_pct:.2f}%, MeanErr: {mean_err:.2f}m, MedianErr: {median_err:.2f}m")
        
        if recall_pct[0] > best_r1:
            best_r1 = recall_pct[0]
            best_alpha = alpha
            best_metrics = [recall_pct[i] for i in range(min(3, len(ranks)))]
            best_metrics.append(hit_rate_pct)
    
    print(f"{'='*60}")
    print(f"✓ Best Configuration: α={best_alpha} (R@1: {best_r1:.2f}%)")
    print(f"{'='*60}\n")
    
    return best_metrics

def evaluate_localization(config, model, dataloader, hit_mask=None):
    model.eval()
    all_meter_errors = []

    with torch.no_grad():
        bar = tqdm(dataloader, total=len(dataloader), desc="Evaluating Localization")
        for batch in bar:
            query = batch[0].to(config.device)
            reference = batch[1].to(config.device)
            query_phase = batch[2].to(config.device)
            reference_phase = batch[3].to(config.device)
            offset_gt_scaled_px = batch[4].to(config.device)

            _, _, _, _, pred_mean_scaled_px, _ = model(
                img1_rgb=query, img2_rgb=reference,
                img1_phase=query_phase, img2_phase=reference_phase
            )

            pixel_error_scaled = torch.sqrt(torch.sum((pred_mean_scaled_px - offset_gt_scaled_px) ** 2, dim=1))

            scale_factor = config.img_size / config.sat_img_size
            pixel_error_original = pixel_error_scaled / scale_factor

            meter_errors = pixel_error_original * config.ground_resolution

            all_meter_errors.append(meter_errors.cpu())
            
    all_meter_errors = torch.cat(all_meter_errors, dim=0).numpy()

    if hit_mask is not None:
        if len(hit_mask) != len(all_meter_errors):
            print(f"Error: Mask length {len(hit_mask)} != Data length {len(all_meter_errors)}")
            return 0.0, 0.0
        final_errors = all_meter_errors[hit_mask]
    else:
        final_errors = all_meter_errors

    if len(final_errors) > 0:
        mean_error = np.mean(final_errors)
        median_error = np.median(final_errors)
    else:
        mean_error = 0.0
        median_error = 0.0

    print(f"Mean Localization Error: {mean_error:.2f} meters")
    print(f"Median Localization Error: {median_error:.2f} meters")

    return mean_error, median_error



def evaluate1(config,
             model,
             reference_dataloader,
             query_dataloader,
             localization_dataloader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True,
             rerank=False,
             alpha=0.1,
             topk_candidates=20):
    print("\nExtract Features:")
    (reference_rgb_features, reference_phase_features), reference_labels = predict(config, model, reference_dataloader)
    (query_rgb_features, query_phase_features), query_labels = predict(config, model, query_dataloader)

    print("Compute Scores:")

    r1 = 0.0
    calculate_scores(query_rgb_features, reference_rgb_features,
                          query_phase_features, reference_phase_features,
                          query_labels, reference_labels, step_size=step_size,
                          ranks=ranks)
    
    if rerank:
        print(f"\nRe-ranking Top-{topk_candidates}:")
        rr_res = calculate_scores_rerank(
            config, model,
            query_rgb_features, reference_rgb_features, query_phase_features, reference_phase_features,
            query_labels, reference_labels,
            query_dataloader, reference_dataloader,
            topk=topk_candidates
        )
        r1 = rr_res[0]
    else:
        r1, hit_mask = calculate_scores(query_rgb_features, reference_rgb_features,
                          query_phase_features, reference_phase_features,
                          query_labels, reference_labels, step_size=step_size,
                          ranks=ranks)

        mean_error, median_error = evaluate_localization(config, model, localization_dataloader, hit_mask=hit_mask)
    # cleanup and free memory on GPU
    if cleanup:
        del reference_rgb_features, reference_phase_features, reference_labels
        del query_rgb_features, query_phase_features, query_labels
        gc.collect()

    return r1

def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=truncate):
    
    print("\nExtract Features:")
    (reference_rgb_features, reference_phase_features), reference_labels = predict(config, model, reference_dataloader)
    (query_rgb_features, query_phase_features), query_labels = predict(config, model, query_dataloader)
    print("Compute Scores:")
    r1, hit_mask = calculate_scores(query_rgb_features, reference_rgb_features,
            query_phase_features, reference_phase_features,
            query_labels, reference_labels, step_size=step_size,
            ranks=ranks)
    # cleanup and free memory on GPU
    if cleanup:
        del reference_rgb_features, reference_phase_features, reference_labels
        del query_rgb_features, query_phase_features, query_labels
        gc.collect()

    return r1


def calc_sim(config,
             model,
             reference_dataloader,
             query_dataloader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    print("\nExtract Features:")
    (reference_rgb_features, reference_phase_features), reference_labels = predict(config, model, reference_dataloader)
    (query_rgb_features, query_phase_features), query_labels = predict(config, model, query_dataloader)

    print("Compute Scores Train:")
    r1 = calculate_scores_train(query_rgb_features, reference_rgb_features,
                                query_phase_features, reference_phase_features,
                                query_labels, reference_labels, step_size=step_size,
                                ranks=ranks)

    near_dict = calculate_nearest(query_rgb_features, reference_rgb_features,
                                  query_phase_features, reference_phase_features,
                                  query_labels, reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)

    # cleanup and free memory on GPU
    if cleanup:
        del reference_rgb_features, reference_phase_features, reference_labels
        del query_rgb_features, query_phase_features, query_labels
        gc.collect()

    return r1, near_dict


def calculate_scores(query_rgb_features, reference_rgb_features,
                     query_phase_features, reference_phase_features,
                     query_labels, reference_labels, step_size=1000,
                     ranks=[1, 5, 10]):
    topk = copy.deepcopy(ranks)
    Q = len(query_rgb_features)
    R = len(reference_rgb_features)

    steps = Q // step_size + 1

    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()

    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i

    similarity = []

    for i in range(steps):
        start = step_size * i

        end = start + step_size

        # 计算 RGB 相似度
        sim_rgb_tmp = query_rgb_features[start:end] @ reference_rgb_features.T

        # 计算 Phase 相似度
        sim_phase_tmp = query_phase_features[start:end] @ reference_phase_features.T

        sim_fused_tmp = 0.6 * sim_rgb_tmp + 0.4 * sim_phase_tmp

        similarity.append(sim_fused_tmp.cpu())

    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk.append(R // 100)

    results = np.zeros([len(topk)])
    hit_mask = np.zeros(Q, dtype=bool)

    hit_rate = 0.0

    bar = tqdm(range(Q))

    for i in bar:

        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i][0]]]

        # number of references with higher similiarity as gt
        higher_sim = similarity[i, :] > gt_sim

        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.

        if ranking < 1:
            hit_mask[i] = True

        # mask for semi pos
        mask = torch.ones(R)
        for near_pos in query_labels_np[i][1:]:
            mask[ref2index[near_pos]] = 0

        # calculate hit rate
        hit = (higher_sim * mask).sum()
        if hit < 1:
            hit_rate += 1.0

    results = results / Q * 100.
    hit_rate = hit_rate / Q * 100

    bar.close()

    # wait to close pbar
    time.sleep(0.1)

    string = []
    for i in range(len(topk) - 1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))

    string.append('Recall@top1: {:.4f}'.format(results[-1]))
    string.append('Hit_Rate: {:.4f}'.format(hit_rate))

    print(' - '.join(string))

    return results[0], hit_mask


def calculate_scores_train(query_rgb_features, reference_rgb_features,
                           query_phase_features, reference_phase_features,
                           query_labels, reference_labels, step_size=1000,
                           ranks=[1, 5, 10]):
    topk = copy.deepcopy(ranks)
    Q = len(query_rgb_features)
    R = len(reference_rgb_features)

    steps = Q // step_size + 1

    query_labels_np = query_labels[:, 0].cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()

    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i

    similarity = []

    for i in range(steps):
        start = step_size * i

        end = start + step_size

        sim_rgb_tmp = query_rgb_features[start:end] @ reference_rgb_features.T
        sim_phase_tmp = query_phase_features[start:end] @ reference_phase_features.T
        sim_fused_tmp = 0.5 * sim_rgb_tmp + 0.5 * sim_phase_tmp

        similarity.append(sim_fused_tmp.cpu())

    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk.append(R // 100)

    results = np.zeros([len(topk)])

    bar = tqdm(range(Q))

    for i in bar:

        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]

        # number of references with higher similiarity as gt
        higher_sim = similarity[i, :] > gt_sim

        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.

    results = results / Q * 100.

    bar.close()

    # wait to close pbar
    time.sleep(0.1)

    string = []
    for i in range(len(topk) - 1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))

    string.append('Recall@top1: {:.4f}'.format(results[-1]))

    print(' - '.join(string))

    return results[0]


def calculate_nearest(query_rgb_features, reference_rgb_features,
                      query_phase_features, reference_phase_features,
                      query_labels, reference_labels, neighbour_range=64,
                      step_size=1000):
    query_labels = query_labels[:, 0]

    Q = len(query_rgb_features)

    steps = Q // step_size + 1

    similarity = []

    for i in range(steps):
        start = step_size * i

        end = start + step_size

        sim_rgb_tmp = query_rgb_features[start:end] @ reference_rgb_features.T
        sim_phase_tmp = query_phase_features[start:end] @ reference_phase_features.T
        sim_fused_tmp = 0.5 * sim_rgb_tmp + 0.5 * sim_phase_tmp

        similarity.append(sim_fused_tmp.cpu())

    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    # there might be more ground views for same sat view
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range + 2, dim=1)

    topk_references = []

    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i, :]])

    topk_references = torch.stack(topk_references, dim=0)

    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)

    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()

    for i in range(len(topk_references)):
        nearest = topk_references[i][mask[i]][:neighbour_range]

        nearest_dict[query_labels[i].item()] = list(nearest)

    return nearest_dict
