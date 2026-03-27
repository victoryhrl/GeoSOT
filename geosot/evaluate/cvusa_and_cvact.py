import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict


def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    (reference_rgb_features, reference_phase_features), reference_labels = predict(config, model, reference_dataloader)
    (query_rgb_features, query_phase_features), query_labels = predict(config, model, query_dataloader)
    
    print("Compute Scores:")
    r1 = calculate_scores(query_rgb_features, reference_rgb_features,
                          query_phase_features, reference_phase_features,
                          query_labels, reference_labels, 
                          step_size=step_size, ranks=ranks)
        
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
    r1 = calculate_scores(query_rgb_features, reference_rgb_features,
                          query_phase_features, reference_phase_features,
                          query_labels, reference_labels, 
                          step_size=step_size, ranks=ranks) 
    
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

def evaluate_test(config,
                  model,
                  reference_dataloader,
                  query_dataloader,
                  ranks=[1, 5, 10],
                  step_size=1000,
                  cleanup=True):
    """
    Memory-efficient evaluation for large test sets (e.g. CVACT Test ~92K samples).
    Uses chunked similarity computation to avoid OOM.
    """

    print("\nExtract Features:")
    (reference_rgb_features, reference_phase_features), reference_labels = predict(config, model, reference_dataloader)
    (query_rgb_features, query_phase_features), query_labels = predict(config, model, query_dataloader)

    print("Compute Scores:")
    r1 = calculate_scores_chunked(query_rgb_features, reference_rgb_features,
                                  query_phase_features, reference_phase_features,
                                  query_labels, reference_labels,
                                  step_size=step_size, ranks=ranks)

    # cleanup and free memory on GPU
    if cleanup:
        del reference_rgb_features, reference_phase_features, reference_labels
        del query_rgb_features, query_phase_features, query_labels
        gc.collect()

    return r1


def calculate_scores_chunked(query_rgb_features, reference_rgb_features,
                             query_phase_features, reference_phase_features,
                             query_labels, reference_labels,
                             step_size=1000, ranks=[1, 5, 10]):
    """
    Memory-efficient version of calculate_scores.
    Computes rankings per-chunk without materializing the full Q×R similarity matrix.
    Peak memory: step_size × R (e.g. 1000 × 92800 ≈ 370MB) instead of Q × R (~34GB).
    """

    topk = copy.deepcopy(ranks)
    Q = len(query_rgb_features)
    R = len(reference_rgb_features)

    steps = Q // step_size + 1

    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()

    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i

    topk.append(R // 100)

    results = np.zeros([len(topk)])

    bar = tqdm(range(steps), desc="Chunked Eval")

    for chunk_idx in bar:
        start = step_size * chunk_idx
        end = min(start + step_size, Q)

        if start >= Q:
            break

        # Compute similarity for this chunk only: [chunk_size, R]
        sim_rgb = query_rgb_features[start:end] @ reference_rgb_features.T
        sim_phase = query_phase_features[start:end] @ reference_phase_features.T
        sim_fused = (0.5 * sim_rgb + 0.5 * sim_phase).cpu()

        # Free intermediate tensors immediately
        del sim_rgb, sim_phase

        # Process each query in this chunk
        for local_i in range(end - start):
            global_i = start + local_i
            gt_idx = ref2index[query_labels_np[global_i]]

            # Similarity value of GT reference
            gt_sim = sim_fused[local_i, gt_idx]

            # Number of references with higher similarity than GT
            ranking = (sim_fused[local_i, :] > gt_sim).sum().item()

            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.

        # Discard this chunk's similarity matrix
        del sim_fused

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


def calculate_scores(query_rgb_features, reference_rgb_features,
                     query_phase_features, reference_phase_features, 
                     query_labels, reference_labels, 
                     step_size=1000, ranks=[1,5,10]):

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
          
        sim_rgb = query_rgb_features[start:end] @ reference_rgb_features.T
        sim_phase = query_phase_features[start:end] @ reference_phase_features.T

        sim_fused = 0.6 * sim_rgb + 0.4 * sim_phase
        
        similarity.append(sim_fused.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        
    results = results/ Q * 100.
 
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))            
        
    print(' - '.join(string)) 

    return results[0]
    

def calculate_nearest(query_rgb_features, reference_rgb_features,
                      query_phase_features, reference_phase_features, 
                      query_labels, reference_labels, 
                      neighbour_range=64, step_size=1000):


    Q = len(query_rgb_features)
    
    steps = Q // step_size + 1
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_rgb = query_rgb_features[start:end] @ reference_rgb_features.T
        sim_phase = query_phase_features[start:end] @ reference_phase_features.T

        sim_fused = 0.5 * sim_rgb + 0.5 * sim_phase
        
        similarity.append(sim_fused.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+1, dim=1)

    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
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