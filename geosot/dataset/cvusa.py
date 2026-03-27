import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time
import albumentations as A
 
class CVUSADatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
                 transforms_query_geometric=None,
                 transforms_reference_geometric=None,
                 transforms_rgb=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        
        self.transforms_query_geometric = transforms_query_geometric
        self.transforms_reference_geometric = transforms_reference_geometric
        self.transforms_rgb = transforms_rgb
        
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        
        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
        

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))
        
        self.idx2pair = dict()
        train_ids_list = list()
        
        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)
            
        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)
            
    def get_spatial_phase_image(self, img_rgb):
        if isinstance(img_rgb, torch.Tensor):
            img_np = img_rgb.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = np.array(img_rgb)

        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        gray_channel = img_gray.astype(np.float32) / 255.0
        
        fft = np.fft.fft2(gray_channel)
        magnitude = np.abs(fft) + 1e-8
        phase_only_spectrum = fft / magnitude
        spatial_phase = np.fft.ifft2(phase_only_spectrum).real
        spatial_phase = np.abs(spatial_phase)
        mean_val = np.mean(spatial_phase)
        std_val = np.std(spatial_phase)
        threshold = mean_val + 3 * std_val
        spatial_phase = np.clip(spatial_phase, 0, threshold)
        
        p_min, p_max = spatial_phase.min(), spatial_phase.max()
        if p_max - p_min > 1e-8:
            phase_norm = (spatial_phase - p_min) / (p_max - p_min)
        else:
            phase_norm = np.zeros_like(spatial_phase)
        
        mixed_img = np.stack([phase_norm, phase_norm, gray_channel], axis=-1)
        
        return mixed_img.astype(np.float32)

    def __getitem__(self, index):
        
        idx, sat, ground = self.idx2pair[self.samples[index]]
        
        # load query -> ground image
        query_img = cv2.imread(f'{self.data_folder}/{ground}')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/{sat}')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        query_img_phase = self.get_spatial_phase_image(query_img)
        reference_img_phase = self.get_spatial_phase_image(reference_img)

            
        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1) 
            query_img_phase = cv2.flip(query_img_phase, 1)
            reference_img_phase = cv2.flip(reference_img_phase, 1)
        
        # image transforms
        if self.transforms_query_geometric is not None:
            transformed_rgb_q = self.transforms_query_geometric(image=query_img)
            query_img = transformed_rgb_q['image']
            replay_params_q = transformed_rgb_q['replay']
            
            transformed_phase_q = A.ReplayCompose.replay(replay_params_q, image=query_img_phase)
            query_img_phase = transformed_phase_q['image']
            
        if self.transforms_reference_geometric is not None:
            
            transformed_rgb_ref = self.transforms_reference_geometric(image=reference_img)
            reference_img = transformed_rgb_ref['image']
            replay_params_ref = transformed_rgb_ref['replay']
            
            transformed_phase_ref = A.ReplayCompose.replay(replay_params_ref, image=reference_img_phase)
            reference_img_phase = transformed_phase_ref['image']

        if self.transforms_rgb is not None:
            query_img = self.transforms_rgb(image=query_img)['image']
            reference_img = self.transforms_rgb(image=reference_img)['image']

        query_img_phase = torch.from_numpy(query_img_phase).permute(2, 0, 1)
        query_img_phase = (query_img_phase - 0.5) / 0.5
        reference_img_phase = torch.from_numpy(reference_img_phase).permute(2, 0, 1)
        reference_img_phase = (reference_img_phase - 0.5) / 0.5
                
        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2)) 
            reference_img_phase = torch.rot90(reference_img_phase, k=r, dims=(1, 2)) 
            
            # use roll for ground view if rotate sat view
            c, h, w = query_img.shape
            shifts = - w//4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)  
            query_img_phase = torch.roll(query_img_phase, shifts=shifts, dims=2)  
            
        label = torch.tensor(idx, dtype=torch.long)  
        
        return query_img, reference_img, query_img_phase, reference_img_phase, label
    
    def __len__(self):
        return len(self.samples)
        
        
            
    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            idx_pool = copy.deepcopy(self.train_ids)
        
            neighbour_split = neighbour_select // 2
            
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)
                
            # Shuffle pairs order
            random.shuffle(idx_pool)
           
            # Lookup if already used in epoch
            idx_epoch = set()   
            idx_batch = set()
     
            # buckets
            batches = []
            current_batch = []
            
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()
    
            while True:
                
                pbar.update()
                
                if len(idx_pool) > 0:
                    idx = idx_pool.pop(0)

                    
                    if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:
                    
                        idx_batch.add(idx)
                        current_batch.append(idx)
                        idx_epoch.add(idx)
                        break_counter = 0
                      
                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                            
                            near_similarity = similarity_pool[idx][:neighbour_range]
                            
                            near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
                            
                            far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
                            
                            random.shuffle(far_neighbours)
                            
                            far_neighbours = far_neighbours[:neighbour_split]
                            
                            near_similarity_select = near_neighbours + far_neighbours
                            
                            for idx_near in near_similarity_select:
                           
                                # check for space in batch
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break
                                
                                # check if idx not already in batch or epoch
                                if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near:
                            
                                    idx_batch.add(idx_near)
                                    current_batch.append(idx_near)
                                    idx_epoch.add(idx_near)
                                    similarity_pool[idx].remove(idx_near)
                                    break_counter = 0
                                    
                    else:
                        # if idx fits not in batch and is not already used in epoch -> back to pool
                        if idx not in idx_batch and idx not in idx_epoch:
                            idx_pool.append(idx)
                            
                        break_counter += 1
                        
                    if break_counter >= 1024:
                        break
                   
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []

            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            print("idx_pool:", len(idx_pool))
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.train_ids) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))  

            
       
class CVUSADatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms_geometric=None, 
                 transforms_rgb=None
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms_geometric = transforms_geometric
        self.transforms_rgb = transforms_rgb
        
        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        else:
            self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)
        
        self.df = self.df.rename(columns={0:"sat", 1:"ground", 2:"ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
    
        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values
            
        elif self.img_type == "query":
            self.images = self.df.ground.values
            self.label = self.df.idx.values 
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
                
    def get_spatial_phase_image(self, img_rgb):
        if isinstance(img_rgb, torch.Tensor):
            img_np = img_rgb.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = np.array(img_rgb)

        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        gray_channel = img_gray.astype(np.float32) / 255.0
        
        fft = np.fft.fft2(gray_channel)
        magnitude = np.abs(fft) + 1e-8
        phase_only_spectrum = fft / magnitude
        spatial_phase = np.fft.ifft2(phase_only_spectrum).real
        spatial_phase = np.abs(spatial_phase)
        mean_val = np.mean(spatial_phase)
        std_val = np.std(spatial_phase)
        threshold = mean_val + 3 * std_val
        spatial_phase = np.clip(spatial_phase, 0, threshold)
        
        p_min, p_max = spatial_phase.min(), spatial_phase.max()
        if p_max - p_min > 1e-8:
            phase_norm = (spatial_phase - p_min) / (p_max - p_min)
        else:
            phase_norm = np.zeros_like(spatial_phase)
        
        mixed_img = np.stack([phase_norm, phase_norm, gray_channel], axis=-1)
        
        return mixed_img.astype(np.float32)

    def __getitem__(self, index):
        
        img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_phase = self.get_spatial_phase_image(img)
        
        # image transforms
        if self.transforms_geometric is not None:
            transformed_rgb = self.transforms_geometric(image=img)
            img = transformed_rgb['image']
            replay_params = transformed_rgb['replay']
                
            # # 对 Phase 重放变换
            transformed_phase = A.ReplayCompose.replay(replay_params, image=img_phase)
            img_phase = transformed_phase['image']

        if self.transforms_rgb is not None:
            img = self.transforms_rgb(image=img)['image']
            
        img_phase = torch.from_numpy(img_phase).permute(2, 0, 1)
        img_phase = (img_phase - 0.5) / 0.5
        
        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, img_phase, label

    def __len__(self):
        return len(self.images)

            





