import cv2
import numpy as np
from torch.utils.data import Dataset
import random
import copy
import torch
from tqdm import tqdm
import time
import scipy.io as sio
import os
from glob import glob
import albumentations as A

class CVACTDatasetTrain(Dataset):
    
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
        
        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')
        
        ids = anuData['panoIds']

        train_ids = ids[anuData['trainSet'][0][0][1]-1]
        
        train_ids_list = []
        train_idsnum_list = []
        self.idx2numidx = dict()
        self.numidx2idx = dict()
        self.idx_ignor = set()
        i = 0

        for idx in train_ids.squeeze():
            
            idx = str(idx)
            
            grd_path = f'ANU_data_small/streetview/{idx}_grdView.jpg'
            sat_path = f'ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
            
            if not os.path.exists(f'{self.data_folder}/{grd_path}') or not os.path.exists(f'{self.data_folder}/{sat_path}'):
                self.idx_ignor.add(idx)
            else:
                self.idx2numidx[idx] = i
                self.numidx2idx[i] = idx
                train_ids_list.append(idx)
                train_idsnum_list.append(i)
                i+=1
        
        print("IDs not found in train images:", self.idx_ignor)
        
        self.train_ids = train_ids_list
        self.train_idsnum = train_idsnum_list
        self.samples = copy.deepcopy(self.train_idsnum)
            
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
        
        idnum = self.samples[index]
        
        idx = self.numidx2idx[idnum]
        
        # load query -> ground image
        query_img = cv2.imread(f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg')
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
                   
            
        label = torch.tensor(idnum, dtype=torch.long)  
        
        return query_img, reference_img, query_img_phase, reference_img_phase, label
    
    def __len__(self):
        return len(self.samples)
        
        
            
    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            idx_pool = copy.deepcopy(self.train_idsnum)
        
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
                      
                        # check if near sat views within margine
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
                                
                                # check if idx not already in batch or epoch and not in ignor list (missing image)
                                if idx_near not in idx_batch and idx_near not in idx_epoch:
                            
                                    idx_batch.add(idx_near)
                                    current_batch.append(idx_near)
                                    idx_epoch.add(idx_near)
                                    similarity_pool[idx].remove(idx_near)
                                    break_counter = 0
                                    
                    else:
                        # if idx fits not in batch and is not already used in epoch -> back to pool
                        if idx not in idx_epoch:
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

            
       
class CVACTDatasetEval(Dataset):
    
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
        
        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')
        
        ids = anuData['panoIds']
        
        if split != "train" and split != "val":
            raise ValueError("Invalid 'split' parameter. 'split' must be 'train' or 'val'")  
            
        if img_type != 'query' and img_type != 'reference':
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")


        ids = ids[anuData[f'{split}Set'][0][0][1]-1]
        
        ids_list = []
       
        self.idx2label = dict()
        self.idx_ignor = set()
        
        i = 0
        
        for idx in ids.squeeze():
            
            idx = str(idx)
            
            grd_path = f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg'
            sat_path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
   
            if not os.path.exists(grd_path) or not os.path.exists(sat_path):
                self.idx_ignor.add(idx)
            else:
                self.idx2label[idx] = i
                ids_list.append(idx)
                i+=1
        
        #print(f"IDs not found in {split} images:", self.idx_ignor)

        self.samples = ids_list

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
        
        idx = self.samples[index]
        
        if self.img_type == "reference":
            path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
        elif self.img_type == "query":
            path = f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_phase = self.get_spatial_phase_image(img)
        
        # image transforms
        if self.transforms_geometric is not None:
            transformed_rgb = self.transforms_geometric(image=img)
            img = transformed_rgb['image']
            replay_params = transformed_rgb['replay']
                
            # 对 Phase 重放变换
            transformed_phase = A.ReplayCompose.replay(replay_params, image=img_phase)
            img_phase = transformed_phase['image']

        if self.transforms_rgb is not None:
            img = self.transforms_rgb(image=img)['image']
            
        img_phase = torch.from_numpy(img_phase).permute(2, 0, 1)
        img_phase = (img_phase - 0.5) / 0.5
            
        label = torch.tensor(self.idx2label[idx], dtype=torch.long)

        return img, img_phase, label

    def __len__(self):
        return len(self.samples)

            
class CVACTDatasetTest(Dataset):
    
    def __init__(self,
                 data_folder,
                 img_type,
                 transforms_geometric=None, 
                 transforms_rgb=None
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.img_type = img_type
        self.transforms_geometric = transforms_geometric
        self.transforms_rgb = transforms_rgb
        
        files_sat = glob(f'{self.data_folder}/ANU_data_test/satview_polish/*_satView_polish.jpg')
        files_ground = glob(f'{self.data_folder}/ANU_data_test/streetview/*_grdView.jpg')
        
        sat_ids = []
        for path in files_sat:
        
            idx = path.split("/")[-1][:-19]
            sat_ids.append(idx)
        
        ground_ids = []
        for path in files_ground:
            idx = path.split("/")[-1][:-12]
            ground_ids.append(idx)  
            
        # only use intersection of sat and ground ids   
        test_ids = set(sat_ids).intersection(set(ground_ids))
        
        self.test_ids = list(test_ids)
        self.test_ids.sort()
        
        self.idx2num_idx = dict()
        
        for i, idx in enumerate(self.test_ids):
            self.idx2num_idx[idx] = i

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
        
        idx = self.test_ids[index]
        
        if self.img_type == "reference":
            path = f'{self.data_folder}/ANU_data_test/satview_polish/{idx}_satView_polish.jpg'
        else:
            path = f'{self.data_folder}/ANU_data_test/streetview/{idx}_grdView.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_phase = self.get_spatial_phase_image(img)
        
        # image transforms
        if self.transforms_geometric is not None:
            transformed_rgb = self.transforms_geometric(image=img)
            img = transformed_rgb['image']
            replay_params = transformed_rgb['replay']
                
            # 对 Phase 重放变换
            transformed_phase = A.ReplayCompose.replay(replay_params, image=img_phase)
            img_phase = transformed_phase['image']

        if self.transforms_rgb is not None:
            img = self.transforms_rgb(image=img)['image']
            
        img_phase = torch.from_numpy(img_phase).permute(2, 0, 1)
        img_phase = (img_phase - 0.5) / 0.5
            
        label = torch.tensor(self.idx2num_idx[idx], dtype=torch.long)

        return img, img_phase, label

    def __len__(self):
        return len(self.test_ids)