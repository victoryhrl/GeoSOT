import cv2
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
from collections import defaultdict
import time
import torchvision.transforms as T
import albumentations as A


from geosot.robustness_transforms import get_dark_transform, get_blur_noise_transform, AddGaussianNoise, \
    get_low_resolution_transform, get_rain_transform, get_snow_transform


class VigorDatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
                 same_area=True,
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
        self.phase_data_folder = '/app/data/VIGOR_phase'
        self.sat_img_size = 640
        self.target_sat_size = 384
        
        self.transforms_query_geometric = transforms_query_geometric
        self.transforms_reference_geometric = transforms_reference_geometric
        self.transforms_rgb = transforms_rgb
        
        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            self.cities = ['NewYork', 'Seattle'] 

        # load sat list 
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/satellite_list.txt', header=None, delim_whitespace=True)
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))
        
        
        # ground dependent on mode 'train' or 'test'
        ground_list = []
        for city in self.cities:

            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_train.txt', header=None, delim_whitespace=True)
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, delim_whitespace=True)
            
            df_tmp = df_tmp.loc[:, [0, 1, 2, 3, 4, 7, 10]].rename(columns={0:  "ground",
                                                                     1:  "sat",
                                                                     2:  "off_x",
                                                                     3:  "off_y",
                                                                     4:  "sat_np1",
                                                                     7:  "sat_np2",
                                                                     10: "sat_np3"})
            
            df_tmp["path_ground"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/panorama/{x.ground}', axis=1)
            df_tmp["path_sat"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1)
            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
                
            ground_list.append(df_tmp) 
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)
        
        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))
                
      
        self.pairs = list(zip(self.df_ground.index, self.df_ground.sat))
        self.idx2pairs = defaultdict(list)
        
        # for a unique sat_id we can have 1 or 2 ground views as gt
        for pair in self.pairs:      
            self.idx2pairs[pair[1]].append(pair)
            
            
        self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 
        
        self.samples = copy.deepcopy(self.pairs)
        self.to_tensor = T.ToTensor()

    def phase_img_getter(self, img):
        """生成相位图像（适配通道优先）"""
        # 输入处理
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()  # 从 (C, H, W) 转为 (H, W, C)
        else:
            img_np = np.array(img)  # PIL.Image的shape是 (H, W, C)

        # 计算FFT相位
        img_fft = np.fft.fft2(img_np, axes=(0, 1))  # 在空间维度计算FFT
        img_pha = np.angle(img_fft)  # 形状 (H, W, C)

        # 调整img_pha为通道优先 (C, H, W)
        img_pha = np.transpose(img_pha, (2, 0, 1))  # 新形状 (3, H, W)

        # 缩放因子形状 (3, 1, 1) [C, H, W]
        scale_factor = np.array([50000, 50000, 50000]).reshape(3, 1, 1)

        # 计算相位图像
        phase_img = scale_factor * (np.e ** (1j * img_pha))  # 广播成功
        phase_img = np.real(np.fft.ifft2(phase_img, axes=(1, 2)))  # 在H,W维度反变换

        # 调整回通道最后并转换为PIL图像
        phase_img = np.transpose(phase_img, (1, 2, 0))  # (H, W, C)
        phase_img = np.clip(phase_img, 0, 255).astype(np.uint8)
        phase_img = Image.fromarray(phase_img)

        # 转换为张量 (C, H, W)
        phase_img = self.to_tensor(phase_img)
        return phase_img

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
            
    def phase_angle(self, img):
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(img)
            
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        img_fft = np.fft.fft2(img_gray)
        img_pha = np.angle(img_fft)  # 形状 (H, W), 范围 [-pi, pi]
        
        phase_map = (img_pha + np.pi) / (2 * np.pi)  # 形状 (H, W), 范围 [0, 1]
        
        phase_map_3channel = np.stack([phase_map] * 3, axis=-1)
        return phase_map_3channel.astype(np.float32)

    def phase_sincos(self, img): 
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_fft = np.fft.fft2(img_gray)
        img_pha = np.angle(img_fft)
        phase_sin = (np.sin(img_pha) + 1) / 2.0
        phase_cos = (np.cos(img_pha) + 1) / 2.0
        gray_norm = img_gray / 255.0
        phase_map_float = np.stack([phase_sin, phase_cos, gray_norm], axis=-1)
        return phase_map_float.astype(np.float32)

    def __getitem__(self, index):
        
        idx_ground, idx_sat = self.samples[index]
        ground_row = self.df_ground.loc[idx_ground]
        # print(self.idx2ground_path[idx_ground])
        
        # load query -> ground image
        query_img_path = self.idx2ground_path[idx_ground]
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img_path = self.idx2sat_path[idx_sat]
        reference_img = cv2.imread(reference_img_path)
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        query_img_phase = self.get_spatial_phase_image(query_img)
        reference_img_phase = self.get_spatial_phase_image(reference_img)

        offset_x_px_orig = ground_row['off_x']
        offset_y_px_orig = ground_row['off_y']
        
        scale = self.target_sat_size / self.sat_img_size
        offset_x_px_scaled = offset_x_px_orig * scale
        offset_y_px_scaled = offset_y_px_orig * scale
        offset_gt_scaled_px = torch.tensor([offset_x_px_scaled, offset_y_px_scaled], dtype=torch.float)

            
        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)
            query_img_phase = cv2.flip(query_img_phase, 1)
            reference_img_phase = cv2.flip(reference_img_phase, 1)
            offset_gt_scaled_px[0] *= -1

        
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
            # reference_img_tmp = torch.rot90(reference_img_tmp, k=r, dims=(1, 2)) 
            reference_img_phase = torch.rot90(reference_img_phase, k=r, dims=(1, 2)) 
            if r == 1: offset_gt_scaled_px = torch.tensor([offset_gt_scaled_px[1], -offset_gt_scaled_px[0]])
            elif r == 2: offset_gt_scaled_px = -offset_gt_scaled_px
            elif r == 3: offset_gt_scaled_px = torch.tensor([-offset_gt_scaled_px[1], offset_gt_scaled_px[0]])
            
            # use roll for ground view if rotate sat view
            c, h, w = query_img.shape
            shifts = - w//4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)   
            # query_img_tmp = torch.roll(query_img_tmp, shifts=shifts, dims=2)   
            query_img_phase = torch.roll(query_img_phase, shifts=shifts, dims=2)   
                   
            
        label = torch.tensor(idx_sat, dtype=torch.long)
        
        return query_img, reference_img, label, query_img_phase, reference_img_phase, offset_gt_scaled_px
    
    def __len__(self):
        return len(self.samples)
        
        
            
    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            pair_pool = copy.deepcopy(self.pairs)
            idx2pair_pool = copy.deepcopy(self.idx2pairs)
            
            neighbour_split = neighbour_select // 2
                        
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)                
            
            # Shuffle pairs order
            random.shuffle(pair_pool)
           
            
            # Lookup if already used in epoch
            pairs_epoch = set()   
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
                
                if len(pair_pool) > 0:
                    pair = pair_pool.pop(0)
                    
                    _, idx = pair
                    
                    if idx not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                        
                        idx_batch.add(idx)
                        current_batch.append(pair)
                        pairs_epoch.add(pair)
                        
                        # remove from pool used for sim-sampling
                        idx2pair_pool[idx].remove(pair)

                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                            
                            near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])
                            near_always = copy.deepcopy(near_similarity[:neighbour_split]) 
                            near_random = copy.deepcopy(near_similarity[neighbour_split:])
                            random.shuffle(near_random)
                            near_random = near_random[:neighbour_split]
                            near_similarity_select = near_always + near_random

                            
                            for idx_near in near_similarity_select:
                            
                            
                                # check for space in batch
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break
                            
                                # no check for pair in epoch necessary cause all we add is removed from pool
                                if idx_near not in idx_batch:
                            
                                    near_pairs = copy.deepcopy(idx2pair_pool[idx_near])
                                    
                                    # up to 2 for one sat view 
                                    random.shuffle(near_pairs)
                                
                                    for near_pair in near_pairs:
                                                                                    
                                        idx_batch.add(idx_near)
                                        current_batch.append(near_pair)
                                        pairs_epoch.add(near_pair)
                                        
                                        idx2pair_pool[idx_near].remove(near_pair)
                                        similarity_pool[idx].remove(idx_near)
                                        
                                        # only select one view
                                        break
                             
                        break_counter = 0
                        
                    else:
                        # if pair fits not in batch and is not already used in epoch -> back to pool
                        if pair not in pairs_epoch:
                            pair_pool.append(pair)
                            
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
            print("pair_pool:", len(pair_pool))
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))  

       
class VigorDatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 same_area=True,
                 transforms_geometric=None, # For 'query' or 'reference' mode
                 transforms_rgb=None,       # For all modes
                 transforms_query_geometric=None, # For 'query_and_ref_pair' mode
                 transforms_reference_geometric=None
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.to_tensor = T.ToTensor()
        self.split = split
        self.img_type = img_type
        self.phase_data_folder = '/app/data/VIGOR_phase'
        self.sat_img_size = 640
        self.target_sat_size = 384
        self.transforms_geometric = transforms_geometric
        self.transforms_rgb = transforms_rgb
        self.transforms_query_geometric = transforms_query_geometric
        self.transforms_reference_geometric = transforms_reference_geometric
            
        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            if split == "train":
                self.cities = ['NewYork', 'Seattle'] 
            else:
                self.cities = ['Chicago', 'SanFrancisco'] 
               
        # load sat list 
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/satellite_list.txt', header=None, delim_whitespace=True)
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))
        
        
        # ground dependent on mode 'train' or 'test'
        ground_list = []
        for city in self.cities:
            
            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_{split}.txt', header=None, delim_whitespace=True)
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, delim_whitespace=True)
  
            
            df_tmp = df_tmp.loc[:, [0, 1, 2, 3, 4, 7, 10]].rename(columns={0:  "ground",
                                                                     1:  "sat",
                                                                     2:  "off_x",
                                                                     3:  "off_y",
                                                                     4:  "sat_np1",
                                                                     7:  "sat_np2",
                                                                     10: "sat_np3"})
            
            df_tmp["path_ground"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/panorama/{x.ground}', axis=1)
            df_tmp["path_sat"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1)
            
            df_tmp["path_sat_np1"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat_np1}', axis=1)
            df_tmp["path_sat_np2"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat_np2}', axis=1)
            df_tmp["path_sat_np3"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat_np3}', axis=1)

            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
                
            ground_list.append(df_tmp) 
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)
        
        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))
        
        
        if self.img_type == "reference":
            if split == "train":
                # only sat images we really train on
                self.label = self.df_ground["sat"].unique()
                self.images = []
                for idx in self.label:
                    self.images.append(self.idx2sat_path[idx])
            else:
                # all sat images of cities in split
                self.images = self.df_sat["path"].values
                self.label = self.df_sat.index.values
            
        elif self.img_type == "query":
            self.images = self.df_ground["path_ground"].values
            self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 

        elif self.img_type == "query_and_ref_pair":
            pass # Length is determined by df_ground

        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")

    def phase_img_getter(self, img):
        """生成相位图像（适配通道优先）"""
        # 输入处理
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()  # 从 (C, H, W) 转为 (H, W, C)
        else:
            img_np = np.array(img)  # PIL.Image的shape是 (H, W, C)

        # 计算FFT相位
        img_fft = np.fft.fft2(img_np, axes=(0, 1))  # 在空间维度计算FFT
        img_pha = np.angle(img_fft)  # 形状 (H, W, C)

        # 调整img_pha为通道优先 (C, H, W)
        img_pha = np.transpose(img_pha, (2, 0, 1))  # 新形状 (3, H, W)

        # 缩放因子形状 (3, 1, 1) [C, H, W]
        scale_factor = np.array([50000, 50000, 50000]).reshape(3, 1, 1)

        # 计算相位图像
        phase_img = scale_factor * (np.e ** (1j * img_pha))  # 广播成功
        phase_img = np.real(np.fft.ifft2(phase_img, axes=(1, 2)))  # 在H,W维度反变换

        # 调整回通道最后并转换为PIL图像
        phase_img = np.transpose(phase_img, (1, 2, 0))  # (H, W, C)
        phase_img = np.clip(phase_img, 0, 255).astype(np.uint8)
        phase_img = Image.fromarray(phase_img)

        # 转换为张量 (C, H, W)
        phase_img = self.to_tensor(phase_img)
        return phase_img
    
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

    def phase_angle(self, img):
        """
        直接将相位角作为特征输入，通过线性映射和通道复制。
        """
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(img)
            
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        img_fft = np.fft.fft2(img_gray)
        img_pha = np.angle(img_fft)  # 形状 (H, W), 范围 [-pi, pi]
        
        phase_map = (img_pha + np.pi) / (2 * np.pi)  # 形状 (H, W), 范围 [0, 1]
        
        phase_map_3channel = np.stack([phase_map] * 3, axis=-1)
        return phase_map_3channel.astype(np.float32)

    def phase_sincos(self, img): 
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_fft = np.fft.fft2(img_gray)
        img_pha = np.angle(img_fft)
        phase_sin = (np.sin(img_pha) + 1) / 2.0
        phase_cos = (np.cos(img_pha) + 1) / 2.0
        gray_norm = img_gray / 255.0
        phase_map_float = np.stack([phase_sin, phase_cos, gray_norm], axis=-1)
        return phase_map_float.astype(np.float32)

    def __getitem__(self, index):

        if self.img_type == "query_and_ref_pair":
            ground_row = self.df_ground.loc[index]
            query_img_path = self.idx2ground_path[index]
            ref_idx = ground_row['sat']
            reference_img_path = self.idx2sat_path[ref_idx]
            
            query_img = cv2.cvtColor(cv2.imread(query_img_path), cv2.COLOR_BGR2RGB)
            reference_img = cv2.cvtColor(cv2.imread(reference_img_path), cv2.COLOR_BGR2RGB)

            query_img_phase = self.get_spatial_phase_image(query_img)
            reference_img_phase = self.get_spatial_phase_image(reference_img)

            # For eval, provide the scaled pixel offset for direct comparison in the evaluator
            offset_x_px_orig, offset_y_px_orig = ground_row['off_x'], ground_row['off_y']
            scale = self.target_sat_size / self.sat_img_size
            offset_gt_scaled_px = torch.tensor([offset_x_px_orig * scale, offset_y_px_orig * scale], dtype=torch.float)

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

            return query_img, reference_img, query_img_phase, reference_img_phase, offset_gt_scaled_px

        else:
            img_path = self.images[index]
            label = self.label[index]

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_phase = self.get_spatial_phase_image(img)
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
            #img = self.robustness_transform_tensor(img)
            label = torch.tensor(label, dtype=torch.long)

            return img, label, img_phase

    def __len__(self):
        if self.img_type == "query_and_ref_pair":
            return len(self.df_ground)
        return len(self.images)