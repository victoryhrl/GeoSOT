import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from geosot.GeoSOT_v import GeoSOT
from geosot.dataset.vigor import VigorDatasetEval
from geosot.transforms import get_transforms_val
from geosot.temp_eval_vigor import evaluate, evaluate_localization


@dataclass
class Configuration:

    # Model
    model: str = '/app/project/Sample4Geo/dinov3-convnext-base-pretrain-lvd1689m'
    
    # Override model image size
    img_size: int = 384
    sat_img_size: int = 640
    ground_resolution: float = 0.114
    
    # Evaluation
    batch_size_eval: int = 32
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    
    # Dataset
    data_folder = "/app/data/VIGOR"
    same_area: bool = True             # True: same | False: cross
    ground_cutting = 0                 # cut ground upper and lower
    

    checkpoint_start = None
  
    # set num_workers to 0 if on Windows
    num_workers: int = 16 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = GeoSOT(model_path=config.model, img_size=config.img_size)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    
    image_size_sat = (img_size, img_size)

    new_width = img_size*2    
    new_hight = int(((1024 - 2 * config.ground_cutting) / 2048) * new_width)
    img_size_ground = (new_hight, new_width)
     
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        checkpoint = torch.load(config.checkpoint_start, map_location=config.device, weights_only=False)
        
        # 从字典中提取 model_state_dict
        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    sat_geom_val, ground_geom_val, rgb_final_val = get_transforms_val(image_size_sat,
                                                                      img_size_ground,
                                                                      mean=mean,
                                                                      std=std,
                                                                      ground_cutting=config.ground_cutting)

    # Reference Satellite Images Test
    reference_dataset_test = VigorDatasetEval(data_folder=config.data_folder,
                                              split="test",
                                              img_type="reference",
                                              same_area=config.same_area,
                                              transforms_geometric=sat_geom_val,  # Use satellite geometric transforms
                                              transforms_rgb=rgb_final_val
                                              )

    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=False)

    # Query Ground Images Test
    query_dataset_test = VigorDatasetEval(data_folder=config.data_folder,
                                          split="test",
                                          img_type="query",
                                          same_area=config.same_area,
                                          transforms_geometric=ground_geom_val,  # Use ground geometric transforms
                                          transforms_rgb=rgb_final_val,
                                          )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=False)

    localization_eval_dataset = VigorDatasetEval(data_folder=config.data_folder,
                                                 split="test",
                                                 img_type="query_and_ref_pair",
                                                 same_area=config.same_area,
                                                 transforms_query_geometric=ground_geom_val,
                                                 transforms_reference_geometric=sat_geom_val,
                                                 transforms_rgb=rgb_final_val,
                                                 )

    localization_eval_dataloader = DataLoader(
        localization_eval_dataset, batch_size=config.batch_size_eval, num_workers=config.num_workers,
        shuffle=False, pin_memory=False
    )

    print("Query Images Test:", len(query_dataset_test))
    print("Reference Images Test:", len(reference_dataset_test))
    print("Localization Eval Pairs:", len(localization_eval_dataset))
    

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#

    print("\n{}[{}]{}".format(30*"-", "VIGOR Same", 30*"-"))  

    r1_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=reference_dataloader_test,
                       query_dataloader=query_dataloader_test, 
                       localization_dataloader=localization_eval_dataloader,
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True,
                       rerank=False,
                       topk_candidates=10)
