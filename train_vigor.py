import os
import time
import math
import shutil
import sys
import torch
import pickle
from dataclasses import dataclass
# Use new API
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup

from geosot.dataset.vigor import VigorDatasetEval, VigorDatasetTrain
from geosot.transforms import get_transforms_train, get_transforms_val
from geosot.utils import get_parameter_groups, setup_system, Logger, EarlyStopping
from geosot.temp_trainer import train
from geosot.temp_eval_vigor import evaluate, calc_sim, evaluate_localization
from geosot.loss import InfoNCE
from geosot.model import TimmModel
from geosot.GeoSOT_v import GeoSOT


@dataclass
class Configuration:
    # Model
    model: str = '/app/project/Sample4Geo/dinov3-convnext-base-pretrain-lvd1689m'
    img_size: int = 384
    sat_img_size: int = 640

    # Training
    mixed_precision: bool = True
    seed = 1
    epochs: int = 100
    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    verbose: bool = True
    gpu_ids: tuple = (0,)

    # Similarity Sampling
    custom_sampling: bool = True
    gps_sample: bool = True
    sim_sample: bool = True
    neighbour_select: int = 64
    neighbour_range: int = 128
    gps_dict_path: str = "./data/VIGOR/gps_dict_cross.pkl"  # gps_dict_cross.pkl | gps_dict_same.pkl

    # Eval
    batch_size_eval: int = 32
    eval_every_n_epoch: int = 4  # eval every n Epoch
    normalize_features: bool = True

    # Optimizer
    clip_grad = 1.0  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing

    # Lossq
    label_smoothing: float = 0.1
    weight_decay = 0.05
    layer_decay: float = 0.8

    # Learning Rate 2e-5
    lr: float = 2e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 2 
    lr_end: float = 1e-6
    ground_resolution = 0.114
    lambda_reg = 0.03

    # Dataset
    data_folder = "/app/data/VIGOR"
    same_area: bool = False  # True: same | False: cross
    ground_cutting = 0  # cut ground upper and lower

    # Augment Images
    prob_rotate: float = 0.75
    prob_flip: float = 0.5

    model_path: str = "./vigor_cross_phase"

    # Eval before training
    zero_shot: bool = False

    early_stopping_patience: int = 5


    checkpoint_start = None
    # set num_workers to 0 for debugging, otherwise 4 is good
    num_workers: int = 16

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#

config = Configuration()

if __name__ == '__main__':

    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        verbose=True
    )

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))
    model = GeoSOT(model_path=config.model, img_size=config.img_size)

    # CRITICAL: Add the learnable logit_scale parameter to the model
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    print(r"mean is {}, std is {}", mean, std)

    image_size_sat = (img_size, img_size)

    new_width = img_size * 2
    new_hight = int(((1024 - 2 * config.ground_cutting) / 2048) * new_width)
    img_size_ground = (new_hight, new_width)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

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

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Transforms
    sat_geom_train, ground_geom_train, rgb_color_train = get_transforms_train(image_size_sat,
                                                                         img_size_ground,
                                                                         mean=mean,
                                                                         std=std,
                                                                         ground_cutting=config.ground_cutting)

    # Train
    train_dataset = VigorDatasetTrain(data_folder=config.data_folder,
                                      same_area=config.same_area,
                                      transforms_query_geometric=ground_geom_train,
                                      transforms_reference_geometric=sat_geom_train,
                                      transforms_rgb=rgb_color_train,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size
                                      )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=False)

    # Eval
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
                                              transforms_geometric=sat_geom_val, # Use satellite geometric transforms
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
                                          transforms_geometric=ground_geom_val, # Use ground geometric transforms
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
    # print("Localization Eval Pairs:", len(localization_eval_dataset))

    # -----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    # -----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    # -----------------------------------------------------------------------------#
    # Sim Sample + Eval on Train                                                  #
    # -----------------------------------------------------------------------------#

    if config.sim_sample:
        # Query Ground Images Train for simsampling
        query_dataset_train = VigorDatasetEval(data_folder=config.data_folder,
                                               split="train",
                                               img_type="query",
                                               same_area=config.same_area,
                                               transforms_geometric=ground_geom_val,
                                               transforms_rgb=rgb_final_val,
                                               )

        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=False)

        # Reference Satellite Images Train for simsampling
        reference_dataset_train = VigorDatasetEval(data_folder=config.data_folder,
                                                   split="train",
                                                   img_type="reference",
                                                   same_area=config.same_area,
                                                   transforms_geometric=sat_geom_val,
                                                   transforms_rgb=rgb_final_val,
                                                   )

        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=False)

        print("\nQuery Images Train:", len(query_dataset_train))
        print("Reference Images Train (unique):", len(reference_dataset_train))

    # -----------------------------------------------------------------------------#
    # Loss                                                                        #
    # -----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )

    if config.mixed_precision:
        # Use new API
        scaler = GradScaler(init_scale=2. ** 10)
    else:
        scaler = None

    # -----------------------------------------------------------------------------#
    # optimizer                                                                   #
    # -----------------------------------------------------------------------------#

    if config.layer_decay < 1.0:
        print(f"\nUsing Layer-wise Learning Rate Decay with decay rate: {config.layer_decay}")
        optimizer_parameters = get_parameter_groups(
            model, 
            weight_decay=config.weight_decay, 
            skip_list=("bias", "LayerNorm.bias", "norm.bias", "norm.weight"),
            base_lr=config.lr,
            layer_decay=config.layer_decay
        )
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)

    elif config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay, # Use config.weight_decay instead of hardcoded 0.01
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # -----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    # -----------------------------------------------------------------------------#

    # Adjust steps for gradient accumulation
    train_steps = len(train_dataloader) // config.gradient_accumulation_steps * config.epochs
    warmup_steps = len(train_dataloader) // config.gradient_accumulation_steps * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    # -----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    # -----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30 * "-", "Zero Shot", 30 * "-"))

        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train,
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          cleanup=True)

    # -----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    # -----------------------------------------------------------------------------#
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)

    # -----------------------------------------------------------------------------#
    # Train                                                                       #
    # -----------------------------------------------------------------------------#
    start_epoch = 1
    best_score = 0

    if config.checkpoint_start is not None:
        if os.path.isfile(config.checkpoint_start):
            print("=> Resuming from checkpoint '{}'".format(config.checkpoint_start))
            checkpoint = torch.load(config.checkpoint_start, map_location=config.device, weights_only=False)

            # 1. Restore starting Epoch
            start_epoch = checkpoint['epoch'] + 1

            # 2. Restore model weights
            #    Need to handle 'module.' prefix potentially added by DataParallel
            model_state_dict = checkpoint['model_state_dict']
            is_data_parallel_checkpoint = all(k.startswith('module.') for k in model_state_dict)
            is_currently_data_parallel = isinstance(model, torch.nn.DataParallel)

            if is_data_parallel_checkpoint and not is_currently_data_parallel:
                # If checkpoint is saved with multi-GPU but currently loaded with single-GPU, remove prefix
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            elif not is_data_parallel_checkpoint and is_currently_data_parallel:
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    name = 'module.' + k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            else:
                # State is consistent, load directly
                model.load_state_dict(model_state_dict)

            # 3. Restore optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 4. Restore scheduler state
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            early_stopping.best_score = checkpoint.get('best_score', None)

            print("=> Loaded checkpoint '{}' (epoch {})".format(config.checkpoint_start, checkpoint['epoch']))
            if early_stopping.best_score is not None:
                print(f"=> Resumed with best score so far: {early_stopping.best_score:.4f}")

        else:
            print("=> No checkpoint found at '{}'".format(config.checkpoint_start))

    for epoch in range(start_epoch, config.epochs + 1):

        print("\n{}[Epoch: {}]{}".format(30 * "-", epoch, 30 * "-"))
        # if epoch < config.lambda_reg_warmup_epochs:
        #     # Linearly increase lambda_reg from 0 to its final value over the warmup period
        #     current_lambda_reg = config.lambda_reg * (epoch / config.lambda_reg_warmup_epochs)
        #     # current_lambda_reg = 0.001
        # else:
        #     # After warmup, use the final value
        #     current_lambda_reg = config.lambda_reg
        
        # print(f"Current regression loss weight (lambda_reg): {current_lambda_reg:.4f}")
        warmup_epochs = 8
        lambda_floor = 0.001
        lambda_max = config.lambda_reg
        if epoch <= warmup_epochs:
           progress = epoch / warmup_epochs
           current_lambda_reg = lambda_floor + (lambda_max - lambda_floor) * (progress ** 2)
        else:
           current_lambda_reg = lambda_max

        # The call remains the same, but the logic is now handled by the modified trainer.py
        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler,
                           current_lambda_reg = current_lambda_reg)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30 * "-", "Evaluate Retrieval", 30 * "-"))

            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test,
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)

            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config,
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train,
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True)
            
            # print("\n{}[{}]{}".format(30 * "-", "[Evaluate Fine-Grained Localization]", 30 * "-"))
            # mean_err, median_err = evaluate_localization(
            #     config=config, model=model, dataloader=localization_eval_dataloader
            # )

            is_best = early_stopping(r1_test)

            if is_best:
                print(f"New best score: {r1_test:.4f}. Saving model for epoch {epoch}...")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model,
                                                                                torch.nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_score': early_stopping.best_score,
                }
                save_path = '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test)
                torch.save(checkpoint, save_path)

            if early_stopping.early_stop:
                print("Early stopping triggered. Halting training.")
                break  # Exit training loop
            # if r1_test > best_score:

            #     best_score = r1_test

            #     if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
            #         torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
            #     else:
            #         torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)
    if early_stopping.best_score is not None:
        print(f"Best validation score achieved: {early_stopping.best_score:.4f}")
    # if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    #     torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    # else:
    #     torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
