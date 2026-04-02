import os
import sys
import random
import errno
import time
import torch
import numpy as np
from datetime import timedelta

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def setup_system(seed, cudnn_benchmark=True, cudnn_deterministic=True) -> None:
    '''
    Set seeds for for reproducible training
    '''
    # python
    random.seed(seed)
    
    # numpy
    np.random.seed(seed)
    
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
      
        
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def sec_to_min(seconds):
    
    seconds = int(seconds)
    minutes = seconds // 60
    seconds_remaining = seconds % 60
    
    if seconds_remaining < 10:
        seconds_remaining = '0{}'.format(seconds_remaining)
    
    return '{}:{}'.format(minutes, seconds_remaining)

def sec_to_time(seconds):
    return "{:0>8}".format(str(timedelta(seconds=int(seconds))))

def print_time_stats(t_train_start, t_epoch_start, epochs_remaining, steps_per_epoch):
    
    elapsed_time = time.time() - t_train_start
    speed_epoch = time.time() - t_epoch_start 
    speed_batch = speed_epoch / steps_per_epoch
    eta = speed_epoch * epochs_remaining
        
    print("Elapsed {}, {} time/epoch, {:.2f} s/batch, remaining {}".format(
                sec_to_time(elapsed_time), sec_to_time(speed_epoch), speed_batch, sec_to_time(eta)))
    
class EarlyStopping:
    
    def __init__(self, patience=7, verbose=True, delta=0, trace_func=print):
        """
        Args:
            patience (int): Number of evaluation cycles to tolerate no performance improvement.
            verbose (bool): Whether to print status information.
            delta (float): Minimum change to qualify as an improvement.
            trace_func (function): Print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_score):
        """
        Call this object with the new validation score.

        Returns:
            bool: Return True if the score is the new best score.
        """
        score = val_score
        is_best = False

        if self.best_score is None:
            self.best_score = score
            is_best = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} (Best: {self.best_score:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                self.trace_func(f'Validation score improved ({self.best_score:.4f} --> {score:.4f}).')
            self.best_score = score
            self.counter = 0
            is_best = True
        
        return is_best

def get_parameter_groups(
    model,
    base_lr=1e-4,
    weight_decay=1e-5,
    layer_decay=0.8,
    skip_list=(),
    verbose=True
):

    num_backbone_layers = 5  # stem + 4 stages

    # backbone: smaller lr for lower layers
    layer_scales = {
        i: layer_decay ** (num_backbone_layers - i)
        for i in range(num_backbone_layers)
    }
    layer_scales[5] = 1.0  # head MUST be base LR

    if verbose:
        print("LLRD Layer Scales:")
        for k, v in layer_scales.items():
            print(f"  Layer {k}: {v:.6f}")

    param_groups = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Matches both model_rgb and model_phase
        if "stem" in name:
            layer_id = 0
        elif "stages.0" in name:
            layer_id = 1
        elif "stages.1" in name:
            layer_id = 2
        elif "stages.2" in name:
            layer_id = 3
        elif "stages.3" in name:
            layer_id = 4
        else:
            layer_id = 5  # head / projection

        no_decay = (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or name in skip_list
        )

        group_key = (layer_id, no_decay)

        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [],
                "lr": base_lr * layer_scales[layer_id],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "layer_id": layer_id,
            }

        param_groups[group_key]["params"].append(param)

    sorted_groups = sorted(param_groups.values(), key=lambda x: x["layer_id"])
    
    final_groups = []
    for g in sorted_groups:
        final_groups.append({
            "params": g["params"],
            "weight_decay": g["weight_decay"],
            "lr": g["lr"],
            "name": f"layer_{g['layer_id']}_{'no_decay' if g['weight_decay']==0 else 'decay'}"
        })

    return final_groups