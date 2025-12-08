# stock_forecast/utils.py
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def device_info():
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU"