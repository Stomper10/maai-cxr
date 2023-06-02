import os
import torch
from datetime import datetime

def make_dir(directory:str):
    if not os.path.exists(directory):
        os.makedirs(directory) 

class CFG:
    def __init__(self):
        return
configs = CFG()
configs.dataset_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-small'

configs.batch_size = 4
configs.learning_rate = 0.0001
configs.weight_decay = 0.00005

configs.num_classes = 14

configs.image_size = (1024, 1024)
configs.image_add_size = 64


configs.backbone = 'efficientnetv2_s'
configs.result_dir = './results'
make_dir(configs.result_dir)
configs.wandb_name = f"{configs.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
configs.wandb_project = 'AAI'

configs.num_devices = torch.cuda.device_count()
configs.num_workers = 4

configs.num_epochs = 30