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
# configs.dataset_dir = f'/home/n1/gyuseonglee/workspace/datasets/chexpert-small'
configs.dataset_dir = f'/home/n1/gyuseonglee/workspace/datasets/chexpert/CheXpert-v1.0'
configs.dataset_name = 'CheXpert-v1.0-small'
configs.dataset_name = 'CheXpert-v1.0'

configs.batch_size = 32
configs.learning_rate = 0.0002
configs.weight_decay = 0.00001

configs.num_classes = 5 # target = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural', 'Effusion']

configs.image_size = (512, 512)
configs.image_add_size = 32


configs.feature_dim = 1280 # for efficientnet_b0
# configs.feature_dim = 2240 # for regnety_120
# configs.feature_dim = 3024 # for regnety_160
# configs.backbone = 'regnetx_064'
# configs.backbone = 'regnety_160'
# configs.backbone = 'regnety_120'
configs.backbone = 'efficientnet_b0'
configs.result_dir = './results'
make_dir(configs.result_dir)
configs.wandb_name = f"{configs.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
configs.wandb_project = 'AAI'

configs.num_devices = torch.cuda.device_count()
configs.num_workers = 8

configs.num_epochs = 30