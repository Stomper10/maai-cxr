import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import glob
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm as tq
import tensorflow as tf
from tensorflow.keras import mixed_precision
import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger

# private
from modules.model import A2IModel
from modules.lr_scheduler import CustomOneCycleSchedule, LearningRateLogger
from cfg import configs
import functions


configs.model.backbone = 'densenet'
# configs.model.backbone = 'convnext'

if __name__ == '__main__':
    weights = glob.glob(f"./{configs.model.backbone}*.h5")
    # argument parsing : [ cluster, backbone type (densenet, convnext), head ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cluster', action='store', default='akmu')
    # parser.add_argument('-b', '--backbone', action='store', default='densenet')

    args = parser.parse_args()
    cluster = args.cluster
    if cluster == 'gsds-ab':
        configs.dataset.data_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized'
    if cluster == 'gsds-c':
        configs.dataset.data_dir = '/home/gyuseonglee/workspace/dataset/chexpert-resized'
    if cluster == 'akmu':
        configs.dataset.data_dir = '/data/s1/gyuseong/chexpert-resized'
    # configs.model.backbone = args.backbone
    configs.model.classifier.add_expert = False
    configs.dataset.cutoff = 10
    configs.wandb.use_wandb = False
    configs.wandb.run_name = None
    configs.general.distributed = False
    configs.general.epochs = 1
    configs.general.batch_size = 16
    configs.saved_model_path = "./" + configs.model.backbone + "_best_model_{epoch:02d}-{val_loss:.2f}.h5" 

    # multi-gpu training strategy
    strategy = tf.distribute.MirroredStrategy()
    num_devices = strategy.num_replicas_in_sync
    configs.general.batch_size = configs.general.batch_size*num_devices # global batch size

    # mixed precision policy
    policy = mixed_precision.Policy('mixed_float16' if configs.general.precision == 16 else 'float32')
    mixed_precision.set_global_policy(policy)

    # load datasets
    train_dataset, valid_dataset, test_dataset = functions.load_datasets(configs) 



    best_weights = None
    best_auc = -1
    for path in weights:
        model, _ = functions.set_model_callbacks(
            model_class=A2IModel,
            weights_path=path,
            configs=configs,
            training=False,
        )

        losses = model.evaluate(test_dataset)
        targets = ['atel', 'card', 'cons', 'edem', 'plef']
        average_auc = np.round(np.mean(losses[2:]), 4)

        # evaluation
        print("-- result of : ", path)
        print( str([f"val_loss : {np.round(losses[0], 4)}"] + [ f"{targets[idx]} : {np.round(losses[2:][idx], 4)}"  for idx in range(5) ]))
        print("-- average AUC : ", average_auc)

        if average_auc > best_auc:
            best_auc = average_auc
            best_weights = path


    print("\n[RESULT]")
    print(f"best model : {best_weights}")
    print(f"best auc   : {best_auc}")