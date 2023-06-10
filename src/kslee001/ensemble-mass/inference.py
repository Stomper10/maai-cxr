import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import time as t
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
# from cfg import configs
import functions


# configs.model.backbone = 'densenet'
# configs.model.backbone = 'convnext'

if __name__ == '__main__':
    # argument parsing : [ cluster, backbone type (densenet, convnext), head ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cluster', action='store', default='akmu')
    parser.add_argument('-b', '--backbone', action='store', default='densenet')
    parser.add_argument('-t', '--backbonetype', action='store', default='121')
    parser.add_argument('-s', '--seed', action='store', default=1005, type=int)

    args = parser.parse_args()
    cluster = args.cluster
    if (args.backbone == 'densenet') & (args.backbonetype == '121'):
        from cfg_121 import configs
        configs.model.backbone = 'densenet'
    if (args.backbone == 'densenet') & (args.backbonetype == '169'):
        from cfg_169 import configs
        configs.model.backbone = 'densenet'
    if (args.backbone == 'convnext') & (args.backbonetype == 'small'):
        from cfg_conv_small import configs
        configs.model.backbone = 'convnext'
    if (args.backbone == 'convnext') & (args.backbonetype == 'base'):
        from cfg_conv_base import configs
        configs.model.backbone = 'convnext'
    if cluster == 'gsds-ab':
        configs.dataset.data_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized'
    if cluster == 'gsds-c':
        configs.dataset.data_dir = '/home/gyuseonglee/workspace/dataset/chexpert-resized'
    if cluster == 'akmu':
        configs.dataset.data_dir = '/data/s1/gyuseong/chexpert-resized'
    configs.general.seed = int(args.seed)

    configs.model.classifier.add_expert = False
    configs.dataset.cutoff = 10
    configs.wandb.use_wandb = False
    configs.wandb.run_name = None
    configs.general.distributed = False
    configs.general.epochs = 1
    configs.general.batch_size = 16
    configs.saved_model_path = "./" + configs.model.backbone + "_{epoch:02d}-{val_loss:.2f}.h5" 

    # multi-gpu training strategy
    strategy = tf.distribute.MirroredStrategy()
    num_devices = strategy.num_replicas_in_sync
    configs.general.batch_size = configs.general.batch_size*num_devices # global batch size

    # mixed precision policy
    policy = mixed_precision.Policy('mixed_float16' if configs.general.precision == 16 else 'float32')
    mixed_precision.set_global_policy(policy)

    # load datasets
    train_dataset, valid_dataset, test_dataset = functions.load_datasets(configs) 

    # load weights
    print("current : ", configs.model.backbone+args.backbonetype)
    weights = sorted(glob.glob(f"./{configs.model.backbone}{args.backbonetype}*.h5"))

    best_weights = None
    best_auc = -1
    best_result = None
    for path in weights:
        model, _ = functions.set_model_callbacks(
            model_class=A2IModel,
            weights_path=path,
            configs=configs,
            training=False,
        )
        start_time = t.time()
        losses = model.evaluate(test_dataset, verbose=0)
        end_time = t.time()
        duration = np.round(end_time-start_time, 6)
        process_images_per_sec = np.round(202/duration, 6)
        single_image_processing = np.round(1/process_images_per_sec, 6)

        targets = ['atel', 'card', 'cons', 'edem', 'plef']
        average_auc = np.round(np.mean(losses[2:]), 4)

        # evaluation
        # print("[RESULT] of : ", path)
        # print( str([f"val_loss : {np.round(losses[0], 4)}"] + [ f"{targets[idx]} : {np.round(losses[2:][idx], 4)}"  for idx in range(5) ]))
        # print("-- average AUC : ", average_auc)
        # print(f"-- DURATION : {duration} | 1 image : {single_image_processing} | 1 sec : {process_images_per_sec}\n")
        if average_auc > best_auc:
            best_auc = average_auc
            best_weights = path
            best_result = losses[1:]

    print("\n[RESULT]")
    print(f"SEED        : {configs.general.seed}")
    print(f"best model  : {best_weights}")
    print(f"best auc    : {best_auc}")
    print(f"best result : {best_result}" )