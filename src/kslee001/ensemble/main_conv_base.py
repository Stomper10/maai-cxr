import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm as tq
import tensorflow as tf
from tensorflow.keras import mixed_precision
import wandb

# private
from modules.model import A2IModel
from cfg_conv_base import configs
import functions


if __name__ == '__main__':
    # argument parsing : [ cluster, backbone type (densenet, convnext), head ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cluster', action='store', default='akmu')
    parser.add_argument('-f', '--backbone', action='store', default='convnext')
    parser.add_argument('-a', '--add_expert', action='store_true')
    parser.add_argument('-e', '--epochs', action ='store', type=int, default=5)
    parser.add_argument('-b', '--batch', action='store', type=int, default=16)
    parser.add_argument('-p', '--progress_bar', action ='store_false')
    parser.add_argument('-s', '--seed', action='store', default=1005)

    # for debugging
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-g', '--single_gpu', action ='store_true')
    parser.add_argument('-w', '--wandb_off', action='store_false')

    args = parser.parse_args()
    configs.general.progress_bar = 1 if args.progress_bar==True else 2 # one line per epoch
    cluster = args.cluster
    if cluster == 'gsds-ab':
        configs.dataset.data_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized'
    if cluster == 'gsds-c':
        configs.dataset.data_dir = '/home/gyuseonglee/workspace/dataset/chexpert-resized'
    if cluster == 'akmu':
        configs.dataset.data_dir = '/data/s1/gyuseong/chexpert-resized'
    configs.model.backbone = args.backbone
    configs.general.seed = int(args.seed)
    configs.model.classifier.add_expert = bool(args.add_expert)
    configs.dataset.cutoff = 1000 if args.test == True else None
    configs.wandb.use_wandb = args.wandb_off
    configs.wandb.run_name = f'final-{configs.general.seed}-{configs.model.backbone}base-{configs.dataset.image_size[0]}'
    configs.general.distributed = True if args.single_gpu == False else False
    configs.general.epochs = int(args.epochs)
    configs.general.batch = int(args.batch)
    configs.saved_model_path = "./" + f"{configs.model.backbone}base_{configs.general.seed}_" + "{epoch:02d}-{val_loss:.2f}.h5" 

    print(f"[TRAINING] current seed : {configs.general.seed}")

    # wandb initialization
    if configs.wandb.use_wandb == True :
        wandb.login()
        run = wandb.init(project=configs.wandb.project_name, name=configs.wandb.run_name, config=configs)

    # multi-gpu training strategy
    strategy = tf.distribute.MirroredStrategy()
    num_devices = strategy.num_replicas_in_sync
    configs.general.batch_size = configs.general.batch_size*num_devices # global batch size

    # mixed precision policy
    policy = mixed_precision.Policy('mixed_float16' if configs.general.precision == 16 else 'float32')
    mixed_precision.set_global_policy(policy)

    # load datasets
    train_dataset, valid_dataset, test_dataset = functions.load_datasets(configs) 
    configs.general.steps_per_epoch = train_dataset.steps_per_epoch

    # settings 
    if args.single_gpu == False:
        with strategy.scope():
            model, callbacks = functions.set_model_callbacks(
                model_class=A2IModel, 
                configs=configs
            )
    else:
        model, callbacks = functions.set_model_callbacks(
            model_class=A2IModel, 
            configs=configs
        )

    # training
    total_parameters = model.count_params()
    model.save_weights(f"sample_model_{configs.general.seed}.h5")
    file_size = os.path.getsize(f"sample_model_{configs.general.seed}.h5") / (1024 * 1024)
    print("[TRAINING INFO]")
    print(f"-- Model Feature extractor : {configs.model.backbone}")
    print(f"-- Total parameters        : {format(total_parameters, ',')}")
    print(f"-- Expected weight size    : {np.round(file_size, 2)} MB")
    model.fit(
        train_dataset, 
        epochs=configs.general.epochs, 
        validation_data=valid_dataset,
        callbacks=callbacks,
        workers=configs.general.num_workers,
        verbose=configs.general.progress_bar, 
        shuffle=True,
    )
    losses = model.evaluate(test_dataset)

    # evaluation
    print(losses)
