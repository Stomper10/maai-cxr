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
from wandb.keras import WandbCallback, WandbMetricsLogger

# private
from modules.model import A2IModel
from modules.lr_scheduler import CustomOneCycleSchedule, LearningRateLogger
from cfg import configs
import functions


if __name__ == '__main__':
    # argument parsing : [ cluster, backbone type (densenet, convnext), head ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cluster', action='store', default='akmu')
    parser.add_argument('-b', '--backbone', action='store', default='densenet')
    parser.add_argument('-a', '--add_expert', action='store_true')
    parser.add_argument('-e', '--epochs', action ='store', type=int, default=5)
    parser.add_argument('-p', '--progress_bar', action ='store_false')

    # for debugging
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-s', '--single_gpu', action ='store_true')
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
    configs.model.classifier.add_expert = bool(args.add_expert)
    configs.dataset.cutoff = 1000 if args.test == True else None
    configs.wandb.use_wandb = args.wandb_off
    configs.wandb.project_name = f'a2i-{cluster}-{configs.model.backbone}-{configs.dataset.image_size[0]}'
    configs.general.distributed = True if args.single_gpu == False else False
    configs.general.epochs = int(args.epochs)
    saved_model_path = "./" + configs.model.backbone + "_best_model_{epoch:02d}-{val_loss:.2f}.h5" 

    # wandb initialization
    if configs.wandb.use_wandb == True :
        wandb.login()
        run = wandb.init(project=configs.wandb.project_name, name=configs.wandb.run_name, config=configs)

    # multi-gpu training strategy
    strategy = tf.distribute.MirroredStrategy()

    # mixed precision policy
    policy = mixed_precision.Policy('mixed_float16' if configs.general.precision == 16 else 'float32')
    mixed_precision.set_global_policy(policy)

    # load datasets
    train_dataset, valid_dataset, test_dataset = functions.load_datasets(configs) 

    # settings 
    if args.single_gpu == False:
        with strategy.scope():
            model = A2IModel(configs=configs)
            model.initialize()
            scheduler = CustomOneCycleSchedule(
                max_lr=configs.optimizer.learning_rate, 
                epochs=configs.general.epochs,
                steps_per_epoch=train_dataset.steps_per_epoch,
                start_lr=None, end_lr=None, warmup_fraction=configs.optimizer.warm_up_rate,
            )
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=scheduler,
                weight_decay=configs.optimizer.weight_decay,
                beta_1=configs.optimizer.beta_1,
                beta_2=configs.optimizer.beta_2,
                ema_momentum=configs.optimizer.ema_momentum,
            )
            criterion = tf.keras.losses.CategoricalCrossentropy(
                # from_logits=True,
                from_logits=False, 
                label_smoothing=configs.model.label_smoothing,
                reduction=tf.keras.losses.Reduction.SUM
            )
            model.compile(optimizer=optimizer, loss=criterion) 
            callbacks = [
                # model checkpoint
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=saved_model_path,
                    monitor='val_loss', 
                    save_best_only=False,  # save all models (True : Save only the best model based on the monitored metric)
                    save_weights_only=True,  # save the entire model (including architecture)
                    # save_format='tf',
                    mode='min', 
                    verbose=0  # do not print messages during saving
                ),
                # learning rate logger
                LearningRateLogger(wandb=configs.wandb.use_wandb),
            ]
            # wandb logger
            if configs.wandb.use_wandb == True:
                callbacks += [ WandbCallback(save_model=False), WandbMetricsLogger(log_freq='batch') ]
    else:
        model = A2IModel(configs=configs)
        model.initialize()
        scheduler = CustomOneCycleSchedule(
            max_lr=configs.optimizer.learning_rate, 
            epochs=configs.general.epochs,
            steps_per_epoch=train_dataset.steps_per_epoch,
            start_lr=None, end_lr=None, warmup_fraction=configs.optimizer.warm_up_rate,
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=scheduler,
            weight_decay=configs.optimizer.weight_decay,
            beta_1=configs.optimizer.beta_1,
            beta_2=configs.optimizer.beta_2,
            ema_momentum=configs.optimizer.ema_momentum,
        )
        criterion = tf.keras.losses.CategoricalCrossentropy(
            # from_logits=True,
            from_logits=False, 
            label_smoothing=configs.model.label_smoothing,
            # reduction=tf.keras.losses.Reduction.SUM
        )
        model.compile(optimizer=optimizer, loss=criterion) 
        callbacks = [
            # model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath=saved_model_path,
                monitor='val_loss', 
                save_best_only=False,  # save all models (True : Save only the best model based on the monitored metric)
                save_weights_only=True,  # save the entire model (including architecture)
                # save_format='tf',
                mode='min', 
                verbose=0  # do not print messages during saving
            ),
            # learning rate logger
            LearningRateLogger(wandb=configs.wandb.use_wandb),
        ]
        # wandb logger
        if configs.wandb.use_wandb == True:
            callbacks += [ WandbCallback(save_model=False), WandbMetricsLogger(log_freq='batch') ]


    # training
    total_parameters = model.count_params()
    model.save_weights("sample_model.h5")
    file_size = os.path.getsize("sample_model.h5") / (1024 * 1024)
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
