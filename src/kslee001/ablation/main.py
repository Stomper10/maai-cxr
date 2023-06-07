import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import argparse
import pandas as pd
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
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-s', '--single_gpu', action ='store_true')
    parser.add_argument('-e', '--epochs', action ='store', type=int, default=5)
    args = parser.parse_args()
    configs.dataset.data_dir = '/data/s1/gyuseong/chexpert-resized'
    configs.dataset.cutoff = 1000 if args.test == True else None
    configs.general.distributed = True if args.single_gpu == False else False

    # multi-gpu training strategy
    strategy = tf.distribute.MirroredStrategy()
    # mixed precision policy
    policy = mixed_precision.Policy('mixed_float16' if configs.general.precision == 16 else 'float32')
    mixed_precision.set_global_policy(policy)


    # settings 
    if args.single_gpu == False:
        with strategy.scope():
            model = A2IModel(configs=configs)
            model.initialize()
            model.summary()
            scheduler = CustomOneCycleSchedule(
                max_lr=configs.optimizer.learning_rate, 
                epochs=configs.general.epochs,
                steps_per_epoch=500,
                start_lr=None, end_lr=None, warmup_fraction=configs.optimizer.warm_up_rate,
            )
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=scheduler,
                weight_decay=configs.optimizer.weight_decay,
                beta_1=configs.optimizer.beta_1,
                beta_2=configs.optimizer.beta_2,
                ema_momentum=configs.optimizer.ema_momentum,
            )
            model.compile(optimizer=optimizer) 
            model.load_weights('/home/n7/gyuseong/workspace/maai-cxr/src/kslee001/version6/densenet_best_model_05-nan.h5')

        print(model.summary())

    else:
        model = A2IModel(configs=configs)
        model.initialize()
        model.summary()
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
        model.compile(optimizer=optimizer) 
        model.load_weights('/home/n7/gyuseong/workspace/maai-cxr/src/kslee001/version6/densenet_best_model_05-nan.h5')
