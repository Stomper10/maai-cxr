import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm as tq
import tensorflow as tf
from tensorflow.keras import mixed_precision
import wandb

from modules.model import Expert as A2IModel
from cfg import configs
import functions


configs.model.backbone = 'densenet'
configs.model.densenet.size = '121'
configs.wandb.use_wandb = False
configs.general.distributed = False
configs.general.epochs = 1
configs.general.batch_size = 16 # 원하는 경우 batch size 변경 가능


LABEL = 'atel' # 중요 !! [atel, card, cons, edem, plef] 중 하나 선택

DATA_DIRECTORY = '/home/gyuseonglee/workspace/dataset/chexpert-resized'
WEIGHT_DIRECTORY = '/home/gyuseonglee/workspace/maai-cxr/src/kslee001/experimental/atel_densenet121_317_30-9.68.h5'

if __name__ == '__main__':
    
    configs.general.label = LABEL
    configs.dataset.data_dir = DATA_DIRECTORY
    configs.dataset.cutoff = 500 # train 데이터 수 cutoff 만큼만 불러오기


    # load dataset
    train_dataset, valid_dataset, test_dataset = functions.load_datasets(configs) 
    configs.general.steps_per_epoch = train_dataset.steps_per_epoch


    # load model
    model = A2IModel(configs=configs)
    model.initialize()
    model.load_weights(WEIGHT_DIRECTORY)
    criterion = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.1,
        reduction='auto',
    )
    model.compile(loss=criterion)
    model.summary()

    # TODO TODO TODO TODO

