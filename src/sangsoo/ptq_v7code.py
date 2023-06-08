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

model = A2IModel(configs=configs)
model.initialize()
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
model.load_weights('/home/n1/sangsooim/2_AAI/project/version7/densenet_best_model_15-47.51.h5')

train_dataset= functions.load_datasets(configs)

#PTQ
def representative_data_gen():
    for img, label in train_dataset.batch(1).take(100):
        img = tf.reshape(img, shape=[1, 320, 320, 1])
        yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 옵션들
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen # 함수로 설정해야 한다.
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # 이 모델은 integer 연산만 할 것
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8 # or tf.int8
converter.inference_output_type = tf.uint8 # or tf.int8
# 옵션 끝

post_quant_tflite_model = converter.convert()

with tf.io.gfile.GFile('baseline_quant_model.tflite', 'wb') as f:
    f.write(post_quant_tflite_model)