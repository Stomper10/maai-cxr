import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

from modules_label_ensemble.model import Expert as A2IModel
from cfg_label_ensemble import configs
import functions


configs.model.backbone = 'densenet'
configs.model.densenet.size = '121'
configs.wandb.use_wandb = False
configs.general.distributed = False
configs.general.epochs = 5
configs.general.batch_size = 16 # 원하는 경우 batch size 변경 가능



DATA_DIRECTORY = '/home/gyuseonglee/workspace/dataset/chexpert-resized'



if __name__ == '__main__':
    
    # dataset directory
    configs.dataset.data_dir = DATA_DIRECTORY
    configs.dataset.cutoff = 500 # train 데이터 수 cutoff 만큼만 불러오기


    # load dataset
    train_dataset, valid_dataset, test_dataset = functions.load_datasets(configs) 
    configs.general.steps_per_epoch = train_dataset.steps_per_epoch

    

    targets = ['atel', 'card', 'cons', 'edem', 'plef']
    for target in targets:
        weights = glob.glob(f'/home/gyuseonglee/workspace/maai-cxr/src/kslee001/quantization/weights/label_ensemble/{target}/*.h5')
        for weight in tq(weights) : 
            
            configs.general.label = target # 중요
            SAVE_NAME = f'./quantized_weights/label_ensemble/{target}/' + weight.rsplit('/', 1)[1].rsplit('.h5', 1)[0]
            # load model

            print(f"\n[INFO] current model will be saved as {SAVE_NAME}\n")

            model = A2IModel(configs=configs)
            model.initialize()
            model.load_weights(weight)
            criterion = tf.keras.losses.CategoricalCrossentropy(
                from_logits=False,
                label_smoothing=0.1,
                reduction='auto',
            )
            model.compile(loss=criterion)

            #PTQ
            def representative_data_gen():
                for img, label in train_dataset :
                    # img = tf.reshape(img, shape=[1, 320, 320, 1])
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

            with tf.io.gfile.GFile(f'{SAVE_NAME}.tflite', 'wb') as f:
                f.write(post_quant_tflite_model)