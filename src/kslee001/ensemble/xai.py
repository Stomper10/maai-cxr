import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shap
#import shap.explainers.deep.deep_tf
import lime
from lime.lime_image import LimeImageExplainer
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Sequential
import ssl
from skimage.segmentation import mark_boundaries

import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


from modules.model import A2IModel
from modules.lr_scheduler import CustomOneCycleSchedule, LearningRateLogger
import functions
from cfg_121 import configs


data_path = '/home/gyuseonglee/workspace/dataset/chexpert-resized'
# data_path = YOUR_DATA_DIR
full_model_weight_path = '/home/gyuseonglee/workspace/maai-cxr/src/kslee001/ensemble/densenet121_1005_21-44.09.h5'
quant_model_wieght_path = '/home/gyuseonglee/workspace/maai-cxr/src/kslee001/ensemble/densenet121_1005.tflite'
# path = YOUR_MODEL_PATH

if __name__ == '__main__':
    configs.model.backbone = 'densenet'
    configs.wandb.use_wandb = False
    configs.general.batch_size = 1
    
    # ===== CHECK YOUR DATA DIRECTORY ===== 
    configs.dataset.data_dir = data_path
    # ===== CHECK YOUR DATA DIRECTORY ===== 
    
    configs.dataset.cutoff = 1000


    train_dataset, valid_dataset, test_dataset = functions.load_datasets(configs) 
    configs.general.steps_per_epoch = train_dataset.steps_per_epoch
    
    
    """
    LOAD FULL MODEL (TF)
    """
    model = A2IModel(configs=configs)
    model.initialize()
    
    # ===== CHECK YOUR FULL MODEL WEIGHT DIRECTORY ===== 
    model.load_weights(full_model_weight_path)
    # ===== CHECK YOUR FULL MODEL WEIGHT DIRECTORY ===== 

    
    criterion = tf.keras.losses.CategoricalCrossentropy(
        # from_logits=True,
        from_logits=False, 
        label_smoothing=configs.model.label_smoothing,
        reduction=tf.keras.losses.Reduction.SUM if configs.general.distributed else 'auto'
    )
    model.compile(loss=criterion)
    
    
    
    """
    LOAD QUANTIZED MODEL (TF-LITE)
    """
    
    # ===== CHECK YOUR QUANT MODEL WEIGHT DIRECTORY ===== 
    interpreter = tf.lite.Interpreter(model_path=quant_model_wieght_path)
    # ===== CHECK YOUR QUANT MODEL WEIGHT DIRECTORY ===== 


    """
    SETTING FOR INTERPRETER
    """
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    

    """
    INFERENCE (TF-LITE MODEL USING INTERPRETER)
    """
    sum_correct = 0.0
    for idx, (x, y) in enumerate(valid_dataset):
        img = (x+1.0)/2.0*255.0
        #image = tf.expand_dims(img, axis=0)
        img = tf.cast(img, tf.uint8)
        #print(image)
        #break
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_index)
        if np.argmax(pred) == np.argmax(y):
            sum_correct += 1.0
        
        
    mean_acc = sum_correct/float(idx+1)
    print(mean_acc)