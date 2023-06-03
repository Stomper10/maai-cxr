import os
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras import layers
# from tensorflow import keras
# from tensorflow.keras import layers, regularizers
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import (
#     Add, ReLU, Input, Dense, Dropout, Activation, Flatten, 
#     Conv2D, MaxPooling2D, InputLayer, Reshape, DepthwiseConv2D, BatchNormalization, GlobalAveragePooling2D,
#     Layer, InputSpec)
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.callbacks import CSVLogger
# from tensorflow.python.keras import backend
# from tensorflow.python.keras.utils import layer_utils


class ConvBnAct(tf.keras.layers.Layer):
    def __init__(
        self, filters, kernel_size=3, strides=(1,1),   padding='same', use_bias=False, seed=1005):
        super().__init__()
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(4e-5),
        
        self.conv = tf.keras.layers.Conv2D(
            filters, 
            kernel_size=kernel_size, 
            strides=strides,
            padding=padding,
            use_bias=use_bias,
        )
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)



# Toy model for experiment
def TestModel(num_classes):
    model= tf.keras.models.Sequential()
    model.add(layers.Conv2D(32, 3, (2,2),))
    model.add(layers.BatchNormalization(axis=-1, epsilon=1e-6, momentum=0.999))
    model.add(layers.ReLU(max_value=6))
    model.add(ConvBnAct(64, strides=(2,2)))
    model.add(ConvBnAct(128, strides=(2,2)))
    model.add(ConvBnAct(256))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes))
    # model.add(layers.Activation('sigmoid'))
    return model    

