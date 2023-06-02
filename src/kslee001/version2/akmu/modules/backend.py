import os
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Add, ReLU, Input, Dense, Dropout, Activation, Flatten \
    , Conv2D, MaxPooling2D, InputLayer, Reshape, DepthwiseConv2D, BatchNormalization, GlobalAveragePooling2D \
    , Layer, InputSpec
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import layer_utils


def ConvBnAct(
    x, 
    num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=False,
    activation=layers.ReLU,
    seed=1005,
    ):
    kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
    kernel_regularizer=tf.keras.regularizers.l2(4e-5),
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regulaizer,
        )(x)
    x = layers.BatchNormalization(
        axis=-1, momentum=0.999, epsilon=0.0001,
    )(x)
    x = activation(6)(x) 
    return x


# Toy model for experiment
def TestModel(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    x = ConvBnAct(32, strides=(2,2))
    x = ConvBnAct(64, strides=(2,2))
    x = ConvBnAct(128, strides=(2,2))
    x = ConvBnAct(256)
    x = ConvBnAct(512)
    print(x.shape)

    return None


