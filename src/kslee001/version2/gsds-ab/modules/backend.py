import os
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras import layers
from tensorflow.python.keras import backend
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
        self, filters, kernel_size=3, strides=(1,1), padding='same', use_bias=False, seed=1005):
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
        self.act = layers.ReLU(6)

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.act(x)



class InvertedBottleneck(tf.keras.layers.Layer):
    def __init__(
        self, 
        in_channels, out_channels,
        strides=1,
        expansion=6,
        drop_rate=0.,
        regularization=4e-5,
        seed=1005,
        ):
        super().__init__()
        self.initializer = tf.keras.initializers.HeNormal(seed=seed)
        self.regularizer = tf.keras.regularizers.l2(regularization)
        self.drop_path = DropPath(drop_rate=drop_rate)

        assert type(strides) == int
        self.residual = (in_channels == out_channels) & (strides==1)
        self.act = tf.keras.activations.gelu

        # input -> bottleneck (Expanding)
        self.conv1 = layers.Conv2D(
            filters=expansion*in_channels,
            kernel_size=1, strides=1, padding='same',
            use_bias=False, activation=None,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
        )
        self.bn1 = layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=0.0001)

        # bottleneck -> Depthwise Convolution
        self.conv2 = layers.DepthwiseConv2D(
            kernel_size=3, strides=strides, padding='same',
            use_bias=False, activation=None,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
        )
        self.bn2 = layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=0.0001)

        # bottleneck -> output (Projection)
        self.conv3 = layers.Conv2D(
            filters=out_channels, 
            kernel_size=1, strides=1, padding='same',
            use_bias=False, activation=None,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
        )
        self.bn3 = layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=0.0001)

    def call(self, x, training=False):
        # into the bottleneck
        out = self.act(self.bn1(self.conv1(x), training=training))
        # in the bottleneck
        out = self.act(self.bn2(self.conv2(out), training=training))
        # projection
        out = self.bn3(self.conv3(out), training=training)

        if self.residual:
            return x + self.drop_path(out)
        else:
            return self.drop_path(out)




class DropPath(tf.keras.layers.Layer):
    # borrowed from https://github.com/rishigami/Swin-Transformer-TF/blob/main/swintransformer/model.py
    def __init__(self, drop_rate=None):
        super().__init__()
        self.drop_rate = drop_rate

    def call(self, x, training=None):
        return self.drop_path(x, self.drop_rate, training)

    def drop_path(self, inputs, drop_rate, is_training):
        # borrowed from https://github.com/rishigami/Swin-Transformer-TF/blob/main/swintransformer/model.py
        if (not is_training) or (drop_rate == 0.):
            return inputs

        # Compute keep_prob
        keep_prob = 1.0 - drop_rate

        # Compute drop_connect tensor
        random_tensor = keep_prob
        shape = (tf.shape(inputs)[0],) + (1,) * \
            (len(tf.shape(inputs)) - 1)
        random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(inputs, keep_prob) * binary_tensor
        return output