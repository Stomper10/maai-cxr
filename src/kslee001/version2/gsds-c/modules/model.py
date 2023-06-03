import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
# import (Conv2D, Dense, DepthwiseConv2D,
#                                      GlobalAveragePooling2D, Layer,
#                                      LayerNormalization)
from tensorflow.keras.models import Model

# private
from .backend import ConvBnAct


# Toy model for experiment
def TestModel(input_shape=(384,384), num_classes=5):
    model= tf.keras.models.Sequential()
    model.add(layers.Input(shape=(*input_shape, 3)))
    model.add(layers.Conv2D(32, 3, (2,2)))
    model.add(layers.BatchNormalization(axis=-1, epsilon=1e-6, momentum=0.999))
    model.add(layers.ReLU(max_value=6))
    model.add(ConvBnAct(64, strides=(2,2)))
    model.add(ConvBnAct(128, strides=(2,2)))
    model.add(ConvBnAct(256))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes))
    # model.add(layers.Activation('sigmoid'))
    return model    
