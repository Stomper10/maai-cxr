import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
# import (Conv2D, Dense, DepthwiseConv2D,
#                                      GlobalAveragePooling2D, Layer,
#                                      LayerNormalization)
from tensorflow.keras.models import Model
from .densenet import DenseNet, DenseNetExpert


class A2IModelBase(tf.keras.Model):
    def __init__(self, img_size=(384, 384), num_classes=5, blocks=[6, 12, 48, 32], conv_filters=[1280, 1440], use_aux_information=True, drop_rate=0., reg=0.00001, seed=1005):
        super().__init__()
        initializer = tf.keras.initializers.HeNormal(seed=seed)
        regularizer = tf.keras.regularizers.l2(reg)
        self.img_size = img_size
        self.num_classes = num_classes
        self.use_aux_information = use_aux_information
        self.drop_rate = drop_rate

        self.densenet = DenseNet(
            blocks=blocks,
            input_shape=(*img_size,3),
            seed=seed,
            reg=reg,
        )
        
        if self.use_aux_information:
            self.auxiliary_layer = None

    def call(self, inputs, training=False):
        x, x_aux = inputs
        
        x = self.densenet(x)
        # feature = [self.experts[idx](feature) for idx in range(self.num_classes)]
        # out = tf.concat(feature, axis=-1)

        # if self.use_aux_information:
        #     information = self.auxiliary_layer(x_aug)
        #     out = out + information
        return x

    def initialize(self):
        self((tf.zeros((1, *self.img_size, 3)), tf.zeros((1, 2))))

