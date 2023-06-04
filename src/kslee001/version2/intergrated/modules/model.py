import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
# import (Conv2D, Dense, DepthwiseConv2D,
#                                      GlobalAveragePooling2D, Layer,
#                                      LayerNormalization)
from tensorflow.keras.models import Model

# private
from .backend import ConvBnAct, InvertedBottleneck


# strides=2 * 6 : 384 x 384  -> 6 x 6
def get_feature_extractor(img_size=(384,384), drop_rate=0., seed=1005, regularization=4e-5):
    extractor = tf.keras.models.Sequential([
            # layers.Input(shape=(*img_size, 3)),
            ConvBnAct(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False, seed=seed),    
            InvertedBottleneck( 32,  32, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck( 32,  32, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),

            InvertedBottleneck( 32,  64, strides=2, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck( 64,  64, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck( 64,  64, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            
            InvertedBottleneck( 64, 128, strides=2, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(128, 128, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(128, 128, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(128, 128, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),

            InvertedBottleneck(128, 196, strides=2, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(196, 196, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(196, 196, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(196, 196, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),

            InvertedBottleneck(196, 256, strides=2, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(256, 256, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(256, 256, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(256, 256, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),

            InvertedBottleneck(256, 512, strides=2, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(512, 512, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(512, 512, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            InvertedBottleneck(512, 512, strides=1, seed=seed, drop_rate=drop_rate,regularization=regularization),
            
            InvertedBottleneck(512, 960, strides=1, drop_rate=drop_rate, seed=seed, regularization=regularization),
            InvertedBottleneck(960, 960, strides=1, drop_rate=drop_rate, seed=seed, regularization=regularization),
            InvertedBottleneck(960, 960, strides=1, drop_rate=drop_rate, seed=seed, regularization=regularization),
            InvertedBottleneck(960, 960, strides=1, drop_rate=drop_rate, seed=seed, regularization=regularization),

        ])
    
    return extractor#, extractor.output


def get_expert(img_size=(384,384), drop_rate=0., seed=1005, regularization=4e-5):
    # previous_output_img_size = previous_output.shape[1:]
    kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
    kernel_regularizer = tf.keras.regularizers.l2(regularization)
    
    extractor = tf.keras.models.Sequential([
        # 6 x 6 -> 3 x 3
        
        # 3 x 3 -> 1 x 1
        InvertedBottleneck(960, 256, strides=1, drop_rate=drop_rate, seed=seed, regularization=regularization),
        InvertedBottleneck(256, 64, strides=1, drop_rate=drop_rate, seed=seed, regularization=regularization),

        layers.Flatten(),
        layers.Dense(1, 
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )
    ])

    return extractor


def get_aux_layer(num_classes=5, seed=1005, regularization=4e-5):
    kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
    kernel_regularizer = tf.keras.regularizers.l2(regularization)
    extractor = tf.keras.models.Sequential([
        # layers.Input((1, 2)),
        layers.Dense(16,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        ),
        layers.Activation('gelu'),
        layers.Dense(16,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        ),
        layers.Activation('gelu'),
        layers.Dense(num_classes,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        ),
    ])
    return extractor


class A2IModel(tf.keras.Model):
    def __init__(self, img_size=(384, 384), num_classes=5, use_aux_information=True, drop_rate=0., regularization=4e-5, seed=1005):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.use_aux_information = use_aux_information
        self.drop_rate = drop_rate

        self.feature_extractor = get_feature_extractor(
            img_size=img_size, 
            drop_rate=drop_rate, 
            seed=seed, 
            regularization=regularization)

        self.experts = [
            get_expert(
                img_size=img_size, 
                drop_rate=drop_rate, 
                seed=seed, 
                regularization=regularization) for _ in range(num_classes)]

        if self.use_aux_information:
            self.auxiliary_layer = get_aux_layer(
                num_classes=num_classes, 
                seed=seed, 
                regularization=regularization)

    def call(self, inputs, training=False):
        x, x_aug = inputs
        
        feature = self.feature_extractor(x)
        feature = [self.experts[idx](feature) for idx in range(self.num_classes)]
        out = tf.concat(feature, axis=-1)

        if self.use_aux_information:
            information = self.auxiliary_layer(x_aug)
            out = out + information
        return out

    def initialize(self):
        self((tf.zeros((1, *self.img_size, 3)), tf.zeros((1, 2))))

