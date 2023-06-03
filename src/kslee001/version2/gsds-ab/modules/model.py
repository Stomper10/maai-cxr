import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
# import (Conv2D, Dense, DepthwiseConv2D,
#                                      GlobalAveragePooling2D, Layer,
#                                      LayerNormalization)
from tensorflow.keras.models import Model

# private
from .backend import ConvBnAct, InvertedBottleneck


# Toy model for experiment
class A2IModel(tf.keras.Model):
    def __init__(self, img_size=(384, 384), num_classes=5, use_information=True, drop_rate=0., regularization=4e-5, seed=1005):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.use_information = use_information
        self.drop_rate = drop_rate

        self.feature_extractor = FeatureExtractor(drop_rate=drop_rate, regularization=regularization, seed=seed)
        self.experts = [Expert(drop_rate=drop_rate, regularization=regularization, seed=seed) for _ in range(num_classes)]

        if self.use_information:
            self.auxiliary_layer = AuxiliaryLayer(
                num_classes=num_classes, 
                regularization=regularization,
                seed=seed
            )

    def call(self, inputs, training=False):
        x, x_aug = inputs
        
        feature = self.feature_extractor(x)
        feature = [self.experts[idx](feature) for idx in range(self.num_classes)]
        out = tf.concat(feature, axis=-1)

        if self.use_information:
            information = self.auxiliary_layer(x_aug)
            out = out + information
        return out

    def initialize(self):
        self((tf.zeros((1, *self.img_size, 3)), tf.zeros((1, 2))))


class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, drop_rate=0., regularization=4e-5, seed=1005):
        super().__init__()
        kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
        kernel_regularizer = tf.keras.regularizers.l2(regularization)

        self.drop_rate = drop_rate

        self.extractor = tf.keras.models.Sequential([
            ConvBnAct(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False, seed=seed),    
            InvertedBottleneck(32, 32, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(32, 32, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(32, 32, strides=1, seed=seed, regularization=regularization),

            InvertedBottleneck(32, 64, strides=2, seed=seed, regularization=regularization),
            InvertedBottleneck(64, 64, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(64, 64, strides=1, seed=seed, regularization=regularization),

            InvertedBottleneck(64, 128, strides=2, seed=seed, regularization=regularization),
            InvertedBottleneck(128, 128, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(128, 128, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(128, 128, strides=1, seed=seed, regularization=regularization),

            InvertedBottleneck(128, 196, strides=2, seed=seed, regularization=regularization),
            InvertedBottleneck(196, 196, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(196, 196, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(196, 196, strides=1, seed=seed, regularization=regularization),

            InvertedBottleneck(196, 256, strides=2, seed=seed, regularization=regularization),
            InvertedBottleneck(256, 256, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(256, 256, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(256, 256, strides=1, seed=seed, regularization=regularization),

            InvertedBottleneck(256, 512, strides=2, seed=seed, regularization=regularization),
            InvertedBottleneck(256, 512, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(512, 512, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(512, 512, strides=1, seed=seed, regularization=regularization),


            InvertedBottleneck(512, 768, strides=2, seed=seed, regularization=regularization),
            InvertedBottleneck(768, 768, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(768, 768, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(768, 768, strides=1, seed=seed, regularization=regularization),

            InvertedBottleneck(768, 1280, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(1280, 1280, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(1280, 1280, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(1280, 1280, strides=1, seed=seed, regularization=regularization),

            InvertedBottleneck(1280, 1440, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(1440, 1440, strides=1, seed=seed, regularization=regularization),
            
        ])

    def call(self, x, training=False):
        return self.extractor(x, training=training)




class Expert(tf.keras.layers.Layer):
    def __init__(self, drop_rate=0., regularization=4e-5, seed=1005):
        super().__init__()
        kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
        kernel_regularizer = tf.keras.regularizers.l2(regularization)
        self.extractor = tf.keras.models.Sequential([
            InvertedBottleneck(1280, 512, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(512, 128, strides=1, seed=seed, regularization=regularization),
            InvertedBottleneck(128, 64, strides=1, seed=seed, regularization=regularization),

            layers.Flatten(),
            layers.Dense(1, 
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )
        ])

    def call(self, x, training=False):
        return self.extractor(x, training=training)


class AuxiliaryLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes=5, regularization=4e-5, seed=1002):
        super().__init__()
        kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
        kernel_regularizer = tf.keras.regularizers.l2(regularization)

        self.extractor = tf.keras.models.Sequential([
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

    def call(self, x, training=False):
        return self.extractor(x, training=training)


def TestModel_old(input_shape=(384,384), num_classes=5, regularization=4e-5, seed=1005):
    model = tf.keras.models.Sequential([
        layers.Input(shape=(*input_shape, 3)),
        ConvBnAct(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False, seed=seed),    
        InvertedBottleneck(32, 32, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(32, 32, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(32, 32, strides=1, seed=seed, regularization=regularization),

        InvertedBottleneck(32, 64, strides=2, seed=seed, regularization=regularization),
        InvertedBottleneck(64, 64, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(64, 64, strides=1, seed=seed, regularization=regularization),

        InvertedBottleneck(64, 128, strides=2, seed=seed, regularization=regularization),
        InvertedBottleneck(128, 128, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(128, 128, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(128, 128, strides=1, seed=seed, regularization=regularization),

        InvertedBottleneck(128, 196, strides=2, seed=seed, regularization=regularization),
        InvertedBottleneck(196, 196, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(196, 196, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(196, 196, strides=1, seed=seed, regularization=regularization),

        InvertedBottleneck(196, 256, strides=2, seed=seed, regularization=regularization),
        InvertedBottleneck(256, 256, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(256, 256, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(256, 256, strides=1, seed=seed, regularization=regularization),

        InvertedBottleneck(256, 512, strides=2, seed=seed, regularization=regularization),
        InvertedBottleneck(256, 512, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(512, 512, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(512, 512, strides=1, seed=seed, regularization=regularization),


        InvertedBottleneck(512, 768, strides=2, seed=seed, regularization=regularization),
        InvertedBottleneck(768, 768, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(768, 768, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(768, 768, strides=1, seed=seed, regularization=regularization),

        InvertedBottleneck(768, 1280, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(1280, 1280, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(1280, 1280, strides=1, seed=seed, regularization=regularization),
        InvertedBottleneck(1280, 1280, strides=1, seed=seed, regularization=regularization),

        # layers.Flatten(),
        # layers.Dense(1, 
        #     # kernel_initializer=kernel_initializer,
        #     # kernel_regularizer=kernel_regularizer,
        # )
    ])
    return model    

