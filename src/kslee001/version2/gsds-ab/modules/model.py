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
class TestModel(tf.keras.Model):
    def __init__(self, num_classes=5, regularization=4e-5, seed=1005):
        super().__init__()
        self.num_classes = num_classes
        self.experts = [Expert(regularization=regularization, seed=seed) for _ in range(num_classes)]
        self.auxiliary_layer = AuxiliaryLayer(
            num_classes=num_classes, 
            regularization=regularization,
            seed=seed
        )

    def call(self, inputs, training=False):
        x, x_aug = inputs
        
        feature = [self.experts[idx](x) for idx in range(self.num_classes)]
        feature = tf.concat(feature, axis=-1)

        information = self.auxiliary_layer(x_aug)

        out = feature + information
        return out


class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, regularization=4e-5, seed=1005):
        super().__init__()
        kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
        kernel_regularizer = tf.keras.regularizers.l2(regularization)

        self.extractor = tf.keras.models.Sequential([
            layers.Conv2D(32, 3, (2,2), 
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            ),
            InvertedBottleneck(32, 64, (2,2),
                seed=seed, regularization=regularization
            ),
            InvertedBottleneck(64, 128, (2,2),
                seed=seed, regularization=regularization
            ),
            InvertedBottleneck(128, 256, (2,2),
                seed=seed, regularization=regularization
            ),    

        ])




class Expert(tf.keras.layers.Layer):
    def __init__(self, regularization=4e-5, seed=1005):
        super().__init__()
        kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
        kernel_regularizer = tf.keras.regularizers.l2(regularization)
        self.extractor = tf.keras.models.Sequential([
            layers.Conv2D(32, 3, (2,2), 
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            ),
            InvertedBottleneck(32, 64, (2,2),
                seed=seed, regularization=regularization
            ),
            InvertedBottleneck(64, 128, (2,2),
                seed=seed, regularization=regularization
            ),
            InvertedBottleneck(128, 256, (2,2),
                seed=seed, regularization=regularization
            ),
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


def TestModel_old(input_shape=(384,384), num_classes=5):
    model = tf.keras.models.Sequential()
    model.add(layers.Input(shape=(*input_shape, 3)))
    model.add(layers.Conv2D(32, 3, (2,2)))
    model.add(InvertedBottleneck(32, 64, (2,2)))
    model.add(InvertedBottleneck(64, 128, (2,2)))
    model.add(InvertedBottleneck(128, 256))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes))

    # model= tf.keras.models.Sequential()
    # model.add(layers.BatchNormalization(axis=-1, epsilon=1e-6, momentum=0.999))
    # model.add(layers.ReLU(max_value=6))
    # model.add(ConvBnAct(64, strides=(2,2)))
    # model.add(ConvBnAct(128, strides=(2,2)))
    # model.add(ConvBnAct(256))
    # model.add(layers.Activation('sigmoid'))
    return model    

