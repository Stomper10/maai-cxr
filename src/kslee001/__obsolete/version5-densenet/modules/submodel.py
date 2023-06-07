import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model


def main():
    image_shape = (320, 320)
    image_channels = 1

    model = DenseNet(
        blocks=[6,12,48,32],
        num_classes=0,
        image_shape=image_shape,
        image_channels=image_channels,
        activation='sigmoid',
        seed=1005,
        reg=0.
    )
    
    model.summary()
    return



class DenseNet(tf.keras.Model):
    def __init__(self,
            blocks=[6, 12, 48, 32, 16, 8], # DenseNet 201 + additional blocks
            num_classes=5, 
            image_shape=(320, 320),
            image_channels=1,
            activation='sigmoid',
            seed=1005,
            reg=0.
        ):
        super().__init__()
        initializer = tf.keras.initializers.HeNormal(seed=seed)
        regularizer = tf.keras.regularizers.l2(reg)
        img_input = layers.Input(shape=(*image_shape, image_channels))

        # configuration
        self.image_shape = image_shape
        self.image_channels = image_channels
        self.num_classes = num_classes

        # layers
        self.stem = tf.keras.Sequential([
            layers.ZeroPadding2D(padding=((3, 3), (3, 3))),
            layers.Conv2D(64, 7, strides=2, use_bias=False,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
            ),
            layers.BatchNormalization(axis=-1, epsilon=1.001e-5),
            layers.Activation("relu"),
            layers.ZeroPadding2D(padding=((1,1), (1,1))),
            layers.MaxPooling2D(3, strides=2),
        ])
        img_input = self.stem(img_input)

        self.dense0 = DenseBlock(num_blocks=blocks[0], seed=seed, reg=reg)
        img_input = self.dense0(img_input)

        self.trans0 = TransitionBlock(
            num_input_filters=img_input.shape[-1],
            reduction=0.5,seed=seed,reg=reg)
        img_input = self.trans0(img_input)

        self.dense1 = DenseBlock(num_blocks=blocks[1], seed=seed, reg=reg)
        img_input = self.dense1(img_input)
        
        self.trans1 = TransitionBlock(
            num_input_filters=img_input.shape[-1],
            reduction=0.5,seed=seed,reg=reg)
        img_input = self.trans1(img_input)

        self.dense2 = DenseBlock(num_blocks=blocks[2], seed=seed, reg=reg)
        img_input = self.dense2(img_input)

        self.trans2 = TransitionBlock(
            num_input_filters=img_input.shape[-1],
            reduction=0.5,seed=seed,reg=reg)
        img_input = self.trans2(img_input)

        self.dense3 = DenseBlock(num_blocks=blocks[3], seed=seed, reg=reg)
        img_input = self.dense3(img_input)
        # no transition block for last dense block

        if self.num_classes > 0:
            self.head = tf.keras.Sequential([
                layers.GlobalAveragePooling2D(),
                layers.Dense(num_classes, 
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                    activation=activation,
                )
            ])
            img_input = self.head(img_input)

        # initialize model
        self.initialize()

    def call(self, inputs):
        x = self.stem(inputs)

        x = self.dense0(x)
        x = self.trans0(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)

        if self.num_classes > 0:
            return self.head(x)
        return x    

    def initialize(self):
        self.build(input_shape=(1,*(self.image_shape),self.image_channels))
        

class ConvBlock(tf.keras.Model):
    def __init__(self, growth_rate, seed=1005, reg=0.):
        super().__init__()

        initializer = tf.keras.initializers.HeNormal(seed=seed)
        regularizer = tf.keras.regularizers.l2(reg)

        self.bn1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5)
        self.act1 = layers.Activation("relu")
        self.conv1 = layers.Conv2D(
            4*growth_rate, 1, use_bias=False,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
        )
        self.bn2 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5)
        self.act2 = layers.Activation("relu")
        self.conv2 = layers.Conv2D(
            growth_rate, 3, padding="same", use_bias=False, 
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
        )
        self.concat = layers.Concatenate(axis=-1)

    def call(self, inputs):
        x = self.conv1(self.act1(self.bn1(inputs)))
        x = self.conv2(self.act2(self.bn2(x)))

        return self.concat([inputs, x])
        

class DenseBlock(tf.keras.Model):
    def __init__(self, num_blocks, seed=1005, reg=0.):
        super().__init__()
        initializer = tf.keras.initializers.HeNormal(seed=seed)
        regularizer = tf.keras.regularizers.l2(reg)
        self.num_blocks = num_blocks

        self.blocks = tf.keras.Sequential([
            ConvBlock(32, seed=seed, reg=reg) for _ in range(self.num_blocks)
        ])
    

    def call(self, inputs):
        return self.blocks(inputs)


class TransitionBlock(tf.keras.Model):
    def __init__(self, num_input_filters, reduction, seed=1005, reg=0.):
        super().__init__()
        initializer = tf.keras.initializers.HeNormal(seed=seed)
        regularizer = tf.keras.regularizers.l2(reg)
        self.num_input_filters = num_input_filters
        self.reduction = reduction

        self.bn = layers.BatchNormalization(axis=-1, epsilon=1.001e-5)
        self.act = layers.Activation("relu")
        self.conv = layers.Conv2D(
            int(num_input_filters * reduction),
            1,
            use_bias=False,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,)
        self.pool = layers.AveragePooling2D(2, strides=2)

    def call(self, inputs):
        return self.pool(self.conv(self.act(self.bn(inputs))))












if __name__ == '__main__':

    main()