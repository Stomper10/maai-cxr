import tensorflow.compat.v2 as tf

from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
layers = VersionAwareLayers()




def dense_block(x, blocks, name, seed=1005, reg=0.):
    """A dense block.

    Args:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + "_block" + str(i + 1), seed=seed, reg=reg)
    return x


def transition_block(x, reduction, name, seed=1005, reg=0.):
    """A transition block.

    Args:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.

    Returns:
      output tensor for the block.
    """
    initializer = tf.keras.initializers.HeNormal(seed=seed)
    regularizer = tf.keras.regularizers.l2(reg)

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_relu")(x)
    x = layers.Conv2D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        name=name + "_conv",
        kernel_initializer=initializer,
        # kernel_regularizer=regularizer,
    )(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + "_pool")(x)
    return x


def conv_block(x, growth_rate, name, seed=1005, reg=0.):
    """A building block for a dense block.

    Args:
      x: input tensor.
      growth_rate: float, growth rate at dense layers.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    initializer = tf.keras.initializers.HeNormal(seed=seed)
    regularizer = tf.keras.regularizers.l2(reg)
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
    )(x)
    x1 = layers.Activation("relu", name=name + "_0_relu")(x1)
    x1 = layers.Conv2D(
        4 * growth_rate, 1, use_bias=False, name=name + "_1_conv",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
    )(x1)
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn"
    )(x1)
    x1 = layers.Activation("relu", name=name + "_1_relu")(x1)
    x1 = layers.Conv2D(
        growth_rate, 3, padding="same", use_bias=False, name=name + "_2_conv",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
    )(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + "_concat")([x, x1])
    return x


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

        out = self.concat([inputs, x])
        return out


class TransitionBlock