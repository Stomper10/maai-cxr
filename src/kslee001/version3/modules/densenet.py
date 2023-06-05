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


def DenseNetExpert(    
    input_shape,
    idx,
    seed=1005,
    reg=0.
):
    if input_shape.ndims == 4:
        input_shape = input_shape[1:]
    initializer = tf.keras.initializers.HeNormal(seed=seed)
    regularizer = tf.keras.regularizers.l2(reg)

    # input shape processing
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    img_input = layers.Input(shape=input_shape)    

    # layers
    x = layers.Flatten()(img_input)
    x = layers.Dense(1, 
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
    )(x)
    
    model = training.Model(img_input, x, name=f"densenet_expert{idx}")
    return model


def DenseNet(
    blocks,
    input_shape=None,
    seed=1005,
    reg=0.
):
    initializer = tf.keras.initializers.HeNormal(seed=seed)
    regularizer = tf.keras.regularizers.l2(reg)

    # Determine proper input shape
    img_input = layers.Input(shape=input_shape)
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    # layers
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name="conv1/conv",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name="conv1/bn"
    )(x)
    x = layers.Activation("relu", name="conv1/relu")(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1")(x)

    x = dense_block(x, blocks[0], name="conv2", seed=seed, reg=reg)
    x = transition_block(x, 0.5, name="pool2")
    x = dense_block(x, blocks[1], name="conv3", seed=seed, reg=reg)
    x = transition_block(x, 0.5, name="pool3")
    x = dense_block(x, blocks[2], name="conv4", seed=seed, reg=reg)
    x = transition_block(x, 0.5, name="pool4")
    x = dense_block(x, blocks[3], name="conv5", seed=seed, reg=reg)
    x = transition_block(x, 0.5, name="pool5")
    x = dense_block(x, blocks[4], name="conv6", seed=seed, reg=reg)
    x = transition_block(x, 0.5, name="pool6")
    x = dense_block(x, blocks[5], name="conv7", seed=seed, reg=reg)

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
    x = layers.Activation("relu", name="relu")(x)

    # Create model.
    model = training.Model(img_input, x, name="densenet")

    return model

