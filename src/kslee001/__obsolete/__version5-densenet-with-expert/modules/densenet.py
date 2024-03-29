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


def Expert(    
    input_shape,
    num_classes,
    conv_filters=[1280, 512],
    name='',
    activation=None,
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
    x = ConvBnAct(filters=conv_filters[0], kernel_size=3, strides=(2,2), padding='same', use_bias=False, seed=seed)(img_input)
    x = ConvBnAct(filters=conv_filters[1], kernel_size=3, strides=(2,2), padding='same', use_bias=False, seed=seed)(x)

    x = layers.GlobalAveragePooling2D(name='convnext_expert_avg_pool')(x)
    x = layers.Dense(num_classes, activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,)(x)
    
    model = training.Model(img_input, x, name=f"convnext_expert_{name}")
    return model


def DenseNet(
    blocks=[6, 12, 48, 32, 16, 8], # DenseNet 201 + additional blocks
    num_classes=5, 
    activation='sigmoid',
    input_shape=None,
    image_channels=1,
    seed=1005,
    reg=0.
):
    initializer = tf.keras.initializers.HeNormal(seed=seed)
    regularizer = tf.keras.regularizers.l2(reg)

    # Determine proper input shape
    img_input = layers.Input(shape=(*input_shape, image_channels))
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    # layers
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name="stem/conv",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name="stem/bn"
    )(x)
    x = layers.Activation("relu", name="stem/relu")(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name="stem_pool")(x)

    # densenet blocks
    for b_idx in range(len(blocks)-1):
        x = dense_block(x, blocks[b_idx], name=f"block{b_idx}", seed=seed, reg=reg)
        x = transition_block(x, 0.5, name=f"pool{b_idx}")
    
    # final densenet block
    x = dense_block(x, blocks[-1], name=f"block{len(blocks)}",
        seed=seed, reg=reg)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
    x = layers.Activation("relu", name="relu")(x)

    # classifier head
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(num_classes, activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,)(x)

    # Create model.
    model = training.Model(img_input, x, name="densenet")

    return model



def drop_path(inputs, drop_rate, is_training):
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


class DropPath(tf.keras.layers.Layer):
    # borrowed from https://github.com/rishigami/Swin-Transformer-TF/blob/main/swintransformer/model.py
    def __init__(self, drop_rate=None):
        super().__init__()
        self.drop_rate = drop_rate

    def call(self, x, training=None):
        return drop_path(x, self.drop_rate, training)


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