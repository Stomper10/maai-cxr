import numpy as np
import tensorflow.compat.v2 as tf

from keras import backend
from keras import layers
from keras import utils
from keras.applications import imagenet_utils
from keras.engine import sequential
from keras.engine import training
from tensorflow.python.util.tf_export import keras_export


def ConvNeXt(
    depths=[3, 3, 27, 3],
    projection_dims=[128, 256, 512, 1024],
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    precision=16, 
    input_shape=(384, 384),
    image_channels=1,
):
    model_name="convnext"
    
    # Determine proper input shape.
    img_input = layers.Input(shape=(*input_shape, image_channels))
    x = img_input
    
    # Stem block.
    stem = sequential.Sequential(
        [
            layers.Conv2D(
                projection_dims[0],
                kernel_size=4,
                strides=4,
                name=model_name + "_stem_conv",
            ),
            layers.LayerNormalization(
                epsilon=1e-6, name=model_name + "_stem_layernorm"
            ),
        ],
        name=model_name + "_stem",
    )

    # Downsampling blocks.
    downsample_layers = []
    downsample_layers.append(stem)

    num_downsample_layers = 3
    for i in range(num_downsample_layers):
        downsample_layer = sequential.Sequential(
            [
                layers.LayerNormalization(
                    epsilon=1e-6,
                    name=model_name + "_downsampling_layernorm_" + str(i),
                ),
                layers.Conv2D(
                    projection_dims[i + 1],
                    kernel_size=2,
                    strides=2,
                    name=model_name + "_downsampling_conv_" + str(i),
                ),
            ],
            name=model_name + "_downsampling_block_" + str(i),
        )
        downsample_layers.append(downsample_layer)

    # Stochastic depth schedule.
    # This is referred from the original ConvNeXt codebase:
    # https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L86
    depth_drop_rates = [
        float(x) for x in np.linspace(0.0, drop_path_rate, sum(depths))
    ]
    
    # First apply downsampling blocks and then apply ConvNeXt stages.
    cur = 0
    num_convnext_blocks = 4
    for i in range(num_convnext_blocks):
        x = downsample_layers[i](x)
        for j in range(depths[i]):
            x = ConvNeXtBlock(
                projection_dim=projection_dims[i],
                drop_path_rate=depth_drop_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value,
                precision=precision,
                name=model_name + f"_stage_{i}_block_{j}",
            )(x)
        cur += depths[i]

    # x = Head(num_classes=classes, name=model_name)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    model = training.Model(inputs=img_input, outputs=x, name=model_name)

    return model



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




def ConvNeXtBlock(
    projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, precision=16, name=None
):
    """ConvNeXt block.
    Args:
      projection_dim (int): Number of filters for convolution layers. In the
        ConvNeXt paper, this is referred to as projection dimension.
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].
      layer_scale_init_value (float): Layer scale value. Should be a small float
        number.
      name: name to path to the keras layer.

    Returns:
      A function representing a ConvNeXtBlock block.
    """
    if name is None:
        name = "prestem" + str(backend.get_uid("prestem"))

    def apply(inputs):
        x = inputs

        x = layers.Conv2D(
            filters=projection_dim,
            kernel_size=7,
            padding="same",
            groups=projection_dim,
            name=name + "_depthwise_conv",
        )(x)
        x = layers.LayerNormalization(epsilon=1e-6, name=name + "_layernorm")(x)
        x = layers.Dense(4 * projection_dim, name=name + "_pointwise_conv_1")(x)
        x = layers.Activation("gelu", name=name + "_gelu")(x)
        x = layers.Dense(projection_dim, name=name + "_pointwise_conv_2")(x)

        if layer_scale_init_value is not None:
            x = LayerScale(
                layer_scale_init_value,
                projection_dim,
                precision=precision,
                name=name + "_layer_scale",
            )(x)
        if drop_path_rate:
            layer = StochasticDepth(
                drop_path_rate, precision=precision, name=name + "_stochastic_depth"
            )
        else:
            layer = layers.Activation("linear", name=name + "_identity")

        return inputs + layer(x)

    return apply



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
    


class StochasticDepth(layers.Layer):
    """Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
      - https://github.com/rwightman/pytorch-image-models

    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].

    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, precision=16, **kwargs):
        super().__init__(**kwargs)
        self.precision = precision
        self.drop_path_rate = drop_path_rate
        if self.precision == 16:
            self.drop_path_rate = tf.cast(self.drop_path_rate, tf.float16)
        

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            uniform = tf.random.uniform(shape, 0, 1)
            if self.precision == 16:
                uniform = tf.cast(uniform, tf.float16)
            random_tensor = keep_prob + uniform
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


class LayerScale(layers.Layer):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239

    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.

    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, precision, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
        self.precision = precision

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )
        if self.precision == 16:
            self.gamma = tf.cast(self.gamma, tf.float16)
        
    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config
    
    
def Head(num_classes=1000, name=None):
    """Implementation of classification head of RegNet.

    Args:
      num_classes: number of classes for Dense layer
      name: name prefix

    Returns:
      Classification head function.
    """
    if name is None:
        name = str(backend.get_uid("head"))

    def apply(x):
        x = layers.GlobalAveragePooling2D(name=name + "_head_gap")(x)
        x = layers.LayerNormalization(
            epsilon=1e-6, name=name + "_head_layernorm"
        )(x)
        x = layers.Dense(num_classes, name=name + "_head_dense")(x)
        return x

    return apply