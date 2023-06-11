import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import numpy as np
import tensorflow as tf
from keras import initializers
import tensorflow.keras.layers as layers


def main():

    x = layers.Input(shape=(320, 320, 1))
    model = Temp(input_shape=(320, 320, 1))
    out = model(x)
    print(out.shape)
    # print(model.layers_names)

    return

class Temp(tf.keras.Model):
    def __init__(self, input_shape=(320, 320, 1), seed=1005):
        super().__init__()
        self.augmentation, augmentation_output_shape = Augmentation(
            model=self, 
            translation_height_factor=(-0.1,0.1), translation_width_factor=(-0.1,0.1),
            rotation_factor=(-0.1,0.1),
            zoom_height_factor=(-0.1,0.1), zoom_width_factor=(-0.1,0.1),
            input_shape=input_shape,
            seed=seed
        )
        self.feature_extractor, feature_extractor_output_shape = DenseNet(
            model=self,
            input_shape=augmentation_output_shape,
            blocks=[6, 12, 48, 32], # DenseNet 201 + additional blocks
            growth_rate=32,
            num_classes=0, 
            activation=None,
            seed=1005,
            reg=0.
        )
        self.classifier, classifier_output_shape = Classifier(
            model=self, 
            input_shape=feature_extractor_output_shape,
            num_classes=2,
            activation=None,
            add_expert=False,
            expert_filters=[512, 256, 128],
            seed=1005,
            reg=0.
        )

        self.all_layers_names = self.augmentation + self.feature_extractor + self.classifier

    def call(self, x):
        print(f"input : {x.shape}")
        for idx in range(len(self.all_layers_names)):
            name = self.all_layers_names[idx]
            x = getattr(self, name)(x)
            print(f"-- {name} | type : {getattr(self, name).__class__} | output : {x.shape}")
        return x



# ========= Modules ====================================================================================
def Augmentation(
    model, 
    translation_height_factor, translation_width_factor,
    rotation_factor,
    zoom_height_factor, zoom_width_factor,
    input_shape=(320, 320, 1),
    seed=1005):
    if len(input_shape) > 3: input_shape = input_shape[-3:]

    layers_names = []
    img_input = layers.Input(shape=(*input_shape,))

    def set_layer(img_input, name, layer, args:dict):
        setattr(model, name, layer(**args, name=name))
        # pass dummy img input --> to get output shape of the layer
        img_input = getattr(model, name)(img_input) 
        layers_names.append(name)
        # print(f"-- {name} | type : {getattr(model, name).__class__} | output : {img_input.shape}")
        return img_input

    # random translation
    img_input = set_layer(
        img_input=img_input, name='densenet_augmentation_random_translation', 
        layer=layers.RandomTranslation, 
        args={
            'height_factor':translation_height_factor,
            'width_factor' :translation_width_factor,
            'fill_mode':'reflect',
            'interpolation':'bilinear',
            'seed':seed,
            'fill_value':0.0,
    })
    # random rotation
    img_input = set_layer(
        img_input=img_input, name='densenet_augmentation_random_rotation', 
        layer=layers.RandomRotation, 
        args={
            'factor':rotation_factor,
            'fill_mode':'reflect',
            'interpolation':'bilinear',
            'seed':seed,
            'fill_value':0.0,
    })
    # random scaling -> scaling tensor values?? scaling tensor size ???
    img_input = set_layer(
        img_input=img_input, name='densenet_augmentation_random_translation', 
        layer=layers.RandomZoom, 
        args={
            'height_factor':zoom_height_factor,
            'width_factor' :zoom_width_factor,
            'fill_mode':'reflect',
            'interpolation':'bilinear',
            'seed':seed,
            'fill_value':0.0,
    })

    output_shape = tuple(img_input.shape)
    return layers_names, output_shape



def DenseNet(
    model,
    input_shape=None,
    blocks=[6, 12, 48, 32], # DenseNet 201 + additional blocks
    growth_rate=32,
    num_classes=0, 
    activation=None,
    seed=1005,
    reg=0.):
    if len(input_shape) > 3: input_shape = input_shape[-3:]

    layers_names = []
    img_input = layers.Input(shape=(*input_shape,))

    initializer = tf.keras.initializers.HeNormal(seed=seed)
    regularizer = tf.keras.regularizers.l2(reg)

    def set_layer(img_input, name, layer, args:dict):
        setattr(model, name, layer(**args, name=name))
        # pass dummy img input --> to get output shape of the layer
        img_input = getattr(model, name)(img_input) 
        layers_names.append(name)
        # print(f"-- {name} | type : {getattr(model, name).__class__} | output : {img_input.shape}")
        return img_input

    """Stem network""" 
    img_input = set_layer(img_input=img_input, name='densenet_stem_zeropad1', layer=layers.ZeroPadding2D, args={'padding':((3,3), (3,3))})
    img_input = set_layer(img_input=img_input, name='densenet_stem_conv', layer=layers.Conv2D, args={
        'filters':64,
        'kernel_size':7,
        'strides':2,
        'use_bias':False,
        'kernel_initializer':initializer,
        'kernel_regularizer':regularizer,})
    img_input = set_layer(img_input=img_input, name='densenet_stem_bn', layer=layers.BatchNormalization, args={'axis':-1, 'epsilon':1.001e-5})
    img_input = set_layer(img_input=img_input, name='densenet_stem_act', layer=layers.Activation, args={'activation':'relu'})
    img_input = set_layer(img_input=img_input, name='densenet_stem_zeropad2', layer=layers.ZeroPadding2D, args={'padding':((1,1), (1,1))})
    img_input = set_layer(img_input=img_input, name='densenet_stem_maxpool', layer=layers.MaxPooling2D, args={'pool_size':3, 'strides':2})


    """Dense Block & Transition Block"""
    # dense block -> 0-th
    for block_idx in range(len(blocks)-1): # not including last dense block
        for layer_idx in range(blocks[block_idx]):
            # residual connection
            img_input_for_skip = img_input[:]
            """Dense Block : [ Conv Block x blocks[block_idx] ] """
            # bn act conv 1
            img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{block_idx}_{layer_idx}_bn1', layer=layers.BatchNormalization, args={'axis':-1, 'epsilon':1.001e-5})
            img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{block_idx}_{layer_idx}_act1', layer=layers.Activation, args={'activation':'relu'})
            img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{block_idx}_{layer_idx}_conv1', layer=layers.Conv2D, args={
                'filters':4*growth_rate, 'kernel_size':1, 'use_bias':False, 
                'kernel_initializer':initializer,
                'kernel_regularizer':regularizer,})
            # bn act conv 2
            img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{block_idx}_{layer_idx}_bn2', layer=layers.BatchNormalization, args={'axis':-1, 'epsilon':1.001e-5})
            img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{block_idx}_{layer_idx}_act2', layer=layers.Activation, args={'activation':'relu'})
            img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{block_idx}_{layer_idx}_conv2', layer=layers.Conv2D, args={
                'filters':growth_rate, 'kernel_size':3, 'use_bias':False, 'padding':'same',
                'kernel_initializer':initializer,
                'kernel_regularizer':regularizer,})
            # concatenate !
            img_input = set_layer(img_input=[img_input, img_input_for_skip], name=f'densenet_denseblock_{block_idx}_{layer_idx}_concatenate', layer=layers.Concatenate, args={'axis':-1})

        """Transition Block"""
        img_input = set_layer(img_input=img_input, name=f'densenet_transition_{block_idx}_bn', layer=layers.BatchNormalization, args={'axis':-1, 'epsilon':1.001e-5})
        img_input = set_layer(img_input=img_input, name=f'densenet_transition_{block_idx}_act', layer=layers.Activation, args={'activation':'relu'})
        img_input = set_layer(img_input=img_input, name=f'densenet_transition_{block_idx}_conv', layer=layers.Conv2D, args={
            'filters':int(img_input.shape[-1]*0.5), 'kernel_size':1, 'use_bias':False, 
            'kernel_initializer':initializer,
            'kernel_regularizer':regularizer,})
        img_input = set_layer(img_input=img_input, name=f'densenet_transition_{block_idx}_pool', layer=layers.AveragePooling2D, args={'pool_size':2, 'strides':2})
    

    """Last Dense Block"""
    for layer_idx in range(blocks[-1]):
        """Dense Block"""
        # bn act conv 1
        img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{len(blocks)-1}_{layer_idx}_bn1', layer=layers.BatchNormalization, args={'axis':-1, 'epsilon':1.001e-5})
        img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{len(blocks)-1}_{layer_idx}_act1', layer=layers.Activation, args={'activation':'relu'})
        img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{len(blocks)-1}_{layer_idx}_conv1', layer=layers.Conv2D, args={
            'filters':4*growth_rate, 'kernel_size':1, 'use_bias':False, 
            'kernel_initializer':initializer,
            'kernel_regularizer':regularizer,})
        # bn act conv 2
        img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{len(blocks)-1}_{layer_idx}_bn2', layer=layers.BatchNormalization, args={'axis':-1, 'epsilon':1.001e-5})
        img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{len(blocks)-1}_{layer_idx}_act2', layer=layers.Activation, args={'activation':'relu'})
        img_input = set_layer(img_input=img_input, name=f'densenet_denseblock_{len(blocks)-1}_{layer_idx}_conv2', layer=layers.Conv2D, args={
            'filters':4*growth_rate, 'kernel_size':1, 'use_bias':False, 
            'kernel_initializer':initializer,
            'kernel_regularizer':regularizer,})


    """Head definition"""
    if num_classes > 0:
        """ Linear Head """
        img_input = set_layer(img_input=img_input, name='densenet_head_globalpool', layer=layers.GlobalAveragePooling2D, args={})
        img_input = set_layer(img_input=img_input, name='densenet_head_linear', layer=layers.Dense, args={
            'units':num_classes, 'activation':activation,
            'kernel_initializer':initializer,
            'kernel_regularizer':regularizer,})

    output_shape = img_input.shape
    return layers_names, output_shape



def ConvNeXt(
    model,
    input_shape=None,
    depths=[3, 3, 27, 3],
    projection_dims=[128, 256, 512, 1024],
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    num_classes=0, 
    activation=None,
    seed=1005,
    reg=0.
    ):
    if len(input_shape) > 3: input_shape = input_shape[-3:]
    
    layers_names = []
    img_input = layers.Input(shape=(*input_shape,))
    
    initializer = tf.keras.initializers.HeNormal(seed=seed)
    regularizer = tf.keras.regularizers.l2(reg)

    def set_layer(img_input, name, layer, args:dict):
        setattr(model, name, layer(**args, name=name))
        # pass dummy img input --> to get output shape of the layer
        img_input = getattr(model, name)(img_input) 
        layers_names.append(name)
        # print(f"-- {name} | type : {getattr(model, name).__class__} | output : {img_input.shape}")
        return img_input
    


    """ConvNeXt blocks"""
    depth_drop_rates = [float(x) for x in np.linspace(0.0, drop_path_rate, sum(depths))]
    
    cur = 0
    num_convnext_blocks = 4
    for i in range(num_convnext_blocks): # 4
        if i == 0: # stem
            """Stem network"""
            img_input = set_layer(img_input=img_input, name='convnext_stem_conv', layer=layers.Conv2D, args={
                'filters':projection_dims[i],
                'kernel_size':4,
                'strides':4,
                # 'use_bias':False,
                'kernel_initializer':initializer,
                'kernel_regularizer':regularizer,})
            img_input = set_layer(img_input=img_input, name='convnext_stem_ln', layer=layers.LayerNormalization, args={'epsilon':1e-6})
        
        else: # downsample
            """Down Sample"""
            img_input = set_layer(img_input=img_input, name=f'convnext_downsample_{i-1}_ln', layer=layers.LayerNormalization, args={'epsilon':1e-6})
            img_input = set_layer(img_input=img_input, name=f'convnext_downsample_{i-1}_conv', layer=layers.Conv2D, args={
                'filters':projection_dims[i],
                'kernel_size':2,
                'strides':2,
                # 'use_bias':False,
                'kernel_initializer':initializer,
                'kernel_regularizer':regularizer,})
            
        for j in range(depths[i]):
            # # residucal : 
            # residual = img_input[:] # save
            """ConvNeXt blocks"""
            img_input = set_layer(img_input=img_input, name=f'convnext_block_{i}_depthwise_conv', layer=layers.Conv2D, args={
                'filters':projection_dims[i],
                'kernel_size':7,
                # 'strides':2,
                'padding':'same',
                'groups':projection_dims[i],
                # 'use_bias':False,
                'kernel_initializer':initializer,
                'kernel_regularizer':regularizer,})    
            img_input = set_layer(img_input=img_input, name=f'convnext_block_{i}_ln', layer=layers.LayerNormalization, args={'epsilon':1e-6})
            img_input = set_layer(img_input=img_input, name=f'convnext_block_{i}_pointwise_conv_1', layer=layers.Dense, args={
                'units':4*projection_dims[i],
                'kernel_initializer':initializer,
                'kernel_regularizer':regularizer,})
            img_input = set_layer(img_input=img_input, name=f'convnext_block_{i}_gelu', layer=layers.Activation, args={'activation':'gelu'})
            img_input = set_layer(img_input=img_input, name=f'convnext_block_{i}_pointwise_conv_2', layer=layers.Dense, args={
                'units':projection_dims[i],
                'kernel_initializer':initializer,
                'kernel_regularizer':regularizer,})
            
            if layer_scale_init_value is not None:
                img_input = set_layer(img_input=img_input, name=f'convnext_block_{i}_layerscale', 
                                      layer=LayerScale, args={
                                          'init_values':layer_scale_init_value, 'projection_dim':projection_dims[i]
                                      })
            img_input = set_layer(img_input=img_input, name=f'convnext_block_{i}_stochastic_depth', 
                                    layer=StochasticDepth, args={'drop_path_rate':depth_drop_rates[cur+j],})

        cur += depths[i]

    if num_classes > 0:
        """ Linear Head """
        img_input = set_layer(img_input=img_input, name='convnext_head_globalpool', layer=layers.GlobalAveragePooling2D, args={})
        img_input = set_layer(img_input=img_input, name='convnext_head_ln', layer=layers.LayerNormalization, args={'epsilon':1e-6})
        img_input = set_layer(img_input=img_input, name='convnext_head_linear', layer=layers.Dense, args={
            'units':num_classes, 'activation':activation,
            'kernel_initializer':initializer,
            'kernel_regularizer':regularizer,})

    output_shape = img_input.shape
    return layers_names, output_shape






def Classifier(
    model, 
    input_shape,
    target_name,
    num_classes,
    activation=None,
    add_expert=False,
    expert_filters=[512, 256, 128],
    seed=1005,
    reg=0.
    ):
    if len(input_shape) > 3: input_shape = input_shape[-3:]

    layers_names = []
    img_input = layers.Input(shape=(*input_shape,))

    initializer = tf.keras.initializers.HeNormal(seed=seed)
    regularizer = tf.keras.regularizers.l2(reg)

    def set_layer(img_input, name, layer, args:dict):
        setattr(model, name, layer(**args, name=name))
        # pass dummy img input --> to get output shape of the layer
        img_input = getattr(model, name)(img_input) 
        layers_names.append(name)
        # print(f"-- {name} | type : {getattr(model, name).__class__} | output : {img_input.shape}")
        return img_input

    """Add Conv Head (when add_expert == True)"""
    if (add_expert==True) & (len(expert_filters)>0):
        for conv_idx in range(len(expert_filters)):
            img_input = set_layer(img_input=img_input, name=f'{target_name}_classifier_expert_{conv_idx}_pool', layer=layers.MaxPool2D, args={'pool_size':(2,2), 'strides':2})
            img_input = set_layer(img_input=img_input, name=f'{target_name}_classifier_expert_{conv_idx}_conv', layer=layers.Conv2D, args={
            'filters':expert_filters[conv_idx], 'kernel_size':3, 'strides':(1,1), 'use_bias':False, 
            'kernel_initializer':initializer,
            'kernel_regularizer':regularizer,})
            img_input = set_layer(img_input=img_input, name=f'{target_name}_classifier_expert_{conv_idx}_bn', layer=layers.BatchNormalization, args={'axis':-1, 'epsilon':1.001e-5})
            img_input = set_layer(img_input=img_input, name=f'{target_name}_classifier_expert_{conv_idx}_act', layer=layers.Activation, args={'activation':'relu'})

    """ Classifier definition (same as Head definition in feature extractor)"""
    """ Linear Head """
    img_input = set_layer(img_input=img_input, name=f'{target_name}_classifier_globalpool', layer=layers.GlobalAveragePooling2D, args={})
    img_input = set_layer(img_input=img_input, name=f'{target_name}_classifier_linear', layer=layers.Dense, args={
        'units':num_classes, 'activation':activation,
        'kernel_initializer':initializer,
        'kernel_regularizer':regularizer,})

    output_shape = img_input.shape
    return layers_names, output_shape


# ========= Modules ====================================================================================


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

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=initializers.Constant(self.init_values),
            trainable=True,
        )

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

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config
    

if __name__ == '__main__':
    main()
    
