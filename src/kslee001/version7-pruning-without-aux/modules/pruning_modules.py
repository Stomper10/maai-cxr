import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import tensorflow as tf
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
    num_classes=5, 
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

    ):
    # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 

    return





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



if __name__ == '__main__':
    main()
    
