from easydict import EasyDict
from functions import make_dir
import numpy as np
import tensorflow as tf



configs = EasyDict()
configs.seed = 1005

# directory & wandb setting
configs.data_dir = 100 # defined at runtime
configs.dataset_name = 'CheXpert-v1.0'
configs.wandb_project = 'AAI'
configs.wandb_name = None # defined at runtime
configs.model_name = 'ConvNeXtExperts' # defined at runtime

# training configuration
configs.cutoff = 1000  # for test ('None' for full model training)
configs.batch_size = 16
configs.epochs = 5
configs.valid_ratio = 0.1
configs.warm_up_rate = 0.1

# optimizer configuration
configs.learning_rate = 0.0001
configs.weight_decay = 0.0004 #  
configs.beta_1 = 0.9
configs.beta_2 = 0.999
configs.ema_momentum = 0.99

# model configuration
configs.blocks = [4, 4, 4, 4] # densenet
configs.depth = [3, 3, 27, 3]
configs.projection_dims = [128, 256, 512, 1024]  # number of convolution blocks in each stage
configs.conv_filters = [1280, 1440] # not implemented : number of convolution filters in each expert 
configs.drop_path_rate = 0.1
configs.regularization = 5e-5
configs.use_aux_information = True
configs.label_smoothing = 0.1

configs.layer_scale_init_value = 1e-6
configs.precision = 16
configs.tf_dtype = tf.float16 if configs.precision==16 else tf.float32


# data configuration
configs.auxiliary_columns = ['Sex', 'Age']
configs.target_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
configs.num_classes = len(configs.target_columns)
configs.image_size = (512, 512) # (384, 384), (320, 320)
configs.image_channels = 1

# misc.
configs.num_workers = 16
configs.saved_model_path = "./" + configs.model_name + "_best_model_{epoch:02d}-{val_loss:.2f}.h5" 

# data augmentation configuration
configs.translation_height_factor = (-0.01, 0.01) # -3% ~ 3%
configs.translation_width_factor  = (-0.01, 0.01)
configs.zoom_height_factor = (-0.01, 0.01) # -3% ~ 3%
configs.zoom_width_factor  = (-0.01, 0.01)
configs.rotation_factor = (-0.01, 0.01)  # -3% ~ 3%




MODEL_CONFIGS = {
    "tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
        "default_size": 224,
    },
    "small": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
        "default_size": 224,
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
        "default_size": 224,
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
        "default_size": 224,
    },
    "xlarge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
        "default_size": 224,
    },
}