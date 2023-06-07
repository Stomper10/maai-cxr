from easydict import EasyDict
from functions import make_dir
import numpy as np
import tensorflow as tf


configs = EasyDict()
"""genearl configuration"""
configs.general = EasyDict()
configs.general.seed = 1005
configs.general.batch_size = 16
configs.general.epochs = 5
configs.general.precision = 32 # fp32
configs.general.num_workers = 16
configs.general.tf_dtype = tf.float16 if configs.general.precision==16 else tf.float32
configs.general.distributed = True

"""dataset configuration"""
configs.dataset = EasyDict()
configs.dataset.data_dir = None # defined at runtime
configs.dataset.dataset_name = 'CheXpert-v1.0'
configs.dataset.valid_ratio = 0.1
configs.dataset.auxiliary_columns = ['Sex', 'Age']
configs.dataset.target_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
configs.dataset.num_classes = len(configs.dataset.target_columns)
configs.dataset.image_size = (320, 320) # (384, 384), (320, 320)
configs.dataset.image_channels = 1
configs.dataset.cutoff = None # 10000 for test, 'None' for full model training

"""data augmentation configuration"""
configs.augmentation = EasyDict()
configs.augmentation.translation_height_factor = (-0.01, 0.01) # -3% ~ 3%
configs.augmentation.translation_width_factor  = (-0.01, 0.01)
configs.augmentation.zoom_height_factor = (-0.01, 0.01) # -3% ~ 3%
configs.augmentation.zoom_width_factor  = (-0.01, 0.01)
configs.augmentation.rotation_factor = (-0.01, 0.01)  # -3% ~ 3%

"""wandb logger"""
configs.wandb = EasyDict()
configs.wandb.use_wandb = False
configs.wandb.project_name = 'AAI'
configs.wandb.run_name = None # defined at runtime

"""optimizer configuration"""
configs.optimizer = EasyDict()
configs.optimizer.warm_up_rate = 0.1
configs.optimizer.learning_rate = 0.0001
configs.optimizer.weight_decay = 0.0004 #  
configs.optimizer.beta_1 = 0.9
configs.optimizer.beta_2 = 0.999
configs.optimizer.ema_momentum = 0.99

"""model configuration"""
# model - general
configs.model = EasyDict()
configs.model.backbone = 'densenet'
configs.model.regularization = 5e-5
configs.model.label_smoothing = 0.1
configs.model.use_aux_information = False
# model - densenet
configs.model.densenet = EasyDict()
configs.model.densenet.blocks = [6, 12, 48, 32] 
# model - convnext
configs.model.convnext = EasyDict()
configs.model.convnext.drop_path_rate = 0.1
configs.model.convnext.layer_scale_init_value = 1e-6
configs.model.convnext.depth = [3, 3, 27, 3]
configs.model.convnext.projection_dims = [128, 256, 512, 1024]
# model - classifier
configs.model.classifier = EasyDict()
configs.model.classifier.add_expert = False
configs.model.classifier.filter_sizes = [512, 256, 128]





MODEL_CONFIGS = {
    #convnext
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

DENSENET_CONFIGS = {
    "densenet121":{
        "blocks":[6, 12, 24, 16]
    },
    "densenet169":{
        "blocks":[6, 12, 32, 32]
    },
    "densenet201":{
        "blocks":[6, 12, 48, 32]
    }
}