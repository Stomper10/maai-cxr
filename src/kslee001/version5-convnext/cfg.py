from easydict import EasyDict
from functions import make_dir


configs = EasyDict()
configs.seed = 1005

# directory & wandb setting
configs.data_dir = 100 # defined at runtime
configs.dataset_name = 'CheXpert-v1.0'
configs.wandb_project = 'AAI'
configs.wandb_name = None # defined at runtime
configs.wandb_name_model = 'ConvNeXt' # defined at runtime

# training configuration
configs.cutoff = 1000  # for test ('None' for full model training)
configs.batch_size = 32
configs.epochs = 15
configs.valid_ratio = 0.1
configs.warm_up_rate = 0.1

# optimizer configuration
configs.learning_rate = 0.0001
configs.weight_decay = 0.0004 #  
configs.beta_1 = 0.9
configs.beta_2 = 0.999
configs.ema_momentum = 0.99

# model configuration
configs.blocks = [6, 12, 48, 32]  # number of convolution blocks in each stage
configs.conv_filters = [1280, 1440] # not implemented : number of convolution filters in each expert 
configs.drop_rate = 0.25
configs.regularization = 5e-5
configs.use_aux_information = True
configs.label_smoothing = 0.1

# data configuration
configs.auxiliary_columns = ['Sex', 'Age']
configs.target_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
configs.num_classes = len(configs.target_columns)
configs.image_size = (512, 512) # (384, 384), (320, 320)

# misc.
configs.num_workers = 16
configs.saved_model_path = "./best_model.h5"

# data augmentation configuration
configs.translation_height_factor = (-0.01, 0.01) # -3% ~ 3%
configs.translation_width_factor  = (-0.01, 0.01)
configs.zoom_height_factor = (-0.01, 0.01) # -3% ~ 3%
configs.zoom_width_factor  = (-0.01, 0.01)
configs.rotation_factor = (-0.01, 0.01)  # -3% ~ 3%
