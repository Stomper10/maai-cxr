from easydict import EasyDict
from functions import make_dir


configs = EasyDict()
configs.seed = 1005
# directory & wandb setting
configs.data_dir = None # defined at runtime
configs.dataset_name = 'CheXpert-v1.0'
configs.wandb_project = 'AAI'
configs.wandb_name = None # defined at runtime

# training configuration
configs.batch_size = 8
configs.epochs = 10
configs.learning_rate = 0.00005
configs.valid_ratio = 0.1
configs.warm_up_rate = 0.1

# model configuration
configs.drop_rate = 0.25
configs.regularization = 4e-5
configs.use_aux_information = True

# data configuration
configs.auxiliary_columns = ['Sex', 'Age']
configs.target_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
configs.num_classes = len(configs.target_columns)
configs.image_size = (384, 384)

configs.saved_model_path = None

