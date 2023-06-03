from easydict import EasyDict
from functions import make_dir


configs = EasyDict()
configs.seed = 1005
configs.batch_size = 8
configs.epochs = 10
configs.learning_rate = 0.00005
configs.warm_up_rate = 0.1
configs.valid_ratio = 0.1

configs.data_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized'
configs.dataset_name = 'CheXpert-v1.0'
configs.auxiliary_columns = ['Sex', 'Age']
configs.target_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
configs.num_classes = len(configs.target_columns)
configs.checkpoint_path = "./checkpoints"

configs.image_size = (384, 384)

configs.saved_model_path = None


