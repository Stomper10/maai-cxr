import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import pandas as pd
from tqdm.auto import tqdm as tq
# import tensorflow as tf

# private
# from modules.model import TestModel
from cfg import configs
import functions

if __name__ == '__main__':
    # load datasets : tf tensor dataset, simlilar to torch dataloader
    configs.data_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized'
    train_data = pd.read_csv(f"{configs.data_dir}/train.csv").fillna(0.0)
    print(train_data['Path'][0])

    train_data['Path'] = train_data['Path'].str.replace("/", "_", regex=False)
    print(train_data['Path'][0])

    train_data['Path'] = train_data['Path'].str.replace(configs.dataset_name+'_train_', configs.data_dir+'/train_512/', regex=False)
    print(train_data['Path'][0])

    # print(train_data.columns)
    

    example = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized/valid_512/patient64541_study1_view1_frontal.jpg'