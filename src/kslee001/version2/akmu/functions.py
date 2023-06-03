import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def csv_to_tensor(configs):
    data_dir = configs.data_dir

    # load csv with desired columns
    data = pd.read_csv(f"{configs.data_dir}/train.csv")[['Path'] + configs.target_columns]
    
    # fix image path
    data['Path'] = data['Path'].str.replace(configs.dataset_name, configs.data_dir, regex=False)

    # drop uncertain values (-1)
    for target_column in configs.target_columns:
        data[target_column] = data[target_column]

    return data




    
