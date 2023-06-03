import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.AUTOTUNE


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def load_datasets(configs):
    # load csv with desired columns
    data_dir = configs.data_dir
    train_data = pd.read_csv(f"{configs.data_dir}/train.csv")[['Path', 'Frontal/Lateral'] + configs.auxiliary_columns + configs.target_columns].fillna(0.0)
    test_data = pd.read_csv(f"{configs.data_dir}/valid.csv")[['Path', 'Frontal/Lateral'] + configs.auxiliary_columns + configs.target_columns].fillna(0.0)

    # drop Lateral images
    train_data = train_data[train_data['Frontal/Lateral']=='Frontal'].reset_index(drop=True)
    test_data = test_data[test_data['Frontal/Lateral']=='Frontal'].reset_index(drop=True)
    del train_data['Frontal/Lateral']
    del test_data['Frontal/Lateral']

    # fix image path
    train_data['Path'] = train_data['Path'].str.replace(configs.dataset_name, configs.data_dir, regex=False)
    test_data['Path'] = test_data['Path'].str.replace(configs.dataset_name, configs.data_dir, regex=False)

    # get uncertain rows (-1) : for train dataset
    uncertain = set()
    for column in configs.target_columns:
        uncertain.update(train_data[train_data[column]==-1].index.tolist())
    

    # use certain data only : for train dataset
    certain = sorted(set(train_data.index) - set(uncertain))
    train_data     = train_data.iloc[certain].reset_index(drop=True)
    uncertain_data = train_data.iloc[uncertain].reset_index(drop=True)

    # NOTE: Only 135,494 images remain in 'train_data', 
    # indicating that approximately 80,000 images have been dropped.
    # 'uncertain_data' has been defined for potential future requirements

    # train-valid split
    train_data, valid_data = train_test_split(train_data, test_size=configs.valid_ratio)
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)

    # get tf tensors
    X_train = train_data.iloc[:, 0].values
    Y_train = train_data.iloc[:, 1:].astype(np.float32)
    X_valid = valid_data.iloc[:, 0].values
    Y_valid = valid_data.iloc[:, 1:].astype(np.float32)
    X_test = test_data.iloc[:, 0].values
    Y_test = test_data.iloc[:, 1:].astype(np.float32)
    
    # dataset
    # note that map method and batch method should be applied in sequence
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.map(process_path_train, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(configs.batch_size)

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
    valid_dataset = valid_dataset.map(process_path_test, num_parallel_calls=AUTOTUNE)
    valid_dataset = valid_dataset.batch(configs.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = test_dataset.map(process_path_test, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(configs.batch_size)

    return train_dataset, valid_dataset, test_dataset




def process_path_train(image_path, label):
    # Read the image from the path
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # ==== augmentation =======================================
    image = tf.image.random_flip_left_right(image)  # Randomly flip the image horizontally
    image = tf.image.random_brightness(image, max_delta=0.2)  # Randomly adjust brightness
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # ==== augmentation =======================================

    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def process_path_test(image_path, label):
    # Read the image from the path
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0    
    return image, label