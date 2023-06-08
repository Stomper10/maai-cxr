import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    if configs.general.precision == 16:
        dtype = np.float16
        tf_dtype = tf.float16
    elif configs.general.precision == 32:
        dtype = np.float32
        tf_dtype = tf.float32
        
    # def process_path(X, X_aux, label):
    def process_path(X, label):
        # Read the image from the path
        image = tf.io.read_file(X)
        image = tf.image.decode_jpeg(image, channels=configs.dataset.image_channels)
        image = tf.cast(image, tf_dtype) / 255.0
        # return (image, X_aux), label
        return image, label
    
    # load csv with desired columns
    # (auxiliary data : Sex, Age)
    data_dir = configs.dataset.data_dir
    train_data = pd.read_csv(f"{configs.dataset.data_dir}/train.csv")[['Path', 'Frontal/Lateral'] + configs.dataset.auxiliary_columns + configs.dataset.target_columns].fillna(0.0)
    test_data = pd.read_csv(f"{configs.dataset.data_dir}/valid.csv")[['Path', 'Frontal/Lateral'] + configs.dataset.auxiliary_columns + configs.dataset.target_columns].fillna(0.0)

    # drop Lateral images
    train_data = train_data[train_data['Frontal/Lateral']=='Frontal'].reset_index(drop=True)
    test_data = test_data[test_data['Frontal/Lateral']=='Frontal'].reset_index(drop=True)
    del train_data['Frontal/Lateral']
    del test_data['Frontal/Lateral']

    # fix image path
    if configs.dataset.image_size[0] == 384:
        train_data['Path'] = train_data['Path'].str.replace(configs.dataset.dataset_name, configs.dataset.data_dir, regex=False)
        test_data['Path'] = test_data['Path'].str.replace(configs.dataset.dataset_name, configs.dataset.data_dir, regex=False)
    if (configs.dataset.image_size[0] == 512) | (configs.dataset.image_size[0] == 320):
        train_data['Path'] = train_data['Path'].str.replace("/", "_", regex=False)
        train_data['Path'] = train_data['Path'].str.replace(configs.dataset.dataset_name+'_train_', configs.dataset.data_dir+f'/train_{configs.dataset.image_size[0]}/', regex=False)

        test_data['Path'] = test_data['Path'].str.replace("/", "_", regex=False)
        test_data['Path'] = test_data['Path'].str.replace(configs.dataset.dataset_name+'_valid_', configs.dataset.data_dir+f'/valid_{configs.dataset.image_size[0]}/', regex=False)

    # convert Sex to int format (auxiliary)
    train_data['Sex'] = np.where(train_data['Sex']=='Male', 0, 1)
    train_auxiliary = train_data[['Sex', 'Age']]
    test_data['Sex'] = np.where(test_data['Sex']=='Male', 0, 1)
    test_auxiliary = test_data[['Sex', 'Age']]

    # train-valid split
    if configs.dataset.cutoff is not None:
        # use small part of dataset
        train_data = train_data[:configs.dataset.cutoff] 

    """ train / valid split by patients """
    if configs.dataset.image_size[0] in [512, 320]:
        patients = train_data['Path'].str.split(f"{str(configs.dataset.image_size[0])}/", expand=True)[1]
        patients = patients.str.split('_study', expand=True)[0] # patients array
    elif configs.dataset.image_size[0] == 384:
        patients = train_data['Path'].str.split(f"train/", expand=True)[1]
        patients = patients.str.split("/study", expand=True)[0]
    train_data['patient'] = patients

    patients = patients.drop_duplicates().reset_index(drop=True)
    train_patients, valid_patients = train_test_split(patients, test_size=configs.dataset.valid_ratio)

    # valid data must be defined first ! (train_data = train_data ~~)
    valid_data = train_data[train_data['patient'].isin(valid_patients)].reset_index(drop=True)
    train_data = train_data[train_data['patient'].isin(train_patients)].reset_index(drop=True)
    del train_data['patient'], valid_data['patient']

    """scaler """
    # normalize 'Age' using StandardScaler after train-valid-test split
    # note that valid data or test data MUST NOT be fitted 
    scaler = StandardScaler()
    train_data['Age'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))
    valid_data['Age'] = scaler.transform(valid_data['Age'].values.reshape(-1, 1))
    test_data['Age'] = scaler.transform(test_data['Age'].values.reshape(-1, 1))

    # get tf tensors
    X_train = train_data.iloc[:, 0].values
    X_train_aux = train_data.iloc[:, 1:3].values.astype(dtype)
    Y_train = train_data.iloc[:, 3:].values.astype(dtype)

    X_valid = valid_data.iloc[:, 0].values
    X_valid_aux = valid_data.iloc[:, 1:3].values.astype(dtype)
    Y_valid = valid_data.iloc[:, 3:].values.astype(dtype)

    X_test = test_data.iloc[:, 0].values
    X_test_aux = test_data.iloc[:, 1:3].values.astype(dtype)
    Y_test = test_data.iloc[:, 3:].values.astype(dtype)

    # dataset
    # note that map method and batch method should be applied in sequence
    # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train_aux, Y_train))
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(configs.general.batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(configs.general.batch_size)
    train_dataset.steps_per_epoch = len(X_train) // configs.general.batch_size

    # valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, X_valid_aux, Y_valid))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
    valid_dataset = valid_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    valid_dataset = valid_dataset.batch(configs.general.batch_size)
    valid_dataset = valid_dataset.prefetch(configs.general.batch_size)

    # test_dataset = tf.data.Dataset.from_tensor_slices((X_test, X_test_aux, Y_test))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = test_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(configs.general.batch_size)
    test_dataset = test_dataset.prefetch(configs.general.batch_size)

    return train_dataset, valid_dataset, test_dataset






