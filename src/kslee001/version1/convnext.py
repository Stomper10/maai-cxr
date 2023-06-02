import os
import random
import numpy as np
# make error code invisible
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0  # param size :  7.1 M  -- for testing
from tensorflow.keras.applications import ConvNeXtBase      # param size : 88.5 M  
from tensorflow.keras.applications import EfficientNetV2L, ConvNeXtLarge # not available -- gpu capacity ㅠㅠ
BACKBONE = ConvNeXtBase
BACKBONE_STR = 'ConvNeXtBase'
CUTOFF = None #1000 # for debugging
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import roc_auc_score

import pandas as pd
import wandb
from wandb.keras import WandbCallback


# configurations
SEED = 1005
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.0001
WARM_UP_RATE = 0.1
MULTI_DEVICE = True
MULTI_DEVICE_STR = 'multi' if MULTI_DEVICE else 'single'

TOTAL_STEPS = None # defined at runtime
WARMUP_STEPS = None # defined at runtime

DATASET_DIR = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized'
DATASET_NAME = 'CheXpert-v1.0'
TARGET_COLUMNS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
NUM_CLASSES = len(TARGET_COLUMNS)
TEST_SIZE = 0.1
IMAGE_SIZE = [384, 384]

CHECKPOINT_PATH = "./checkpoints-single/cp-{epoch:04d}.ckpt"  # Path for saving checkpoints
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

AUTOTUNE = tf.data.AUTOTUNE


def main():
    # fix seed
    set_seed(SEED)
    
    # wandb logging
    wandb.init(project='aai', name=f'tf-{BACKBONE_STR}-{MULTI_DEVICE_STR}')

    # Load data
    df = csv_to_df() if not CUTOFF else csv_to_df()[:CUTOFF]
    train_df, valid_df = train_test_split(df, test_size=TEST_SIZE)
    TOTAL_STEPS = int(EPOCHS * len(train_df) / BATCH_SIZE)
    WARMUP_STEPS = int(TOTAL_STEPS * WARM_UP_RATE)

    # Dataset for training
    list_ds_train = tf.data.Dataset.from_tensor_slices((train_df['Path'].values, train_df.iloc[:,1:].values))
    ds_train = list_ds_train.map(process_path, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(AUTOTUNE)

    # Dataset for validation
    list_ds_valid = tf.data.Dataset.from_tensor_slices((valid_df['Path'].values, valid_df.iloc[:,1:].values))
    ds_valid = list_ds_valid.map(process_path, num_parallel_calls=AUTOTUNE)
    ds_valid = ds_valid.batch(BATCH_SIZE)
    ds_valid = ds_valid.prefetch(AUTOTUNE)

    # Multi-gpu setting
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    if MULTI_DEVICE:
        with strategy.scope():
            # Load base model
            base_model = BACKBONE(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))        
            # freeze first 3 layers
            for layer in base_model.layers[:3]:
                layer.trainable = False
            for layer in base_model.layers[3:]:
                layer.trainable = True

            # Add new layers
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            output = Dense(NUM_CLASSES, activation='sigmoid')(x)

            # model setting
            model = Model(inputs=base_model.input, outputs=output)
            lr_schedule = CosineDecayWithWarmup(LEARNING_RATE, TOTAL_STEPS, WARMUP_STEPS)
            tf.keras.optimizers.schedules.serialize(lr_schedule)

            # Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr_schedule), 
                loss='binary_crossentropy', 
                metrics=['accuracy', AUROC(name='auroc')], # empty
            )

    else:
        base_model = BACKBONE(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))        
        # freeze first 3 layers
        for layer in base_model.layers[:3]:
            layer.trainable = False
        for layer in base_model.layers[3:]:
            layer.trainable = True

        # Add new layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        output = Dense(NUM_CLASSES, activation='sigmoid')(x)

        # model setting
        model = Model(inputs=base_model.input, outputs=output)
        lr_schedule = CosineDecayWithWarmup(LEARNING_RATE, TOTAL_STEPS, WARMUP_STEPS)
        tf.keras.optimizers.schedules.serialize(lr_schedule)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr_schedule), 
            loss='binary_crossentropy', 
            metrics=['accuracy', AUROC(name='auroc')], # empty
        )



    # callbacks
    custom_metrics_callback = CustomMetricsCallback()
    wandb_callback = CustomWandbCallback()
    model_checkpoint = ModelCheckpoint(
        f"{BACKBONE_STR}-best_model.h5", 
        save_best_only=True,
    )

    # Train
    model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_valid,
        callbacks=[
            model_checkpoint,
            custom_metrics_callback, 
            wandb_callback, 
        ]
    )

    # save model (TFLite)
    with CustomObjectScope({
            'CosineDecayWithWarmup': CosineDecayWithWarmup,
            'AUROC': AUROC_convert, 
        }):
        model = tf.keras.models.load_model(f"/home/n1/gyuseonglee/workspace/AmbientAI-2023/project/{BACKBONE_STR}-best_model.h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("converted_{BACKBONE_STR}.tflite", 'wb') as f:
        f.write(tflite_model)


    wandb.finish()



def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


# return : dataframe with columns [Path, label_1, label_2, ... label_n]
def csv_to_df():
    # TODO : load data into pandas dataframe
    data = pd.read_csv(f"{DATASET_DIR}/train.csv").fillna(0.0)

    # TODO : fix image path: 
    # ex. CheXpert-v1.0/patient0000... 
    # ex.       > /mnt/e/dataset/chexpert/CheXpert-v1.0/patient0000...
    data['Path'] = data['Path'].str.replace(DATASET_NAME, DATASET_DIR, regex=False)

    # TODO : fill null values
    for col_idx in range(5, len(data.columns)):
        data[data.columns[col_idx]] = data[data.columns[col_idx]].astype(str)
        data[data.columns[col_idx]] = data[data.columns[col_idx]].str.replace("-1.0", "0.0").astype(float).fillna(0.0)

    # TODO : column reduction
    target_columns = ['Path'] + TARGET_COLUMNS
    dataframe = data[target_columns].reset_index(drop=True)

    return dataframe


# Function to load and preprocess each image
def process_path(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


# lr scheduler
class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, total_steps, warmup_steps=0):
        super(CosineDecayWithWarmup, self).__init__()
        assert warmup_steps < total_steps
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.cosine_decay = tf.keras.experimental.CosineDecay(
            learning_rate, total_steps - warmup_steps)

        
    @tf.function
    def __call__(self, step):
        if step < self.warmup_steps:
            return (self.learning_rate / 
                    tf.cast(self.warmup_steps, tf.float32) * 
                    tf.cast((step + 1), tf.float32))
        return self.cosine_decay(step - self.warmup_steps)

    def get_config(self):
        return {"learning_rate": np.array(self.learning_rate),
                "total_steps": np.array(self.total_steps),
                "warmup_steps": np.array(self.warmup_steps)}


class CustomMetricsCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            wandb.log({'train_loss_step': logs.get('loss'), 'train_acc_step': logs.get('accuracy')}, commit=False)

    def on_test_batch_end(self, batch, logs=None):
        if logs is not None:
            wandb.log({'val_loss_step': logs.get('loss'), 'val_acc_step': logs.get('accuracy')}, commit=False)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            wandb.log({'train_loss_epoch': logs.get('loss'), 'train_acc_epoch': logs.get('accuracy'),
                       'val_loss_epoch': logs.get('val_loss'), 'val_acc_epoch': logs.get('val_accuracy')})



class CustomWandbCallback(WandbCallback):
    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            wandb.log({'train_loss': logs.get('loss'), 'train_accuracy': logs.get('accuracy')}, commit=False)
        super().on_train_batch_end(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        if logs is not None:
            wandb.log({'val_loss': logs.get('loss'), 'val_accuracy': logs.get('accuracy')}, commit=False)
        super().on_test_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if logs is not None:
            wandb.log({'train_loss': logs.get('loss'), 'val_loss': logs.get('val_loss'),
                       'train_accuracy': logs.get('accuracy'), 'val_accuracy': logs.get('val_accuracy')})


class AUROC(AUC):
    def __init__(self, name='auroc', **kwargs):
        super().__init__(name=name, curve='ROC', **kwargs)

class AUROC_convert(AUC):
    def __init__(self, name='auroc', **kwargs):
        super().__init__(name=name, **kwargs)

def auroc(y_true, y_pred):
    return AUC(curve='ROC')(y_true, y_pred)



if __name__ == '__main__':
    main()
