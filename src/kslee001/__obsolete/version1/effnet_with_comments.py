"""
라이브러리 임포트 하는 부분 :    
    BACKBONE, BACKBONE_STR 에서 어떤 모델 쓸 것인지를 정했는데, (tensorflow.keras.applications 에 있는 pretrain된 모델들)
    다음에 내가 올릴 모델은 아마도 직접 짜는 모델이라 이거 신경 안써도 될 것임!
"""

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
BACKBONE = EfficientNetV2L
BACKBONE_STR = 'EfficientNetV2L'
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


"""
configuration 정하는 부분 :    
"""
# configurations
SEED = 1005
BATCH_SIZE = 8
EPOCHS = 10
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

    """
    랜덤시드 고정하는 함수. SEED 넣으면 랜덤 시드 고정됨
    """
    set_seed(SEED)
    
    """
    wandb 로그인 하는 부분
    """
    # wandb logging
    wandb.init(project='aai', name=f'tf-{BACKBONE_STR}-{MULTI_DEVICE_STR}')


    """
    csv_to_df() 라는 함수를 이용해서 train.csv를 가져옴 ! 
    [:CUTOFF] 를 넣은 이유는 테스트용... 테스트할 떄 데이터 다가져오면 넘 느려서 일부만 짤라서 하려고 넣은것
    (실제 학습할 때는 이거 None임)

    train_test_split은 train_df, valid_df 나누는 것임
    """
    # Load data
    df = csv_to_df() if not CUTOFF else csv_to_df()[:CUTOFF]
    train_df, valid_df = train_test_split(df, test_size=TEST_SIZE)
    TOTAL_STEPS = int(EPOCHS * len(train_df) / BATCH_SIZE)
    WARMUP_STEPS = int(TOTAL_STEPS * WARM_UP_RATE)


    """
    이 부분은 tensorflow용 dataset 만드는 부분
    pytorch의 dataset이랑 비슷한 것임
    - train_df['Path']는 이미지들의 경로들이 들어가 있고
    - train_df.iloc[:, 1:].values 는 label들이 들어가 있음 
        e.g., np.array([[1,0,0,0,1], [0,0,0,1,1], ...])
    AUTOTUNE은 나도 잘 모르겠음... 뭔가 parallel training을 위한 사전 준비작업인 것으로 보임
    ds_train (train dataset)에만 augmentation + normalization이 적용되고
    ds_valid는 augmentation없이 normalization만 적용됨 !!
    """
    # Dataset for training
    list_ds_train = tf.data.Dataset.from_tensor_slices((train_df['Path'].values, train_df.iloc[:,1:].values))
    ds_train = list_ds_train.map(process_path_train, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(AUTOTUNE)

    # Dataset for validation
    list_ds_valid = tf.data.Dataset.from_tensor_slices((valid_df['Path'].values, valid_df.iloc[:,1:].values))
    ds_valid = list_ds_valid.map(process_path_validation, num_parallel_calls=AUTOTUNE)
    ds_valid = ds_valid.batch(BATCH_SIZE)
    ds_valid = ds_valid.prefetch(AUTOTUNE)


    """
    이부분은 data parallel로 학습하는 부분
    """
    # Multi-gpu setting
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    if MULTI_DEVICE:
        """
        multi device training
        """
        with strategy.scope():
            # Load base model
            """
            BACKBONE은 위에서 정의한 efficientnetv2B0임.
            imagenet으로 학습된 weight 불러온다는 의미고, IMAGE_SIZE는 (384,384)임
            """
            base_model = BACKBONE(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))    
            base_model.summary()    
            # for layer in base_model.layers[:3]:
            #     layer.trainable = False

            """
            이 부분은 모델의 모든 layer를 학습 가능한 상태로 바꾼다는 것
            (torch의 requires_grad=True 랑 동일)
            """
            for layer in base_model.layers:
                layer.trainable = True

            # Add new layers
            """
            모델이 뱉는 것은 우리가 원하는 형태 (5개의 label)가 아니기 때문에, 우리가 원하는 형태로 바꿔주는 부분임. 
            NUM_CLASSES 는 5
            """
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            output = Dense(NUM_CLASSES, activation='sigmoid')(x)

            # model setting
            """
            모델 컴파일하고 learning rate 스케쥴러 정의하는 부분
            """
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
        """
        이건 single-gpu용 코드인데 우리는 무조건 multi-gpu로 할거니까 굳이 안봐도 되긴 함!
        구조는 똑같음
        """
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




    """
    tf model용 callback 정의하는건데
    아마 quantization이나 tflite 변환 떄 문제가 생기는 부분은
    위에 있는 model compile이나 이부분(call back 정의) 일것임

    custom callback이나 metric같은게 들어가기 때문에....
    우리 본 모델 훈련 때는 최대한 customized된 모듈 없이 한방에 quantization 되는 방식으로 쓸 예정 !!!!
    """
    # callbacks
    custom_metrics_callback = CustomMetricsCallback()
    wandb_callback = CustomWandbCallback()
    model_checkpoint = ModelCheckpoint(
        f"{BACKBONE_STR}-best_model.h5", 
        save_best_only=True,
    )


    """
    모델 훈련하는 부분 
    """
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

    """
    이 부분은 학습된 모델 저장된 것 (모델은 학습 과정에서 callback이 자동으로 저장함) 불러오는 부분
    with CustomObjectScope 같은 괴상한게 들어간 이유는 내가 custom module을 정의했기 때문...
    (CosineDecayWithWarmpup, AUROC 같은 애들)
    그냥 AUROC도 아니고 왜 AUROC_convert를 with 문에 집어넣었느냐? 하면, 그냥 AUROC는 
    'curve="ROC"' 
    이 부분 때문에 변환이 안됨 (tf 자체적인 문제인듯)
    이거는 내가 어떻게든 해결방법을 찾아 보겠슴다.......
    """
    # save model (TFLite)
    with CustomObjectScope({
            'CosineDecayWithWarmup': CosineDecayWithWarmup,
            'AUROC': AUROC_convert, 
        }):
        model = tf.keras.models.load_model(f"/home/n1/gyuseonglee/workspace/AmbientAI-2023/project/{BACKBONE_STR}-best_model.h5")


    """
    tf lite로 변환하는 부분인데, 이것도 결국은 다시 짜야 함
    그래서 지금 당장은 effnet으로 quantization하는 것이 어려울 수 있어서... 
    일단 toy model로 방법 알려주면 effnet에 내가 코드 짜서 그대로 적용해 볼게요 !!
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(f"converted_{BACKBONE_STR}.tflite", 'wb') as f:
        f.write(tflite_model)

    wandb.finish()




"""
이 아래 부분 함수는 굳이 자세하게 알 필요가 없음 !!!!!!

"""



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

def process_path_train(image_path, label):
    # Read the image from the path
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize(image, [224, 224])
    
    image = tf.image.random_flip_left_right(image)  # Randomly flip the image horizontally
    image = tf.image.random_brightness(image, max_delta=0.2)  # Randomly adjust brightness
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

def process_path_validation(image_path, label):
    # Read the image from the path
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

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
