import os

import tensorflow as tf
### Q1. Import modules ###
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Multiply, ReLU, Input, Dense, Activation, Flatten, Conv2D, \
    DepthwiseConv2D, BatchNormalization, GlobalAveragePooling2D, Dropout, AveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import hard_sigmoid
#########################

"""
Make sure your runtime type is GPU!
"""
physical_devices = tf.config.list_physical_devices('GPU')
print('Num_GPUs:{}, List:{}'.format(len(physical_devices), physical_devices))

REGULARIZATION = 4e-5
initializer = tf.keras.initializers.HeNormal(seed=22364)

def h_swish(x):
    return Multiply()([x, Activation(hard_sigmoid)(x)])

def _inverted_res_block(inputs, expansion, filters, strides):
    x = inputs
    in_chnls = inputs.shape[-1]

    # Expansion
    if expansion != 1:
        x = Conv2D(kernel_size=1, filters=in_chnls*expansion, strides=1,
                   padding='same', use_bias=False, kernel_regularizer=l2(REGULARIZATION),
                   kernel_initializer=initializer
                  )(x)
        x = BatchNormalization(momentum=0.999, epsilon=0.001)(x)
        x = Activation(h_swish)(x)

    # Depthwise convolution
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same',
                       use_bias=False, depthwise_regularizer=l2(REGULARIZATION),
                       kernel_initializer=initializer
                       )(x)
    x = BatchNormalization(momentum=0.999, epsilon=0.001)(x)
    x = Activation(h_swish)(x)


    # Linear bottleneck
    x = Conv2D(kernel_size=1, filters=filters, strides=1, padding='same', use_bias=False, kernel_regularizer=l2(REGULARIZATION),
              kernel_initializer=initializer
              )(x)
    x = BatchNormalization(momentum=0.999, epsilon=0.001)(x)


    # Residual connection
    if in_chnls == filters and strides == 1:
        x = Add()([inputs, x])

    return x #return output of layer


def MobileNetV2plus(input_shape, classes):
    inputs = Input(shape=input_shape)

    # ====== stem stage ======
    # 32 x 32 x 3
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(inputs)
    x = BatchNormalization(momentum=0.999, epsilon=0.001)(x)
    x = Activation(h_swish)(x)


    # ====== inverted residual blocks ======
    # 16x16x32 | 1 | 16 | 1      x 1
    x = _inverted_res_block(inputs=x, expansion=1, filters=16, strides=1)

    # 16x16x16 | 6 | 24 | 2 |    x 2
    x = _inverted_res_block(inputs=x, expansion=6, filters=24, strides=1)
    x = _inverted_res_block(inputs=x, expansion=6, filters=24, strides=1)

    #  8x8x24 | 6 | 32 | 2 |    x 3
    x = _inverted_res_block(inputs=x, expansion=6, filters=32, strides=2)
    x = _inverted_res_block(inputs=x, expansion=6, filters=32, strides=1)
    x = _inverted_res_block(inputs=x, expansion=6, filters=32, strides=1)

    #  4x4x32 | 6 | 64 | 2 |    x 4
    x = _inverted_res_block(inputs=x, expansion=6, filters=64, strides=2)
    x = _inverted_res_block(inputs=x, expansion=6, filters=64, strides=1)
    x = _inverted_res_block(inputs=x, expansion=6, filters=64, strides=1)
    x = _inverted_res_block(inputs=x, expansion=6, filters=64, strides=1)

    #  2x2x64 | 6 | 96 | 1 |    x 3
    x = _inverted_res_block(inputs=x, expansion=6, filters=96, strides=2)
    x = _inverted_res_block(inputs=x, expansion=6, filters=96, strides=1)
    x = _inverted_res_block(inputs=x, expansion=6, filters=96, strides=1)

    #  2x2x96 | 6 | 160 | 2 |    x 3 -> last block
    x = _inverted_res_block(inputs=x, expansion=6, filters=160, strides=1)
    x = _inverted_res_block(inputs=x, expansion=6, filters=160, strides=1)
    x = _inverted_res_block(inputs=x, expansion=6, filters=160, strides=1)

    # ====== last stage ======
    x = Conv2D(kernel_size=1, filters=960, strides=1,
                padding='same', use_bias=False, kernel_regularizer=l2(REGULARIZATION),
              kernel_initializer=initializer
              )(x)
    x = BatchNormalization(momentum=0.999, epsilon=0.001)(x)
    x = Activation(h_swish)(x)

    x = GlobalAveragePooling2D(keepdims=True)(x)

    x = Conv2D(kernel_size=1, filters=1280, strides=1,
                padding='same', use_bias=False, kernel_regularizer=l2(REGULARIZATION),
              kernel_initializer=initializer
              )(x)
    x = Activation(h_swish)(x)
    x = Flatten()(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)

    # ====== FC layer ======
    outputs = Dense(classes, activation='softmax', kernel_initializer=initializer)(x)

    return Model(inputs=inputs, outputs=outputs)


my_mobilenet = MobileNetV2plus((32,32,3),classes=10)


import tensorflow_datasets as tfds
# Load  cifar10 dataset
(dataset_train, dataset_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# The dataset is split int train/test dataset
# Let's split the trainset into train/validation dataset
# Set train/validation split as 0.8:0.2
train_size = int(ds_info.splits['train'].num_examples * 0.9)
val_size = ds_info.splits['train'].num_examples - train_size

# Use take method to retrieve (train_size) data as New training data
ds_train = dataset_train.take(train_size)

# Use skip method to retrieve remaining data as validation data
ds_val = dataset_train.skip(train_size)


### Q3. Preporcessing ###
rescaling = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
])
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
    tf.keras.layers.RandomRotation(0.2),
])


### Q4. Model compile ###

BATCH_SIZE = 4
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
EPOCHS = 20
WARMUP_PERCENTILE = 20


ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_val = ds_val.batch(BATCH_SIZE)
ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)
ds_test = dataset_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

optimizer_fn = tf.keras.optimizers.Adam(
    lr=LEARNING_RATE,
    decay=WEIGHT_DECAY,
  )
my_mobilenet = tf.keras.Sequential([
    rescaling,
    data_augmentation,
    my_mobilenet
])
my_mobilenet.compile(optimizer=optimizer_fn, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
#########################

### Q5. Callbacks ###
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
import math


# learning rate schedule function
def learning_rate_scheduler(epoch, lr_max, lr_min, n_epochs):
    if epoch < n_epochs//WARMUP_PERCENTILE:
        lr = lr_min + (epoch/(n_epochs//WARMUP_PERCENTILE))*lr_max
        return lr
    cos_inner = (math.pi * (epoch % n_epochs)) / n_epochs
    cos_out = math.cos(cos_inner) + 1
    lr = lr_min + 0.5 * (lr_max - lr_min) * cos_out
    return lr


# ModelCheckpoint callback
checkpoint_path = "./checkpoints"
cps_prefix = os.path.join(checkpoint_path, "cp_{epoch}")
model_checkpoint_callback = ModelCheckpoint(
    filepath=cps_prefix,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


lr_scheduler_callback = LearningRateScheduler(
    lambda epoch: learning_rate_scheduler(epoch, LEARNING_RATE, LEARNING_RATE/10, EPOCHS))

# log_dir = './root'
callbacks = []
callbacks.append(model_checkpoint_callback)
callbacks.append(lr_scheduler_callback)

#####################

train1 = my_mobilenet.fit(ds_train, batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          callbacks=callbacks,
                          validation_data=ds_val)
# train2 = my_mobilenet.fit(ds_train, batch_size=BATCH_SIZE,
#                           epochs=EPOCHS,
#                           callbacks=callbacks,
#                           validation_data=ds_val)
