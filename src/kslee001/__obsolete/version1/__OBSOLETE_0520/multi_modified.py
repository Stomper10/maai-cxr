import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
from tqdm.auto import tqdm as tq
import pandas as pd
import numpy as np
import json
from joblib import Parallel, delayed

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
	Conv2D, 
	MaxPooling2D, Flatten, 
	Dense, 
	Dropout,
)
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M as effnet

import cv2
# import gdcm
# import pydicom
import matplotlib.pyplot as plt


class CFG:
	def __init__(self):
		return
strategy = tf.distribute.MirroredStrategy()	


configs = CFG()
# directory
configs.dataset_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-small'
# configs.mode = 'train'
# configs.data_dir = f"{configs.dataset_dir}/{configs.mode}.csv"

# train setting
configs.num_epochs = 10
configs.batch_size = 64
GLOBAL_BATCH_SIZE = (configs.batch_size 
                     * strategy.num_replicas_in_sync)

configs.image_size = (256, 256)
configs.num_classes = 14


class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, configs, mode='train', shuffle=True):
        self.configs = configs
        self.mode = mode
        self.shuffle = shuffle
        self.data_dir = f"{configs.dataset_dir}/{mode}.csv"
        self.batch_size = configs.batch_size
        self.image_size = configs.image_size
        self._init_dataset()

    def _init_dataset(self):
        data = pd.read_csv(self.data_dir).fillna(0.0)
        data['Path'] = data['Path'].str.replace("CheXpert-v1.0-small", self.configs.dataset_dir)
        for col_idx in range(5, len(data.columns)):
            data[data.columns[col_idx]] = data[data.columns[col_idx]].astype(str)
            data[data.columns[col_idx]] = data[data.columns[col_idx]].str.replace("-1", "0").astype(float)
        self.X = data['Path'].values
        self.Y = data.values[:, 5:].astype(np.float32)

        self.dataset_size = len(self.X)
        self.indices = np.arange(self.dataset_size)

        self.ds = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
        self.ds = self.ds.shuffle(buffer_size=self.dataset_size, reshuffle_each_iteration=self.shuffle)
        self.ds = self.ds.map(self._load_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.ds = self.ds.batch(self.batch_size)

    def _load_image(self, path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def __len__(self):
        return int(np.ceil(self.dataset_size / self.batch_size))

    def __getitem__(self, idx):
        return next(iter(self.ds.skip(idx)))

    def on_epoch_end(self):
        self.ds = self.ds.shuffle(buffer_size=self.dataset_size, reshuffle_each_iteration=self.shuffle)
        self.ds = self.ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def create_model(configs):
	model = effnet(
		include_top=True,
		weights=None,
		input_shape=(
      configs.image_size[0], 
      configs.image_size[1], 
      3),
		classes=configs.num_classes,
		classifier_activation=None, #'sigmoid'
		include_preprocessing=True,
	)
	return model


if __name__ == '__main__':

    # load dataset
    train_sequence = ImageSequence(configs, mode='train', shuffle=True)
    valid_sequence = ImageSequence(configs, mode='valid', shuffle=False)
    train_dataset = tf.data.Dataset.from_generator(
        generator=lambda: train_sequence,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            (configs.batch_size, configs.image_size[0], configs.image_size[1], 3),
            (configs.batch_size, configs.num_classes))
    )
    valid_dataset = tf.data.Dataset.from_generator(
        generator=lambda: valid_sequence,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            (configs.batch_size, configs.image_size[0], configs.image_size[1], 3),
            (configs.batch_size, configs.num_classes)
        )
    )
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    # loss function
    with strategy.scope():
        loss_object = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )
        def compute_loss(labels, preds):
            per_ex_loss = loss_object(labels, preds)
            return tf.nn.compute_average_loss(
                per_ex_loss, 
                global_batch_size=GLOBAL_BATCH_SIZE
            )
        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        train_accuracy = tf.keras.metrics.BinaryAccuracy(
            name='train_accuracy'
        )
        valid_accuracy = tf.keras.metrics.BinaryAccuracy(
            name='valid_accuracy'
        )
        
    # model, optimizer, checkpoint
    with strategy.scope():
        model = create_model(configs)
        optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer, model=model
        )
        
    # steps
    def train_step(batch):
        x, y = batch
        with tf.GradientTape() as tape:
            yhat = model(x, training=True)
            loss = compute_loss(y, yhat)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(
            grads, model.trainable_variables
        ))
        train_accuracy.update_state(y, yhat)
        return loss

    def valid_step(batch):
        x, y = batch
        yhat = model(x, training=False)
        v_loss = loss_object(y, yhat)
        valid_loss.update_state(v_loss)
        valid_accuracy.update_state(y, yhat)
        
    @tf.function
    def distributed_train_step(batch):
        per_replica_losses = strategy.run(
            train_step, 
            args=(batch,)
        )    
        return strategy.reduce(
                tf.distribute.ReduceOp.SUM, 
                per_replica_losses,
                axis=None
                )
        
    @tf.function
    def distributed_valid_step(batch):
        return strategy.run(valid_step, agrs=(batch,))
        
        
    # epoch
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    for epoch in range(configs.num_epochs):
        total_loss = 0.0
        num_batches = 0
        # train
        for idx, batch in enumerate(train_dist_dataset):
            total_loss += distributed_train_step(batch)
            num_batches +=1
            print(f"current batch idx : {idx} | current_loss = {total_loss:.4f}", end ='\r')
        train_loss = total_loss / num_batches

        # validation
        for batch in valid_dist_dataset:
            distributed_valid_step(batch)
        
        if epoch %2 == 0:
            checkpoint.save(checkpoint_prefix)

        print(f"EPOCH {epoch} | train loss : {train_loss:.4f} | valid loss : {valid_loss:.4f} | train accuracy : {train_accuracy.result()*100} | valid accuracy = {valid_accuracy.result()*100}")
        
        valid_loss.reset_states()
        train_accuracy.reset_states()
        valid_accuracy.reset_states()
        