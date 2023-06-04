import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import pandas as pd
from tqdm.auto import tqdm as tq
import tensorflow as tf
from tensorflow.keras import mixed_precision

# private
from modules.model import A2IModel, TestModel_old
from cfg import configs
import functions

if __name__ == '__main__':
    # multi-gpu training strategy
    strategy = tf.distribute.MirroredStrategy()

    # mixed precision policy
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # load datasets
    train_dataset, valid_dataset, test_dataset, uncertain_dataset = functions.load_datasets(configs) 

    # settings 
    with strategy.scope():
        model = A2IModel(num_classes=5)
        model.initialize()
        model.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=configs.learning_rate)
        criterion = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        metrics = [tf.keras.metrics.AUC(multi_label=True, num_labels=5)]
        model.compile(optimizer=optimizer, loss=criterion, metrics=metrics) 

    # training
    model.fit(train_dataset, epochs=configs.epochs, validation_data=valid_dataset)
    loss, auc = model.evaluate(test_dataset)

    # evaluation
    print("Test Loss:", loss)
    print("Test AUC:", auc)