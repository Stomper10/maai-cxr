import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import pandas as pd
from tqdm.auto import tqdm as tq
import tensorflow as tf

# private
from modules.model import TestModel, tf
from cfg import configs
import functions

if __name__ == '__main__':
    # multi-gpu training strategy
    strategy = tf.distribute.MirroredStrategy()

    # load datasets
    train_dataset, valid_dataset, test_dataset, uncertain_dataset = functions.load_datasets(configs) 

    model = TestModel_old()
    model.summary()
    exit()

    # settings 
    # with strategy.scope():
    model = TestModel(num_classes=5)
    model((tf.zeros((1, *configs.image_size, 3)), tf.zeros((1, 2))))
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=configs.learning_rate)
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.AUC(multi_label=True, num_labels=5)]
    model.compile(optimizer=optimizer, loss=criterion, metrics=metrics) 

    # training
    model.fit(train_dataset, epochs=configs.epochs, validation_data=valid_dataset)
    loss, auc = model.evaluate(test_dataset)

    # evaluation
    print("Test Loss:", loss)
    print("Test AUC:", auc)