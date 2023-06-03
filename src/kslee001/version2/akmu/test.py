import os
# make error code invisible
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import random
import numpy as np
import tensorflow as tf

# private
from modules.backend import TestModel


# loss : https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC
if __name__ == '__main__':
    batch_size = 8
    X = tf.random.normal((256, 128, 128, 3))
    Y = tf.floor(tf.random.uniform(shape=(256, 5), minval=0, maxval=2))
    dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)
    
    model = TestModel(num_classes=5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_auc = tf.keras.metrics.AUC(multi_label=True, num_labels=5)

        for batch, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                # forward
                yhat = model(x_batch, training=True)
                loss = criterion(y_batch, yhat) # y gt first
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss)

            y_pred_prob = tf.nn.sigmoid(yhat)
            epoch_auc.update_state(y_batch, y_pred_prob)


        print("Epoch {}: Loss = {}, AUC = {}".format(epoch, epoch_loss_avg.result(), epoch_auc.result()))


    #metric = tf.keras.metrics.AUC(
    #     multi_label=True,
    #     num_labels=5,
    #     from_logits=True,
    # )




