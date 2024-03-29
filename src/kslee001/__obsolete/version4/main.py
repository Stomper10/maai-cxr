import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import argparse
import pandas as pd
from tqdm.auto import tqdm as tq
import tensorflow as tf
from tensorflow.keras import mixed_precision
import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger

# private
from modules.model import A2IModelBase as A2IModel
from modules.lr_scheduler import CustomOneCycleSchedule, LearningRateLogger
from cfg import configs
import functions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cluster', action='store')
    args = parser.parse_args()
    cluster = args.cluster
    if cluster == 'gsds-ab':
        configs.data_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized'
    if cluster == 'gsds-c':
        configs.data_dir = '/home/gyuseonglee/workspace/dataset/chexpert-resized'
    if cluster == 'akmu':
        configs.data_dir = '/data/s1/gyuseong/chexpert-resized'
    configs.wandb_name = f'a2i-{cluster}-{configs.image_size[0]}'

    # wandb initialization
    wandb.login()
    run = wandb.init(project=configs.wandb_project, name=configs.wandb_name, config=configs)

    # multi-gpu training strategy
    strategy = tf.distribute.MirroredStrategy()

    # mixed precision policy
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)

    # load datasets
    train_dataset, valid_dataset, test_dataset, uncertain_dataset = functions.load_datasets(configs) 
    
    # settings 
    with strategy.scope():
        model = A2IModel(
            img_size=configs.image_size, 
            num_classes=configs.num_classes, 
            blocks=configs.blocks,
            conv_filters=configs.conv_filters,
            use_aux_information=configs.use_aux_information, 
            drop_rate=configs.drop_rate, 
            reg=configs.regularization, 
            seed=configs.seed)
        model.initialize()
        model.summary()
        scheduler = CustomOneCycleSchedule(
            max_lr=configs.learning_rate, 
            epochs=configs.epochs,
            steps_per_epoch=train_dataset.steps_per_epoch,
            start_lr=None, end_lr=None, warmup_fraction=configs.warm_up_rate,
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=scheduler,
            # weight decay not included now...
        )
        criterion = tf.keras.losses.BinaryCrossentropy(from_logits=False) # True : raw score / False : probability score (from a sigmoid function)
        metrics = [tf.keras.metrics.AUC(multi_label=True, num_labels=configs.num_classes)]
        model.compile(optimizer=optimizer, loss=criterion, metrics=metrics) 

        callbacks = [
            WandbCallback(save_model=False), # wandb - system
            WandbMetricsLogger(log_freq='batch'), # wandb - metrics

            # model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath=configs.saved_model_path,
                monitor='val_loss', 
                save_best_only=False,  # Save only the best model based on the monitored metric
                save_weights_only=False,  
                # Save just the weights. do not save the entire model (architecture)
                mode='min', 
                verbose=0  # do not print messages during saving
            ),

            # learning rate logger
            LearningRateLogger(),
        ]

    # training
    model.fit(
        train_dataset, 
        epochs=configs.epochs, 
        validation_data=valid_dataset,
        callbacks=callbacks,
        workers=configs.num_workers,
    )
    loss, auc = model.evaluate(test_dataset)

    # evaluation
    print("Test Loss:", loss)
    print("Test AUC:", auc)