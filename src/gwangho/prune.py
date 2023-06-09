import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm as tq
import tensorflow as tf
from tensorflow.keras import mixed_precision
import tensorflow_model_optimization as tfmot # pip install 필요!
import wandb

# private
from modules.model import A2IModel
from cfg import configs
import functions


if __name__ == '__main__':
    # argument parsing : [ cluster, backbone type (densenet, convnext), head ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cluster', action='store', default='akmu')
    parser.add_argument('-f', '--backbone', action='store', default='densenet')
    parser.add_argument('-a', '--add_expert', action='store_true')
    parser.add_argument('-e', '--epochs', action ='store', type=int, default=5)
    parser.add_argument('-b', '--batch', action='store', type=int, default=16)
    parser.add_argument('-p', '--progress_bar', action ='store_false')

    # for debugging
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-s', '--single_gpu', action ='store_true')
    parser.add_argument('-w', '--wandb_off', action='store_false')

    args = parser.parse_args()
    configs.general.progress_bar = 1 if args.progress_bar==True else 2 # one line per epoch
    cluster = args.cluster
    configs.dataset.data_dir = "."
    # if cluster == 'gsds-ab':
    #     configs.dataset.data_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized'
    # if cluster == 'gsds-c':
    #     configs.dataset.data_dir = '/home/gyuseonglee/workspace/dataset/chexpert-resized'
    # if cluster == 'akmu':
    #     configs.dataset.data_dir = '/data/s1/gyuseong/chexpert-resized'
    configs.model.backbone = args.backbone
    configs.model.classifier.add_expert = bool(args.add_expert)
    configs.dataset.cutoff = 500 if args.test == True else None
    configs.wandb.use_wandb = False
    configs.wandb.run_name = f'a2i-{configs.model.backbone}-{configs.dataset.image_size[0]}'
    configs.general.distributed = True if args.single_gpu == False else False
    configs.general.epochs = int(args.epochs)
    configs.general.batch = int(args.batch)
    configs.saved_model_path = "./" + configs.model.backbone + "_best_model_{epoch:02d}-{val_loss:.2f}.h5" 

    # wandb initialization
    if configs.wandb.use_wandb == True :
        wandb.login()
        run = wandb.init(project=configs.wandb.project_name, name=configs.wandb.run_name, config=configs)

    # multi-gpu training strategy
    strategy = tf.distribute.MirroredStrategy()
    num_devices = strategy.num_replicas_in_sync
    configs.general.batch_size = configs.general.batch_size*num_devices # global batch size

    # mixed precision policy
    policy = mixed_precision.Policy('mixed_float16' if configs.general.precision == 16 else 'float32')
    mixed_precision.set_global_policy(policy)

    # load datasets
    train_dataset, valid_dataset = functions.load_datasets(configs) 
    configs.general.steps_per_epoch = train_dataset.steps_per_epoch

    # settings 
    if args.single_gpu == False:
        with strategy.scope():
            model, callbacks = functions.set_model_callbacks(
                model_class=A2IModel, 
                configs=configs
            )
    else:
        model, callbacks = functions.set_model_callbacks(
            model_class=A2IModel, 
            configs=configs
        )

    # pruning
    with strategy.scope():
        model.load_weights('final_densenet_320.h5') # base model weight 불러오기
        layers = [layer for layer in model.layers]
        model = tf.keras.Sequential(layers)
        
        print("MODEL LOAD SUCCESS!")
        
        # model.summary()
       
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        target_sparsity = 0.80 # !!!!!!!!!!!!!!!!!! 0.80 /0.90 / 0.95 세팅 바꿔서 돌리기 configuration 반영 플리즈!

        end_step = np.ceil(len(train_dataset) / configs.general.batch_size).astype(np.int32) * configs.general.epochs

        pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                        final_sparsity=target_sparsity,
                                                                        begin_step=0,
                                                                        end_step=end_step)
        }

        def apply_pruning_to_dense(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                return tfmot.sparsity.keras.prune_low_magnitude(layer)
            if isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.sparsity.keras.prune_low_magnitude(layer)
            return layer
        
        model_for_pruning = tf.keras.models.clone_model(
                model,
                clone_function=apply_pruning_to_dense,
            )
        

        # recompile
        model_for_pruning, callbacks = functions.set_model_callbacks(model_class=A2IModel, 
                                                                    configs=configs
                                                                    )
        # model_for_pruning.summary()
        print("PRUNING RECOMPILE DONE!")
        print("START TRAINING...")
        model_for_pruning.fit(
            train_dataset, 
            epochs=configs.general.epochs, 
            validation_data=valid_dataset,
            callbacks=callbacks,
            workers=configs.general.num_workers,
            verbose=configs.general.progress_bar, 
            shuffle=True,
        )

        model.summary() 

        model.save("pruned_{target_sparsity}.h5", save_format='h5')
    


