import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import glob
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm as tq
import tensorflow as tf
import tensorflow_model_optimization as tfmot # pip install 필요!
from tensorflow.keras import mixed_precision
import wandb

from modules_label_ensemble.model import Expert as A2IModel
from cfg_label_ensemble import configs
import functions


configs.model.backbone = 'densenet'
configs.model.densenet.size = '121'
configs.wandb.use_wandb = False
configs.general.distributed = False
configs.general.epochs = 5
configs.general.batch_size = 16 # 원하는 경우 batch size 변경 가능

DATA_DIRECTORY = '/home/gyuseonglee/workspace/dataset/chexpert-resized'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', action='store', default=1005)
    parser.add_argument('-t', '--target_sparsity', action='store', default=0.80)
    args = parser.parse_args()
    configs.general.seed = int(args.seed)
    target_sparsity = float(args.target_sparsity)

    # dataset directory
    configs.dataset.data_dir = DATA_DIRECTORY
    configs.dataset.cutoff = None # train 데이터 수 cutoff 만큼만 불러오기

    # load dataset
    train_dataset, valid_dataset, test_dataset = functions.load_datasets(configs) 
    configs.general.steps_per_epoch = train_dataset.steps_per_epoch


    targets = ['atel', 'card', 'cons', 'edem', 'plef']
    # targets = ['plef']
    for target in targets:
        configs.general.label = target
        weights = glob.glob(f'./weights/label_ensemble/{target}/*{configs.general.seed}*.h5')

        for weight in (weights):

            SAVE_NAME = f"./pruned_weights/label_ensemble/{target}/pruned_{str(target_sparsity)}_{weight.rsplit('/', 1)[1].rsplit('.h5', 1)[0]}"
            configs.saved_model_path = f"{SAVE_NAME}.h5" 
            
            print(f"\n[INFO] current model will be saved as {SAVE_NAME}\n")

            # load model
            model = A2IModel(configs=configs)
            model.initialize()
            model.load_weights(weight) # base model weight 불러오기
            criterion = tf.keras.losses.CategoricalCrossentropy(
                from_logits=False,
                label_smoothing=0.1,
                reduction='auto',
            )
            model.compile(loss=criterion)

            layers = [layer for layer in model.layers]
            model = tf.keras.Sequential(layers)
            
            print("MODEL LOAD SUCCESS!")
            
            # model.summary()
            
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            # target_sparsity = 0.80 # !!!!!!!!!!!!!!!!!! 0.80 /0.90 / 0.95 세팅 바꿔서 돌리기 configuration 반영 플리즈!

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
                epochs=5, 
                validation_data=valid_dataset,
                callbacks=callbacks,
                workers=configs.general.num_workers,
                verbose=1, 
                shuffle=True,
            )

            # model_for_pruning.save(f"pruned_{target_sparsity}.h5", save_format='h5')



