import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm as tq
import tensorflow as tf
import tensorflow_model_optimization as tfmot # pip install 필요!
from tensorflow.keras import mixed_precision
import wandb

from modules.model import A2IModel
from cfg import configs
import functions

from pruning_modules import Augmentation, DenseNet, ConvNeXt, Classifier


configs.model.backbone = 'densenet'
configs.model.densenet.size = '121'
configs.wandb.use_wandb = False
configs.general.distributed = False
configs.general.epochs = 1
configs.general.batch_size = 16 # 원하는 경우 batch size 변경 가능

DATA_DIRECTORY = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized'



class Prune(tf.keras.layers.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, x):
        return self.model(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', action='store', default=1005)
    parser.add_argument('-t', '--target_sparsity', action='store', default=0.80)
    args = parser.parse_args()
    configs.general.seed = int(args.seed)
    target_sparsity = float(args.target_sparsity)

    # dataset directory
    configs.dataset.data_dir = DATA_DIRECTORY
    configs.dataset.cutoff = 500 # train 데이터 수 cutoff 만큼만 불러오기

    # load dataset
    train_dataset, valid_dataset, test_dataset = functions.load_datasets(configs) 
    configs.general.steps_per_epoch = train_dataset.steps_per_epoch

    WEIGHT_DIRECTORY = f'/home/n1/gyuseonglee/workspace/maai-cxr/src/kslee001/pruning/weights/ensemble/densenet121_{configs.general.seed}_test.h5'

    configs.saved_model_path = "./pruned_weights/ensemble/" + f"pruned_{str(target_sparsity)}_{configs.model.backbone}121_{configs.general.seed}_test.h5" 

    # load model
    model_ = A2IModel(configs=configs)
    model_.initialize()
    model_.load_weights(WEIGHT_DIRECTORY) # base model weight 불러오기
    
    model = Prune(model=model_)


    
    criterion = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.1,
        reduction='auto',
    )

    # model.compile(loss=criterion)

    
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
    
    # model_for_pruning = tf.keras.models.clone_model(
    #         model,
    #         clone_function=apply_pruning_to_dense,
    #     )
    

    # recompile
    # model_for_pruning, callbacks = functions.set_model_callbacks(model_class=A2IModel, 
    #                                                             configs=configs
    #                                                             )
    # model_for_pruning.summary()
    print("PRUNING RECOMPILE DONE!")
    print("START TRAINING...")
    scheduler = CustomOneCycleSchedule(
        max_lr=configs.optimizer.learning_rate, 
        epochs=configs.general.epochs,
        steps_per_epoch=configs.general.steps_per_epoch,
        start_lr=None, end_lr=None, warmup_fraction=configs.optimizer.warm_up_rate,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=scheduler,
        weight_decay=configs.optimizer.weight_decay,
        beta_1=configs.optimizer.beta_1,
        beta_2=configs.optimizer.beta_2,
        ema_momentum=configs.optimizer.ema_momentum,
    )    

    for idx, batch in tq(enumerate(train_dataset)):
        x, y = batch
        with tf.GradientTape() as tape:
            yhat = model(x)
            atel_loss = criterion(y[:, 0], yhat[0])
            card_loss = criterion(y[:, 1], yhat[1])
            cons_loss = criterion(y[:, 2], yhat[2])
            edem_loss = criterion(y[:, 3], yhat[3])
            plef_loss = criterion(y[:, 4], yhat[4])
            total_loss = atel_loss + card_loss + cons_loss + edem_loss + plef_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        

    # model.fit(
    #     train_dataset, 
    #     epochs=5, 
    #     validation_data=valid_dataset,
    #     # callbacks=callbacks,
    #     workers=configs.general.num_workers,
    #     verbose=1, 
    #     shuffle=True,
    # )

    model.save(f"pruned_{target_sparsity}.h5", save_format='tf')
    model_for_export = tfmot.sparsity.keras.strip_pruning(model)

    _, pruned_keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)