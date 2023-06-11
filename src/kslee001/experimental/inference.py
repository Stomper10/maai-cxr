import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import time as t
import glob
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm as tq
import tensorflow as tf
from tensorflow.keras import mixed_precision
import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger

# private
# from modules.model import A2IModel
from modules.model import Expert as A2IModel
from modules.lr_scheduler import CustomOneCycleSchedule, LearningRateLogger
from cfg import configs
import functions


# configs.model.backbone = 'densenet'
# configs.model.backbone = 'convnext'


def main():
    # argument parsing : [ cluster, backbone type (densenet, convnext), head ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cluster', action='store', default='akmu')
    parser.add_argument('-b', '--backbone', action='store', default='densenet')
    parser.add_argument('-t', '--backbonetype', action='store', default='121')
    parser.add_argument('-s', '--seed', action='store', default=1005, type=int)

    args = parser.parse_args()
    cluster = args.cluster
    configs.model.backbone = 'densenet'
    configs.model.densenet.size = '121'
    if cluster == 'gsds-ab':
        configs.dataset.data_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized'
    if cluster == 'gsds-c':
        configs.dataset.data_dir = '/home/gyuseonglee/workspace/dataset/chexpert-resized'
    if cluster == 'akmu':
        configs.dataset.data_dir = '/data/s1/gyuseong/chexpert-resized'
    configs.general.seed = int(args.seed)
    functions.set_seed(configs.general.seed)

    configs.model.classifier.add_expert = False
    configs.dataset.cutoff =  1000 #None
    configs.wandb.use_wandb = False
    configs.wandb.run_name = None
    configs.general.distributed = False
    configs.general.epochs = 1
    configs.general.batch_size = 256
    configs.saved_model_path = "./" + configs.model.backbone + "_{epoch:02d}-{val_loss:.2f}.h5" 

    # mixed precision policy
    policy = mixed_precision.Policy('mixed_float16' if configs.general.precision == 16 else 'float32')
    mixed_precision.set_global_policy(policy)

    # load datasets
    train_dataset, valid_dataset, test_dataset = functions.load_datasets(configs) 


    # load weights
    print("current model   : ", configs.model.backbone+args.backbonetype)
    for label in ['atel', 'card', 'cons', 'edem', 'plef']:
        print("current target : ", label)
        weights = sorted(glob.glob(f"./{label}_{configs.model.backbone}{args.backbonetype}_{configs.general.seed}*.h5"))

        test_best_weights = None
        test_best_auc = -1
        valid_best_weights = None
        valid_best_auc = -1
        valid_metric_score_for_testset = None

        for path in tq(weights):
            configs.general.label = label

            test_model, _ = functions.set_model_callbacks(
                model_class=A2IModel,
                weights_path=path,
                configs=configs,
                training=False,
            )
            valid_model, _ = functions.set_model_callbacks(
                model_class=A2IModel,
                weights_path=path,
                configs=configs,
                training=False,
            )
            test_metric_score = test_model.evaluate(test_dataset, verbose=0)
            test_average_auc = test_metric_score[2]
            valid_metric_score = valid_model.evaluate(valid_dataset, verbose=0)
            valid_average_auc = valid_metric_score[2]

            if test_average_auc > test_best_auc:
                test_best_auc = test_average_auc
                test_best_weights = path

            if valid_average_auc > valid_best_auc:
                valid_best_auc = valid_average_auc
                valid_best_weights = path
                valid_metric_score_for_testset = valid_model.evaluate(test_dataset, verbose=0)


        test_best_auc = np.round(test_best_auc, 4)
        valid_best_auc = np.round(valid_best_auc, 4)
        valid_metric_score_for_testset = np.round(valid_metric_score_for_testset, 4)
        print("\n[RESULT]")
        print(f"# ===== TEST BEST MODEL ({label}) of seed {configs.general.seed}=====")
        print(f"best model : {test_best_weights}")
        print(f"best auc   : {test_best_auc}")

        print(f"# ===== VALID BEST MODEL ({label}) of seed {configs.general.seed}=====")
        print(f"best model : {valid_best_weights}")
        print(f"best auc   : {valid_best_auc}")
        print(f"auc for test set : {valid_metric_score_for_testset}\n\n")
        print(f"difference : {test_best_auc - valid_metric_score_for_testset}")





def label_processing(y, target):
    """label (Y) setting"""
    # atel : ones (replace -1 with 1) (0.858)
    if target == 'atel':
        atel_gt = y[:, 0]
        atel_gt = tf.where(atel_gt == tf.constant(-1.0, dtype=configs.general.tf_dtype), 
                        tf.constant(1.0, dtype=configs.general.tf_dtype), 
                        atel_gt) # float values needed !
        atel_gt = tf.cast(atel_gt, dtype=tf.int32) # onehot : integer needed
        atel_gt = tf.one_hot(atel_gt, depth=2)
        return atel_gt
    
    # card : multi (replace -1 with 2) (0.854)
    if target == 'card':
        card_gt = y[:, 1]
        card_gt = tf.where(card_gt == tf.constant(-1.0, dtype=configs.general.tf_dtype), 
                        tf.constant(2.0, dtype=configs.general.tf_dtype), 
                        card_gt) 
        card_gt = tf.cast(card_gt, dtype=tf.int32) # onehot : integer needed
        card_gt = tf.one_hot(card_gt, depth=3)
        return card_gt

    # cons : ignore (0.937) -> U-zeros (0.932) adopted instead 
    if target == 'cons':
        cons_gt = y[:, 2]
        cons_gt = tf.where(cons_gt == tf.constant(-1.0, dtype=configs.general.tf_dtype), 
                        tf.constant(0.0, dtype=configs.general.tf_dtype), 
                        cons_gt) # float values needed !
        cons_gt = tf.cast(cons_gt, dtype=tf.int32) # onehot : integer needed
        cons_gt = tf.one_hot(cons_gt, depth=2)
        return cons_gt

    # edem : ones (0.941)
    if target == 'edem':
        edem_gt = y[:, 3]
        edem_gt = tf.where(edem_gt == tf.constant(-1.0, dtype=configs.general.tf_dtype), 
                        tf.constant(1.0, dtype=configs.general.tf_dtype), 
                        edem_gt)
        edem_gt = tf.cast(edem_gt, dtype=tf.int32) # onehot : integer needed
        edem_gt = tf.one_hot(edem_gt, depth=2)
        return edem_gt
    
    # plef : multi (0.936)
    if target == 'plef':
        plef_gt = y[:, 4]
        plef_gt = tf.where(plef_gt == tf.constant(-1.0, dtype=configs.general.tf_dtype), 
                        tf.constant(2.0, dtype=configs.general.tf_dtype), 
                        plef_gt) 
        plef_gt = tf.cast(plef_gt, dtype=tf.int32)
        plef_gt = tf.one_hot(plef_gt, depth=3)
        return plef_gt



if __name__ == '__main__':
    main()

