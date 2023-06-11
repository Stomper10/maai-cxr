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
from modules.model import A2IModel
from modules.lr_scheduler import CustomOneCycleSchedule, LearningRateLogger
from cfg import configs
import functions
configs.model.backbone = 'densenet'
configs.model.densenet.size = '121'

# configs.model.backbone = 'convnext'

if __name__ == '__main__':
    # argument parsing : [ cluster, backbone type (densenet, convnext), head ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cluster', action='store', default='akmu')
    parser.add_argument('-b', '--backbone', action='store', default='densenet')
    parser.add_argument('-t', '--backbonetype', action='store', default='121')
    parser.add_argument('-s', '--seed', action='store', default=1005, type=int)

    args = parser.parse_args()
    cluster = args.cluster
    if (args.backbone == 'densenet') & (args.backbonetype == '121'):
        from cfg_121 import configs
        configs.model.backbone = 'densenet'
    if (args.backbone == 'densenet') & (args.backbonetype == '169'):
        from cfg_169 import configs
        configs.model.backbone = 'densenet'
    if (args.backbone == 'convnext') & (args.backbonetype == 'small'):
        from cfg_conv_small import configs
        configs.model.backbone = 'convnext'
    if (args.backbone == 'convnext') & (args.backbonetype == 'base'):
        from cfg_conv_base import configs
        configs.model.backbone = 'convnext'
    if cluster == 'gsds-ab':
        configs.dataset.data_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-resized'
    if cluster == 'gsds-c':
        configs.dataset.data_dir = '/home/gyuseonglee/workspace/dataset/chexpert-resized'
    if cluster == 'akmu':
        configs.dataset.data_dir = '/data/s1/gyuseong/chexpert-resized'
    configs.general.seed = int(args.seed)

    configs.model.classifier.add_expert = False
    configs.dataset.cutoff = None
    configs.wandb.use_wandb = False
    configs.wandb.run_name = None
    configs.general.distributed = False
    configs.general.epochs = 1
    configs.general.batch_size = 16
    configs.saved_model_path = "./" + configs.model.backbone + "_{epoch:02d}-{val_loss:.2f}.h5" 

    # multi-gpu training strategy
    strategy = tf.distribute.MirroredStrategy()
    num_devices = strategy.num_replicas_in_sync
    configs.general.batch_size = configs.general.batch_size*num_devices # global batch size

    # mixed precision policy
    policy = mixed_precision.Policy('mixed_float16' if configs.general.precision == 16 else 'float32')
    mixed_precision.set_global_policy(policy)

    # load datasets
    train_dataset, valid_dataset, test_dataset = functions.load_datasets(configs) 

    # load weights
    print("current : ", configs.model.backbone+args.backbonetype)
    weights = sorted(glob.glob(f"./{configs.model.backbone}{args.backbonetype}_{configs.general.seed}*.h5"))


    best_weights = None
    best_auc = -1
    best_result = None

    test_best_weights = None
    test_best_auc = -1
    valid_best_weights = None
    valid_best_auc = -1
    valid_metric_score_for_testset = None
    targets = ['atel', 'card', 'cons', 'edem', 'plef']

    for path in tq(weights):
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
        start_time = t.time()
        test_metric_score = test_model.evaluate(test_dataset, verbose=0)
        end_time = t.time()
        duration = np.round(end_time - start_time, 4)
        processing_time_1_image = np.round(duration/202, 4) # 전체시간 / 이미지 개수
        processing_time_1_sec = np.floor(202/duration)
        test_average_auc = test_metric_score[2:]


        valid_metric_score = valid_model.evaluate(valid_dataset, verbose=0)
        valid_average_auc = valid_metric_score[2:]

        test_average_auc = np.round(np.mean(test_average_auc[2:]), 4)
        valid_average_auc = np.round(np.mean(valid_average_auc[2:]), 4)

        print("[RESULT] of : ", path)
        print("-- TEST average AUC  : ", test_average_auc)
        print( str([f"test_loss : {np.round(test_metric_score[0], 4)}"] + [ f"{targets[idx]} : {np.round(test_metric_score[2:][idx], 4)}"  for idx in range(5) ]))
        print("-- VALID average AUC : ", valid_average_auc)
        print( str([f"val_loss  : {np.round(valid_metric_score[0], 4)}"] + [ f"{targets[idx]} : {np.round(valid_metric_score[2:][idx], 4)}"  for idx in range(5) ]))
        print(f"-- INFERENCE TIME (TEST DATASET) : {duration} | 1 image : {processing_time_1_image} | 1 sec : {processing_time_1_sec}\n")

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
    print(f"# ===== TEST BEST MODEL of seed {configs.general.seed}=====")
    print(f"best model : {test_best_weights}")
    print(f"best auc   : {test_best_auc}")

    print(f"# ===== VALID BEST MODEL of seed {configs.general.seed}=====")
    print(f"best model : {valid_best_weights}")
    print(f"best auc   : {valid_best_auc}\n\n")
    print(f"auc for test set : {valid_metric_score_for_testset}\n\n")
    print(f"difference : {test_best_auc - valid_metric_score_for_testset}")