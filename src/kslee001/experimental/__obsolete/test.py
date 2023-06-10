import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# # os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# import pandas as pd
# from tqdm.auto import tqdm as tq
import tensorflow as tf

# # private
# # from modules.model import TestModel
# from cfg import configs
# from modules.model import A2IModelBase as A2IModel
# import functions

if __name__ == '__main__':
    a = tf.random.uniform(shape=(16, 1), minval=-1, maxval=2, dtype=tf.int32)
    a = tf.cast(a, dtype=tf.float32)
    print(a)
    a = tf.where(a == tf.constant(a==-1.0))


    exit()
    atel_gt = tf.where(atel_gt == tf.constant(-1.0, dtype=self.configs.general.tf_dtype), 
                        tf.constant(1.0, dtype=self.configs.general.tf_dtype), 
                        atel_gt) # float values needed !
    atel_gt = tf.cast(atel_gt, dtype=tf.int32) # onehot : integer needed
    atel_gt = tf.one_hot(atel_gt, depth=2)