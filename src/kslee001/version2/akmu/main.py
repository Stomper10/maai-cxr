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


if __name__ == '__main__':
    x = tf.random.normal((1, 128, 128, 3))

    model = TestModel(num_classes=5)
    out = model(x)
    print(inp.shape)
    print(out.shape)
