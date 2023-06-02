import os
import random
import numpy as np


# private
from modules.backend import TestModel


if __name__ == '__main__':
    
    model = TestModel(input_shape=(384, 384), num_classes=5)
