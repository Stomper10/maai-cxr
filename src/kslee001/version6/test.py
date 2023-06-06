# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# # os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# import pandas as pd
# from tqdm.auto import tqdm as tq
# import tensorflow as tf

# # private
# # from modules.model import TestModel
# from cfg import configs
# from modules.model import A2IModelBase as A2IModel
# import functions

if __name__ == '__main__':
    # load datasets : tf tensor dataset, simlilar to torch dataloader

    # model = A2IModel(configs=configs)
    # model.summary()
    import os
    import glob

    files = glob.glob('/data/s1/gyuseong/chexpert-resized/train_320/*.jpg')

    idx = 0
    for f in files:
        print(f)
        idx += 1
        if idx == 100:
            break

