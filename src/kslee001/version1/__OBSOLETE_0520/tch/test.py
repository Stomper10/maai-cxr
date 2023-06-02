import pandas as pd
target = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
data = pd.read_csv(f'/home/n1/gyuseonglee/workspace/datasets/chexpert-small/train.csv')
cols = data.columns
for idx, c in enumerate(cols):
    if c in target:
        print(f"{idx} : {c}")
    
