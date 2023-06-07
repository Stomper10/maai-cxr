import pandas as pd

dataset_dir = '/home/n1/gyuseonglee/workspace/datasets/chexpert-small'
dataset_name = 'CheXpert-v1.0-small'
data = pd.read_csv(dataset_dir + '/train.csv')
data['Path'] = data['Path'].str.replace(dataset_name, dataset_dir)

print(data.iloc[:5])
print(data.iloc[0]['Path'])