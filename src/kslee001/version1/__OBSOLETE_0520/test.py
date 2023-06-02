import pandas as pd

if __name__ =='__main__':
    
    dataset_dir = f'/mnt/e/dataset/chexpert/CheXpert-v1.0'
    dataset_name = 'CheXpert-v1.0'
    
    path = '/mnt/e/dataset/chexpert/CheXpert-v1.0'
    data = pd.read_csv(f"{path}/train.csv").fillna(0.0)
    data['Path'] = data['Path'].str.replace(dataset_name, dataset_dir, regex=False)
    
    for col_idx in range(5, len(data.columns)):
        data[data.columns[col_idx]] = data[data.columns[col_idx]].astype(str)
        data[data.columns[col_idx]] = data[data.columns[col_idx]].str.replace("-1.0", "0.0").astype(float).fillna(0.0)
        # data[data.columns[col_idx]] = data[data.columns[col_idx]].astype(int)
    
    # store data
    target = ['Path'] + ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    data = data[target].reset_index(drop=True)

    print(data.Path[0])