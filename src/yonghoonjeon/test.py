import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


TARGET_COLUMNS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
NUM_CLASSES = len(TARGET_COLUMNS)
AUTOTUNE = tf.data.AUTOTUNE


# return : dataframe with columns [Path, label_1, label_2, ... label_n]
def csv_to_df(dataset_dir, dataset_name):
    # TODO : load data into pandas dataframe
    data = pd.read_csv(f"{dataset_dir}/valid.csv").fillna(0.0)

    # TODO : fix image path:
    # ex. CheXpert-v1.0/patient0000...
    # ex.       > /mnt/e/dataset/chexpert/CheXpert-v1.0/patient0000...
    data['Path'] = data['Path'].str.replace(dataset_name, dataset_dir, regex=False)

    # TODO : fill null values
    for col_idx in range(5, len(data.columns)):
        data[data.columns[col_idx]] = data[data.columns[col_idx]].astype(str)
        data[data.columns[col_idx]] = data[data.columns[col_idx]].str.replace("-1.0", "0.0").astype(float).fillna(0.0)

    # TODO : column reduction
    target_columns = ['Path'] + TARGET_COLUMNS
    dataframe = data[target_columns].reset_index(drop=True)

    return dataframe

def process_path_validation(image_path, label):
    # Read the image from the path
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label

def auroc(y_true, y_pred):
    return AUC(curve='ROC')(y_true, y_pred)


# lr scheduler
class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, total_steps, warmup_steps=0):
        super(CosineDecayWithWarmup, self).__init__()
        assert warmup_steps < total_steps
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.cosine_decay = tf.keras.experimental.CosineDecay(
            learning_rate, total_steps - warmup_steps)

    @tf.function
    def __call__(self, step):
        if step < self.warmup_steps:
            return (self.learning_rate /
                    tf.cast(self.warmup_steps, tf.float32) *
                    tf.cast((step + 1), tf.float32))
        return self.cosine_decay(step - self.warmup_steps)

    def get_config(self):
        return {"learning_rate": np.array(self.learning_rate),
                "total_steps": np.array(self.total_steps),
                "warmup_steps": np.array(self.warmup_steps)}


class AUROC(AUC):
    def __init__(self, name='auroc', **kwargs):
        super().__init__(name=name, **kwargs)


class AUROC_convert(AUC):
    def __init__(self, name='auroc', **kwargs):
        super().__init__(name=name, **kwargs)


if __name__ == "__main__":
    # Input Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/home/n0/a2i002/ambient/project/valid', help='dataset directory')
    parser.add_argument('--dataset_name', type=str, default='CheXpert-v1.0-small', help='dataset name')
    parser.add_argument('--model_dir', type=str, default='/home/n0/a2i002/ambient/project/models/EfficientNetV2B0-best_model.h5', help='model directory')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    args = parser.parse_args()

    # GPU Setting
    physical_devices = tf.config.list_physical_devices('GPU')
    print('Num_GPUs:{}, List:{}'.format(len(physical_devices), physical_devices))

    # load df from csv
    df = csv_to_df(args.dataset_dir, args.dataset_name)

    # Filter filenames ending with "_frontal.jpg"
    valid_df = df[df['Path'].str.endswith('_frontal.jpg')]

    # Dataset for validation
    list_ds_valid = tf.data.Dataset.from_tensor_slices((valid_df['Path'].values, valid_df.iloc[:, 1:].values))
    ds_valid = list_ds_valid.map(process_path_validation, num_parallel_calls=AUTOTUNE)
    ds_valid = ds_valid.batch(args.batch_size)
    ds_valid = ds_valid.prefetch(AUTOTUNE)

    # load model
    model = load_model(args.model_dir, custom_objects={'CosineDecayWithWarmup': CosineDecayWithWarmup, 'AUROC': AUROC})

    # Perform inference on the validation dataset
    predictions = model.predict(ds_valid)

    # Convert the predictions to labels
    predicted_labels = (predictions > 0.5).astype(int)

    # Get the true labels from the validation dataset
    true_labels = np.array([label.numpy() for _, label in ds_valid])
    true_labels = np.reshape(true_labels, (-1, NUM_CLASSES))  # Reshape to 2D array

    # Compute AUROC for each class
    auc_scores = []
    for i in range(NUM_CLASSES):
        auc = roc_auc_score(true_labels[:, i], predictions[:, i])
        auc_scores.append(auc)

    # Calculate the F1 scores for each class
    f1_scores = []
    for i in range(NUM_CLASSES):
        f1 = f1_score(true_labels[:, i], predicted_labels[:, i])
        f1_scores.append(f1)

    # Calculate the accuracy for each class
    accuracy_scores = []
    for i in range(NUM_CLASSES):
        accuracy = accuracy_score(true_labels[:, i], predicted_labels[:, i])
        accuracy_scores.append(accuracy)

    # Calculate the average AUROC, F1, and accuracy scores
    avg_auroc = sum(auc_scores) / len(auc_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)

    # Print the AUROC, F1, and accuracy scores for each class
    for i, column in enumerate(TARGET_COLUMNS):
        print(f"{column}: Accuracy={accuracy_scores[i]:.4f}, AUROC={auc_scores[i]:.4f}, F1={f1_scores[i]:.4f}")

    # Print the average scores
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average AUROC: {avg_auroc:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
