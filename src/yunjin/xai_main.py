#%%
# load package
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shap
#import shap.explainers.deep.deep_tf
import lime
from lime.lime_image import LimeImageExplainer
from PIL import Image
#import PIL.Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as c_map
#from IPython.display import Image, display
import keras.backend as K
from keras.models import Sequential
import ssl
from skimage.segmentation import mark_boundaries

import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

#print(np.__version__) 1.24.3


TARGET_COLUMNS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
NUM_CLASSES = len(TARGET_COLUMNS)
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = [384, 384]
BATCH_SIZE = 1

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

def transform(dataset_dir):
    # make train_images
    train_images = []
    for i in range(64541, 64741):
        img_path = f"{dataset_dir}/valid/patient{i}/study1/view1_frontal.jpg"
        img = image.load_img(img_path, target_size=(384, 384))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        train_images.append(x)

    # Convert the list to a numpy array
    # (number_of_images, height, width, channels). (48, 384, 384, 3).
    train_images = np.concatenate(train_images, axis=0) 
    #print(train_images.shape)
    
    return train_images


def auroc(y_true, y_pred):
    return AUC(curve='ROC')(y_true, y_pred)

# def predict_fn(images):
# 	return session.run(probabilities, feed_dict={processed_images: images})

def get_model_predictions(data):
    model_prediction = model.predict(data)
    print(f"The predicted class is : {decode_predictions(model_prediction, top=1)[0][0][1]}")
    return decode_predictions(model_prediction, top=1)[0][0][1]

def generate_prediction_sample(exp, exp_class, weight = 0.1, show_positive = False, hide_background = False):
    '''
    Method to display and highlight super-pixels used by the black-box model to make predictions
    '''
    image, mask = exp.get_image_and_mask(exp_class, 
                                         positive_only=show_positive, 
                                         num_features=6, 
                                         hide_rest=hide_background,
                                         min_weight=weight
                                        )
    plt.imshow(mark_boundaries(image, mask))
    plt.savefig('lime_pred.png')
    #plt.axis('off')
    #plt.show()

def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    return K.get_session().run(model.layers[layer].input, feed_dict)

    
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
    parser.add_argument('--dataset_dir', type=str, default='/home/n0/a2i006/xai/dataset', help='dataset directory')
    parser.add_argument('--dataset_name', type=str, default='CheXpert-v1.0-small', help='dataset name')
    parser.add_argument('--model_dir', type=str, default='/home/n0/a2i006/xai/models/EfficientNetV2B0-best_model.h5', help='model directory')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    args = parser.parse_args()
    

    # GPU Setting
    physical_devices = tf.config.list_physical_devices('GPU')
    print('Num_GPUs:{}, List:{}'.format(len(physical_devices), physical_devices))
    
    # load df from csv
    df = csv_to_df(args.dataset_dir, args.dataset_name)

    # Filter filenames ending with "_frontal.jpg"
    valid_df = df[df['Path'].str.endswith('_frontal.jpg')]
    #print(valid_df)


    # Dataset for validation
    list_ds_valid = tf.data.Dataset.from_tensor_slices((valid_df['Path'].values, valid_df.iloc[:, 1:].values))
    #print(list_ds_valid)
    ds_valid = list_ds_valid.map(process_path_validation, num_parallel_calls=AUTOTUNE)
    ds_valid = ds_valid.batch(args.batch_size)
    ds_valid = ds_valid.prefetch(AUTOTUNE)

    # load model
    #model = load_model(args.model_dir, custom_objects={'CosineDecayWithWarmup': CosineDecayWithWarmup})
    model = load_model(args.model_dir, custom_objects={'CosineDecayWithWarmup': CosineDecayWithWarmup, 'AUROC': AUROC})

    # Perform inference on the validation dataset
    predictions = model.predict(ds_valid)

    # Convert the predictions to labels
    predicted_labels = (predictions > 0.5).astype(int)

    # Get the true labels from the validation dataset
    true_labels = np.array([label.numpy() for _, label in ds_valid])
    true_labels = np.reshape(true_labels, (-1, NUM_CLASSES))  # Reshape to 2D array

    #image = Image.open('/home/n0/a2i006/xai/dataset/valid/patient64541/study1/view1_frontal.jpg')
    dataset_dir= '/home/n0/a2i006/xai/dataset'
    images = transform(dataset_dir)
    
    #plt.imshow(images[0])
    #pred_orig = get_model_predictions(image)
        
    explainer = LimeImageExplainer()
    explainer_l = explainer.explain_instance(images[0], model.predict, hide_color=0, top_labels=2, num_samples=1000)
    #temp, mask = explanation.get_image_and_mask(240, positive_only=True, num_features=5, hide_rest=True)
    #plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.imshow(explainer_l.segments)
    plt.savefig('lime.png')
    
    #generate_prediction_sample(exp, exp.top_labels[0], show_positive = True, hide_background = True)
    generate_prediction_sample(explainer_l, explainer_l.top_labels[0], show_positive = False, hide_background = False)
    
    # Initialize a list to store dataset elements
    elements = []

    # Iterate over the dataset and collect elements
    for element in ds_valid:
        resized_image = tf.image.resize(element[0].numpy(), IMAGE_SIZE)
        print(element[1])
        elements.append(resized_image.numpy())

    # Convert the elements list to a NumPy array
    elements = np.array(elements)
    # Reshape the elements array
    X = np.squeeze(elements, axis=1)
    
    # define a masker that is used to mask out partitions of the input image, this one uses a blurred background
    masker = shap.maskers.Image("inpaint_telea", X[0].shape)
    
    def f(X):
        tmp = X.copy()    
        tmp = tmp / 255.0
        return model(tmp)
    
    #explainer_s = shap.DeepExplainer(model, map2layer(preprocess_input(X.copy()), 7))
    #shap_values,indexes = e.shap_values(map2layer(to_explain, 7), ranked_outputs=2)
    
    # select backgroud for shap
    #background = images[np.random.choice(images.shape[0], 1, replace=False)]
    # DeepExplainer to explain predictions of the model
    explainer_s = shap.Explainer(f, masker, output_names=TARGET_COLUMNS)
    #explainer_sd = shap.DeepExplainer(f, masker)
    # compute shap values
    #shap_values = explainer_s.shap_values(images[0])
    shap_values = explainer_s(X[1:3], max_evals = 100, batch_size=BATCH_SIZE, outputs=shap.Explanation.argsort.flip[:1])

    #print(shap_values)
    shap.image_plot(shap_values)

    # multi-label