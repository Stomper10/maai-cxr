import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5  # Number of label columns

# Load CSV data
df = pd.read_csv("/home/train/labels.csv")

# Split the dataset into train and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2)

# Image generator with normalization
datagen = ImageDataGenerator(rescale=1./255.)

# Load images and labels from dataframe
print("Preparing training dataset....")
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="/home/train/images",
    x_col="image_id",
    y_col=["label1", "label2", "label3", "label4", "label5"],
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=IMAGE_SIZE)

print("Preparing validation dataset....")
valid_generator = datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory="/home/train/images",
    x_col="image_id",
    y_col=["label1", "label2", "label3", "label4", "label5"],
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=IMAGE_SIZE)

# Load base model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))

# Freeze the base model
base_model.trainable = False

# Add new layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(NUM_CLASSES, activation='sigmoid')(x)

# Create new model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          epochs=EPOCHS,
          validation_data=valid_generator)
