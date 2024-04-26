import os
from turtle import pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
# from keras.applications import EfficientNetB0, preprocess_input
# from tensorflow.keras.applications import EfficientNetB0, preprocess_input
# patches = layers.Permute((1, 3, 2, 4))(patches)


# Constants for LeNet-5 data preprocessing
IMAGE_SIZE = (64, 64)  # LeNet-5 is designed for 32x32 grayscale images
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 5

# Data preprocessing and augmentation using LeNet-5 approach
train_datagen = ImageDataGenerator(
# preprocessing_function=preprocess_input,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    brightness_range=[0.5, 1.5],  # Adjust brightness
    channel_shift_range=50.0,  # Adjust channel intensity
    vertical_flip=True,  # Flip vertically
    featurewise_center=False,
    featurewise_std_normalization=False,
    zca_whitening=False
)



# Data preparation for LeNet-5
def create_train_data():
    original_count = 0
    augmented_count = 0
    training_data = []
    for folders in tqdm(os.listdir('C:/Users/monasser/Desktop/nural_project/dataset/train')):
        num_of_folder = 'C:/Users/monasser/Desktop/nural_project/dataset/train' + "/" + str(folders)
        for img in tqdm(os.listdir(num_of_folder)):
            original_count += 1  # Increment original count for each image
            path = os.path.join(num_of_folder, img)
            img_data = cv2.imread(path, 0)
            img_data = cv2.resize(img_data, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
            img_data = img_data.reshape((1,) + img_data.shape + (1,))  # Reshape for augmentation

            # Generate augmented images
            i = 0
            for batch in train_datagen.flow(img_data, batch_size=1):
                augmented_count += 1  # Increment augmented count for each augmentation
                image = batch[0].reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
                label = np.zeros(NUM_CLASSES)  # Create an array of zeros
                label[int(folders) - 1] = 1  # Set the appropriate index to 1
                training_data.append([np.array(image), label])
                i += 1
                if i >= 5:  # Number of augmented images per original image
                    break

    shuffle(training_data)
    print("Original data count:", original_count)
    print("Augmented data count:", augmented_count)
    return training_data

import joblib # save model with joblib
from keras.models import load_model
import h5py


if (os.path.exists('train_data_cnn.npy')): # If you have already created the dataset:
    train_data =joblib.load('train_data_cnn.npy')
else: # If dataset is not created:
    train_data = create_train_data()
    joblib.dump(train_data, "train_data_cnn.npy")


# Splitting data into train and test sets
train, test = train_test_split(train_data, test_size=0.1, random_state=42)

# Reshape and preprocess data for LeNet-5
X_train = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
Y_train = np.array([i[1] for i in train])

X_test = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
Y_test = np.array([i[1] for i in test])


cnn_model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')

])
import tensorflow as tf
# from keras.applications import EfficientNetB0, preprocess_input

# Load pre-trained ViT model
# transformer_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

import tensorflow as tf
from keras import layers
import tensorflow as tf
# from keras.layers import Patches

# def MultiHeadSelfAttention(embed_dim, num_heads):
#     return layers.MultiHeadAttention(
#         key_dim=embed_dim // num_heads,
#         num_heads=num_heads,
#         dropout=0.1
#     )

def TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.1):
    inputs = tf.keras.Input(shape=(None, embed_dim))
    attention = layers.MultiHeadAttention(key_dim=embed_dim // num_heads, num_heads=num_heads,dropout=0.1)(inputs, inputs)
    attention = layers.Dropout(rate)(attention)
    attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    ff = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(attention)
    ff = layers.Dropout(rate)(ff)
    ff = layers.Conv1D(filters=embed_dim, kernel_size=1)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(attention + ff)

def VisionTransformer(image_size, patch_size, num_classes, num_blocks, embed_dim, num_heads, ff_dim, rate=0.1):
    input_shape = (image_size, image_size, 3)
    inputs = tf.keras.Input(shape=input_shape)

    # Reshape the input image into patches
    patch_height = image_size // patch_size
    patch_width = image_size // patch_size
    patches = layers.Reshape((patch_height * patch_width, patch_size, patch_size, 3))(inputs)
    patches = layers.Reshape((patch_height * patch_width, patch_size * patch_size * 3))(patches)

    # Projection and Positional Embeddings
    embeddings = layers.Dense(embed_dim, activation="linear")(patches)
    embeddings += tf.keras.Input(shape=(embed_dim,))

    # Transformer Encoder
    for _ in range(num_blocks):
        embeddings = TransformerBlock(embed_dim, num_heads, ff_dim, rate)(embeddings)

    # Applying global average pooling to get a fixed-size representation
    representation = layers.GlobalAveragePooling1D()(embeddings)

    # Classifier head
    logits = layers.Dense(num_classes, activation="softmax")(representation)

    return tf.keras.Model(inputs=inputs, outputs=logits)
# Example usage
image_size = IMAGE_SIZE[0]
patch_size = 16
num_classes = 5
num_blocks = 12
embed_dim = 768
num_heads = 12
ff_dim = 3072
num_patches = 16

#  cnn => vit => model

vit_model = VisionTransformer(image_size, patch_size, num_classes, num_blocks, embed_dim, num_heads, ff_dim)
vit_model.summary()
#################################################################################################3

# Flatten the output of the CNN
cnn_model_output = cnn_model.output

#
#
# # Assuming you have a positional embedding layer
# class PositionalEmbedding(tf.keras.layers.Layer):
#     def _init_(self, sequence_length, embedding_dim, **kwargs):
#         super(PositionalEmbedding, self)._init_(**kwargs)
#         self.sequence_length = sequence_length
#         self.embedding_dim = embedding_dim
#         self.embedding = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=embedding_dim)
#
#     def call(self, inputs):
#         position_indices = tf.range(start=0, limit=self.sequence_length, delta=1)
#         positional_embeddings = self.embedding(position_indices)
#         return inputs + positional_embeddings
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
# positional_embeddings = PositionalEmbedding(num_patches, embed_dim)(flattened_output)
# positional_embeddings = layers.Reshape((num_patches, embed_dim))(positional_embeddings)
#
# # Reshape the output of the CNN to (num_patches, embed_dim)
# reshape_output = layers.Reshape((num_patches, embed_dim))(flattened_output)
#
# # Concatenate or add positional embeddings to the CNN output
# combined_features = layers.Concatenate(axis=1)([reshape_output, positional_embeddings])
# # Reshape for ViT model input
# combined_features = layers.Reshape((-1, embed_dim))(combined_features)
combined_features =layers.Reshape((num_patches, embed_dim))(cnn_model_output)

# Example ViT model
transformer_output = vit_model(combined_features)


combined_model = tf.keras.Model(inputs=cnn_model.input, outputs=transformer_output)

combined_model.compile(optimizer='adam', loss='categorical_crossentropy',learning_rate=.001, metrics=['accuracy'])

combined_model.fit(X_train, Y_train, batch_size=64, validation_data=(X_test, Y_test),
                            epochs=5, verbose=1)


# importing required module
import csv

# opening the file
with open("final.csv", "w", newline="") as f:
    # creating the writer
    writer = csv.writer(f)
    # using writerow to write individual record one by one
    # writer.writerow(["Image", "Label", "Index"])
    writer.writerow(["image_id", "label"])
    # Make predictions
    label = ["apple", "banana", "grapes", "mango", "stra"]
    for image in tqdm(os.listdir('C:/Users/monasser/Desktop/nural_project/dataset/test')):
        img = cv2.imread(os.path.join('C:/Users/monasser/Desktop/nural_project/dataset/test', image), 0)
        img_test = cv2.resize(img, (IMAGE_SIZE))
        img_test = img_test.reshape(1, IMAGE_SIZE[0],IMAGE_SIZE[1], 1)  # Reshape for model input
        prediction = combined_model.predict(img_test)[0]
        max_index = np.argmax(prediction)
        print(image.split('.')[0], "  ", label[max_index])
        # writer.writerow([image.split('.')[0],  label[max_index], max_index+1])
        writer.writerow([image.split('.')[0],max_index+1])