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

# Constants for LeNet-5 data preprocessing
IMAGE_SIZE = (64, 64)  # LeNet-5 is designed for 32x32 grayscale images
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 5

# Data preprocessing and augmentation using LeNet-5 approach
train_datagen = ImageDataGenerator(
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
                label = np.zeros(NUM_CLASSES)  # Create an array of zeros 0 0 0 0 0
                label[int(folders) - 1] = 1  # Set the appropriate index to 1      0 1 0 0 0

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







# 50    *         50
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

from keras.callbacks import LearningRateScheduler



# Compile the model with a lower learning rate and categorical crossentropy loss

print('Model Details are : ')
print(cnn_model.summary())

if (os.path.exists('C:/Users/monasser/PycharmProjects/pythonProject4/modelllll.tfl')):
    cnn_model=joblib.load('model.tfl')
else:
    def lr_schedule(epoch):
        lr = 0.001
        if epoch > 30:
            lr *= 0.5
        elif epoch > 20:
            lr *= 0.7
        return lr
    reduce_lr = LearningRateScheduler(lr_schedule)
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
    # cnn_model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ThisModel = cnn_model.fit(X_train, Y_train, batch_size=64, validation_data=(X_test, Y_test),
                                epochs=5, callbacks=[reduce_lr], verbose=1)
    joblib.dump(cnn_model, "cnntest5.tfl")

# Save the trained LeNet-5 model as an .h5 file
# cnn_model.save('lenet_custom_model.h5')
# Train the LeNet-5 model


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
        prediction = cnn_model.predict(img_test)[0]
        # .11 .1 .55 .23 .44
        max_index = np.argmax(prediction)
        print(image.split('.')[0], "  ", label[max_index])
        # writer.writerow([image.split('.')[0],  label[max_index], max_index+1])
        writer.writerow([image.split('.')[0], max_index+1])
