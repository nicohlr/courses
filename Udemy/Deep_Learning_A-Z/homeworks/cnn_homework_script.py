import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Initializing our CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3,3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection, add hidden layer (fully connected layer)
classifier.add(Dense(units=128, activation='relu'))

# Step 4 - Full connection - add output layer
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

path_train = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))), 
                          'ressources/Convolutional_Neural_Networks/dataset/training_set')
path_test = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))), 
                          'ressources/Convolutional_Neural_Networks/dataset/test_set')

# code for image augmentation found here : https://keras.io/preprocessing/image/

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) # Apply image augmentation on train images

test_datagen = ImageDataGenerator(rescale=1./255) # Do not apply transformation on test images

# target size is the same as choosen in our cnn architecture
training_set = train_datagen.flow_from_directory(path_train, target_size=(64, 64),batch_size=32, 
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(path_test, target_size=(64, 64), batch_size=32, 
                                            class_mode='binary')

# fit model
classifier.fit_generator(training_set, 
                         steps_per_epoch=8000, # number of images in training_set
                         epochs=25, 
                         validation_data=test_set, 
                         validation_steps=2000) # number of images in training_set

deeper_classifier = Sequential()
deeper_classifier.add(Conv2D(32, (3,3), input_shape=(64, 64, 3), activation='relu'))
deeper_classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add a second convolutionnal layer, input_shape is not necessary because there is other layer before
deeper_classifier.add(Conv2D(32, (3,3), activation='relu'))
deeper_classifier.add(MaxPooling2D(pool_size=(2,2)))

deeper_classifier.add(Flatten())
deeper_classifier.add(Dense(units=128, activation='relu'))
deeper_classifier.add(Dense(units=1, activation='sigmoid'))
deeper_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

deeper_classifier.fit_generator(training_set, 
                         steps_per_epoch=8000, # number of images in training_set
                         epochs=25, 
                         validation_data=test_set, 
                         validation_steps=2000) # number of images in training_set