
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle 
import matplotlib.pyplot as plt
import os
import cv2
import random

# make sure to change filepath for images and unzip file

# creating data path to training data
DATADIR = "/Users/scottmorris/Desktop/model/Training"
CATEGORIES = ["Black", "East Asian", "Indian", "Latino", "Middle Eastern", "Southeast Asian", "White"]

for category in CATEGORIES: 
    path = os.path.join(DATADIR, category) # path to each classification directory
    for img in os.listdir(path): # loop through each image in each classification directory
        img_array = cv2.imread(os.path.join(path, img)) 
        plt.imshow(img_array, cmap='gray')
        plt.show() 
        break
    break

print(img_array) 

training_data = []

def create_training_data(): # looping through each folder
    for category in CATEGORIES: 
        path = os.path.join(DATADIR, category) 
        class_num = CATEGORIES.index(category) # assigning number to each classification            
        for img in os.listdir(path): # loop through each image in each folder
            try:
                img_array = cv2.imread(os.path.join(path, img)) 
                training_data.append([img_array, class_num]) # add to training data
            except Exception as e:
                pass

print(len(training_data)) # confirm training data is correct

# shuffle data so not just learning from one label at a time
random.shuffle(training_data)
                
X = [] # features
y = [] # labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 224, 224, 3)

# use pickle to save data
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# to reload data
X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0


# create model object and layers
# 2 x 64 CNN
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:])) # 64 nodes, 3x3 convolutional filter
model.add(Activation('relu')) # rectify linear
model.add(MaxPooling2D(pool_size=(2,2))) # 2x2 pooling

model.add(Conv2D(64, (3,3))) # 3x3 convolutional filters
model.add(Activation('relu')) # rectify linear
model.add(MaxPooling2D(pool_size=(2,2))) # 2x2 pooling

model.add(Flatten()) # flatten to one dimensional data for dense layer
model.add(Dense(64)) # fully connected layer

# output layer
model.add(Dense(1)) # single output
model.add(Activation('sigmoid')) 

model.compile(loss='categorical_crossentropy',
             optimizer='adam', 
             metrics=['accuracy'])

# model training instructions
model.fit(X, y, batch_size=30, epochs=10, validation_split=0.2) # split data into training and validation sets  (20% validation)

