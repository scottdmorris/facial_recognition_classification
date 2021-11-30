
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

# set data path to training data
DATADIR = "/Users/scottmorris/Desktop/model/Training"
CATEGORIES = ["Black", "East Asian", "Indian", "Latino", # declare labels/classes
            "Middle Eastern", "Southeast Asian", "White"]

for category in CATEGORIES: 
    path = os.path.join(DATADIR, category) # path to each classification directory
    for img in os.listdir(path): # loop through each image in each classification directory
        img_array = cv2.imread(os.path.join(path, img)) 
        #plt.imshow(img_array, cmap='gray')
        #plt.show() # for visualizaition of what computer sees
        break
    break

print(img_array) 
print(img_array.shape)

img_size = 224
resized_array = cv2.resize(img_array, (img_size, img_size)) # resize image to 224x224
plt.imshow(resized_array, cmap='gray')    
plt.show()

training_data = []

def create_training_data(): # looping through each folder
    for category in CATEGORIES: 
        path = os.path.join(DATADIR, category) 
        class_num = CATEGORIES.index(category) # assigning number to each classification            
        for img in os.listdir(path): # loop through each image in each folder
            try:
                img_array = cv2.imread(os.path.join(path, img)) 
                resized_array = cv2.resize(img_array, (img_size, img_size)) # resize image to 224x224
                training_data.append([resized_array, class_num]) # add to training data
            except Exception as e:
                pass

if __name__ == "__main__": 
    create_training_data()
    print(len(training_data)) # confirm training data is correct

    # shuffle data so not just learning from one label at a time
    random.shuffle(training_data)
          
    X = [] # features
    y = [] # labels

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, img_size, img_size, 3)

    pickle_out = open("X.pickle", "wb") # save features for re-use
    pickle.dump(X, pickle_out) 
    pickle_out.close()

    pickle_out = open("y.pickle", "wb") # save labels for re-use 
    pickle.dump(y, pickle_out) 
    pickle_out.close()

    # load features
    pickle_in = open("X.pickle", "rb") 
    X = pickle.load(pickle_in) # load features 
    
    pickle_in = open("y.pickle", "rb")  # load labels
    y = pickle.load(pickle_in)

    X = X/255.0 # normalize features   
   # -------------------------------------------------------------
   # convolutional neural network creation
    
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model = Sequential()
    model.add(Conv2D(64, (3, 3)))    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # converts 3D feature maps to 1D feature vectors
    model.add(Dense(64)) # fully connected layer

    model.add(Dense(1)) # output layer
    model.add(Activation('sigmoid')) # sigmoid activation function
    
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    
    model.fit(X, y, batch_size=32, epochs=10) # train model
    # not sure why it's not working with data


