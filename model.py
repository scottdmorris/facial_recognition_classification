
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle 

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

# add convolutional & pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")

num_classes = 7
outputs = layers.Dense(num_classes, activation="softmax")


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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model training instructions
model.fit(X, y, batch_size=20, epochs=10, validation_split=0.2)

