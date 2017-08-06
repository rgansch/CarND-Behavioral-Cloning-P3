# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 12:43:47 2017

@author: gansc
"""

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D

def build(model):
    ''' Add the LeNet architecture to Keras model
    '''
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.8))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.7))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
