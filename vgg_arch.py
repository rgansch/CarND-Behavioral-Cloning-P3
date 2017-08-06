# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 12:43:47 2017

@author: gansc
"""

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D

def build(model):
    ''' Add the VGG architecture to Keras model
    '''
    # Layer Cluster - 1
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Layer Cluster - 2
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Layer Cluster - 3
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))    

    # Layer Cluster - 4
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2))) 

    # Layer Cluster - 5 - Output Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='relu'))