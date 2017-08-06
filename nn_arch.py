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
    model.add(Conv2D(24, 5, strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(36, 5, strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(48, 5, strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(64, 3, strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    #adding dropout so the model doesnt overfit too much
    model.add(SpatialDropout2D(0.15))
    
    model.add(Conv2D(64, 3, strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    #adding dropout so the model doesnt overfit too much
    model.add(SpatialDropout2D(0.15))
    
    model.add(Flatten())
    
    model.add(Dense(1164))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(50))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(10))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(3))