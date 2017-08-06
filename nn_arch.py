# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 12:43:47 2017

@author: gansc
"""

from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, SpatialDropout2D
from keras.layers import Cropping2D, Lambda
from keras.layers.advanced_activations import LeakyReLU

def build(model):
    ''' Add the nvidia architecture to Keras model
    '''
    # Image preprocessing
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    def normalize(pixel):
        return ((pixel/8.0)-16.0)
    model.add(Lambda(normalize))  
    
    # Conv2D
    model.add(Conv2D(24, 5, strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    
    # Conv2D
    model.add(Conv2D(36, 5, strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    
    # Conv2D
    model.add(Conv2D(48, 5, strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    
    # Conv2D with dropout
    model.add(Conv2D(64, 3, strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(SpatialDropout2D(0.15))
    
    # Conv2D with dropout
    model.add(Conv2D(64, 3, strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(SpatialDropout2D(0.15))
    
    model.add(Flatten())
    
    # 5x Dense to 3 outputs (angle, throttle, brake)
    model.add(Dense(1164))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(50))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(10))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(3))