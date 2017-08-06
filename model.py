# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 12:43:47 2017

@author: gansc
"""

from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers import Cropping2D, Lambda
import numpy as np

from imagegen import ImageGenerator
import nn_arch

class GNet(object):
    def __init__(self):
        self._model = None
        self._build_model()
        pass
    
    def _build_model(self):
        ''' Create the Keras neural network model '''
        self._model = Sequential()
        
        # Image preprocessing
        self._model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
        
        def normalize(pixel):
            return ((pixel/32.0)-64.0)
        self._model.add(Lambda(normalize))        
        
        nn_arch.build(self._model)
        
        self._model.compile(loss='mse', optimizer='adam')
        return self._model
    
    def train(self, train_path, sets):
        ''' Train the model with images in train_path '''
        img_gen = ImageGenerator(train_path, sets, train_val_split = 0.2)
        train_gen = img_gen.train_data_gen()
        train_len = img_gen.train_data_len()
        valid_gen = img_gen.valid_data_gen()
        valid_len = img_gen.valid_data_len()
        
        self._model.fit_generator(train_gen, \
                                  steps_per_epoch = train_len, \
                                  validation_data = valid_gen, \
                                  validation_steps = valid_len, \
                                  epochs=5)
        pass
    
    def save(self):
        ''' Save the model (.h5)
        '''
        self._model.save('model.h5')

    def load(self):
        ''' Load the saved weights (.h5)
        '''
        self._model.load_weights('model.h5')
        
 #############################
    
if __name__ == '__main__':
    nn = GNet()
    #nn.load()
    nn.train(r'data', ['set1', 'set2', 'set3', 'set4', 'set5', 'set6', 'set7', 'set8'])
    nn.save()