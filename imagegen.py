# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 12:43:47 2017

@author: gansc
"""

import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2

class ImageGenerator(object):
    ''' Fetches all images from the driving similar in path + set 
        and returns them as batch for training after preprocessing. 
        Images from left and right camera are returned with a 
        corrected steering angle
    '''
    def __init__(self, path, sets, batch_size=32, train_val_split=0.2):
        ''' Initialize samples with all files from .csv.
        '''
        self._samples = []
        self._batch_size = batch_size
        self._PP_factor = 2 # Preprocessing factor
        self._cameras = ['center']
        for set in sets:
            with open(path + '/' + set + '/driving_log.csv') as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    self._samples.append(line)
       
        self._train_samples, self._valid_samples = train_test_split(self._samples, test_size=train_val_split)
    
    def train_data_gen(self):
        yield from self._data_gen(self._train_samples)
        
    def valid_data_gen(self, batch_size=32):
        yield from self._data_gen(self._valid_samples)
    
    def _data_gen(self, samples):
        ''' Generator to return batches of training / validation data
        '''
        batch_size = self._batch_size
        num_samples = len(samples)
        num_samples_scaled = num_samples * self._PP_factor * len(self._cameras)
        batch_size_scaled = int(batch_size / (self._PP_factor * len(self._cameras)))
        while 1: # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size_scaled):
                batch_samples = samples[offset:offset+batch_size_scaled]             
                images = []
                angles = []
                for batch_sample in batch_samples:
                    for img_index,position in zip(range(len(self._cameras)), self._cameras):  
                        name = './data/' + '/'.join(batch_sample[img_index].replace('\\', '/').split('/')[-3:])
                        image = cv2.imread(name)
                        angle = float(batch_sample[3])
                        
                        img_set, angle_set = self._preprocess_image(image, position, angle)
                        for img_pp, ang_pp in zip(img_set, angle_set):
                            images.append(img_pp)
                            angles.append(ang_pp)
    
                X_train = np.array(images)
                y_train = np.array(angles)

                yield shuffle(X_train, y_train)
    
    def _preprocess_image(self, image, camera_position='center', angle=0):
        ''' Preprocess images from driving simulator to use with model.
            Image is split in left and right half (flipped).
            Steering angle for left and right are corrected with geometry offset
        '''
        image_set = [image, np.fliplr(image)]
    
        if camera_position == 'center':
            a_off = 0
        elif camera_position == 'left':
            a_off = 0.2
        elif camera_position == 'right':
            a_off = -0.2
            
        angle_set = [(angle+a_off), -(angle+a_off)]
        return image_set, angle_set
    
    def train_data_len(self):
        return len(self._train_samples) * self._PP_factor * len(self._cameras) / self._batch_size
    
    def valid_data_len(self):
        return len(self._valid_samples) * self._PP_factor * len(self._cameras) / self._batch_size
   
#############################
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    img_gen = ImageGenerator(r'.\data', ['set1'], batch_size=6)
    train_gen = img_gen.train_data_gen()
    img,angle = next(train_gen)
    
    for i in img:
        plt.imshow(i)
        plt.show()
