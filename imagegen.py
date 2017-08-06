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
        self._correction = 0.09
        self._batch_size = batch_size
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
    
    def _read_image(self, file):
        name = './data/' + '/'.join(file.replace('\\', '/').split('/')[-3:])
        image = cv2.imread(name)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _data_gen(self, samples):
        ''' Generator to return batches of training / validation data
        '''
        batch_size = self._batch_size
        num_samples = len(samples)
        while 1: # Loop forever so the generator never terminates
            shuffle(samples)
            images = []
            cmds = []
            for i in range(batch_size):
                #central camera
                image = self._read_image(samples[i][0])
                angle = float(samples[i][3])
                throttle = float(samples[i][4])
                br = float(samples[i][5])
                images.append(image)
                cmds.append([angle, throttle, br])
                #flipped central camera
                image = np.fliplr(image)
                angle = -angle
                images.append(image)
                cmds.append([angle, throttle, br])
                #left camera
                image = self._read_image(samples[i][1])
                angle = float(samples[i][3]) + self._correction
                images.append(image)
                cmds.append([angle, throttle, br])
                #right camera
                image = self._read_image(samples[i][2])
                angle = float(samples[i][3]) - self._correction
                images.append(image)
                cmds.append([angle, throttle, br])
            X_train = np.array(images)
            y_train = np.array(cmds)

            yield shuffle(X_train, y_train)
    
    def train_data_len(self):
        return len(self._train_samples) / self._batch_size
    
    def valid_data_len(self):
        return len(self._valid_samples) / self._batch_size
   
#############################
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    img_gen = ImageGenerator(r'.\data', ['set1'], batch_size=6)
    train_gen = img_gen.train_data_gen()
    img,angle = next(train_gen)
    
    for i in img:
        plt.imshow(i)
        plt.show()
