#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:29:18 2019

@author: quinn
"""

import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, AvgPool1D, UpSampling1D

#from numpy.random import normal

from main import convert_model

def gen_test_model_1d(input_shape=(None,5)):
    m = Sequential()
    m.add(Conv1D(4, 8, padding='same', input_shape=input_shape, activation='relu'))
    m.add(MaxPool1D(2, padding='same'))
    m.add(Conv1D(4, 8, padding='same', activation='tanh'))
    m.add(UpSampling1D(2))
    m.add(Conv1D(4, 8, padding='same', activation='sigmoid'))
    m.compile('SGD','mse')
    return m
    
if __name__=='__main__':
    import tempfile
    shape = (100,4)
    batch_shape = (100,)+shape
    m = gen_test_model_1d(shape)
#    a, b = normal(0, 1, batch_shape), normal(0, 1, batch_shape)

    path = tempfile.gettempdir()
    m.save(path+'/__test_1d.h5')
    convert_model(path+'/__test_1d.h5', 
                  name='__test_1d_model',
                  path=path, 
                  verbose=False)