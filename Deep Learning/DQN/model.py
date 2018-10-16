# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:23:28 2018

@author: Amit
"""


import tensorflow as tf
import tensorflow.keras as keras

######################
######DQN Model#######
######################




weight_int = keras.initializers.RandomNormal(mean=0.5, stddev=0.25)

model = keras.Sequential()
model.add(keras.layers.Dense(1000, activation='relu', kernel_initializer='zero', bias_initializer=weight_int,input_shape=(1,), )  )
model.add(keras.layers.Dense(10, activation='linear', kernel_initializer='zero',bias_initializer=weight_int))


optimizer = optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)

model.compile(optimizer, loss='mean_squared_error')
model.save('mymodel.h5')