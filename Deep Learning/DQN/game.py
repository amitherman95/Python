# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:32:19 2018

@author: Amit
"""

import tensorflow as tf
import tensorflow.keras as keras
from random import *
import numpy as np
import keras.backend as K


#The purpose of this program is to serve as a simple proof of concept for DQN
#The model plays a simple game: each time it gets a random number and it has to choose
#the corrent cell in its output whose index corresponds to the random number in the input
#if it succeed, it gets 1 point
#else, it losses one point


def annealing(epsilon, annealing_num):
    m = 0.99/annealing_num
    
    if epsilon>0.1:
        return epsilon - m 
    
    return epsilon
    

def bernouli(epsilon):
    X = np.random.binomial(1, epsilon)
    return X






score=0
for i in range(20):
    keras.backend.clear_session()
model=keras.models.load_model('mymodel.h5') 


D_size=10000
up_freq = 100
size=0
clone = keras.models.clone_model(model)
Memory=[]
C=0
greedy=1
m=[]
y=[]
while size<D_size:
    sample=[]
    flag=1
    phi =np.random.randint(10)
    
    while flag:
        sample.clear()
        if bernouli(greedy):
             index = np.random.choice(10)
                
        else:
              Q_values = model.predict((phi,))
             # print(Q_values)
              index=np.argmax(Q_values)
    
    
    
        greedy=annealing(greedy, 1)
    
        sample.append(phi)
        sample.append(index)
    
        if phi==index: #this is the rule to continue
            sample.append(1)
        else:
            sample.append(0)#game over
            flag=0
            
        phi =np.random.randint(10)
        sample.append(phi)
        Memory.append(sample.copy())
        size=size+1
        y = np.empty(0)
        m=np.empty(0)
        A=25#numbr of samples in minibatch
        if(size>100):
            
             exp_replay =[choice(Memory) for v in range(A)]
             states=[exp_replay[v][0] for v in range(A)]
             
             for v in range(A):
                 Q_values =np.reshape( model.predict((states[v],)),10)
                 
                 action = exp_replay[v][1]#action
                 
                 if exp_replay[v][2]==0:##reward=0
                     target = -1
                 else:
                     q_max = np.max(clone.predict((exp_replay[v][3],)))
                     target = 1 + 0.99*q_max
                     
                 Q_values[action]=target#change model prediction
                 y=np.append(y,Q_values.copy())
             y=np.reshape(y,(A,10))
             states=np.reshape(states,(A,1))
             model.fit(x=states, y=y,verbose=0)
            
        C=C+1
        m=m+1
        if(C==up_freq):
            clone=keras.models.clone_model(model)
            C=0
            print([np.argmax(model.predict((i,))) for i in range(10)])
            #Each time we update the target, we print the predictions for each number
            #if each number fits its corrent index(ie [0,1,2,3...]) then the model has successfuly learned the logic behind this game
model.save('mymodel.h5')
      

    
    
    
    