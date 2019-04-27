# -*- coding: utf-8 -*-
"""
File:   run.py
Author: Diandra Prioleau
Date:   27 Nov 2018
Desc:   Training a multi-layer perceptron using backpropagation for a  
        two-dimensional data set  
    
"""

import numpy as np
from sklearn.model_selection import train_test_split
import mlp

#load data
data = np.load('dataSet.npy')

#Set up Neural Network
data_in = data[:,:2]
target_in = data[:,-1].reshape(1,data[:,-1].size)
hidden_layers = 5
NN = mlp.mlp(data_in,target_in.T,hidden_layers)

#Analyze Neural Network Performance
#Before Updating Weights
print("Before Updating Weights", end='\n')
NN.confmat(data_in,target_in.T)

eta = 0.01
niterations = 10000

#Training - updating weights using error backpropagation
NN.mlptrain(data_in,target_in.T,eta,niterations)

#After Updating Weights
print("After Updating Weights", end='\n')
NN.confmat(data_in,target_in.T)       
