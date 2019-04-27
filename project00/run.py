# -*- coding: utf-8 -*-
"""
File:   run.py
Author: Diandra Prioleau
Date:   09 October 2018
Desc:   Runs training and testing functions on sample images 
    
"""
import numpy as np 
from train import trainData
from test import testData

data_train =  np.load('data_train.npy')
data_test = np.load('data_test.npy')
ground_truth = np.load('more_red_cars.npy')

k = 19 # optimal k choosen for KNN Classifier after conducting 4-Fold CV

train_coords,knn_model,val_predictions = trainData(data_train,ground_truth,k) # call trauning function
test_coords,test_predictions,prediction_labels = testData(data_test,knn_model) # call test function
