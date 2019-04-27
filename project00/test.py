# -*- coding: utf-8 -*-
"""
File:   test.py
Author: Diandra Prioleau
Date:   09 October 2018
Desc:   Implementing a function to run testing code on input data set that 
        returns the location of red cars, predictions and labels  
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from skimage import img_as_float, color
from skimage.feature import corner_peaks
from train import isRedCar

def testData(test_data,model):
    image =  img_as_float(test_data) #convert image to floats
   
    '''----------- Code Reference -----------'''
     # Reference: https://stackoverflow.com/questions/21656136/scikit-image-color-filtering-and-changing-parts-of-image-to-array
    black_mask = color.rgb2gray(image) < 0.1
    distance_red = color.rgb2gray(1 - np.abs(image - (1, 0, 0)))
    distance_red[black_mask] = 0
    
    # coordinates of red pixels that are corner; Assuming if a corner is detected, 
    # must be a vehicle 
    coords_car = corner_peaks(distance_red, threshold_rel=0.9, min_distance=50)
    '''----------- End Code Reference -----------'''
    
    test_rgb = test_data[coords_car[:,0],coords_car[:,1],:] # use coordinates from above to extract corresponding RGB value
    
#    plt.figure(figsize=(50,50))
#    plt.imshow(test_data)
#    plt.plot(coords_car[:, 1], coords_car[:, 0], 'ro')
#    plt.axis('image')
#    plt.show()
    
    predictions = model.predict(test_rgb) # predict which classes chosen pixels belong
    
    # label pixel as either 0 (is red vehicle) or 1 (not a red vehicle) depending if it's in 
    # classes 0 - 2 (red vehicle) or classes 3 - 7 (not red vehicle)
    isRedCarLabels = []
    for i in range(predictions.shape[0]):
        isRedCarLabels.append(isRedCar(predictions.item(i)))
        
    return coords_car,predictions,isRedCarLabels  