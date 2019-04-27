# -*- coding: utf-8 -*-
"""
File:   train.py
Author: Diandra Prioleau
Date:   09 October 2018
Desc:   Implementing a function to run training code on an input data set 
        that returns location of red cars, classifier model, validation
        predictions 
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

def trainData(inputData,ground_truth,k):
    data_train =  inputData  
  
    labelled = ground_truth
    
    i = 0
    
    grd_truth_rgb = np.empty([labelled.shape[0]])

    # extracting RGB values using coordinates from ground truth 
    for index in labelled:
        if(i == 0):
            grd_truth_rgb = data_train[index[1]:index[3], index[0]:index[2], :].reshape(-1,data_train[index[1]:index[3], index[0]:index[2], :].shape[-1])            
            i += 1
        else: 
            grd_truth_rgb = np.vstack((grd_truth_rgb,data_train[index[1]:index[3], index[0]:index[2], :].reshape(-1,data_train[index[1]:index[3], index[0]:index[2], :].shape[-1])))
            
    image_bkgd = data_train[:,:,0]
    
    '''----------- Code Reference -----------'''
    #Code Reference: http://flothesof.github.io/removing-background-scikit-image.html
    dark_spots = np.array((image_bkgd < 3).nonzero()).T

    bkgd_rgb = data_train[dark_spots[:,1],dark_spots[:,0],:]
    '''----------- End Code Reference -----------'''
    
#    plt.figure(figsize=(20,20))
#    plt.plot(dark_spots[:, 1], dark_spots[:, 0], 'o')
#    plt.imshow(data_train)
#    plt.title('Background Pixels Choosen in Train Image')
    
    bkgd_rgb = bkgd_rgb[np.random.choice(bkgd_rgb.shape[0],round(len(bkgd_rgb)/4),replace=False),:]
    
    model_redcars = KMeans(n_clusters=3,random_state=0).fit(grd_truth_rgb) # conduct k-means on ground truth (red cars)
    model_bkgd = KMeans(n_clusters=5,random_state=0).fit(bkgd_rgb)# conduct k-means on bacgkround data
    
    clusters_redcars = model_redcars.predict(grd_truth_rgb)
    clusters_bkgd = model_bkgd.predict(bkgd_rgb)
    
    # adding 3 to distinguish labels of background data (labelled 3-7) from red car data (labelled 0-2)
    clusters_bkgd = clusters_bkgd + 3 
    
    grd_truth_data = np.c_[grd_truth_rgb,clusters_redcars]
    
    ncolumns_grd_truth = len(grd_truth_data[1])-1 
    
    bkgd_data = np.c_[bkgd_rgb,clusters_bkgd]
    
    ncolumns_bkgd = len(bkgd_data[1])-1 
    
    '''----------- 4-fold Cross Validation for determing optimal k for KNN classifier -----------'''
    # Removed for submitting code - will take some time if uncommented 
    """
    k_cv = 4
    kf = KFold(n_splits=k_cv,random_state=None,shuffle=True)
    Train_grd_truth = []
    Valid_grd_truth = []
    Train_bkgd = []
    Valid_bkgd = []
    
    for train_index, test_index in kf.split(grd_truth_data):
        #Train_grd_truth[i], Valid_grd_truth[i] = np.array(grd_truth_data[train_index]),np.array(grd_truth_data[test_index])
    
        Train_grd_truth.append(grd_truth_data[train_index])
        Valid_grd_truth.append(grd_truth_data[test_index])
    for train_index, test_index in kf.split(bkgd_data):
        #Train_bkgd[i], Valid_bkgd[i] = np.array(bkgd_data[train_index]),np.array(bkgd_data[test_index])
        Train_bkgd.append(bkgd_data[train_index])
        Valid_bkgd.append(bkgd_data[test_index])

    for k in range (25,27,2):  
        accuracy = []
        for i in range(k_cv):
            
            Train_Data = np.append(Train_grd_truth[i],Train_bkgd[i],axis=0)
            Train_Labels = Train_Data[:,Train_Data.shape[1]-1]
            Train_Data = np.delete(Train_Data,Train_Data.shape[1]-1,axis = 1)
            
            Valid_Data = np.append(Valid_grd_truth[i],Valid_bkgd[i],axis=0)
            Valid_Labels = Valid_Data[:,Valid_Data.shape[1]-1]
            Valid_Data = np.delete(Valid_Data,Valid_Data.shape[1]-1,axis = 1)
        
            model = KNeighborsClassifier(n_neighbors=k,metric='euclidean')
            model.fit(Train_Data,Train_Labels)
            pred = model.predict(Valid_Data)
            
            accuracy.append(accuracy_score(Valid_Labels,pred))
            
        print("Accuracy Mean for k = " + str(k), np.mean(accuracy))
        print("Variance of Accuracy for k = " + str(k), np.var(accuracy))
            
    """        
    '''----------- End of Cross Validation -----------'''
    
    # 70/30 training,validation split for ground truth (red cars)
    X_train_grd_truth, X_valid_grd_truth, label_train_grd_truth, label_valid_grd_truth = train_test_split(grd_truth_data[:,0:ncolumns_grd_truth], grd_truth_data[:,ncolumns_grd_truth], test_size = 0.3)
    
    # 70/30 training,validation split for background 
    X_train_bkgd, X_valid_bkgd, label_train_bkgd, label_valid_bkgd = train_test_split(bkgd_data[:,0:ncolumns_bkgd], bkgd_data[:,ncolumns_bkgd], test_size = 0.3)

    train_data = np.append(X_train_grd_truth,X_train_bkgd,axis=0)
    train_labels = np.append(label_train_grd_truth,label_train_bkgd,axis=0)
    
    X_valid = np.append(X_valid_grd_truth,X_valid_bkgd,axis=0)
    label_valid = np.append(label_valid_grd_truth,label_valid_bkgd,axis=0)
        
    # use KNN algorithm to predict label of validation dataset using euclidean distance     
    knn_model = KNeighborsClassifier(n_neighbors=k,metric='euclidean')
    
    knn_model.fit(train_data,train_labels)
    
    val_predictions = knn_model.predict(X_valid) # predictions of combined validation set (ground truth and background)
    
    print('Accuracy for Predicting Validation Data: ',accuracy_score(label_valid, val_predictions))
    
    # label pixel as either 0 (is red vehicle) or 1 (not a red vehicle) depending if it's in 
    # classes 0 - 2 (red vehicle) or classes 3 - 7 (not red vehicle)
    isRedCarLabels = []
    isRedCarLabels_predicted = []
    for i in range(label_valid.shape[0]):
        isRedCarLabels.append(isRedCar(label_valid.item(i)))
     
    for i in range(val_predictions.shape[0]):
        isRedCarLabels_predicted.append(isRedCar(val_predictions.item(i)))
       
    print('Confusion Matrix of Training Data: ',confusion_matrix(isRedCarLabels,isRedCarLabels_predicted)) # print confusion matrix 
    
    grd_truth_predictions = knn_model.predict(X_valid_grd_truth) # predictions of ground truth validation set
    
    print('Accuracy for Predicting Only Ground Truth (Red Cars): ',accuracy_score(label_valid_grd_truth,grd_truth_predictions))
        
    return labelled,knn_model,val_predictions  

def isRedCar(argument):
    '''isRedCar(argument): Labels whether or not pixel is that of a red vehicle 
       based on if pixel was predicted to be in class labels 0-2 (red vehicle)
       or class lables 3-7 (not red vehicle)'''
    switcher = {
            0: "0",
            1: "0",
            2: "0",
            3: "1",
            4: "1",
            5: "1",
            6: "1",
            7: "1"
        }
    
    return  switcher.get(argument)