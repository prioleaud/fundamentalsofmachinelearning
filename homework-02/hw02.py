# -*- coding: utf-8 -*-
"""
File:   hw02.py
Author: Diandra Prioleau
Date:   25 September 2018
Desc:   Implementing a probabilistic generative classifier and a KNN
        classifier and comparing their results on the same data; Discriminating
        among the classes for each training data and providing classification
        results for each test data 
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
import random
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

plt.close('all') #close any open plots

""" =======================  Import DataSet ========================== """


Train_2D = np.loadtxt('2dDataSetforTrain.txt')
Train_7D = np.loadtxt('7dDataSetforTrain.txt')
Train_HS = np.loadtxt('HyperSpectralDataSetforTrain.txt')

Test_2D = np.loadtxt('2dDataSetforTest.txt')
Test_7D = np.loadtxt('7dDataSetforTest.txt')
Test_HS = np.loadtxt('HyperSpectralDataSetforTest.txt')

""" ======================  Function definitions ========================== """


"""
===============================================================================
===============================================================================
======================== Probabilistic Generative Classfier ===================
===============================================================================
===============================================================================
"""

""" Here you can write functions to estimate the parameters for the training data, 
    and the prosterior probabilistic for the testing data. """

def sumPriors(dataset):
    '''sumPriors(dataset): Sums data points in each class in the dataset   
       '''
    sumPriors = 0
    for classes in range(len(dataset)):
        sumPriors += dataset[classes].shape[0]

    return sumPriors

def calculatePosterior(X_test,mu,cov,prior):
    '''calculatePosterior(X_test,mu,cov,prior): Calculates posterior  
       probability for each class in the dataset'''
       
    sumPosterior = 0
    y = []
    posterior = []

    for classes in range(len(mu)):  
        y.append(multivariate_normal.pdf(X_test, mean=mu[classes], cov=cov[classes])) #pdf for each class 
        sumPosterior += y[classes]*prior[classes]
        
    for classes in range(len(mu)):    
        posterior.append((y[classes]*prior[classes])/(sumPosterior))# posterior for each class
        
    return posterior
 
def diagonalCovariance(train,mu):
    '''diagonalCovariance(train,mu): Compute diagonal covariance matrix
       '''
       
    cov = np.sum(np.square(train - mu), axis=0)/train.shape[0]
    cov_matrix = np.diag(cov)
    
    return cov_matrix    

# Reference Lecture06, Probabilistic Generative Models, from Dr. Zare 
def generativeClassifier(test,train,isDiagonalCov):
    '''generativeClassifier(test,train,isDiagonalCov): Generate probabilistic 
       generative classifier and returns mean and covariance and prior for each 
       class in the dataset'''

    mu = [] 
    cov = [] 
    prior = []
    sumPrior = sumPriors(train)

    for classes in range(len(train)):
        mu.append(np.mean(train[classes], axis=0)) # estimate mean for each class 

        if(isDiagonalCov):
            cov.append(diagonalCovariance(train[classes],mu[classes])) # estimate diagonal covariance for each class
        else:
            cov.append(np.cov(train[classes].T)) # estimate covariance for each class
        
        # regularize the covariance if it is a singular matrix 
        if(np.linalg.det(cov[classes]) == 0.0):
            class_cov = cov[classes]
            for i in range(cov[classes].shape[0]):
                class_cov[i,i] = class_cov[i,i] + 0.01

        prior.append(train[classes].shape[0]/sumPrior) # estimate prior for each class

    return mu,cov,prior

def predictLabels(X_test,posterior):
    '''predictLabels(X_test,posterior): Predicts label for each test 
       sample for datasets with 2 classes using the posterior probability'''
       
    class1 = posterior[0]
    class2 = posterior[1]
    labels = np.empty([len(X_test)])
    
    # comparing the posterior of each class for each test sample 
    for i in range(len(X_test)):
        if(class1[i] > class2[i]):
            labels[i] = "0"
        else:
            labels[i] = "1"
        
    return labels

def predictHSLabels(X_test,posterior):
    '''predictHSLabels(X_test,posterior): Predicts label for each test 
       sample for datasets with 5 classes using the posterior probability'''
       
    class1 = posterior[0]
    class2 = posterior[1]
    class3 = posterior[2]
    class4 = posterior[3]
    class5 = posterior[4]
    
    labels = np.empty([len(X_test)])
    
    # comparing the posterior of each class for each test sample 
    for i in range(len(X_test)):
        if(class1[i] > class2[i] and class1[i] > class3[i] and 
           class1[i] > class4[i] and class1[i] > class5[i]):
            labels[i] = "1"
        elif(class2[i] > class1[i] and class2[i] > class3[i] and 
             class2[i] > class4[i] and class2[i] > class5[i]):
            labels[i] = "2"
        elif(class3[i] > class1[i] and class3[i] > class2[i] and 
             class3[i] > class4[i] and class3[i] > class5[i]):
            labels[i] = "3"
        elif(class4[i] > class1[i] and class4[i] > class2[i] and 
             class4[i] > class3[i] and class4[i] > class5[i]):
            labels[i] = "4"
        elif(class5[i] > class1[i] and class5[i] > class2[i] and 
             class5[i] > class3[i] and class5[i] > class4[i]):
            labels[i] = "5"
        
    return labels

"""
===============================================================================
===============================================================================
============================ KNN Classifier ===================================
===============================================================================
===============================================================================
"""

""" Here you can write functions to achieve your KNN classifier. """

def knnClassifier(k,test,train,train_labels,isHSData):
    '''knnClassifier(k,test,train,train_labels): Generate k-nearest neighbors
       classifier using Euclidean distance'''
       
    dist = np.empty([train.shape[0]])
    test_labels = np.empty([test.shape[0]])
    for i in range(len(test)):
        for j in range(len(train)):
            # compute Euclidean distance for each test data point and all of the training points
            dist[j] = np.linalg.norm(train[j]-test[i]) 
        if(isHSData):
            #np.argsort returns the indices that would sort an array.
            test_labels[i] = labelHSTest(k,train_labels,np.argsort(dist))
        else:
            test_labels[i] = labelTest(k,train_labels,np.argsort(dist))
 
    return test_labels

def labelTest(k,train_labels,label_indices):
    '''labelTest(k,train_labels,label_indices): Predicts label for each test 
       sample for datasets with 2 classes based upon Euclidean distance - 
       how close in distance is test data point to the training data points'''
       
    num_class1 = 0;
    num_class2 = 0;
    for i in range(k):
        index = label_indices[i]
        if(train_labels[index] == 0):
            num_class1 += 1
        else:
            num_class2 += 1
    if(num_class1 > num_class2):
        return "0"
    else:
        return "1"

def labelHSTest(k,train_labels,label_indices):
    '''labelHSTest(k,train_labels,label_indices): Predicts label for each test 
       sample for datasets with 5 classes based upon Euclidean distance - 
       how close in distance is test data point to the training data points'''
       
    num_class1 = 0;
    num_class2 = 0;
    num_class3 = 0;
    num_class4 = 0;
    num_class5 = 0;
        
    for i in range(k):
        index = label_indices[i]
        if(train_labels[index] == 1):
            num_class1 += 1
        elif(train_labels[index] == 2):
            num_class2 += 1
        elif(train_labels[index] == 3):
            num_class3 += 1
        elif(train_labels[index] == 4):
            num_class4 += 1
        elif(train_labels[index] == 5):
            num_class5 += 1

    highestscore_class = max(num_class1,num_class2,num_class3,num_class4,num_class5)
    
    if(num_class1 > num_class2 and num_class1 > num_class3 and 
       num_class1 > num_class4 and num_class1 > num_class5):
        return "1"
    elif(num_class2 > num_class1 and num_class2 > num_class3 and 
         num_class2 > num_class4 and num_class2 > num_class5):
        return "2"
    elif(num_class3 > num_class1 and num_class3 > num_class2 and 
         num_class3 > num_class4 and num_class3 > num_class5):
        return "3"
    elif(num_class4 > num_class1 and num_class4 > num_class2 and 
         num_class4 > num_class3 and num_class4 > num_class5):
        return "4"
    elif(num_class5 > num_class1 and num_class5 > num_class2 and 
         num_class5 > num_class3 and num_class5 > num_class4):
        return "5"
    else: # if tie randomly choose class label among classes that were tied 
        values = [num_class1,num_class2,num_class3,num_class4,num_class5]
        val_rand = []
        for i in range(5):
            if(values[i] == highestscore_class):
                val_rand.append(i)
        index = random.choice(val_rand)
        return values[index]
    
""" ============  Generate Training and validation Data =================== """

""" Here is an example for 2D DataSet, you can change it for 7D and HS 
    Also, you can change the random_state to get different validation data """

# Here you can change your data set
Train = Train_2D
Test = Test_2D
isHSData = False # set to true if using Hyperspectral data set
k_cv = 4 # k-fold cross validation 

kf = KFold(n_splits = k_cv, random_state=None, shuffle=True) # Performs k-fold 

acc_pg = []
acc_pgdiag = []
acc_KNN = []
var_KNN = []
fold = 1

# 4-fold cross validation for each k in range 1-45 (only odd numbers to avoid ties) for KNN classifier
for k in range(1,47,2):  
    
    """ ======================== Cross Validation ============================= """
    print("k parameter for KNN: ", k)
    accuracy = []
    
    for train_index, test_index in kf.split(Train):
        X_train, X_valid = Train[train_index], Train[test_index]
        
        label_train = X_train[:,X_train.shape[1]-1]
        label_valid = X_valid[:,X_valid.shape[1]-1]
        X_train = np.delete(X_train,X_train.shape[1]-1,axis = 1)
        X_valid = np.delete(X_valid,X_valid.shape[1]-1,axis = 1)

        labels = label_train
        Classes = np.sort(np.unique(labels))    
        
        X_train_class = []
        for j in range(Classes.shape[0]):
            jth_class = X_train[label_train == Classes[j],:]
            X_train_class.append(jth_class)
        
        #Visualization of first two dimension of your dataSet
        #for j in range(Classes.shape[0]):
        #    plt.scatter(X_train_class[j][:,0],X_train_class[j][:,1])
        
            
        """ ========================  Train the Classifier ======================== """
        
        """ Here you can train your classifier with your training data """
        """ Only necessary for Probabilistic Generative Classifier - KNN doesn't need training """
        
        if(k == 1): #only need to do 4-fold CV once for Probabilistic Generative Classifier
            mu,cov,prior = generativeClassifier(X_valid,X_train_class,False)
            mu_PGdiag,cov_PGdiag,prior_PGdiag = generativeClassifier(X_valid,X_train_class,True)
            
            """ Here you should test your parameters with validation data """
            posterior_PG = calculatePosterior(X_valid,mu,cov,prior)
            posterior_PGdiag = calculatePosterior(X_valid,mu_PGdiag,cov_PGdiag,prior_PGdiag)    
                    
            """ Call functions to label validation data  """
            if(isHSData):
                print("HS is True")
                predictions_PG = predictHSLabels(X_valid,posterior_PG)
                predictions_PGdiag = predictHSLabels(X_valid,posterior_PGdiag)
            else:
                predictions_PG = predictLabels(X_valid,posterior_PG)
                predictions_PGdiag = predictLabels(X_valid,posterior_PGdiag)
            
            # The accuracy for your validation data for Probabilisitic Generative Classifier
            accuracy_PG = accuracy_score(label_valid, predictions_PG)
            accuracy_PGdiag = accuracy_score(label_valid, predictions_PGdiag)
            
            print("Fold ", fold)
            print("Accuracy with Full Covariance: ",accuracy_PG)
            print("Accuracy with Diagonal Covariance: ",accuracy_PGdiag)

            acc_pg.append(accuracy_PG)
            acc_pgdiag.append(accuracy_PGdiag)
            
            fold += 1 
            
        predictions_KNN = knnClassifier(k,X_valid,X_train,label_train,isHSData)
        
        # The accuracy for your validation data for KNN
        accuracy_KNN = accuracy_score(label_valid, predictions_KNN)
        accuracy.append(accuracy_KNN)
        
        predictions_PG = []
        predictions_PGdiag = []
        predictions_KNN = []
    
    acc_KNN.append(np.mean(accuracy))
    var_KNN.append(np.var(accuracy))
    accuracy = []


print("Probabilistic Generative Classifier Accuracy Mean: ", np.mean(acc_pg))
print("PG Variance of Accuracy: ", np.var(acc_pg))
print("Probabilistic Generative Classifier Accuracy Mean with Diagonal Cov: ", np.mean(acc_pgdiag))
print("PGDiag Variance of Accuracy: ", np.var(acc_pgdiag))

print("Accuracy Mean & Variance of Accuracy for each k of KNN")
for i in range(23):
    print("Accuracy Mean for k = ", str(i*2+1) + ": ", acc_KNN[i])

    print("Variance of Accuracy Mean for k = ", str(i*2+1) + ": ", var_KNN[i])

""" ========================  Test the Model ============================== """
    
""" This is where you should test the testing data with your classifier """

Train = [Train_2D, Train_7D]
Test = [Test_2D, Test_7D]
isDiagCov = [True,True]
testFiles = ["2DforTestLabels.txt","7DforTestLabels.txt"]

# Testing 2D and 7D with Probabilistic Generative Classifier with diagonal covariance
for i in range(2):
    X_train = Train[i]
    X_test = Test[i]
    
    label_train = X_train[:,X_train.shape[1]-1]
    X_train = np.delete(X_train,X_train.shape[1]-1,axis = 1)
    
    labels = label_train
    Classes = np.sort(np.unique(labels))    
    
        
    X_train_class = []
    for j in range(Classes.shape[0]):
        jth_class = X_train[label_train == Classes[j],:]
        X_train_class.append(jth_class)

    mu,cov,prior = generativeClassifier(X_test,X_train_class,isDiagCov[i])
    posterior = calculatePosterior(X_test,mu,cov,prior)
    predictions= predictLabels(X_test,posterior)

    np.savetxt(testFiles[i],predictions)


# Testing Hyperspectral Data with KNN classifier, k=13
X_train = Train_HS
X_test = Test_HS

label_train = X_train[:,X_train.shape[1]-1]
X_train = np.delete(X_train,X_train.shape[1]-1,axis = 1)
   
predictions_KNN = knnClassifier(13,X_test,X_train,label_train,True)
np.savetxt("HyperSpectralforTestLabels.txt",predictions_KNN)


