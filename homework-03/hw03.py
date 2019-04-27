# -*- coding: utf-8 -*-
"""
File:   hw03.py
Author: Diandra Prioleau
Date:   23 Oct 2018 
Desc:   Implement python code that applies principal compenents analysis
        (without dimensionality reduction, only decorrelation) and data
        whitening to generated data set

""" 

""" =======================  Import dependencies ========================== """
import numpy as np 
import matplotlib.pyplot as plt

""" =======================  Create 2D DataSet ========================== """
N = 1000 # number of data points 
mean = [0.5, 3]
cov = [[2,4.5],[4.5,25]]
data = np.random.multivariate_normal(mean, cov, N) #generate data 

print('Mean of Original Data \n',np.mean(data,axis=0))
print('Covariance of Original Data \n',np.cov(data.T))

# plot original data 
plt.scatter(data[:,0],data[:,1])
plt.title('Original Data')

""" =======================  PCA without Dimensionality Reduction ========================== """
data_std = data - np.mean(data,axis=0) # subtract mean from original data

# compute covariance, eigenvalues, and eigenvectors
cov_mat = np.cov(data_std.T)
print(cov_mat)

# Reference Jupyter Notebook Lecture 13 from Dr. Zare 
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True) # sort eigenvals and eigenvecs from largest to smallest

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))

# End Reference 

data_pca = data_std.dot(w) 

cov_mat = np.cov(data_pca.T) # covariance after PCA 
print('Covariance after PCA \n',cov_mat)

# plot PCA data 
plt.figure()
plt.scatter(data_pca[:,0],data_pca[:,1],c='g')
plt.title('After PCA (without dimensionality reduction)')

""" =======================  Data Whitening ========================== """
eigen_val_matrix = np.diag([eigen_pairs[0][0],eigen_pairs[1][0]]) # create diagonal matrix of eigenvalues 

l,h = data.shape[1],data.shape[0]
data_whiten = [[0 for x in range(l)] for y in range(h)]

data_whiten = np.dot(np.sqrt(np.linalg.inv(eigen_val_matrix)),data_pca.T) # data whitening

data_whiten = data_whiten.T 

cov_mat = np.cov(data_whiten.T) # covariance of data after data whitening 
print('Covariance after Data Whitening \n',cov_mat)

# plot data after whitening
plt.figure()
plt.scatter(data_whiten[:,0],data_whiten[:,1],c='r')
plt.title('After Data Whitening')
plt.show()