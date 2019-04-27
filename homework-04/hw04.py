# -*- coding: utf-8 -*-
"""
File:   hw04.py
Author: Diandra Prioleau
Date:   30 Oct 2018
Desc:   Design 2 different neural networks by hand which can discriminate 
        between the classes; neural networks are coded to verify that the 
        networks learn a decision boundary similar to what was designed 
    
"""

import numpy as np
import matplotlib.pyplot as plt

fig   = plt.figure()

""" =======================  Data Set 1 ========================== """

""" =======================  Neural Network 1 for Data Set 1 ========================== """
data =  np.load('dataSet1.npy') # load data set 1

labels = data[:,2] # labels of original data set 

data = np.delete(data,data.shape[1]-1,axis = 1) # delete labels

data = np.hstack((np.ones((data.shape[0],1)),data))

plt.scatter(data[:,1],data[:,2], c=labels, linewidth=0) # plot original data set    
plt.title('Data Set 1')
plt.pause(2)

weights_dataset1 = [[-0.5,0,1],[-1.5,0,1],[0.01,-1,5]] # weights found for neural network 

# Using ReLu function as Activation Function
n1 = data@weights_dataset1[0]
n1[n1<=0] = 0

n2 = data@weights_dataset1[1]
n2[n2<=0] = 0

n_matrix = np.vstack((n1,n2))
n_matrix = n_matrix.T
n_matrix = np.hstack((np.ones((data.shape[0],1)),n_matrix))

output = n_matrix@weights_dataset1[2]
output_labels = output[:] 

# Using Heavyside Step Function as Activation Function 
output_labels[output_labels<=0] = 0 # classify as 0 if less than or equal to 0
output_labels[output_labels>0] = 1 # classify as 0 if greater than 0

plt.scatter(data[:,1],data[:,2],c=output_labels,linewidth=0) # plot labels determined by neural network
# Reference Code from Dr. Zare - Lecture 15 of Jupyter Notebook
# Use for each neural network in each data set
x = np.array([-0.5,2.5])
y1 = -(weights_dataset1[0][0]/weights_dataset1[0][2])-(weights_dataset1[0][1]/weights_dataset1[0][2])*x
y2 = -(weights_dataset1[1][0]/weights_dataset1[1][2])-(weights_dataset1[1][1]/weights_dataset1[1][2])*x
plt.plot(x,y1,'r')
plt.plot(x,y2,'r')
plt.title('Labels from Neural Network 1 of Data Set 1')
# End Reference
plt.pause(2)

""" =======================  Neural Network 2 for Data Set 1 ========================== """
plt.figure()

weights_dataset1 = [[-0.5,1,0],[-1.5,1,0],[0.01,-1,5]] # weights found for neural network 

# Using ReLu function as Activation Function
n1 = data@weights_dataset1[0]
n1[n1<=0] = 0

n2 = data@weights_dataset1[1]
n2[n2<=0] = 0

n_matrix = np.vstack((n1,n2))
n_matrix = n_matrix.T
n_matrix = np.hstack((np.ones((data.shape[0],1)),n_matrix))

output = n_matrix@weights_dataset1[2]
output_labels = output[:]

# Using Heavyside Step Function as Activation Function 
output_labels[output_labels<=0] = 0
output_labels[output_labels>0] = 1

plt.scatter(data[:,1],data[:,2],c=output_labels,linewidth=0) # plot labels determined by neural network
x = np.array([-2.5,2.5])
y1 = -(weights_dataset1[0][0]/weights_dataset1[0][1])-(weights_dataset1[0][2]/weights_dataset1[0][1])*x
y2 = -(weights_dataset1[1][0]/weights_dataset1[1][1])-(weights_dataset1[1][2]/weights_dataset1[1][1])*x
plt.plot(y1,x,'b')
plt.plot(y2,x,'b')
plt.title('Labels from Neural Network 2 of Data Set 1')

""" =======================  Data Set 2 ========================== """
plt.figure()

""" =======================  Neural Network 1 for Data Set 2 ========================== """
data =  np.load('dataSet2.npy') # load data set 2

labels = data[:,2]

data = np.delete(data,data.shape[1]-1,axis = 1) # delete labels

data = np.hstack((np.ones((data.shape[0],1)),data))

plt.scatter(data[:,1],data[:,2], c=labels, linewidth=0) # plot original data set 
plt.title('Data Set 2')
plt.pause(2)

weights_dataset2 = [[0.5,1,0],[-0.5,1,0],[0,1,1],[-1,5,1],[-1,5,2]] # weights found for neural network 

# Using ReLu function as Activation Function
n1 = data@weights_dataset2[0]
n1[n1<=0] = 0 # if output is less than or equal to 0, change to 0

n2 = data@weights_dataset2[1]
n2[n2<=0] = 0 # if output is less than or equal to 0, change to 0

n_matrix = np.vstack((n1,n2))
n_matrix = n_matrix.T
n_matrix = np.hstack((np.ones((data.shape[0],1)),n_matrix))

output = [0 for x in range(3)]
output[0] = n_matrix@weights_dataset2[2]
output[1] = n_matrix@weights_dataset2[3]
output[2] = n_matrix@weights_dataset2[4]
output = np.vstack(output)
output = output.T

# Using Softmax function as Activation Function
# however, function is simplified by just finding the index of the maximum
# value in each row (outputs from output layer)
output_labels = np.argmax(output,axis=1)

plt.scatter(data[:,1],data[:,2],c=output_labels,linewidth=0) # plot labels determined by neural network
x = np.array([-2.5,2.5])
y1 = -(weights_dataset2[0][0]/weights_dataset2[0][1])-(weights_dataset2[0][2]/weights_dataset2[0][1])*x
y2 = -(weights_dataset2[1][0]/weights_dataset2[1][1])-(weights_dataset2[1][2]/weights_dataset2[1][1])*x
plt.plot(y1,x,'r')
plt.plot(y2,x,'r')
plt.title('Labels from Neural Network 1 of Data Set 2')
plt.pause(2)

""" =======================  Neural Network 2 for Data Set 2 ========================== """
plt.figure()
weights_dataset2 = [[-0.4,0,1],[0,1,0],[0,1,0.4],[-1,7,1],[-1,5,2]] # weights found for neural network 

# Using ReLu function as Activation Function
n1 = data@weights_dataset2[0]
n1[n1<=0] = 0 # if output is less than or equal to 0, change to 0

n2 = data@weights_dataset2[1]
n2[n2<=0] = 0 # if output is less than or equal to 0, change to 0

n_matrix = np.vstack((n1,n2))
n_matrix = n_matrix.T
n_matrix = np.hstack((np.ones((data.shape[0],1)),n_matrix))

output = [0 for x in range(3)]
output[0] = n_matrix@weights_dataset2[2]
output[1] = n_matrix@weights_dataset2[3]
output[2] = n_matrix@weights_dataset2[4]
output = np.vstack(output)
output = output.T

# Using Softmax function as Activation Function
# however, function is simplified by just finding the index of the maximum
# value in each row (outputs from output layer)
output_labels = np.argmax(output,axis=1)

plt.scatter(data[:,1],data[:,2],c=output_labels,linewidth=0) # plot labels determined by neural network
x = np.array([-2.5,2.5])
y1 = -(weights_dataset2[0][0]/weights_dataset2[0][2])-(weights_dataset2[0][1]/weights_dataset2[0][2])*x
y2 = -(weights_dataset2[1][0]/weights_dataset2[1][1])-(weights_dataset2[1][2]/weights_dataset2[1][1])*x
plt.plot(x,y1,'b')
plt.plot(y2,x,'b')
plt.title('Labels from Neural Network 2 of Data Set 2')
plt.pause(2)

""" =======================  Data Set 3 ========================== """
plt.figure()

""" =======================  Neural Network 1 for Data Set 3 ========================== """
data = np.load('dataSet3.npy') # load data set 3 

labels = data[:,2]

data = np.delete(data,data.shape[1]-1,axis = 1) # delete labels

data = np.hstack((np.ones((data.shape[0],1)),data))

plt.scatter(data[:,1],data[:,2], c=labels, linewidth=0) # plot original data set 
plt.title('Data Set 3')
plt.pause(2)

weights_dataset3 = [[-2,-2,1],[-0.5,-1,1],[3,-1.5,1],[0.01,-6,3,-1]] # weights found for neural network 

# Using ReLu function as Activation Function
n1 = data@weights_dataset3[0]
n1[n1<=0] = 0 # if output is less than or equal to 0, change to 0

n2 = data@weights_dataset3[1]
n2[n2<=0] = 0 # if output is less than or equal to 0, change to 0

n3 = data@weights_dataset3[2]
n3[n3<=0] = 0 # if output is less than or equal to 0, change to 0

n_matrix = np.vstack((n1,n2,n3))
n_matrix = n_matrix.T
n_matrix = np.hstack((np.ones((data.shape[0],1)),n_matrix))

output = n_matrix@weights_dataset3[3]
output_labels = output[:] 

# Using Heavyside Step Function as Activation Function 
output_labels[output_labels<=0] = 0 # classify as 0 if less than or equal to 0
output_labels[output_labels>0] = 1 # classify as 0 if greater than 0

plt.scatter(data[:,1],data[:,2],c=output_labels,linewidth=0) # plot labels determined by neural network
x = np.array([1,4])
y1 = -(weights_dataset3[0][0]/weights_dataset3[0][2])-(weights_dataset3[0][1]/weights_dataset3[0][2])*x
y2 = -(weights_dataset3[1][0]/weights_dataset3[1][2])-(weights_dataset3[1][1]/weights_dataset3[1][2])*x
y3 = -(weights_dataset3[2][0]/weights_dataset3[2][2])-(weights_dataset3[2][1]/weights_dataset3[2][2])*x
plt.plot(x,y1,'r')
plt.plot(x,y2,'r')
plt.plot(x,y3,'r')
plt.title('Labels from Neural Network 1 of Data Set 3')
plt.pause(2)

""" =======================  Neural Network 2 for Data Set 3 ========================== """
weights_dataset3 = [[-1.5,1,0],[-2.75,1,0],[-3.5,1,0],[0,0.4,-8,25]] # weights found for neural network 

# Using ReLu function as Activation Function
n1 = data@weights_dataset3[0] 
n1[n1<=0] = 0 # if output is less than or equal to 0, change to 0

n2 = data@weights_dataset3[1]
n2[n2<=0] = 0 # if output is less than or equal to 0, change to 0

n3 = data@weights_dataset3[2]
n3[n3<=0] = 0 # if output is less than or equal to 0, change to 0

n_matrix = np.vstack((n1,n2,n3))
n_matrix = n_matrix.T
n_matrix = np.hstack((np.ones((data.shape[0],1)),n_matrix))

output = n_matrix@weights_dataset3[3]
output_labels = output[:] 

# Using Heavyside Step Function as Activation Function 
output_labels[output_labels<=0] = 0 # classify as 0 if less than or equal to 0
output_labels[output_labels>0] = 1 # classify as 1 if greater than 0

plt.scatter(data[:,1],data[:,2],c=output_labels,linewidth=0) # plot labels determined by neural network
x = np.array([-1,5])
y1 = -(weights_dataset3[0][0]/weights_dataset3[0][1])-(weights_dataset3[0][2]/weights_dataset3[0][1])*x
y2 = -(weights_dataset3[1][0]/weights_dataset3[1][1])-(weights_dataset3[1][2]/weights_dataset3[1][1])*x
y3 = -(weights_dataset3[2][0]/weights_dataset3[2][1])-(weights_dataset3[2][2]/weights_dataset3[2][1])*x
plt.plot(y1,x,'b')
plt.plot(y2,x,'b')
plt.plot(y3,x,'b')
plt.title('Labels from Neural Network 2 of Data Set 3')
