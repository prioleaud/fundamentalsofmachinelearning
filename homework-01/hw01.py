# -*- coding: utf-8 -*-
"""
File:   hw01.py
Author: Diandra Prioleau
Date:   13 September 2018
Desc:   Coding questions 1 and 2 for HW01; Plotting the root-mean-square 
        error between the predicted and true value for both a training
        and test set; Computing the ML and MAP solution for the Gaussian mean 
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.close('all') #close any open plots

"""
===============================================================================
===============================================================================
============================ Question 1 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Function definitions ========================== """

def generateUniformData(N, l, u, gVar):
	'''generateUniformData(N, l, u, gVar): Generate N uniformly spaced data points 
    in the range [l,u) with zero-mean Gaussian random noise with variance gVar'''
	# x = np.random.uniform(l,u,N)
	step = (u-l)/(N);
	x = np.arange(l+step/2,u+step/2,step)
	e = np.random.normal(0,gVar,N)
	t = np.sinc(x) + e
	return x,t

def plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    p2 = plt.plot(x2, t2, 'g') #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') #plot training data

    #add title, legend and axes labels
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0]),legend)
        
    """
    This seems like a good place to write a function to learn your regression
    weights!
    
    """
def fitdata(x,t,M):
    '''fitdata(x,t,M): Fit a polynomial of order M to the data (x,t) - 
       Reference Lecture01 from Dr. Zare'''
    X = np.array([x**m for m in range(M+1)]).T
    w = np.linalg.inv(X.T@X)@X.T@t
    return w       

def plotRms(train_x,train_rms,test_x,test_rms,legend=[]):
    '''plotRms(train_x,train_rms,test_x,test_rms,legend=[]): Generate a plot of 
       error root-mean-square of the training and test data for each model 
       order, M'''
    train_plot = plt.plot(train_x, train_rms,marker='o',color='b')
    test_plot = plt.plot(test_x, test_rms,marker='o',color='r')
    
    plt.legend((train_plot[0],test_plot[0]),legend)

def plotEstimates(mle_range,mle,map_range,map_prob,legend=[]):
    '''plotEstimates(mle_range,mle,map_range,map_prob,legend=[]): Generate a 
       plot of maximum likelihood and maximum posterior based on the number 
       of samples'''
    mle_plot = plt.plot(mle_range, mle,color='b') # plot maximum likelihood
    map_plot = plt.plot(map_range, map_prob,color='r') # plot maximum posterior
    
    # add legend
    plt.legend((mle_plot[0],map_plot[0]),legend)

""" ======================  Variable Declaration ========================== """

l = 0 # lower bound on x
u = 10 # upper bound on x
N = 20 # number of samples to generate
gVar = .1 # variance of error distribution
M =  np.array(np.arange(0,12,1)) # regression model order
e_rms_train = np.empty([M.size]) # root-mean-square error for training data
e_rms_test = np.empty([M.size]) # root-mean-square error for test data

print('M:',M)
""" =======================  Generate Training Data ======================= """
data_uniform  = np.array(generateUniformData(N, l, u, gVar)).T

train_x = data_uniform[:,0]
train_t = data_uniform[:,1]

xrange = np.arange(l,u,0.001)  # get equally spaced points in the xrange
true_t = np.sinc(xrange) # compute the true function value

""" ======================== Generate Test Data =========================== """


"""This is where you should generate a validation testing data set.  This 
should be generated with different parameters than the training data!   """
N_test = 125 
test_data = np.array(generateUniformData(N_test,l,u,gVar)).T
    
for i in range(0,M.size):
    """ ========================  Train the Model ============================= """

    w = fitdata(train_x,train_t,M[i]) 
    
    """ ========================  Estimated Polynomial Across xrange (0 to 10) ============================= """
    X = np.array([xrange**m for m in range(w.size)]).T
    predictions = X@w # compute the predicted value
    
    """ ========================  Evaluate Training Data ============================= """

    X_train = np.array([train_x**m for m in range(w.size)]).T
    train_predictions = X_train@w; # compute the predicted value of training data

    err_train = 0.5 * np.square(train_predictions - train_t) # compute the error function of train data
    e_rms_train[i] = np.sqrt(2 * np.sum(err_train) / N) # compute root-mean-square error for train data
    
    
    """ ========================  Test the Model ============================== """
    
    """ This is where you should test the validation set with the trained model """
    test_x = test_data[:,0]
    test_t = test_data[:,1]
    
    X_test = np.array([test_x**m for m in range(w.size)]).T
    test_predictions = X_test@w # compute the predicted value of test data
    
    err_test = 0.5 * np.square(test_predictions - test_t) # compute the error function of test data
    e_rms_test[i] = np.sqrt(2 * np.sum(err_test) / N_test) # compute root-mean-square error for test data


# plot data with respect to the last model order (M=11) in the array
plotData(train_x,train_t,xrange,true_t,xrange,predictions,['Training Data', 'True Function', 'Estimated\nPolynomial'])

# plot root-mean-square errors for training and test data 
plt.figure()
plotRms(M,e_rms_train,M,e_rms_test,['Training','Test'])
plt.ylabel('$E_{RMS}$')
plt.xlabel('M')
"""

===============================================================================
===============================================================================
============================ Question 2 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Variable Declaration ========================== """

# True distribution mean and variance 
trueMu = 4
trueVar = 2

'''Initial prior distribution mean and variance (You should change these 
parameters to see how they affect the ML and MAP solutions)'''
priorMu = 8
priorVar = 4

numDraws = 200 # Number of draws from the true distribution
drawRange = np.arange(1,numDraws+1,1)
drawResult = []
max_likelihood = np.empty([numDraws])
map_prob = np.empty([numDraws])

"""========================== Plot the true distribution =================="""
# plot true Gaussian function
step = 0.01
l = -20
u = 20
x = np.arange(l+step/2,u+step/2,step)
plt.figure()
p1 = plt.plot(x, norm(trueMu,trueVar).pdf(x), color='b')
plt.title('Known "True" Distribution') 

"""========================= Perform ML and MAP Estimates =================="""
for draw in range(numDraws):
    drawResult.append(norm(trueMu,trueVar).rvs(1)[0])
    max_likelihood[draw]  = sum(drawResult)/len(drawResult)
    map_prob[draw] = (((priorVar * sum(drawResult) 
                     / ((len(drawResult)*priorVar) + trueVar))) 
                     + ((priorMu * trueVar)
                     / ((len(drawResult) * priorVar) + trueVar)))

    # calculate posterior and update prior for the given number of draws
    posteriorMu = (((trueVar*priorMu)/((len(drawResult)*priorVar) + trueVar))
                  + ((len(drawResult)*priorVar*max_likelihood[draw])
                  /((len(drawResult)*priorVar) + trueVar)))
    posteriorVar = 1/(len(drawResult)/trueVar + (1/priorVar))
    
    # update prior distribution to be posterior distribution
    priorMu = posteriorMu
    priorVar = posteriorVar
    
    print("Number of Samples:", draw+1)
    print("Mean of MLE: ", max_likelihood[draw])
    print("Mean of MAP: ", map_prob[draw])
    print("Prior Mean:",priorMu)
    print("Prior Variance:",priorVar,end='\n \n')

print("Final Estimates",end='\n')
print("Mean of MLE: ", max_likelihood[draw])
print("Mean of MAP: ", map_prob[draw])

"""
You should add some code to visualize how the ML and MAP estimates change
with varying parameters.  Maybe over time?  There are many different things you could do!
"""
plt.figure()
# plot maximum likelihood mean and maximum posterior mean with respect to number of samples
plotEstimates(drawRange,max_likelihood,drawRange,map_prob,['MLE','MAP']) 
plt.xlabel('N')
plt.title('ML and MAP Estimates by Number of Samples')