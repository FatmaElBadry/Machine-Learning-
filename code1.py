# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 07:26:50 2019

@author: ayman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#The path o f the data file
path = 'C:\\Fatma ElBadry\\ML\ML-GitHub-Projects\\Linear-Regression-1\\ex1data1.txt'

#Read the data using pandas
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

#show data details
print('data shape = \n' ,data.shape)
print('**************************************')
print('data = \n' ,data.head(10) )
print('**************************************')
print('data.describe = \n',data.describe())
print('**************************************')
#draw data
#data.plot(kind='scatter', x='Population', y='Profit', figsize=(6,6))

# adding a new column called ones before the data
data.insert(0, 'Ones', 1)
print('new data = \n' ,data.head(10) )
print('**************************************')

# separate X (training data) from y (target variable)
##Get the number of columns in the data
cols = data.shape[1]
print('No of Column = ',data.shape[1])

X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

print('**************************************')
print('X data = \n' ,X.head(10) )
print('y data = \n' ,y.head(10) )
print('**************************************')

# convert from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
print('X \n',X)
print('X.shape = ' , X.shape)
print('theta \n',theta)
print('theta.shape = ' , theta.shape)
print('y \n',y)
print('y.shape = ' , y.shape)
print('theta.T \n= ' ,theta.T)
print('theta.T.shape = ' ,theta.T.shape)
print('**************************************')

#=========================================================================
# cost function
def computeCost(X, y, theta):
    z = np.power(((X * theta.T) - y), 2)
#    print('z \n',z)
#    print('m ' ,len(X))
    return np.sum(z) / (2 * len(X))

print('computeCost(X, y, theta) = ' , computeCost(X, y, theta))

print('**************************************')

# GD function
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

# initialize variables for learning rate and iterations
alpha = 0.01
iters = 1000

# perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(X, y, theta, alpha, iters)

print('g = ' , g)
print('cost  = ' , cost[0:50] )
print('computeCost = ' , computeCost(X, y, g))
print('**************************************')


#=========================================================================

# get best fit line

x = np.linspace(data.Population.min(), data.Population.max(), 100)

f = g[0, 0] + (g[0, 1] * x)
print('f \n',f)
# draw the line
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

# draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')























