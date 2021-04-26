# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:34:19 2018

@author: Felix
"""

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from copy import copy
#--------------------------------------------
#---------- Helper Functions ----------------

#---------- Read Input Data -----------------
def readDataToArray(trainingRows, testRows, file):
    train_data_array = np.zeros((trainingRows,3))
    test_data_array = np.zeros((testRows,2))
    training_data_string = open(file, "r")
    
    for i in range (0,trainingRows):
        # split line and write to list:
        current_line = training_data_string.readline().rstrip().split(',')

        for j in range (0, 3):
            train_data_array[i,j] = current_line[j]
    
    training_data_string.readline()
    for i in range (0,testRows):
        # split line and write to list:
        current_line = training_data_string.readline().split(',')
        
        for j in range (0, 2):
            test_data_array[i,j] = current_line[j]
    
    training_data_string.close()
    return train_data_array, test_data_array


def getResults(testRows,file):
    test_data_string = open(file, "r")
    result_data_array = np.zeros((testRows,1))
    
    for i in range(0,testRows):
        result_data_array[i,0] = test_data_string.readline().rstrip()

    return result_data_array

#---------- Normalize Input Data -----------------
def normalize(data_array, test_array):
    max_train_value = max(data_array[:,0].max(),data_array[:,1].max())
    min_train_value = min(data_array[:,0].min(),data_array[:,1].min())
    
    max_test_value = max(test_array[:,0].max(),test_array[:,1].max())
    min_test_value = min(test_array[:,0].min(),test_array[:,1].min())

    normalized_train_vec = np.zeros(data_array.shape)
    normalized_test_vec = np.zeros(test_array.shape)
    
    normalized_train_vec[:,2] = data_array[:,2]
    
    for i in range(0, data_array.shape[0]):
        for j in range(0, data_array.shape[1] - 1):
            normalized_train_vec[i,j] = 2 * (data_array[i,j] - min_train_value) / (max_train_value - min_train_value) - 1
    
    for i in range(0, test_array.shape[0]):
        for j in range(0, test_array.shape[1]):
            normalized_test_vec[i,j] = 2 * (test_array[i,j] - min_test_value) / (max_test_value - min_test_value) - 1
    
    return normalized_test_vec, normalized_train_vec
        

#---------- Feed Forward Line -----------------
def feedForward(w, line):
    out = 0
    for i in range(0, 2):
        out += w[0,i] * line[i]
    
    out += w[0,2]
    out = np.tanh(out)
    
    if (out >= 0 ):
        return 1
    else:
        return -1

#---------- Derivative of tanh ------------------
def tanhDeriv(x):
    return 1 / np.power(np.cosh(x),2)

#---------- Calculate z ------------------
def calcZ(w, line):
    out = 0
    for i in range(0, 2):
        out += w[0,i] * line[i]
    
    out += w[0,2]
    return out

#---------- Initialize Network -----------------
def initializePerceptron():
    weights = np.zeros((1, 3))
#    for i in range(0,3):
#        weights[0,i] = rnd.randrange(-1000,1000) / 100000
    weights[0,0] = -.003
    weights[0,1] = .002
    weights[0,2] = -.001
    return weights

#---------- Train Network ----------------------
def training(train_data, w, alpha, visu):
    dW_i = 0
    out = 0
    z = 0
    w_trained = w
    error_vec = np.zeros((train_data.shape[0],1))
    
    if(visu):
        ax = plt.subplot(221)
        ax.set_title("Training Data")
        plotTrainData(train_data)
        axes = plt.gca()
        axes.set_xlim([-1.2,1.2])
        axes.set_ylim([-1.2,1.2])
        plt.show()
        current_boundary = plotBoundary(w_trained)
        new_boundary = copy(current_boundary)
        
    
    for i in range(0, train_data.shape[0]):
        out = feedForward(w_trained, train_data[i,:])
        z = calcZ(w_trained, train_data[i,:])
        current_target = train_data[i,2]
        #print(current_target)
        
        error_vec[i] = .5 * np.power((current_target - out),2)
        
        for j in range(0,2):
            dW_i = - alpha * (current_target - out) *tanhDeriv(z) * train_data[i,j]
            w_trained[0,j] -= dW_i
        
        w_trained[0,2] -= - alpha * (current_target - out) * tanhDeriv(z)
        
        #visualization
        if(visu):
            ax.set_title("Training Data, Iteration: " + str(i+1))
            new_boundary = plotBoundary(w_trained)
            plt.pause(0.01)
            current_boundary.pop(0).remove()
            current_boundary = new_boundary
    
       
    return error_vec, w_trained

# enhancement function
def enhancement(train_data, w,  alpha, n, plot, error_after_train):
    train_error_sum = np.sum(error_after_train)
    error_sum = np.zeros((n + 1,1))
    error_sum[0,0] = train_error_sum
    new_error_vec = np.zeros((train_data.shape[0],1))
    
    if(plot):
        fig, axs = plt.subplots(nrows = n, ncols = 1 , constrained_layout = True)
        plt.show()
    
    for i in range(0, n):
        np.random.shuffle(train_data)
        new_error_vec, w = training(train_data, w, alpha, False)
        error_sum[i + 1,0] = np.sum(new_error_vec)
        
        if(plot):
            plot_nbr = n * 100 + 10 + i + 1
            plt.subplot(plot_nbr)
            plt.plot(new_error_vec, 'r-')
    
    return error_sum
        

#-------------- Plotting Functions ------------------
def plotBoundary(w):
    x = [-2, 2]
    y = [0, 0]

    for i in range(0, 2):
        y[i] = -(x[i]*w[0,0] + w[0,2]) / w[0,1]
    
    lines = plt.plot(x,y,'y',marker = 'x')
    
    return lines
    
    
def plotTrainData(train_data):
    for i in range(0, train_data.shape[0]):       
        if (train_data[i,2] == 1):
            plt.plot(train_data[i,0], train_data[i,1], 'ro')
        else:
            plt.plot(train_data[i,0], train_data[i,1], 'bo')
    

def plotTestData(test_data, real_results):
    for i in range(0, test_data.shape[0]):
        if (test_data[i,2] != real_results[i,0]):
            plt.plot(test_data[i,0], test_data[i,1], 'go')    
        elif (test_data[i,2] == 1):
            plt.plot(test_data[i,0], test_data[i,1], 'ro')
        else:
            plt.plot(test_data[i,0], test_data[i,1], 'bo')

            
def plotResults(error_data, error_sum, train_data, test_data, real_results, w, n):
    
    
#    ax = plt.subplot(221)
#    axes = plt.gca()
#    axes.set_xlim([-1.2,1.2])
#    axes.set_ylim([-1.2,1.2])
#    ax.set_title("Training Data")
#    plotTrainData(train_data)
#    plotBoundary(w)
    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    ax = plt.subplot(222)
    axes = plt.gca()
    axes.set_xlim([-1.2,1.2])
    axes.set_ylim([-1.2,1.2])
    ax.set_title("Test Data")
    plotTestData(test_data, real_results)
    plotBoundary(w)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    
    ax = plt.subplot(223)
    ax.set_title("Error Propagation during Training")
    plt.plot(error_data, 'r-')

    ax = plt.subplot(224)
    ax.set_title("Error Propagation after " + str(n) + " Enhancement Steps")
    plt.plot(error_sum, 'r')
    
    plt.show()
    
    
    
            
        
        
        
        
        
    