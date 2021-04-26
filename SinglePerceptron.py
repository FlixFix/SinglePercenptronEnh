# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 08:57:30 2018

@author: Felix
"""
import numpy as np
import HelperFunctions as helpfct
import matplotlib.pyplot as plt
import random as rnd


# read data
train_data, test_data = helpfct.readDataToArray(600,30,"test_data2.txt")



# normalize data
norm_test_data, norm_train_data = helpfct.normalize(train_data, test_data)


# initialize perceptron
w = helpfct.initializePerceptron()

#create figure
fig = plt.figure(figsize=(15,9))
fig.canvas.set_window_title('Single Perceptron')

# do the training
error_data, w = helpfct.training(norm_train_data,w,0.0002, True)

print(w[0,:])

error_sum = helpfct.enhancement(norm_train_data, w, 0.0002, 10, False, error_data)



#plot everything
fig, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True)
plt.subplot(221)
helpfct.plotTrainData(norm_train_data)

plt.subplot(221)
helpfct.plotBoundary(w)

plt.subplot(223)
plt.plot(error_data, 'r-')

plt.subplot(224)
plt.plot(error_sum, 'r')

#test
real_results = helpfct.getResults(norm_test_data.shape[0], "result_data2.txt")
my_results = np.zeros((norm_test_data.shape[0],3))
my_results[:,0] = norm_test_data[:,0]
my_results[:,1] = norm_test_data[:,1]

for i in range(0, norm_test_data.shape[0]):
    my_results[i,2] = helpfct.feedForward(w,norm_test_data[i,:])
    

helpfct.plotResults(error_data, error_sum, norm_train_data, my_results, real_results, w, 9)

plt.subplot(222)
helpfct.plotTestData(my_results, real_results)
helpfct.plotBoundary(w)
plt.show()

ew_data = helpfct.enhancement(train_data,1)

# enhancement
helpfct.enhancement(norm_train_data, w, 0.0002, 9, False)


# -------------------------test with new data-------------------------------------
train_data2, test_data2 = helpfct.readDataToArray(200,50,"test_data2.txt")

# normalize data
norm_test_data2, norm_train_data2 = helpfct.normalize(train_data2, test_data2)

# initialize perceptron
w2 = helpfct.initializePerceptron()

# do the training
error_data2, w2 = helpfct.training(norm_train_data2,w2,0.0002, True)



        

