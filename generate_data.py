#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:34:20 2017
This is a small project for CNN in KERAS.
This file creates, trains and save a convolutional neural network for
Human Acitivity Recognition. The data we used for this file is released and provided by
Wireless Sensor Data Mining (WISDM) lab and can be found on this link.
http://www.cis.fordham.edu/wisdm/dataset.php  
Feel free to use this code and site this repositry if you use it for your reports or project.
@author: Muhammad Shahnawaz
"""
import matplotlib.pyplot as plt
import numpy as np
# importing libraries and dependecies
import pandas as pd
from scipy import stats

# from keras import backend as K
# K.set_image_dim_ordering('th')
# setting up a random seed for reproducibility
random_seed = 611
np.random.seed(random_seed)

# matplotlib inline
plt.style.use('ggplot')


# defining function for loading the dataset
def readData(filePath):
    # attributes of the dataset
    columnNames = ['user_id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(filePath, header=None, names=columnNames, na_values=';')
    return data


# defining a function for feature normalization
# (feature - mean)/stdiv
def featureNormalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


# defining the function to plot a single axis data
def plotAxis(axis, x, y, title):
    axis.plot(x, y)
    axis.set_title(title)
    axis.xaxis.set_visible(False)
    axis.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    axis.set_xlim([min(x), max(x)])
    axis.grid(True)


# defining a function to plot the data for a given activity
def plotActivity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plotAxis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plotAxis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plotAxis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.9)
    plt.show()
    fig.savefig(activity + '1.png')


# defining a window function for segmentation purposes
def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)


# segmenting the time series
def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = list()
    for (start, end) in windows(data['timestamp'], window_size):
        x = data['x-axis'][start:end]
        y = data['y-axis'][start:end]
        z = data['z-axis'][start:end]
        if (len(data['timestamp'][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels_tmp = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
            labelStr = stats.mode(data['activity'][start:end])[0][0]
            index = labels_tmp.index(labelStr)
            label_groundTruth = np.zeros((len(labels_tmp)))
            label_groundTruth[index] = 1
            labels.append(label_groundTruth)
    return segments, np.array(labels, dtype=np.int8)


''' Main Code '''
# # # # # # # # #   reading the data   # # # # # # # # # # 
# Path of file #
# dataset = readData('aa')
dataset = readData('sensordata.csv')

# plotting a subset of the data to visualize
for activity in np.unique(dataset['activity']):
    subset = dataset[dataset['activity'] == activity][:180]
    plotActivity(activity, subset)
# segmenting the signal in overlapping windows of 90 samples with 50% overlap
segments, labels = segment_signal(dataset)
# defining parameters for the input and network layers
# we are treating each segmeent or chunk as a 2D image (90 X 3)
numOfRows = segments.shape[1]
numOfColumns = segments.shape[2]
numChannels = 1
numFilters = 128  # number of filters in Conv2D layer
# kernal size of the Conv2D layer
kernalSize1 = 2
# max pooling window size
poolingWindowSz = 2
# number of filters in fully connected layers
numNueronsFCL1 = 128
numNueronsFCL2 = 128
# split ratio for test and validation
trainSplitRatio = 0.8
# number of epochs
Epochs = 10
# batchsize
batchSize = 10
# number of total clases
numClasses = labels.shape[1]
# dropout ratio for dropout layer
dropOutRatio = 0.2
# reshaping the data for network input
reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns, 1)

np.save('groundTruth.npy', labels)
np.save('testData.npy', reshapedSegments)
