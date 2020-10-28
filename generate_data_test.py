#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:07:51 2017
This script is written to evaluate a pretrained model saved as  model.h5 using 'testData.npy' 
and 'groundTruth.npy'. This script reports the error as the cross entropy loss in percentage
and also generated a png file for the confusion matrix. 
@author:Muhammad Shahnawaz
"""
# importing the dependencies
from keras.models import load_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os

# defining a function for plotting the confusion matrix
# takes cmNormalized
os.environ['QT_PLUGIN_PATH'] = ''

np.seterr(divide='ignore', invalid='ignore')


def plot_cm(cM, labels, title):
    # normalizing the confusionMatrix for showing the probabilities
    cmNormalized = np.around((cM / cM.sum(axis=1)[:, None]) * 100, 2)
    cmNormalized = np.nan_to_num(cmNormalized)

    # creating a figure object
    fig = plt.figure()
    # plotting the confusion matrix
    plt.imshow(cmNormalized, interpolation=None, cmap=plt.cm.Blues)
    # creating a color bar and setting the limits
    plt.colorbar()
    plt.clim(0, 100)
    # assiging the title, x and y labels
    plt.xlabel('Predicted Values')
    plt.ylabel('Ground Truth')
    plt.title(title + '\n%age confidence')
    # defining the ticks for the x and y axis
    plt.xticks(range(len(labels)), labels, rotation=60)
    plt.yticks(range(len(labels)), labels)
    # number of occurences in the boxes
    width, height = cM.shape
    print('Accuracy for each class is given below.')
    for predicted in range(width):
        for real in range(height):
            color = 'black'
            if (predicted == real):
                color = 'red'
                print(labels[predicted].ljust(12) + ':', cmNormalized[predicted, real], '%')
            plt.gca().annotate(
                '{:d}'.format(int(cmNormalized[predicted, real])), xy=(real, predicted),
                horizontalalignment='center', verticalalignment='center', color=color)
    # making sure that the figure is not clipped
    plt.tight_layout()
    # saving the figure
    fig.savefig(title + '.png')


# loading the pretrained model
model = load_model('model/keras/model.h5')
# loading the testData and groundTruth data
test_x = np.load('testData.npy')
groundTruth = np.load('groundTruth.npy')

# test_x = np.load('testData.npy')
# groundTruth = np.load('groundTruth.npy')
# evaluating the model
# score = model.evaluate(test_x, groundTruth, verbose=2)
# print('Baseline Error: %.2f%%' % (100 - score[1] * 100))
'''
 Creating and plotting a confusion matrix

'''
# defining the class labels
# labels = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
labels = ['Jogging', 'Walking']

# predicting the classes
predictions = model.predict(test_x, verbose=2)
# getting the class predicted and class in ground truth for creation of confusion matrix
predictedClass = np.zeros((predictions.shape[0]))
groundTruthClass = np.zeros((groundTruth.shape[0]))
for instance in range(groundTruth.shape[0]):
    predictedClass[instance] = np.argmax(predictions[instance, :])
    groundTruthClass[instance] = np.argmax(groundTruth[instance, :])

result = np.zeros(shape=(len(labels), len(labels)), dtype=np.int)
for i in range(len(predictedClass)):
    x = int(predictedClass[i])
    y = int(groundTruthClass[i])
    result[y][x] = result[y][x] + 1
print(result)
# plotting the confusion matrix
plot_cm(result, labels, 'Confusion Matrix1')
