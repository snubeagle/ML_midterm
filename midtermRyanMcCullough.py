#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:11:01 2020

@author: ryan
"""

import sklearn.svm as svm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import os

def getData():
    cwpath = "/Data/MNIST/train.csv"
    dpath = os.getcwd() + cwpath
    data = np.array(pd.read_csv(dpath))
    return (
            data[0:7000,1:], 
            data[0:7000,0], 
            data[7000:7900,1:], 
            data[7000:7900,0], 
            data[7900:9900,1:], 
            data[7900:9900,0])
    
def modelController(xTrain, yTrain, xTest, yTest, xVal, yVal):
    models = {1: svcModel, 
              2: lrModel, 3: dtModel
              }
    for i in models:
        models[i](xTrain, yTrain, xTest, yTest, xVal, yVal)
        
def svcModel(xTrain, yTrain, xTest, yTest, xVal, yVal):
    Gamma = .001
    C=1
    model = svm.SVC(kernel='poly', C=C, gamma=Gamma)
    reporter(xTrain, yTrain, xTest, yTest, model, xVal, yVal)
    
def lrModel(xTrain, yTrain, xTest, yTest, xVal, yVal):
    model = LogisticRegression(max_iter=10000)
    reporter(xTrain, yTrain, xTest, yTest, model, xVal, yVal)
    
def dtModel(xTrain, yTrain, xTest, yTest, xVal, yVal):
    model = DecisionTreeClassifier()
    reporter(xTrain, yTrain, xTest, yTest, model, xVal, yVal)
    
        
def reporter(xTrain, yTrain, xTest, yTest, model, xVal, yVal):
    model.fit(xVal, yVal)
    yPred=model.predict(xTest)
    print(classification_report(yTest, yPred))
    Showlist=np.arange(10)
    for i in Showlist:
        sample = xTest[i]
        sample = sample.reshape((28,28))
        plt.imshow(sample,cmap='gray')
        plt.title('The prediction:' + str(yPred[i]))
        plt.show()

if __name__ == '__main__':
    xTrain, yTrain, xTest, yTest, xVal, yVal = getData()
    modelController(xTrain, yTrain, xTest, yTest, xVal, yVal)
    