# -*- coding: utf-8 -*-
"""
Helper functions for assignment 1 in DD2434

#author Henrik Hellström
"""
import numpy as np

"""Loads dataset from path and returns three variables.
dataset = matrix containing data from all attributes except name and type (101 by 16 matrix)
types = 16 by 1 nparray of types
names = 16 by 1 python list of names"""
def loadDataset(path):
    #Loads all attributes except "animal name" and "type"
    dataset = np.loadtxt(path, delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
    #Load types
    types = np.loadtxt(path, delimiter=',', usecols=(17))
    #Load names
    file = open(path, "r")
    names = []
    for line in file:
        name = line.split(',')[0]
        names.append(name)
    file.close()
    return (dataset, types, names)

"""Preprocessing of data matrix. Performs the following actions:
    1. Normalize legs field to be within 0-1
    2. Takes transpose to have each column be one datapoint
    3. Performs centering of data"""
def preprocessing(Y):
    #Normalize legs value to be between 0-1
    Y[:,12] = Y[:,12]/8
    #Transpose Y (each column should be one data point)
    Y = Y.T
    #Center Y
    n = Y.shape[1]
    print(Y.shape)
    Y = Y - 1/n*np.matmul((Y), np.ones((n,n)))
    return Y