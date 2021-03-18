# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:13:47 2020

@author: 钰哥lab
 Email:  huangxy213@163.com
"""
import numpy as np
import random

# the label corresponds to the iris name
label_irisName = {"Iris-setosa":"0",
                  "Iris-versicolor":"1",
                  "Iris-virginica":"2",}
def getDataset_file(file_path):
    '''
    get dataset from file and 
        shuffle the training data set randomly 
    '''   
    dataset = []
    # file_path = 'data\iris.data'
    # file_path = 'data\cc.txt'
    with open(file_path, 'r') as file_object:
        for line in file_object:
            # remove '\n', separate the strings with ',' and 
            # store them in the list
            line=line.strip('\n').split(',') 
            # make sure the list is not empty
            if line[-1] != '':
                # Convert the iris name to label 
                #   using the dictionary (label_irisName)                
                line[-1] = label_irisName[line[-1]]             
                dataset.append(line)
        # shuffle dataset with the random.shuffle method.
        random.seed(17)
        random.shuffle(dataset)
        # Convert dataset to array and 
        #   define the data type as float
        dataset = np.array(dataset,dtype=float)
    return dataset


def getData_CrossValidation(dataset, num_fold):
    '''
    get data from dataset with cross validation
    Parameters:
        dataset: dataset from file
        num_fold: folds number 
    '''  
    # get the number of instances and attributes from dataset
    (num_instance, num_attribute) = dataset.shape
    # number of instances per fold  eg.[30,30,30,30,30]
    num_instance_folds = np.ones(num_fold) * \
                        int(num_instance / num_fold)
    num_instance_folds[0:(num_instance % num_fold)] += 1
    # array index of all instances   eg.[0,1,2...150]
    index = np.arange(num_instance).astype(int)
    idx_previous=0
    for idx in num_instance_folds:        
        # split the dataset and get the index 
        # eg. [0...30],[30...60],[60...90],[90...120],[120...150]
        test_idx = index[idx_previous:idx_previous+int(idx)]
        idx_previous = idx_previous+int(idx)        
        train_idx = np.setdiff1d(index, test_idx)      
        # get the train_data, train_label, test_data, test_label
        train_data = dataset[train_idx, 0:4]
        train_label = dataset[train_idx, 4]
        test_data = dataset[test_idx,0:4]
        test_label = dataset[test_idx,4]
        yield train_data, train_label, test_data, test_label
        