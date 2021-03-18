# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:48:47 2020

@author: 钰哥lab
 Email:  huangxy213@163.com
"""
import pre_data
import MLP_model
import numpy as np
import matplotlib.pyplot as plt
import os

# MLP parameter
n_features = 4
n_hidden = 20
n_classes = 3
learning_rate = 0.001


def one_hot(n_classes,y):
    return np.eye(n_classes)[y]


accuracy_train_epoch = [[],[],[],[],[]]
accuracy_test =[]
losses = [[],[],[],[],[]]

crossValidation = 0
epoches = 300
plt.close('all')

"""""""""""""""""""""""""""""Cross validation"""""""""""""""""""""""""""""
# shuffle the data set randomly and 
#       get data from dataset with cross validation
dataset = pre_data.getDataset_file('data/bezdekIris.data')
dataloader =pre_data.getData_CrossValidation(dataset,5)

for train_data, train_label, test_data, \
                 test_label in dataloader:  
    print ('CrossValidation: %s' %(crossValidation))
    """""""""""""""""""""""""""Training model"""""""""""""""""""""""""""        
    # Initialize w ,learning_rate
    model = MLP_model.MLP_Classification(n_features, n_hidden,
                         n_classes, learning_rate)
    
    # Number of epoches
    for epoch in range(epoches):
        loss = 0
        # The amount of training data
        N = np.shape(train_data)[0]
        for num_data in range(N):
            # Choose the pattern xn
            x = train_data[num_data,:].reshape(1,n_features)
            label = one_hot(n_classes,
                            train_label[num_data].astype(int)).reshape(1,n_classes)
            # Forward the input through the network
            # calculate the gradient of the network weight
            # Update network weights
            loss_ = model.train( x, label)
            loss = loss + loss_

        # accuracy rate of train data per epoch
        x = train_data
        label = one_hot(n_classes,train_label.astype(int))    
        accuracy = model.accuracy(x, label)            
        accuracy_train_epoch[crossValidation].append(accuracy)
        
        # record loss per epoch
        losses[crossValidation].append(loss/N) 
        
    # save model
    path = os.path.join('MLP_Parameters/model%s/' % (crossValidation))
    folder = os.getcwd() +'/'+ path
    if not os.path.exists(folder):
        os.makedirs(folder)
    model.save(path)
    
    """""""""""""""""""""""""""""""test model"""""""""""""""""""""""""""""""
    # load model
    model = MLP_model.MLP_Classification(n_features, n_hidden,
                         n_classes, learning_rate)
    model.load(path)
    
    # accuracy rate of test data
    x = test_data
    label = one_hot(n_classes,test_label.astype(int))    
    accuracy = model.accuracy(x, label)
    accuracy_test.append(accuracy)    
    print ('  test accuracy: %.2f' %(accuracy))
    # 5 fold cross validation train data iter
    crossValidation+=1
print ('5 fold cross validation average accuracy : %.2f' %(np.mean(accuracy_test)))

# figure show
accuracy_train_epoch = np.array(accuracy_train_epoch,dtype=float)

# The loss of model trained independently by five different training sets per epoch
plt.figure(num="loss")
plt.xlabel("The number of iterations")
plt.ylabel("Loss")
for i in range(len(losses)):
    plt.plot(losses[i],label="model"+str(i)+" - training set"+str(i))   
    plt.legend() 
    plt.title("The loss of five different models per epoch");
plt.savefig('figures/loss.png')


# accuracy rate train data
plt.figure(num="accuracy rate train data")  
plt.xlabel("The number of iterations")
plt.ylabel("Accuracy")
for i in range(len(accuracy_train_epoch)):
    plt.plot(accuracy_train_epoch[i],label="model"+str(i)+" - training set"+str(i))   
    plt.legend() 
    plt.title("accuracy rate train data");
plt.savefig('figures/accuracy_rate_train_data.png')
  