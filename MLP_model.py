# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:43:36 2020

@author: 钰哥lab
 Email:  huangxy213@163.com
"""
import numpy as np
import os
    


# define the MLP class
class MLP_Classification():
    
    # initialize the parameters of the network
    def __init__(self,input_node= 4,hidden_node= 20,output_node= 3, learning_rate = 0.001):
        self.input_node = input_node
        self.hidden_node = hidden_node
        self.output_node = output_node
        self.W_ij=np.random.uniform(
            size=(input_node+1,hidden_node),
            high= np.sqrt(6/(input_node+output_node+1)),
            low= -np.sqrt(6/(input_node+output_node+1)))

        self.W_jk = np.random.uniform(
            size=(hidden_node, output_node), 
            high= np.sqrt(6/(input_node+output_node)), 
            low= -np.sqrt(6/(input_node+output_node)))                                                                # low= -np.sqrt(6/(input_node+output_node+1)))

        self.learning_rate = learning_rate
    # forward propagation implementation   
    def forward(self,X):
        
        # input: (1,4),  W_ij:{Bias(0,20),W_ij(1:,20)}
        a_j = np.dot(X,self.W_ij[1:,:]) + self.W_ij[0,:]
        # hidden_activations
        z_j = self.tanh(a_j)
              
        # input: (1,hidden),  W_jk:(hidden,output) 
        y_k = np.dot(z_j, self.W_jk)
        
        return y_k, z_j, a_j
        
    # loss function calculation
    def loss_function(self, y_k, y_label):       
        loss = np.sum(np.square(y_k - y_label))/ 2  
        return loss
    
    # weight gradient calculation 
    def calculate_gradient(self, x, y_k, z_j, y_label):
        
        # the gradient of the output 
        # input: y_k(1,3) y_label(1,3)  ,output: y_k (1,3)
        gradient_y_k = y_k - y_label
        
        # the gradient of the w_jk  input: z_j(1,20) gradient_y_k(1,3), output:(20,3)
        gradient_w_jk = np.outer(z_j , gradient_y_k ) 
        
        # the gradient of the tanh 
        # input: z_j(1,20) , output: d_tanh(1,20)
        d_tanh = 1 - np.square(z_j) 
        
        # the gradient of the z_j input: gradient_y_k(1,3) W_jk(20,3), output:(1,20)
        gradient_z_j = np.dot(gradient_y_k, np.transpose(self.W_jk))   
        
        # the gradient of the a_j input: gradient_z_j(1,20) d_tanh(1,20), output:(1,20)
        gradient_a_j = gradient_z_j* d_tanh
        
        # the gradient of the w_ij  input: x(1,4) gradient_a_j(1,20), output:(4,20)
        gradient_w_ij = np.outer(x, gradient_a_j)
        # the gradient of the W_i_j_1(Bias), input: gradient_a_j(1,20), output:(1,20)
        gradient_W_i_j_1 = gradient_a_j
        
        
        return gradient_w_jk, gradient_w_ij, gradient_W_i_j_1
    
    def tanh(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    # weight updates
    def train(self, x, y_label):
        # forward propagation  
        y_k, z_j, a_j = self.forward(x)
        # loss calculation
        loss = self.loss_function(y_k, y_label)
        # weight gradient calculation
        gradient_w_jk, gradient_w_ij, gradient_W_i_j_1 = self.calculate_gradient( x, y_k, z_j, y_label)
        
        # weight update
        self.W_jk = self.W_jk - self.learning_rate*gradient_w_jk

        self.W_ij[1:,:] = self.W_ij[1:,:] - self.learning_rate*gradient_w_ij
        self.W_ij[0,:] = self.W_ij[0,:] - self.learning_rate*gradient_W_i_j_1

        
        return loss


    def predict_label(self, X):

        y_k, z_j, a_j = self.forward(X)
            
        return y_k 

    def accuracy(self, X, y_label):
        y_predict_label = self.predict_label(X)
        y_preds = np.argmax(y_predict_label, axis=1)
        y_k = np.argmax(y_label, axis=1)
        return np.mean(y_preds == y_k)
    
    def save(self, path='MLP_Parameters/model1/'):

        # save W_jk, W_ij
        data_path = os.path.join('%sW_jk-%s-%s-%s.npy' % (path, self.input_node,self.hidden_node,self.output_node))
        np.save(data_path,self.W_jk)
        data_path = os.path.join('%sW_ij-%s-%s-%s.npy' % (path, self.input_node,self.hidden_node,self.output_node))
        np.save(data_path,self.W_ij)
        
    def load(self, path='MLP_Parameters/model1/'):

        # save W_jk, W_ij
        data_path = os.path.join('%sW_jk-%s-%s-%s.npy' % (path, self.input_node,self.hidden_node,self.output_node))
        self.W_jk = np.load(data_path)
        data_path = os.path.join('%sW_ij-%s-%s-%s.npy' % (path, self.input_node,self.hidden_node,self.output_node))
        self.W_ij = np.load(data_path)