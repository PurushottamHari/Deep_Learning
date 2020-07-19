# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 02:58:21 2020

@author: PURUSHOTTAM

Restrictive Boltzmann Machines(RBM)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable 

#Import the dataset
movies = pd.read_csv('Desktop/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 5 - Boltzmann Machines (BM)/Boltzmann_Machines/ml-1m/movies.dat', sep='::', header = None, engine ='python', encoding = 'latin-1')
users = pd.read_csv('Desktop/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 5 - Boltzmann Machines (BM)/Boltzmann_Machines/ml-1m/users.dat', sep='::', header = None, engine ='python', encoding = 'latin-1')
ratings = pd.read_csv('Desktop/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 5 - Boltzmann Machines (BM)/Boltzmann_Machines/ml-1m/ratings.dat', sep='::', header = None, engine ='python', encoding = 'latin-1')

#Preparing the training and test set
training_set = pd.read_csv('Desktop/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 5 - Boltzmann Machines (BM)/Boltzmann_Machines/ml-100k/u1.base',delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('Desktop/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 5 - Boltzmann Machines (BM)/Boltzmann_Machines/ml-100k/u1.test',delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


#Now we will make a matrix according to our dataset to fit the model properly with proper rows and columns
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

def convert(data):
    new_data = []
    for id_users in range(1,nb_users+1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        # Add 0 as rating of movies that were not rated by the user
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings 
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)


# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        #The biases for hidden and visible resp
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())   #Matrix Multiplication
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):  #According to the research paper 
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        

#Determine the inputs for Rbm        
nv = len(training_set[0])
nh = 50    # Number of features to be detected, basically the hidden nodes
batch_size = 100

#Make the object
rbm = RBM(nv, nh)

#Training 
nb_epoch = 10


for epoch in range(1 , nb_epoch + 1):
    train_loss = 0
    s = 0.   #Counter which is a float
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user : id_user + batch_size]
        v0 = training_set[id_user : id_user + batch_size]
        ph0, _ = rbm.sample_h(v0)
        #Contrastive Divergence
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]   #Training only on the nodes were input was given 
        
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+ str(epoch) + ' loss: '+ str(train_loss/s))    
    

#Testing the RBM
test_loss = 0
s = 0.   #Counter which is a float
for id_user in range(0, nb_users):
    v = training_set[id_user : id_user+1]  #They are the inputs to get the predictive ratings of the test set
    vt = test_set[id_user : id_user+1]

    #Contrastive Divergence with just one step
    if len(vt[vt>=0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('testLoss: '+ str(test_loss/s)) 
   




        
        