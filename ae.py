# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 02:36:02 2020

@author: PURUSHOTTAM
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
movies = pd.read_csv('C:/Users/PURUSHOTTAM/Desktop/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 6 - AutoEncoders (AE)/AutoEncoders/ml-1m/movies.dat', sep='::', header = None, engine ='python', encoding = 'latin-1')
users = pd.read_csv('C:/Users/PURUSHOTTAM/Desktop/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 6 - AutoEncoders (AE)/AutoEncoders/ml-1m/users.dat', sep='::', header = None, engine ='python', encoding = 'latin-1')
ratings = pd.read_csv('C:/Users/PURUSHOTTAM/Desktop/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 6 - AutoEncoders (AE)/AutoEncoders/ml-1m/ratings.dat', sep='::', header = None, engine ='python', encoding = 'latin-1')

#Preparing training and test set
training_set = pd.read_csv('C:/Users/PURUSHOTTAM/Desktop/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 6 - AutoEncoders (AE)/AutoEncoders/ml-100k/u1.base',delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('C:/Users/PURUSHOTTAM/Desktop/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 6 - AutoEncoders (AE)/AutoEncoders/ml-100k/u1.test',delimiter = '\t')
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

#Creating the architechure (Stacked Auto Encoder)
class SAE(nn.Module):   #Inheriting a super class nn.Module
    def __init__(self, ):
        super(SAE, self).__init__()   #The super constructor
        self.fc1 = nn.Linear(nb_movies, 20)  #1st Hidden layer is 20 neurons
        self.fc2 = nn.Linear(20, 10)   #2nd hidden layer is 10 nodes
        self.fc3 = nn.Linear(10, 20)   #3rd hidden layer is 20 nodes
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()  #Defining the activation Layer
    
    def forward(self, x):
        x = self.activation(self.fc1(x))   #Going trough the first layer and so on
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)                    #No activation here
        return x
    

#Defining the object
sae = SAE()

criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)  #lr is learning rate


#Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    #To exclude users who havent rated any movie...memory optimization
    s = 0.   #Float Value
    
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)   #The batch parameter which pytorch requires
        target = input.clone()
        #Atleast one rating is larger than zero
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False   #Trick to save up some memory 
            output[target == 0] = 0   #They wont count in the error
            loss = criterion(output, target)  #Calculating the error
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)   #1e-10 so that denominator isnt zero
            loss.backward()   #Decides the direction of weights updating and optimizer.step will determine the intensity
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch:'+str(epoch)+' loss:'+str(train_loss/s))            
            
    
#Testing the SAE
    
test_loss = 0
#To exclude users who havent rated any movie...memory optimization
s = 0.   #Float Value

for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)   #This is related to how the dataset is, hence we are using the training set #The batch parameter which pytorch requires
    target = input.clone()
    #Atleast one rating is larger than zero
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False   #Trick to save up some memory 
        output[target == 0] = 0   #They wont count in the error
        loss = criterion(output, target)  #Calculating the error
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)   #1e-10 so that denominator isnt zero
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
       
print('Test-loss:'+str(test_loss/s))            
            



        
        
