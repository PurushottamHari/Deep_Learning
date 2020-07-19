# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:38:45 2020

@author: PURUSHOTTAM
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Self_Organizing_Maps/Credit_Card_Applications.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#Training SOM
from minisom import MiniSom
som = MiniSom(x=10,y=10, input_len = 15, sigma = 1.0, learning_rate = 0.5)   #10 X 10 Grid, Sigma is the circle radius parameter
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#Visualizing the Data
from pylab import bone, pcolor, colorbar, plot, show
bone() #The window
pcolor(som.distance_map().T)  #Plot the colors according to the mean internueron distances given by the "distance_map" function  [.T means transpose]
colorbar()

# Here if you run the plot...the white one have HIGH interneuron distance and hence are the frauds....
# Now we will mark the customers who were approved and who were not approved for further clarity
markers = ['o','s']  # o-> circle || s->square
colors = ['r','g']   # red and green
 
for i,x in enumerate(X): #i is the index...x is the vector of each customer 
    W = som.winner(x)    #Winner node for customer X
    plot(W[0] + 0.5,     #W[0] and W[1]  are the coordinates of the corner of the winning node....hence we added the values to find the center  
          W[1] + 0.5,
          markers[Y[i]],  #Mark according to the whether customer was approved or not
          markeredgecolor = colors[Y[i]], #Color the edge only
          markerfacecolor = 'None',
          markersize = 10,
          markeredgewidth = 2)

show()     


#Finding the frauds...
mappings = som.win_map(X)  #Dictionary of customers to cooridnates
frauds = np.concatenate((mappings[(3,5)], mappings[(2,6)]) , axis=0) #Looking at the graph plotted (white ones)
#They are still scaled though...hence
frauds = sc.inverse_transform(frauds)



