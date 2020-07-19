# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:43:19 2020

@author: PURUSHOTTAM
"""

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the training set

dataset = pd.read_csv("Recurrent_Neural_Networks/Google_Stock_Price_Train.csv")

training_set = dataset.iloc[:,1:2].values  #Only the first index taken but 1:2 makes sure its a numpy array and not a vector

#Feature Scaling (Standardization and Normalization) [FOr RNN and Sigmoid activation...Normalization is recommended]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))

training_set_scaled = sc.fit_transform(training_set)  #fit will find max and min...wheras transform will use the formula

#TimeSteps basically means how much of previous data should be considered to make prediction for the next one...Here it is 60
#Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, 1258):   #1258 is the size of dataset..training
    X_train.append(training_set_scaled[i-60:i, 0])    #Using these
    y_train.append(training_set_scaled[i, 0])         #To predict this
    
X_train , y_train = np.array(X_train), np.array(y_train)     #Look at the data structures created to understand

#Reshaping (adding a new dimension)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 1))   #In Keras Doc: "3D tensor with shape (batch_size, timesteps, input_dim)"  where input_dim is the factor which might help in predicting...we are only using 1 here 

#Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 

regressor = Sequential()

# 4 Layers
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))   #Return Sequences = True if not final layer
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

#Output Layer
regressor.add(Dense(units = 1))


#Compiling
regressor.compile(optimizer = 'adam', loss= 'mean_squared_error')

#Visualizing the Model
from keras.utils.vis_utils import plot_model
plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


#Fitting
regressor.fit(X_train, y_train, epochs= 20, batch_size = 32)


#Prediction and Visualization
test_dataset = pd.read_csv("Recurrent_Neural_Networks/Google_Stock_Price_Test.csv")

test_set = test_dataset.iloc[:,1].values

'''
Key points:
1- We need 60 points prior to make a predicition....hence the final model for prediction will have 
the points of training as well as test set to achieve 60 prior points throughout
2- We cannot concatenate the trainnig and test directly and then scale...because the values will become 
different from the ones trained
'''

dataset_total = pd.concat((dataset['Open'],test_dataset['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(test_dataset) - 60:].values

inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)   #Accoring to previously done measures

#Making the input the way it needs to be accepted according to Keras

X_test = []

for i in range(60, 80):   
    X_test.append(inputs[i-60:i, 0])    
 
X_test = np.array(X_test) 
   
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))   #In Keras Doc: "3D tensor with shape (batch_size, timesteps, input_dim)"  where input_dim is the factor which might help in predicting...we are only using 1 here 

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing
plt.plot(test_set, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
