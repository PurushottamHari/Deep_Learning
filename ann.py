# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 19:33:14 2020

@author: PURUSHOTTAM
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Artificial_Neural_Networks\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])

labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')

X = np.array(columnTransformer.fit_transform(X), dtype = np.str)

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Create your classifier here
classifier = Sequential()

#Adding te layers
classifier.add(Dense(output_dim = 6, init = 'uniform',activation='relu',input_dim=11))  #input dim only once
classifier.add(Dropout(p = 0.1)) #Drop 10% Nodes
classifier.add(Dense(output_dim = 8, init = 'uniform',activation='relu'))   #2nd hidden ayer
classifier.add(Dense(output_dim = 1,init ='uniform',activation='sigmoid'))  #output layer

#Compiling
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])   #the cost function is the logarithmic one and has 2 values....if multi use 'categorical_crossentropy'


#Fitting the ann
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Homework
y_pred_hw = classifier.predict(sc.fit_transform(np.array([[0,0,600,1,400,3,60000,2,1,1,50000]])))
y_pred_hw = (y_pred_hw>0.5)


# Evaluating the ANN (K_Fold Cross Validation)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)  #K=10 in K-Folds and nb_jobs(#No of CPUs parallel) = -1 (all of them)
mean = accuracies.mean()
variance = accuracies.std()


#Tuning the ANN (Choosing the best HyperParameters)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizerInput):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizerInput, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size':[25,32],          #HyperParameters to compare
              'nb_epoch':[100, 500],
              'optimizerInput': ['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10
                           )

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_ 

