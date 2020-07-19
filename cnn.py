# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:55:18 2020

@author: PURUSHOTTAM
"""
import tensorflow as tf
import keras
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 5} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#CNN Building
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json


#Initializing the CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3), activation = 'relu'))   # number of feature detectors,number of rows, number of columns, input shape{for theano backend=>(3,64,64)}

#Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))  #The size of filter is 2X2

#Flattening
classifier.add(Flatten())  

classifier.add(Dense(output_dim=5))

#Full Connection....Starting with ANN
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))  #Final Output

#Compiling the Model
classifier1.compile(optimizer = 'adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])

#Fitting the image

from keras.preprocessing.image import ImageDataGenerator
 
#Code taken from Keras Documentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/PURUSHOTTAM/Desktop/val',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


test_set = test_datagen.flow_from_directory(
        'Convolutional_Neural_Networks/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier1.fit_generator(
        training_set,
        steps_per_epoch=8000,    #Total images we have
        epochs=10,
        #validation_data=test_set,
        #validation_steps=2000
        )   #Total images we have


#-------------------------------------------------------------------------------------------
#APPARENTLY THIS PERFORMS OKAYISH AND IS NOT THAT SATISFACTORY, HENCE LETS TWEAK SOME STUFF

#WE WILL ADD ANOTHER CONVOLUTION/POOLING LAYER AS FOLLOWS
classifier1 = Sequential()

#Convolution1
classifier1.add(Convolution2D(32,3,3,input_shape = (64,64,3), activation = 'relu'))   # number of feature detectors,number of rows, number of columns, input shape{for theano backend=>(3,64,64)}

#Pooling1
classifier1.add(MaxPooling2D(pool_size = (2,2)))  #The size of filter is 2X2

#Convolution2
classifier1.add(Convolution2D(32,3,3, activation = 'relu'))   # number of feature detectors,number of rows, number of columns, {NO NEED HERE=>}input shape{for theano backend=>(3,64,64)}

#Pooling2
classifier1.add(MaxPooling2D(pool_size = (2,2)))  #The size of filter is 2X2

#Flattening
classifier1.add(Flatten())  

#classifier1.add(Dense(output_dim=5))

#Full Connection....Starting with ANN
classifier1.add(Dense(output_dim = 128, activation = 'relu'))
classifier1.add(Dense(output_dim = 5, activation = 'softmax'))  #Final Output

#THIS WILL PROVIDE A MUCH BETTER RESULT
#---------------------------------------------------------------------------------------------


#Predicting an image...sharky
'''
#Self done
import os, sys
from PIL import Image

im = Image.open('sharky.jpeg')
im = im.resize((64,64), Image.ANTIALIAS)
im.save('sharky_resizied.jpg')
'''

#Saving the model
# serialize model to JSON
model_json = classifier1.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier1.save_weights("model.h5")
print("Saved model to disk")



#Loading the Model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier1 = model_from_json(loaded_model_json)
# load weights into new model
classifier1.load_weights("model.h5")
print("Loaded model from disk")
 



#Method in course
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('C:/Users/PURUSHOTTAM/Desktop/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/ben.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)

#Although the input takes 4 dimensions...one of batch as well
test_image = np.expand_dims(test_image,axis = 0)

result = classifier1.predict(test_image)

#training_set.class_indices

print(result[0])
    

#Show the details about the Actor!-----------------------------------------------------------
from bs4 import BeautifulSoup
import requests
import os
import io

class lifeInImages:
    def __init__(self, term):
        self.term = term
        term = term.replace(' ', '+')
        self.nameUrl = 'https://www.google.com/search?q={0}'.format(term)
        self.run()

    def getUrl(self, response):
        soup = BeautifulSoup(response.content, 'html.parser')
        for link in soup.find_all('a', class_='bdCxnb'):
            self.url = link['href']
                        
    def run(self):
        print("1")
        MOBILE_USER_AGENT = "Mozilla/5.0 (Linux; Android 7.0; SM-G930V Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.125 Mobile Safari/537.36"
        print("2")
        headers = {"user-agent" : MOBILE_USER_AGENT}
        print("3")
        nameResponse = requests.get(self.nameUrl, headers=headers)
        print("4")
        self.getUrl(nameResponse)
        print("5")
        response = requests.get(self.url)
        print(response.text)
        print("6")
        with io.open(r"C:/Users/PURUSHOTTAM/Desktop/electron-quick-start/index.html", "w+", encoding="utf-8") as f:
            f.write(response.text)
        print("7")
        
        os.system("npm start --prefix C:/Users/PURUSHOTTAM/Desktop/electron-quick-start")

a = lifeInImages('mindy kaling')




def UseModel(imagePath):
    import numpy as np
    from keras.preprocessing import image
    
    test_image = image.load_img(imagePath,target_size=(64,64))
    test_image = image.img_to_array(test_image)
    
    #Although the input takes 4 dimensions...one of batch as well
    test_image = np.expand_dims(test_image,axis = 0)
    
    result = classifier1.predict(test_image)
    
    #training_set.class_indices
    
    print(result)
    if result[0][0] == 1:
        return "Ben Affleck" 
    elif result[0][1] == 1:
        return "Elton John"
    elif result[0][2] == 1:
        return "Jerry Seinfield"
    elif result[0][3] == 1:
        return "Madonna"
    elif result[0][4] == 1:
        return "Mindy Kaling"
    
    
def SearchStar(imagePath):
    star = UseModel(imagePath)
    lifeInImages(star)
    
    
    
SearchStar("C:/Users/PURUSHOTTAM/Desktop/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/madonna.jpg")    