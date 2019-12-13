# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:11:58 2019

@author: mücahit
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import flatten 
from keras.layers import Dense

#ilkleme
classifier= Sequential()

#step-1 - Convolution

classifier.add(Convolution2d(32 , 3 , 3 , input_shape=64,64,3) , activation='relu')
#step 2 - pooling 
classifier.add(MaxPooling2D(Pool_size=(2 , 2 )))

# layer of convolution 
    
classifier.add(Convolution2D(32,3,3 , activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step-3 flattening
classifier.add(Flatten( ))

#step 4 - Natureal Neureal Network
classifier.add(Dense(output_dim=128 ,  activation='relu'))
classifier.add(Dense(outpu_dim=1 , activation='sigmoid'))

#CNN

classifier.compile(optimizier='adam' , loss='binary_crossentropy' , metric='accuracy' )

#CNN AND İMAGE

from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator()
