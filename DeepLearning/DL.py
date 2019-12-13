# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:53:34 2019

@author: mücahit
"""
# add the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("Churn_Modelling.csv")

# spliting the depend samples and indepent samples

X=veriler.iloc[:,3:13].values
Y=veriler.iloc[:,13].values
print(X)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder() 
X[:,1]=le.fit_transform(X[:,1]) 

le2=LabelEncoder()
X[:,2]=le2.fit_transform(X[:,2])

print(X)

from sklearn.preprocessing import OneHotEncoder
OHE=OneHotEncoder(categorical_features=[1])

X=OHE.fit_transform(X).toarray()
print(X)
X=X[:,1:]

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test=train_test_split(X,Y , test_size=0.33 , random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#3-) yapay sinir ağları
import keras
from keras.models import Sequential # choose the neural network's model 
from keras.layers import Dense # make a neural network

classifier=Sequential() # with use this , play the neural network and add the new paramethres
classifier.add(Dense(6 , init='uniform' , activation='relu' , input_dim=11)) # how much node add to  secret layer  , add first random paratmethres , decide the sort of parametrehs,how much neural add to enterance layer.              
classifier.add(Dense(6 , init='uniform' , activation='relu' ))# ı add the second secret layer without enterance layer
classifier.add(Dense(1 , init='uniform' , activation='sigmoid' )) # exit
classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'] , )
classifier.fit(X_train , y_train , epochs=50)

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test , y_pred)
print(cm)
