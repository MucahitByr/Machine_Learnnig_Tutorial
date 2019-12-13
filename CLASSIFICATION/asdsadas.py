# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:03:44 2019

@author: mücahit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
print(veriler)
x=veriler.iloc[:,2:4].values
y=veriler.iloc[:,4:].values






#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_train=ss.fit_transform(x_train)
X_test=ss.transform(x_test)

from sklearn.linear_model import LogisticRegression

logr=LogisticRegression(random_state=0)
logr.fit(X_train , y_train)

y_pred=logr.predict(X_test)
print(y_pred)
print(y_test)

                    
         





    
    

