# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:41:07 2019

@author: mücahit
"""
# ders 6 : kütüphanelerin yüklenmesi

#kütüphaneler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# kod kismi

#data ekleme
veriler=pd.read_csv('eksikveriler.csv')
print(veriler)

boy=veriler[['boy']]
print(boy)

boykilo=veriler[['boy','kilo']]
print(boykilo)

#missing value

#eksik veriler icin imputer fonk. kullanacagiz

from sklearn.preprocessing import Imputer
yas=veriler.iloc[:,1:4].values
print(yas)
imputer=Imputer(missing_values='NaN' , strategy='mean' , axis=0)
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)

# Nümerik olmayan verileri nümerik verilere çevirme
#Encoder

from sklearn.preprocessing import LabelEncoder , OneHotEncoder

lb= LabelEncoder()
ulke=veriler.iloc[:,0:1].values
print(ulke)

ulke[:,0]=lb.fit_transform(ulke[:,0])
print(ulke)

ohe= OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke[:,0:1]).toarray()
print(ulke)