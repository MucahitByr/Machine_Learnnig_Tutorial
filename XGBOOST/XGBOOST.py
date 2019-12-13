# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:47:40 2019

@author: mücahit
"""

import pandas as pd

url="http://www.bilkav.com/wp-content/uploads/2018/03/satislar.csv"

veriler=pd.read_csv(url)

X=veriler.iloc[:,0:1]
Y=veriler.iloc[:,1]

bolme=0.33
from sklearn import model_selection
X_train, X_test , Y_train , Y_test= model_selection.train_test_split(X,Y,test_size=0.33)

