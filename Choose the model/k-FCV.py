# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 10:51:10 2019

@author: mücahit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("Social_Network_Ads.csv")

X=veriler.iloc[:,[2,3]].values
y=veriler.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=0 )

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.svm import SVC
classifier=SVC(kernel='rbf' , random_state=0)
classifier.fit(X_train , y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test , y_pred)
print(cm)

from sklearn.model_selection import cross_val_score

basari=cross_val_score(estimator=classifier , X=X_train , y=y_train , cv=4)#cv = katlanma sayısı
print(basari.mean())
print(basari.std())