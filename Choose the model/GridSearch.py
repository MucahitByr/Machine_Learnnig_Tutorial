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
'''
1.estimator : classifier(our situation)
2.X
3.y
4.cv: how much layer ? 
'''

basari=cross_val_score(estimator=classifier , X=X_train , y=y_train , cv=4)#cv = katlanma sayısı
print(basari.mean())
print(basari.std())

#parametre optimizasyonu ve algoritma seçimi
from sklearn.model_selection import GridSearchCV
p=[{'C':[1,2,3,4,5],'kernel':['linear']},
    {'C':[1,10,100,1000],'kernel':['rbf'],
     'gamma':[1,0.5,0.1,0.01,0.001]}]

''' 
GS paramrthres

estimator : classifier algorithms (which algorithm you want do optmize )
param_grid :paratmethres / things to try

scoring : according to whom , calculating


'''
gs=GridSearchCV(estimator = classifier , #svm algorithms
                 param_grid= p ,
                 scoring='accuracy',
                 cv=10,
                 n_jobs = -1)
grid_search=gs.fit(X_train, y_train)
eniyisonuc=grid.search.best_score_
bestparamethres=grid.search.best_params_
print(eniyisonuc)
print(bestparamethres)

























