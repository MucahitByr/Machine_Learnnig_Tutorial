# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 08:54:39 2019

@author: mücahit
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("wine.csv")

X=veriler.iloc[:,0:13].values
y=veriler.iloc[:,13:].values

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.2 , random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.decomposition import PCA #reduction THE PRİNCİPLE

pca=PCA(n_components=2)
X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

classifier2=LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

#predictions
y_pred=classifier.predict(X_test)

y_pred2=classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix

print("original / without PCA")
cm=confusion_matrix(y_test , y_pred)
print(cm)

print("original / with PCA")
cm2=confusion_matrix(y_test , y_pred2)
print(cm2)
print(" with use pca and without use pca")
cm3=confusion_matrix(y_pred , y_pred2)
print(cm3)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2) 

X_train_lda=lda.fit_transform(X_train , y_train)
X_test_lda=lda.transform(X_test)

# after USE THE LDA

classifier_lda=LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda , y_train)

#predict to lda's data
y_pred_lda=classifier_lda.predict(X_test_lda)

#after the lda / original
print("lda and original")
cm4=confusion_matrix(y_pred , y_pred_lda )
print(cm4)



























