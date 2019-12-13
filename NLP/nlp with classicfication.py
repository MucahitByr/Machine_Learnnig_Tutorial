# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:44:47 2019

@author: mücahit
"""

#Note: when we compile classification methods , seen that best class is  logistic regression between to those
import pandas as pd
import numpy as np

veriler=pd.read_csv("Restaurant_Reviews.csv")
import re
import nltk

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

nltk.download('stopwords')
derlem=[]
from nltk.corpus import stopwords

for i in range(1000):
    veri=re.sub("[^a-zA-Z]", " " , veriler["Review"][i]) # if sentence has different things without letter , out off those in the sentence
    yorum=veri.lower() #  changing   uppercase letter to  lower case
    yorum=veri.split() # we slipt the sentence
    yorum=[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] # out off meaningless
    yorum=' '.join(yorum)# retry conjugate
    derlem.append(yorum)#add the different list
    
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=2000)

X=cv.fit_transform(derlem).toarray()
y=veriler.iloc[:,1].values

#Machine Learning

from sklearn.model_selection import train_test_split

X_train , X_test, y_train , y_test = train_test_split(X,y , test_size=0.20 , random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()

gnb.fit(X_train, y_train)

y_pred=gnb.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test , y_pred)
print("GNB")
print(cm) #%73 accuracy 

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
cm=confusion_matrix(y_test , y_pred)
print("SS")
print(cm) #%73 ACCURACY 

from sklearn.linear_model import LogisticRegression

logr=LogisticRegression(random_state=0)
logr.fit(X_train , y_train)

y_pred=logr.predict(X_test)

cm=confusion_matrix(y_test , y_pred)
print("LR")
print(cm) #74

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5 , metric='minkowski')
knn.fit(X_train , y_train)

y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print("KNN")
print(cm) #%62.5 accuracy 

from sklearn.svm import SVC
svc=SVC(kernel='linear')
svc.fit(X_train , y_train)
y_pred=svc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

print('SVC')
print(cm)#%71.5


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)

y_pred=dtc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

print('DTC')
print(cm) #%66.5 accuracy 

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)
y_proba=rfc.predict_proba(X_test)
cm=confusion_matrix(y_test,y_pred)
print('RFC')
print(cm) #%71 accuracy 

