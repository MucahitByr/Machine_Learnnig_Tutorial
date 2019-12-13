# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:44:47 2019

@author: mücahit
"""
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
    veri=re.sub("[^a-zA-Z]", "- " , veriler["Review"][i]) # if sentence has different things without letter , out off those in the sentence
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
print(cm)

