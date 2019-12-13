# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:56:46 2019

@author: mücahit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv('musteriler.csv')

X=veriler.iloc[:,3:].values # this is three D sample.


from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=3 , init='k-means++',random_state=123)

kmeans.fit(X)

print(kmeans.cluster_centers_)

sonuclar=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i , init='k-means++' , random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
plt.plot(range(1,11),sonuclar)
    


