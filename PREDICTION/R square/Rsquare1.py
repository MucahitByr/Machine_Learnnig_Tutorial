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

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

print("Linear R2  degeri:")
print(r2_score(Y,lin_reg.predict(X)))
#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()
print("Polynomial R2 degeri:")
print(r2_score(Y,lin_reg2 .predict(poly_reg.fit_transform(X))))
#tahminler
"""
print(lin_reg.predict(11))
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))
"""
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()
print("SVR   R2 degeri:")
print(r2_score(y_olcekli,rf_reg.predict(x_olcekli )))
"""
print(svr_reg.predict(11))
print(svr_reg.predict(6.6))
"""
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
K=X+0.5
Z=X-0.5

plt.scatter(X,Y,color='red')
plt.plot(x,r_dt.predict(X),color='blue')
plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='orange')
plt.show()
print("Decision Tree R2 degeri:")
print(r2_score(Y,r_dt.predict(X)))

from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(n_estimators=10 , random_state=0) #n_estimator komutu kaç adet ağac olucağıyla alakalı komuttur
rf_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,rf_reg.predict(X),color='blue')
plt.plot(x,rf_reg.predict(K),color='green')
plt.plot(x,rf_reg.predict(Z),color='orange')
plt.show()

from sklearn.metrics import r2_score

print("Random Forest R2 degeri:")

print(r2_score(Y,rf_reg.predict(X)))
print(r2_score(Y,rf_reg.predict(K)))
print(r2_score(Y,rf_reg.predict(Z)))
print("-------------")
print("OZET:")

print("Linear R2  degeri:")
print(r2_score(Y,lin_reg.predict(X)))

print("Polynomial R2 degeri:")
print(r2_score(Y,lin_reg2 .predict(poly_reg.fit_transform(X))))

print("SVR   R2 degeri:")
print(r2_score(y_olcekli,rf_reg.predict(x_olcekli )))

print("Decision Tree R2 degeri:")
print(r2_score(Y,r_dt.predict(X)))




    
