# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:19:15 2019

@author: mücahit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv("maaslar.csv")

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:3]
X=x.values
Y=y.values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)



from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)

x_poly=poly_reg.fit_transform(X)

print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

poly_reg3=PolynomialFeatures(degree=4)
x_poly3=poly_reg3.fit_transform(X)

lin_reg3=LinearRegression()
lin_reg3.fit(x_poly3,y)

#görselleştirme


plt.scatter(X,Y , color="red")
plt.plot(X,lin_reg.predict(X),COLOR="blue")
plt.show()

poly_reg=PolynomialFeatures(degree=4)

x_poly=poly_reg.fit_transform(X)

print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y , color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),COLOR="blue")
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color='blue')
"""
print(lin_reg.predict(11))
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))"""

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc1.fit_transform(Y)

from sklearn.svm import SVR

svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli , y_olcekli)

plt.scatter(x_olcekli, y_olcekli , color='red')
plt.plot(x_olcekli , svr_reg.predict(x_olcekli) , color='blue')

print(svr_reg.predict(11))
print(svr_reg.predict(6.6))
