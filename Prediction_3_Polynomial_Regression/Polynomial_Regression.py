# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:54:01 2022

@author: MEHMET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#veri yükleme
df = pd.read_csv("maaslar.csv")

x = df.iloc[:,1:2]
y = df.iloc[:,2:3]
X = x.values
Y = y.values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y, color = "red")
plt.plot(X,lin_reg.predict(X),color ="green")
plt.show()
#burdan grafiğe bakarak verilerimizin linear regression'a uygun olmadığını anlarız.


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)    #1.colum = x^0  2.column = x^1  3.column = x^2

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(X,Y, color = "red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = "green")
plt.show()


#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

