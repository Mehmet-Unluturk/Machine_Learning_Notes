# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 00:38:47 2022

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


from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)

r_dt.fit(X,Y)

plt.scatter(X,Y,color = "red")
plt.plot(X,r_dt.predict(X),color = "green")

#tahmin
print(r_dt.predict([[6.6]]))
print(r_dt.predict([[10.5]]))