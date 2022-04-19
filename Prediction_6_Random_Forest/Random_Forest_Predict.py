# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:38:47 2022

@author: MEHMET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("maaslar.csv")

x = df.iloc[:,1:2]
y = df.iloc[:,2:3]
X = x.values
Y = y.values

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
#n_estimator kaç tane decision tree çizileceğini belirler

rf_reg.fit(X,Y.ravel())


plt.scatter(X,Y,color = "red")
plt.plot(X,rf_reg.predict(X),color = "green")


print(rf_reg.predict([[6.6]]))

