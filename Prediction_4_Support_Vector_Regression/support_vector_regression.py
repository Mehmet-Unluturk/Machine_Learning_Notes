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


#svr kullanmak için verileri scale etmemiz gerekiyor.
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler() 
x_scale = sc1.fit_transform(X) 

sc2 = StandardScaler()
y_scale = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_scale,y_scale)

plt.scatter(x_scale,y_scale,color = "red")
plt.plot(x_scale,svr_reg.predict(x_scale),color = "green")

#tahmin
print(svr_reg.predict([[7.5]]))
print(svr_reg.predict([[11]]))
