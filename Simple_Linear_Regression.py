# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:21:24 2022

@author: MEHMET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("satislar.csv")
print(df)

aylar = df[["Aylar"]]
satislar = df[["Satislar"]]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

"""#makine öğrenmesi için öznitelik ölçeklendirme
from sklearn.preprocessing import StandardScaler

sc = StandardScaler() 

X_train = sc.fit_transform(x_train) 
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train) 
Y_test = sc.fit_transform(y_test)
"""

#Model İnşası (Linear Regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)
#x_train ile y_train'i öğrendi.x_test'i vererek y_test'i ne kadar doğru öğrenmiş öğrenebiliriz.


#Görselleştirme
x_train = x_train.sort_index()
y_train = y_train.sort_index()
print(plt.plot(x_train,y_train))
print(plt.plot(x_test,tahmin)) 
#iki grafik aynı anda run edilirse aralarındaki ilişkiye bakılabilir
plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")











