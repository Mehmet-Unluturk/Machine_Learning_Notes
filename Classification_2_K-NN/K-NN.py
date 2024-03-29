# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:49:29 2022

@author: MEHMET
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=(0))

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5,metric="minkowski")
#n_neighbors =  kullanılacak komşu sayısını belirlemeye yarar.
#Bu veri setinde n_neighbors değerini düşürmek daha doğru bir tahmin yapmamızı sağlar.Örn(1)

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

