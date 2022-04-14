# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 00:10:43 2022

@author: MEHMET
"""

import numpy as np
import pandas as pd

df = pd.read_csv("odev_tenis.csv")


outlook = df.iloc[:,0:1].values
print(outlook)

play = df.iloc[:,-1:].values
print(play)

df["windy"] = df["windy"].astype(int)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
outlook[:,0] = le.fit_transform(df.iloc[:,0])
print(outlook)

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
play[:,0] = le.fit_transform(df.iloc[:,-1])
print(play)


sonuc = pd.DataFrame(data=outlook, index = range(14), columns = ["overcast","rainy","sunny"])
print(sonuc)

sonuc2 = df.iloc[:,1:2]
print(sonuc2)

humidity = df.iloc[:,2:3]
sonuc5 = df.iloc[:,3:4]

sonuc3 = pd.DataFrame(data=play, index = range(14), columns = ["play"])
print(sonuc3)

df1 = pd.concat([sonuc,sonuc2],axis=1)
df2 = pd.concat([df1,sonuc5],axis=1)
dfson = pd.concat([df2,sonuc3],axis=1)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(dfson,humidity,test_size=0.33, random_state=0)


#build model (humidity)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#testing
y_pred = regressor.predict(x_test)


import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int),values=dfson, axis=1)
#çoklu coğrusal regresyon için Beta0 değerlerini eklediğimiz bir array oluşturduk.

X_l = dfson.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)

model =sm.OLS(humidity,X_l).fit() 
print(model.summary())


#en yüksek p value si olan column u çıkartıyoruz.
x_train = x_train.drop("windy",axis=1)
x_test = x_test.drop("windy",axis=1)


#build new model (humidity)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#testing
y_pred = regressor.predict(x_test)   #tahmin doğruluğu arttı.








