# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:23:33 2022

@author: MEHMET
"""

import numpy as np
import pandas as pd


#data preprocessing
df = pd.read_csv("veriler.csv")
print(df)

Yas = df.iloc[:,1:4].values

ulke = df.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(df.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


c = df.iloc[:,-1:].values
print(c)

le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(df.iloc[:,-1])
print(c)

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)


sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ["fr","tr","us"])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns= ["boy","kilo","yaş"])
print(sonuc2)

sonuc3 = pd.DataFrame(data=c[:,:1], index = range(22),columns=["cinsiyet"]) 
print(sonuc3)

s = pd.concat([sonuc,sonuc2],axis = 1)
print(s)

s2 = pd.concat([s,sonuc3],axis = 1)
print(s2)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler() 

X_train = sc.fit_transform(x_train) 
X_test = sc.fit_transform(x_test)


#build model (cinsiyet)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#testing
y_pred = regressor.predict(x_test)


#build model (boy)
boy = s2.iloc[:,3:4].values
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train,x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

#testing
y_pred = r2.predict(x_test)



#Backward Elimination yöntemi
import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int),values=veri, axis=1)
#çoklu coğrusal regresyon için Beta0 değerlerini eklediğimiz bir array oluşturduk.

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)

model =sm.OLS(boy,X_l).fit() 
print(model.summary())
#yapmak istediğimiz şey bu linear regression model'deki kolonların teker teker sonuca etkisini ölçmek.(boy üzerindeki etkisini)
#model summary'den x1,x2 gibi değişkenlerin P>|t| değerlerine bakılır.
#en yüksek p değerine sahip değişken 0.05'den büyükse değişken sistemden kaldırılıp model güncellenir.
#bu işlem 0.05'den büyük p değerine sahip değişken kalmayıncaya kadar devam edilir.




