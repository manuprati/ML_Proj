# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 11:53:00 2020

@author: User
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
data=pd.read_csv('Toyota.csv',na_values=(['??','????']))
# data.drop_duplicates(keep='first',inplace=True,)
# No duplicate reord
data.columns

data=data.drop(['Unnamed: 0'],axis=1)

data.info()

pd.set_option('max_columns',500)

data.describe(include='O')

data['MetColor'].fillna(data['MetColor'].median(),inplace=True)
data['KM'].fillna(data['KM'].median(),inplace=True)
data['Age'].fillna(data['Age'].mean(),inplace=True)
data['HP'].fillna(data['HP'].mean(),inplace=True)
data['FuelType'].fillna(data['FuelType'].mode()[0],inplace=True)
np.unique(data.KM)
np.unique(data.HP)
data.Doors.replace('five',5,inplace=True)
data.Doors.replace('four',4,inplace=True)
data.Doors.replace('three',3,inplace=True)
data.Doors=data.Doors.astype('int')

np.unique(data.MetColor)
np.unique(data.Doors)
np.unique(data.Age)
data.isnull().sum()

data=pd.get_dummies(data,drop_first=True)

features= list(set(data.columns)-set(['Price']))

target= list(['Price'])

x= data.loc[:,features]

y= data.loc[:,target]

#tr_x1=sm.add_constant(tr_x)

tr_x,tst_x,tr_y,tst_y=train_test_split(x,y,test_size=.3,random_state=40)

model= sm.OLS(tr_y, tr_x).fit()

model.summary()

predictions=model.predict(tst_x)

base_pred=np.mean(tst_y)
print(base_pred)

base_pred=np.repeat(base_pred,len(tst_y))

base_rmse=(mean_squared_error(tst_y,base_pred))**0.5

base_rmse

#lr_rmse=np.sqrt(mean_squared_error(tst_y, predictions))

lr_rmse=(mean_squared_error(tst_y, predictions))**0.5

lr_rmse

#PCR

def linreg(X_tr,X_tst,y_tr,y_tst):
    ols=LinearRegression()
    ols.fit(X_tr,y_tr)
    y_pred=ols.predict(X_tst)
    rmse=np.sqrt(((y_tst-y_pred)**2).mean())
    return rmse

from sklearn.decomposition import PCA

def pca_kfold_ols(X,y,nfact):
    Rmse_pcr_lst=list()
    for pc in range(1,nfact,1):
        pca=PCA(n_components=pc)
        X_red = pca.fit_transform(X)
        rmse_pcr_lst=list()
        k= model_selection.KFold(5)
        for tr_idx,tst_idx in k.split(X_red):
            pc_tr,pc_tst=X_red[tr_idx],X_red[tst_idx]
            y_tr,y_tst=y[tr_idx],y[tst_idx]
            rmse_pcr_lst.append(linreg(pc_tr,pc_tst,y_tr,y_tst))
        Rmse_pcr_lst.append(np.array(rmse_pcr_lst).mean())
    return np.array(Rmse_pcr_lst)

pcs=10
tr_x = tr_x.to_numpy()
tr_y = tr_y.to_numpy()
Rmse_pcr=np.append(base_rmse,pca_kfold_ols(tr_x,tr_y,pcs))
Rmse_pcr.shape

plt.plot(range(pcs),Rmse_pcr)
plt.xlabel('No. of PCs')
plt.ylabel('RMSE')
Rmse_pcr[5]
