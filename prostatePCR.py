# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 01:40:07 2020

@author: User
"""

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
data=pd.read_table('prostate.txt',index_col=0,delimiter="\t")
#data.drop_duplicates(keep='first',inplace=True)
# No duplicate reord
data.columns

data=data.drop(['train','age','lbph','lcp'],axis=1)

data.info()

pd.set_option('max_columns',500)

data.describe()

correlation=data.corr()

data['lpsa'].value_counts()
sns.distplot(data['lpsa'])
sns.boxplot(data['lpsa'])
pd.crosstab(columns='count',index=data['lpsa'] )
pd.crosstab(columns=['lbph'],index=data['lpsa'], normalize=True, )

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(data=data,y='lpsa',hue='svi')
plt.subplots(figsize=(8,6))

sns.scatterplot(x='lcavol', y='lpsa', hue='svi', data=data)

data['lcavol'].value_counts()
sns.distplot(data['lcavol'],kde=False)
sns.boxplot(data['lcavol'])

data['lcp'].value_counts()
sns.distplot(data['lcp'],kde=False)
sns.boxplot(data['lcp'])

sns.boxplot(data=data,y='lpsa', x='lcp')
pd.crosstab(columns=data['lpsa'], index=data['lcavol'], margins=True, normalize='index')
#sns.countplot(data=data,x='lpsa', hue='lcavol')
sns.regplot(y='lcavol', x='lcp', data=data)
sns.scatterplot(y='lcavol', x='lcp', hue='lpsa', data=data)
#%%
#This Cell is not relevant
input_columns = list(data.iloc[:,:].columns)
input_columns.sort()
input_data=data[input_columns]
input_data.head()
input_dataa=data.copy()
#input_dataa=input_dataa.drop(['Species'],axis=1)
plt.subplots(figsize=(8,6))
sns.scatterplot(x='lcavol', y='lcp', hue='lpsa', data=input_data)
plt.show()

np.unique(data.lpsa)
#%%

features= list(set(data.columns)-set(['lpsa']))

target= list(['lpsa'])

x= data.loc[:,features]

y= data.loc[:,target]

#tr_x1=sm.add_constant(tr_x)

tr_x,tst_x,tr_y,tst_y=train_test_split(x,y,test_size=.3)

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

pcs=6
tr_x = tr_x.to_numpy()
tr_y = tr_y.to_numpy()
Rmse_pcr=np.append(base_rmse,pca_kfold_ols(tr_x,tr_y,pcs))
Rmse_pcr.shape

plt.plot(range(pcs),Rmse_pcr)
plt.xlabel('No. of PCs')
plt.ylabel('RMSE')
Rmse_pcr[5]
