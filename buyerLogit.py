# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:57:38 2020

@author: User
"""
#%%
# =============================================================================
# CLASSIFYING PERSONAL INCOME 
# =============================================================================
################################# Required packages ############################
# To work with dataframes
import pandas as pd 

# To perform numerical operations
import numpy as np

# To visualize data
import seaborn as sns

# To partition the data
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

#%%
###############################################################################
# =============================================================================
# Importing data
# =============================================================================

#%%2
pd.set_option('max_column',100)
data=pd.read_csv('Train.csv')
# data0=pd.read_csv('Test.csv')
# data0.info()
# data0.insert(8,'time_spent',' ')
data.info()

data2=data.copy()
data2=data2.dropna(axis=0)

sns.set(rc={'figure.figsize':(11.7,8.27)})

#%%
# =============================================================================
# Cleaning Data
# =============================================================================

#dropping the duplicate records

# data2.drop_duplicates(keep='first',inplace=True)
# No duplicate records found
missing = data[data.isnull().any(axis=1)]
sns.boxplot(missing.purchased)
missing.purchased.value_counts()
data['time_spent'].value_counts()

sns.countplot(data2['time_spent'])
print(np.unique(data2['time_spent']))
sns.distplot(data['time_spent'],kde=False)
sns.boxplot(data['time_spent'])
data['time_spent'].describe()
data['time_spent'] = np.where(data['time_spent']>=25000,
      data['time_spent'].median(), data['time_spent'])

#%%

#Filling Missing values

data2.isnull().sum()
# 160 values are missing in client_agent which can be filled by modal value i.e. female
data2['client_agent'].value_counts()
data2['client_agent'].describe(include='O')
sns.boxplot(y=data['client_agent'],x=data['purchased'])
sns.regplot(y=data['client_agent'],x=data['session_number'],scatter=True, fit_reg=False)
sns.pairplot(data)
data['client_agent'].mode()

sns.boxplot(x = data['client_agent'], y = data['session_number'])


data.client_agent.isnull().sum()



data2['session_number'].value_counts().sort_index()
print(np.unique(data2['session_number']))
pd.crosstab(index=data2['session_number'], columns=data2['time_spent'], margins=True, normalize='index')
sns.regplot(data=data2,x='session_number', y='time_spent')
sns.distplot(data2['session_number'],kde=False)
data2['session_number'].describe()

sns.boxplot(data2['session_number'])
sum(data['session_number'] > 3000)
sum(data['session_number'] < 10)
# working range 10-5000
sns.regplot(x='session_number', y='time_spent', scatter=True, 
            fit_reg=False, data=data2)
#time_spent is decreasing with increasing session_number except some Exceptions

data2['session_id'].value_counts()
print(np.unique(data2['session_id']))
data2['session_id']
len_ind = [i for i,value in enumerate(data0.session_id) if len(value)!=32]
# All the session_id are of same length in test and train
data2.session_id.equals(data0.session_id)


data2['device_details'].value_counts()
sns.countplot(data2['lender_count'])
print(np.unique(data2['device_details']))

data2['date'].value_counts()
print(np.unique(data2['date']))
pd.crosstab(columns=data2['date'], index=data2['time_spent'], margins=True, normalize='index')
sns.countplot(data=data2,hue='date', y='time_spent')

data2['purchased'].value_counts()
sns.countplot(data2['purchased'])
pd.crosstab(columns=data2['purchased'], index=data2['time_spent'], margins=True, normalize='index')
sns.boxplot(data=data,x='purchased', y='time_spent')
print(np.unique(data2['purchased']))

data2['added_in_cart'].value_counts()
sns.countplot(data2['added_in_cart'])
pd.crosstab(columns=data2['added_in_cart'], index=data2['time_spent'], margins=True, normalize='index')
sns.boxplot(data=data2,x='added_in_cart', y='time_spent')

data2['checked_out'].value_counts()
sns.countplot(data2['checked_out'])
pd.crosstab(columns=data2['checked_out'], index=data2['time_spent'], margins=True, normalize='index')
sns.boxplot(data=data2,x='checked_out', y='time_spent')


#%% 

#%%
# =============================================================================
# Correlation
# =============================================================================

#Correlation - Realtionship between independent variables

data_select=data2.select_dtypes(exclude=["object"])
correlation=data_select.corr()
data2.describe(include=["object"])
data.describe()
# Loan amount is highly dependent on lender count

#%%
# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================

# Reindexing the salary status names to 0,1

data2 = data2.drop(['session_id','date'],axis=1)
new_data=pd.get_dummies(data2, drop_first=True)

# Storing the column names 
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['time_spent']))
print(features)

# Storing the output values in y
y=new_data['time_spent'].values
print(y)

# Storing the values from input features
x = new_data[features].values
print(x)

tr_x,tst_x,tr_y,tst_y=train_test_split(x,y,test_size=.3,random_state=40)

import statsmodels.api as sm
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

import matplotlib.pyplot as plt

plt.plot(range(pcs),Rmse_pcr)
plt.xlabel('No. of PCs')
plt.ylabel('RMSE')
Rmse_pcr[3]


x1 = data.filter(['purchased','added_in_cart','checked_out'],axis=1)
y1= data.filter(['time_spent'],axis=1)

#%%
# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



# =============================================================================
# END OF SCRIPT
# =============================================================================
