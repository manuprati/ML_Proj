# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:57:38 2020

@author: User
"""
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


#import os
#os.chdir("d:/pandas")

#%%2
pd.set_option('max_column',100)
data=pd.read_csv('Train.csv')
# data0=pd.read_csv('Test.csv')
# data0.info()
# data0.insert(8,'time_spent',' ')
data.info()

#%%
# =============================================================================
# Cleaning Data
# =============================================================================

#dropping the duplicate records

data2=data.copy()
sns.set(rc={'figure.figsize':(11.7,8.27)})
# data2.drop_duplicates(keep='first',inplace=True)
# No duplicate records found
missing = data[data.isnull().any(axis=1)]
sns.boxplot(missing.purchased)
missing.purchased.value_counts()
data['time_spent'].value_counts()
data=data.dropna(axis=0)

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

# Let us impute those missing values using mean based on the output
# varieble 'Churn' – Yes & No

data.groupby(['client_agent']).mean().groupby('client_agent')['purchased'].mean()

data['client_agent'] = data.groupby('client_agent')['time_spent']\
.transform(lambda x: x.fillna(x.mean()))

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
data.describe(include=["object"])
data.describe()
# Loan amount is highly dependent on lender count

#%%



y1= data.filter(['time_spent'],axis=1)

x1 = data.drop(['session_id','time_spent','device_details','date'], axis='columns', inplace=False)
x1=pd.get_dummies(x1,drop_first=True) 
X_train, X_test = train_test_split(x1,test_size=0.3, random_state = 3)

#%%
"""
Linear regression for full model
"""

y2=np.log(y1)

#%%
# Splitting data into test and train
X_train, X_test, y_train_log,y_test_log = train_test_split(x1, y2, test_size=0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train_log.shape, y_test_log.shape)

#%%
def rmse_log(test_y,predicted_y):
    t1=np.exp(test_y)
    t2=np.exp(predicted_y)
    rmse_test=np.sqrt(mean_squared_error(t1,t2))
    #for base rmse
    base_pred = np.repeat(np.mean(t1), len(t1))
    rmse_base = np.sqrt(mean_squared_error(t1, base_pred))
    values={'RMSE-test from model':rmse_test,'Base RMSE':rmse_base}
    return values


#%%
rf = RandomForestRegressor(n_estimators = 220,max_depth=87)
#rf = RandomForestRegressor(max_depth=10)

# Model
model_rf1=rf.fit(X_train,y_train_log)

# Predicting model on test and train set
cars_predictions_rf1_test = rf.predict(X_test)

# RMSE
rmse_log(y_test_log,cars_predictions_rf1_test)

# Rsquared
model_rf1.score(X_train,y_train_log)
#%%

"""
Hyperparameter Tuning
"""

## Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 600,num = 15)]
print(n_estimators)

## Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]


## Minimum number of samples required to split a node
min_samples_split = np.arange(100,1100,100)

## Create the random grid
random_grid1 = {'n_estimators': n_estimators}
random_grid2 = {'max_depth': max_depth}
random_grid3 = {'min_samples_split': min_samples_split}

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}

print(random_grid)

## Use the random grid to search for best hyperparameters

## First create the base model to tune
rf_for_tuning = RandomForestRegressor()

## Random search of parameters, using 3 fold cross validation, 
## search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf_for_tuning, 
                            param_distributions = random_grid, 
                            n_iter = 100,cv = 3, verbose=2, random_state=1)


## Fit the random search model
rf_random.fit(X_train,y_train_log)
print(rf_random.best_params_)

## finding the best model
rf_model_best = rf_random.best_estimator_
print(rf_model_best)

# predicting with the test data on best model
predictions_best = rf_model_best.predict(X_test)
predictions_best_train=rf_model_best.predict(X_train)

# =============================================================================
# END OF SCRIPT
# =============================================================================
