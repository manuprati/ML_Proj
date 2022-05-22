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



x1 = data.filter(['session_number','purchased','added_in_cart','checked_out'],axis=1)
y1= data.filter(['time_spent'],axis=1)

#%%
# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#%%
    
# Measure for multicollinearity
def calculateVIF(data):
    features = list(data.columns)
    num_features = len(features)

    model = LinearRegression()

    result = pd.DataFrame(index = ['VIF'], columns = features)
    result = result.fillna(0)

    for ite in range(num_features):
        x_features = features[:]
        y_featue = features[ite]
        x_features.remove(y_featue)     
        model.fit(data[x_features], data[y_featue])
        result[y_featue] = 1/(1 - model.score(data[x_features], data[y_featue]))

    return result
#%%
"""
Model evaluation
"""

def rmse(test_y,predicted_y):
    rmse_test=np.sqrt(mean_squared_error(test_y, predicted_y))
    #for base rmse
    base_pred = np.repeat(np.mean(test_y), len(test_y))
    rmse_base = np.sqrt(mean_squared_error((test_y), base_pred))
    values={'RMSE-test from model':rmse_test,'Base RMSE':rmse_base}
    return values



#%%
"""
Linear regression model building starts here
"""
X_train2 = sm.add_constant(X_train)
model_lin1 = sm.OLS(y_train, X_train2)
results1=model_lin1.fit()
print(results1.summary())

# Predicting model on test set 
X_test=sm.add_constant(X_test)
cars_predictions_lin1_test = results1.predict(X_test)

#%%
vif_val=calculateVIF(X_train)
vif_val=vif_val.transpose()

#%%
# Model evaluation on predicted and test 
rmse(y_test,cars_predictions_lin1_test)

# For diagnostics, we need to predict the model on the train data
cars_predictions_lin1_train = results1.predict(X_train2)

#%%
"""
Diagnostics
"""

residuals=y_train.iloc[:,0]-cars_predictions_lin1_train

# Residual plot
sns.regplot(x=cars_predictions_lin1_train,y=residuals)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot')

# QQ plot
sm.qqplot(residuals)
plt.title("Normal Q-Q Plot")



#%%
# Plotting the variable price
prices = pd.DataFrame({"1. Before":y1.iloc[:,0], "2. After":np.log(y1.iloc[:,0])})
prices.hist()

# Plotting the variable price
prices = pd.DataFrame({"1. Before":y_train.iloc[:,0], "2. After":np.log(y_train.iloc[:,0])})
prices.hist()

# Plotting the variable price
prices = pd.DataFrame({"1. Before":y_test.iloc[:,0], "2. After":np.log(y_test.iloc[:,0])})
prices.hist()

#%%

# Transforming prices to log
y2=np.log(y1)
y_train_log,y_test_log=train_test_split(y2, test_size=0.3, random_state = 3)


#%%

"""
Linear Regression
Model- log(price)~powerPS+kilometer+Age
"""

X_train2 = sm.add_constant(X_train)
model_lin2 = sm.OLS(y_train_log, X_train2)
results2=model_lin2.fit()
print(results2.summary())

# Predicting model on test set 
X_test=sm.add_constant(X_test)
cars_predictions_lin2_test = results2.predict(X_test)

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

# Model evaluation on predicted and test 
rmse_log(y_test_log,cars_predictions_lin2_test)

# For diagnostics, we need to predict the model on the train data
cars_predictions_lin2_train = results2.predict(X_train2)

#%%
"""
Diagnostics
"""

residuals=y_train_log.iloc[:,0]-cars_predictions_lin2_train

# Residual plot
sns.regplot(x=cars_predictions_lin2_train,y=residuals)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot')

# QQ plot
sm.qqplot(residuals)
plt.title("Normal Q-Q Plot")


#%%

"""
Full model with log(price)~powerPS+kilometer+Age+vehicletype+fuelType+gearbox+notRepairedDamage
"""

x1 = data.drop(['session_id','time_spent','device_details','date'], axis='columns', inplace=False)
x1=pd.get_dummies(x1,drop_first=True) 
X_train, X_test = train_test_split(x1,test_size=0.3, random_state = 3)

#%%
"""
Linear regression for full model
"""

X_train2 = sm.add_constant(X_train)
model_lin3 = sm.OLS(y_train_log, X_train2)
results3=model_lin3.fit()
print(results3.summary())

# Predicting model on test set 
X_test=sm.add_constant(X_test)
cars_predictions_lin3_test = results3.predict(X_test)

#%%

# Model evaluation on predicted and test 
rmse_log(y_test_log,cars_predictions_lin3_test)
    
# For diagnostics, we need to predict the model on the train data
cars_predictions_lin3_train = results3.predict(X_train2)

#%%
"""
Diagnostics
"""

residuals=y_train_log.iloc[:,0]-cars_predictions_lin3_train

# Residual plot
sns.regplot(x=cars_predictions_lin3_train,y=residuals)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot')

# QQ plot
sm.qqplot(residuals)
plt.title("Normal Q-Q Plot")


#%%
# =============================================================================
# END OF SCRIPT
# =============================================================================
