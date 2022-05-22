# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:09:53 2020

@author: HP
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
import os
os.chdir("d:\pandas")
#%%

dataa=sns.load_dataset('anscombe')
x1 = dataa.filter(['x'],axis=1)
x1 =x1[22:33]
y1= dataa.filter(['y'],axis=1)
y1=y1[22:33]
# dataa=pd.read_csv('nyc.csv')
# x1 = dataa.filter(['Food'],axis=1)
# y1= dataa.filter(['Price'],axis=1)
#%%
# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=3)
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
###############################

practicee
a=np.array([1,2,3,4])
b=pd.DataFrame(a)
c=b[(b>2) & (b<4]
c
