# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 01:40:07 2020

@author: User
"""
#%%
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
# data.drop_duplicates(keep='first',inplace=True)
# No duplicate reord
data.columns

#data=data.drop(['train'],axis=1)
#%%
data.info()

pd.set_option('max_columns',500)

data.describe()

correlation=data.corr()

data['lpsa'].value_counts().sort_index
sns.distplot(data['lpsa'])
sns.boxplot(data['lpsa'])
pd.crosstab(index=data['lpsa'],columns=data['svi'], margins=True,normalize=True )
sns.regplot(x='lcavol', y='lpsa', scatter=True, 
            fit_reg=False, data=data)

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(data=data,y='lpsa', hue='svi')
sns.boxplot(data=data,x='lcavol',y='lpsa', hue='svi')
pd.crosstab(columns='lcavol',index=data['svi'], margins=True, normalize=True )

plt.subplots(figsize=(12,9))

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris_data)

data['lcavol'].value_counts()
sns.distplot(data['lcavol'],kde=False)
sns.boxplot(data['lcavol'])
sns.regplot(x='lcavol', y='lpsa', scatter=True, 
            fit_reg=False, data=data)

data['lcp'].value_counts().sort_index
sns.distplot(data['lcp'],kde=False)
sns.boxplot(data['lcp'])
sns.regplot(x='lcp', y='lpsa', scatter=True, 
            fit_reg=False, data=data)

data['svi'].value_counts()
sns.distplot(data['svi'], kde=False)
sns.boxplot(data['svi'])
pd.crosstab(columns='count',index=data['svi'], margins=True, )
sns.regplot(x='svi', y='lpsa', scatter=True, 
            fit_reg=False, data=data)

sns.boxplot(data=data,y='lpsa', x='lcp')
pd.crosstab(columns=data['lpsa'], index=data['lcavol'], margins=True, normalize='index')
#sns.countplot(data=data,x='lpsa', hue='lcavol')
sns.regplot(y='lcavol', x='lcp', data=data)

sns.scatterplot(y='lcavol', x='lcp', hue='svi', data=data)
sns.scatterplot(y='lcavol', x='lpsa', hue='svi', data=data)
sns.scatterplot(y='lpsa', x='lcp', hue='svi', data=data)

data['age'].value_counts()
sns.boxplot(data=data,y='lpsa', x='gleason')
pd.crosstab(index=data['lpsa'], columns=data['gleason'], normalize='index')
sns.distplot(data['age'], kde=False, bins=5)
sns.regplot(x='age', y='lpsa', scatter=True, 
            fit_reg=False, data=data)

sns.boxplot(data['lpsa'])
pd.crosstab(columns='count',index=data['lpsa'], margins=True, )

data.columns
data['lweight'].value_counts().sort_index
sns.boxplot(data=data,y='lpsa', x='lweight')
pd.crosstab(index=data['lpsa'], columns=data['lweight'])
sns.regplot(x='lweight', y='lpsa', scatter=True, 
            fit_reg=False, data=data)

sns.boxplot(data=data,y='lpsa', x='lbph')
pd.crosstab(index=data['lpsa'], columns=data['lbph'], normalize='index')
sns.regplot(x='lbph', y='lpsa', scatter=True, 
            fit_reg=False, data=data)

sns.boxplot(data=data,y='lpsa', x='gleason')
pd.crosstab(index=data['lpsa'], columns=data['gleason'], normalize='index')
sns.regplot(x='gleason', y='lpsa', scatter=True, 
            fit_reg=False, data=data)

sns.boxplot(data=data,y='lpsa', x='pgg45')
pd.crosstab(index=data['lpsa'], columns=data['pgg45'], normalize='index')
sns.regplot(x='pgg45', y='lpsa', scatter=True, 
            fit_reg=False, data=data)

x1 = data.filter(['lcavol','lcp','svi'],axis=1)
y1= data.filter(['lpsa'],axis=1)

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

x1 = data.drop(['lpsa','lweight','age','lbph','train'], axis='columns', inplace=False)
x1=pd.get_dummies(x1,drop_first=True) 
X_train, X_test = train_test_split(x1,test_size=0.3, random_state = 3)

#%%
"""
Linear regression for full model
"""

X_train2 = sm.add_constant(X_train)
model_lin3 = sm.OLS(y_train, X_train2)
results3=model_lin3.fit()
print(results3.summary())

# Predicting model on test set 
X_test=sm.add_constant(X_test)
cars_predictions_lin3_test = results3.predict(X_test)

#%%

# Model evaluation on predicted and test 
rmse(y_test,cars_predictions_lin3_test)
    
# For diagnostics, we need to predict the model on the train data
cars_predictions_lin3_train = results3.predict(X_train2)

#%%
"""
Diagnostics
"""

residuals=y_train.iloc[:,0]-cars_predictions_lin3_train

# Residual plot
sns.regplot(x=cars_predictions_lin3_train,y=residuals)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot')

# QQ plot
sm.qqplot(residuals)
plt.title("Normal Q-Q Plot")

#%%
x1 = cars_omit_data.drop(['price','model','brand'], axis='columns', inplace=False)
x1=pd.get_dummies(x1,drop_first=True) 
x1 = x1.drop(['fuelType_electro','fuelType_hybrid'],axis=1)
X_train, X_test = train_test_split(x1,test_size=0.3, random_state = 3)
x1.columns
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
