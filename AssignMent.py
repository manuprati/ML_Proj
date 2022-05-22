# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:30:24 2020

@author: User
"""
import pandas as pd
df=pd.read_csv('Tips.csv')
df.corr()

df.TotalBill.describe()
max=48.27
min=3.07
max-min
df.isnull().sum()

import numpy as np
x=np.matrix('4,5,16,7;2,-3,2,3;3,4,5,6;4,7,8,9')
np.linalg.matrix_rank(x)
a=np.matrix('1,0,-1,2;0,3,1,-1;2,4,0,3;-3,1,-1,2')
b=np.matrix('1,2;3,-1;0,-1;4,2')
c=np.matrix('3,8,0,5;1,0,-4,8')
d=np.dot(c,a)
D=np.dot(d,b)
np.linalg.det(z)
z=np.matrix('-2,32,24;92,66,25;-80,37,10')
pk=np.matrix('2,1,2;1,2,1;3,1,3')
np.linalg.inv(x)
np.linalg.matrix_rank(x)
A=np.matrix('2,0,0;0,1,0;0,0,3')
e_values,e_vector=np.linalg.eig(A)
e_values
e_vector
B=np.matrix('1,6,1;1,2,3;0,0,3')
np.linalg.eig(B)
E=np.matrix('-6,2;2,6')
np.linalg.eigvals(E)
k=np.random.randint(1,100,20)
q=5*k
np.corrcoef(k,q)
data=pd.read_csv('nyc.csv')
Food=data.Food.values.reshape(-1,1)
Price=data['Price'].values.reshape(-1,1)
from sklearn.linear_model import LinearRegression
lgr=LinearRegression()
model=lgr.fit(Food,Price)
model.intercept_
model.coef_
x=data.drop(['Price'],axis=1)
y=Price
model2=lgr.fit(x,y)
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
RMSE_train=np.sqrt(mean_squared_error(y,model2.predict(x)))
RMSE_train
r2=r2_score(model2.predict(x))
model2.summary()

base_pred=np.repeat(np.mean(y),len(y))
RMSE_base=np.sqrt(mean_squared_error(y,base_pred))
Diff=RMSE_base-RMSE_train
Diff
reduced_x=data.drop(['Price','Service'],axis=1)
model4=lgr.fit(reduced_x,y)
coeff_df=pd.concat([pd.DataFrame(reduced_x.columns),
pd.DataFrame(np.transpose(model4.coef_))],axis=1)
coeff_df
