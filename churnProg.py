# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:10:27 2020

@author: User
"""
import pandas as pd
import numpy as np
import seaborn as sns
data=pd.read_csv("churn.csv",index_col=0)
data.describe()
data[data.customerID.duplicated(keep=False)]
data=data.drop_duplicates()
data1
data.isna().sum()
data.info()
np.unique(data['SeniorCitizen'])
cat_data=data.select_dtypes(include=['object'])
data['customerID'].value_counts()
cat_data.columns
data['senior'].value_counts()
data['SeniorCitizen'].value_counts()
data['SeniorCitizen'].fillna(data['SeniorCitizen'].mode()[0], inplace=True)
data['TotalCharges'].mean()
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
data['MonthlyCharges'].fillna(data['MonthlyCharges'].mean(),inplace=True)
sns.boxplot(data['TotalCharges'],data['Churn'])
sns.distplot(data['TotalCharges'])
sns.regplot(data['TotalCharges'])
sns.barplot(data['TotalCharges'],data['Churn'])
# Logical fallacy
data['customerID'].value_counts()
import re
pattern='^[0-9]{4,4}-[A-Z]{5,5}'
p=re.compile(pattern)
type(p)
q=[i for i,value in enumerate(data['customerID']) if p.match(str(value))==None]
fp1= re.compile('^[A-Z]{5,5}-[0-9]{4,4}')
fp2= re.compile('^[0-9]{4,4}/[A-Z]{5,5}')

for i in q:
    false_str=str(data.customerID[i])
    if (fp1.match(false_str)):
        str_split= false_str.split('-')
        data.customerID[i]=str_split[1]+'-'+str_split[0]
    elif (fp2.match(false_str)):
        data.customerID[i]=false_str.replace('/' , '-')

y=data[(data.InternetService=='No')]        
z=y.iloc[:,13:20]

for i,row in z.iterrows():
    yes_cnt=row.str.count('Yes').sum()
    if yes_cnt>=2:
        z.loc[i].InternetService == 'Yes'
    else:
        z.loc[i,:] = 'No internet service'
churn=pd.merge(y,z, on InternetService)       
y.iloc[:,13:20]=z
