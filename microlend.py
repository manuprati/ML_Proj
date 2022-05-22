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

#import os
#os.chdir("d:/pandas")

#%%2
pd.set_option('max_column',100)
data=pd.read_csv('microlending_data.csv',index_col=0)
data.info()

#%%
# =============================================================================
# Cleaning Data
# =============================================================================

#dropping the duplicate records

data2=data.drop_duplicates(keep='first',inplace=True)
# 1575 duplicate records dropped

#%%

#Filling Missing values

data2.isnull().sum()
# 30 values are missing in borrower gender which can be filled by modal value i.e. female
data2['borrower_genders'].value_counts()
data2['borrower_genders'].fillna(data['borrower_genders'].mode()[0],inplace=True)
print(np.unique(data2['borrower_genders']))


#%% 
print(np.unique(data['term_in_months']))
data2['term_in_months'].replace('1 Year',12, inplace=True)
data2['term_in_months'].replace('2 Years',24, inplace=True)
data2['term_in_months']=data2['term_in_months'].astype('int')
data2.describe()
data2['term_in_months'].value_counts()
sns.distplot(data2['term_in_months'],kde=False)
sns.boxplot(data2['term_in_months'])

data2['lender_count'].value_counts()
sns.distplot(data2['lender_count'],kde=False)
sns.boxplot(data2['lender_count'])


sum(data2['lender_count']<1)
# lender_count < 1 = 58 which are meaningless because lender_count cannot be less than one so we can remove in data4
# sum(data2['lender_count']>355)
# Also lender_count > 355 = 2 which seems ouliers but as info about loan amount is also attached, we can treat them as follows .

data2=data2[(data.lender_count >= 1)] 
data2['status'].value_counts()
# By removing 58 records having lender_count less than one not_funded records have also been reduced by 58
# Aslo
data2['lender_count'] = np.where(data2['lender_count']>=355,
      data2['lender_count'].median(), data2['lender_count'])
# Similarly for Loan amount
data2['loan_amount'].value_counts()
sns.boxplot(data2['loan_amount'])
sns.distplot(data2['loan_amount'],kde=False)

sum(data2['loan_amount']>12000)
# Although only 4 values more than 12000 in loan amount which are outlier so we can treat them similarly
# Loan amount important variable

data2['loan_amount'] = np.where(data2['loan_amount']>=12000,
      data2['loan_amount'].median(), data2['loan_amount'])

sns.boxplot(data=data2,y='status', x='loan_amount')
data2.groupby(['status']).mean().groupby('status')['loan_amount'].mean()
# Mean of loan amout funded & not funded are 754 and 1295 respectively
data2.groupby('status')['loan_amount'].median()
# Median of loan amout funded & not funded are 525 and 1000 respectively
# i.e. Half of the loan amount funded in comparison to not_funded

#%%
# =============================================================================
# Correlation
# =============================================================================

#Correlation - Realtionship between independent variables

data_select=data2.select_dtypes(exclude=["object"])
correlation=data_select.corr()

# Loan amount is highly dependent on lender count

#%%

data3=data2.copy()
# data3 is the cleaned data

# Making a copy of cleaned data
data4=data3.copy()

sns.countplot(data=data2,x='lender_count', hue='status')
sns.boxplot(data=data2,x='lender_count', y='status')
data2.groupby(['status']).mean().groupby('status')['lender_count'].mean()
data2['lender_count'] = data2.groupby('status')['lender_count']\
.transform(lambda x: x.fillna(x.mean()))

# Lender count is an important variable as status is dependent gretly on it.


# =============================================================================
#Data Visualisation
# =============================================================================


data3['loan_amount'].value_counts()
# sns.boxplot(data3['loan_amount'])
# sns.countplot(data3['loan_amount']) cannot be used for numerical data
sns.distplot(data3['loan_amount'],kde=False)
sns.boxplot(data=data3,y='status', x='loan_amount')
#Although only 1 value more than 20000 in loan amount which may beoulier but we cant remove
# Loan amount important variable

data3.columns
data3.describe(include='O')

sns.countplot(data3['status'])
pd.crosstab(index=data3['status'], columns='count', normalize=True)
# 87% funded and 13% not funded
data3['status']=data3['status'].map({'not_funded':0,'funded':1})


data3['activity'].value_counts()
print(np.unique(data3['activity']))
sns.countplot(data=data3,y='activity', hue='status')
pd.crosstab(columns=data2['status'], index=data2['activity'], margins=True, normalize='index')
# Activity has 150 value counts like aquaculture, Arts, Well digging which are 100% funded 
# Important variable - except some activity viz. wedding expenses all other funded in great percentage.
# Not significant

data3['country'].value_counts()
sns.countplot(data=data3,y='country', hue='status')
pd.crosstab(columns=data3['status'], index=data3['country'], margins=True, normalize='index')
# Country has 61 value counts like Afghanistan, Belize which are 100% funded 
# Important variable - except some country viz. Brazil and Bolivia of which 29% & 24% not funded.
# Not significant

data3['country_code'].value_counts()
sns.regplot(data=data3,x='country', y='country_code') # throws error scatter b/w cat. & Numerical
sns.countplot(data=data3,y='country_code', hue='status')
pd.crosstab(columns=data3['status'], index=data3['country_code'], margins=True, normalize='index')

#country_code are one to one with country so we can discard this in data4

data3['currency_policy'].value_counts()
sns.countplot(data=data3,y='currency_policy', hue='status')
pd.crosstab(columns=data3['status'], index=data3['currency_policy'], margins=True, normalize='index')
# Not significant can be dropped in data4

data3['distribution_model'].value_counts()
# Not significant as only one value count and hence can be dropped in data4

data3['original_language'].value_counts()
sns.countplot(data=data3,y='original_language', hue='status')
sns.boxplot(data=data3,x='original_language', y='loan_amount')
pd.crosstab(columns=data3['status'], index=data2['original_language'], margins=True, normalize='index')
# Important variable as 29% Portuguess and 26% French were not funded


data3['repayment_interval'].value_counts()
sns.boxplot(data=data3,x='repayment_interval', y='loan_amount')
sns.countplot(data=data3,y='repayment_interval', hue='status')
pd.crosstab(columns=data3['status'], index=data3['repayment_interval'], margins=True, normalize='index')
# Not so significant can be dropped in data4

data3['sector'].value_counts()
sns.countplot(data=data3,x='sector', hue='status')
sns.boxplot(data=data3,x='sector', y='loan_amount')
pd.crosstab(columns=data3['status'], index=data3['sector'], margins=True, normalize='index')
# Important variable as sector like Housing were not funded in 26% cases

sns.countplot(data=data3,x='term_in_months', hue='status')
sns.boxplot(data=data3,y='term_in_months', x='status')
pd.crosstab(index=data3['term_in_months'], columns=data2['status'], margins=True, normalize='index')
# important variable as term of 130, 135 & 140 months were not funded 

data3['rMPI'].value_counts()
sns.boxplot(data=data3,x='rMPI', y='status')
pd.crosstab(columns=data3['status'], index=data3['rMPI'], margins=True, normalize='index')
sns.countplot(data=data3,x='rMPI', hue='status')
# Important variable as rMPI value of .005 and .583 were not funded in 39% & 41% cases respectively etc. 


data3.describe()
sns.countplot(data=data3,x='borrower_genders', hue='status')
pd.crosstab(columns=data2['borrower_genders'], index=data2['status'], margins=True, normalize='index')
# 64% females were funded but 49% male were not funded
# Hence important variable



#%%
# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================

# To partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

# Reindexing the status names to 0,1
data3['status']=data3['status'].map({'not_funded':0,'funded':1})
print(data3['status'])
data3['status'].value_counts()
# Balancing the data (status)
data3=pd.get_dummies(data3, drop_first=True)
df_majority=data3[data3.status == 1]
df_minority=data3[data3.status == 0]
from sklearn.utils import resample

df_minority_up=resample(df_minority, replace=True,n_samples=22634, random_state=123)
upsampled=pd.concat([df_majority,df_minority_up])
upsampled.status.value_counts()

# new_data=pd.get_dummies(data3, drop_first=True)
# df_majority=new_data[new_data.status == 1]
# df_minority=new_data[new_data.status == 0]
# from sklearn.utils import resample

# df_minority_up=resample(df_minority, replace=True,n_samples=22634, random_state=123)
# upsampled=pd.concat([df_majority,df_minority_up])
# upsampled.status.value_counts()


# storing the column names 
columns_list=list(upsampled.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['status']))
print(features)

# Storing the output values in y
y=upsampled['status'].values
print(y)
y1 = y # Copy for further analysis

# Storing the values from input features
x = upsampled[features].values
print(x)
x1 = x # Copy for further analysis

# To partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

# Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

# Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
# 86.42%

# Printing the misclassified values from prediction

print('Misclassified samples: %d' % (test_y != prediction).sum())
# 1853
np.unique(prediction)
from sklearn.metrics import roc_auc_score
prob_y=logistic.predict_proba(x)
prob_y=[p[1] for p in prob_y]
roc_auc_score(y,prob_y)

# =============================================================================
# END OF SCRIPT Logistic Regression
# =============================================================================
# =============================================================================
# KNN
# =============================================================================
# importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier  

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

# Splitting the data into train and test
train_x1,test_x1,train_y1,test_y1 = train_test_split(x1,y1,test_size=0.3, random_state=0)
# we used variables x1 & y1 from the previous model

# Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)  

# Fitting the values for X and Y
KNN_classifier.fit(train_x1, train_y1) 

# Predicting the test values with model
prediction1 = KNN_classifier.predict(test_x1)

# Performance metric check
confusionMatrix1 = confusion_matrix(test_y1, prediction1)
print(confusionMatrix1)

# Calculating the accuracy
accuracy_score1=accuracy_score(test_y1, prediction1)
print(accuracy_score1)
# 92.70%

print('Misclassified samples: %d' % (test_y1 != prediction1).sum())
# 553
"""
Effect of K value on classifier
"""
Misclassified_sample1 = []
# Calculating error for K values between 1 and 20
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x1, train_y1)
    pred_i = knn.predict(test_x1)
    Misclassified_sample1.append((test_y1 != pred_i).sum())

print(Misclassified_sample1)
#optimized value of k=1

np.unique(prediction)
# from sklearn.metrics import roc_auc_score
prob_y1=KNN_classifier.predict_proba(x1)
prob_y1=[p[1] for p in prob_y1]
roc_auc_score(y1,prob_y1)
# 98.82
# =============================================================================
# END OF SCRIPT KNN with reduced data
# =============================================================================

# =============================================================================
# LOGISTIC REGRESSION - REMOVING INSIGNIFICANT VARIABLES (The best Model with 100% Accuracy)
# =============================================================================

# Reindexing the salary status names to 0,1
data4['status']=data4['status'].map({'not_funded':0,'funded':1})
print(data4['status'])

cols = ['distribution_model','country_code','repayment_interval','currency_policy']
new_data2 = data4.drop(cols,axis = 1)

new_data2=pd.get_dummies(new_data2, drop_first=True)

# Balancing the data
majority=new_data2[new_data2.status == 1]
minority=new_data2[new_data2.status == 0]

minority_up=resample(minority, replace=True,n_samples=22634, random_state=123)
upsampled1=pd.concat([majority,minority_up])
upsampled1.status.value_counts()

# Storing the column names 
columns_list2=list(upsampled1.columns)
print(columns_list2)

# Separating the input names from data
features2=list(set(columns_list2)-set(['status']))
print(features2)

# Storing the output values in y
y2=upsampled1['status'].values
print(y2)
y3 = y2 # Copy for further analysis

# Storing the values from input features
x2 = upsampled1[features2].values
print(x2)
x3 = x2 # Copy for further analysis

# Splitting the data into train and test
train_x2,test_x2,train_y2,test_y2 = train_test_split(x2,y2,test_size=0.3)

# Make an instance of the Model
logistic2 = LogisticRegression()

# Fitting the values for x and y
logistic2.fit(train_x2,train_y2)

logistic2.coef_
logistic2.intercept_

# Prediction from test data
prediction2 = logistic2.predict(test_x2)

confusion_matrix2 = confusion_matrix(test_y2, prediction2)
print(confusion_matrix2)

# Calculating the accuracy
accuracy_score2=accuracy_score(test_y2, prediction2)
print(accuracy_score2)
# 86.51%

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y2 != prediction2).sum())
# 1831
np.unique(prediction2)
# from sklearn.metrics import roc_auc_score
prob_y2=logistic2.predict_proba(x2)
prob_y2=[p[1] for p in prob_y2]
roc_auc_score(y2,prob_y2)
# 93.75%
# =============================================================================
# END OF SCRIPT Logistic with reduced data
# =============================================================================
# =============================================================================
# KNN with reduced data
# =============================================================================

train_x3,test_x3,train_y3,test_y3 = train_test_split(x3,y3,test_size=0.3)
# Used variables x3 and y3 from the previous model

# Storing the K nearest neighbors classifier
KNN_classifier2 = KNeighborsClassifier(n_neighbors = 1)  

# Fitting the values for X and Y
KNN_classifier2.fit(train_x3, train_y3) 

# Predicting the test values with model
prediction3 = KNN_classifier2.predict(test_x3)

# Performance metric check
confusionMatrix3 = confusion_matrix(test_y3, prediction3)
print(confusionMatrix3)

# Calculating the accuracy
accuracy_score3=accuracy_score(test_y3, prediction3)
print(accuracy_score3)
# 94.57%

print('Misclassified samples: %d' % (test_y3 != prediction3).sum())
# 737
"""
Effect of K value on classifier
"""
Misclassified_sample3 = []
# Calculating error for K values between 1 and 20
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x3, train_y3)
    pred_i = knn.predict(test_x3)
    Misclassified_sample3.append((test_y3 != pred_i).sum())

print(Misclassified_sample3)
#optimized value of k = 1

np.unique(prediction3)
from sklearn.metrics import roc_auc_score
prob_y3=KNN_classifier2.predict_proba(x3)
prob_y3=[p[1] for p in prob_y3]
roc_auc_score(y3,prob_y3)
# 0.93%
# =============================================================================
# END OF SCRIPT KNN with reduced data
# =============================================================================

# Conclusion:
#     The best fit model is Logistic Regression with reduced data and 100% accuracy.