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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier  

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

#import os
#os.chdir("d:/pandas")

#%%
pd.set_option('max_column',100)
data=pd.read_csv('microlending_data.csv',index_col=0)
data.info()

data.isnull().sum()
data['borrower_genders'].value_counts()
data['borrower_genders'].fillna(data['borrower_genders'].mode()[0],inplace=True)
print(np.unique(data['borrower_genders']))

data.describe(include='O')

data.drop_duplicates(keep='first',inplace=True)

data.columns
data2=data.copy()

data3=data.copy()

data4=data.copy()

#%%
data['status'].value_counts()
data['activity'].value_counts()
data['country'].value_counts()
data['country_code'].value_counts()
data['currency_policy'].value_counts()
data['distribution_model'].value_counts()
data['lender_count'].value_counts()
data['loan_amount'].value_counts()
data['original_language'].value_counts()
data['repayment_interval'].value_counts()
data['sector'].value_counts()
data['term_in_months'].value_counts()
data['rMPI'].value_counts()

#%%
#Filling Missing values

print(np.unique(data['status']))
print(np.unique(data['activity']))
print(np.unique(data['currency_policy']))
print(np.unique(data['distribution_model']))
print(np.unique(data['lender_count']))
print(np.unique(data['rMPI']))

print(np.unique(data['term_in_months']))
data['term_in_months'].replace('1 Year',12, inplace=True)
data['term_in_months'].replace('2 Years',24, inplace=True)
data['term_in_months']=data['term_in_months'].astype('int')
data.describe()


#%%
data.describe()
pd.crosstab(columns=data['borrower_genders'], index=data['status'], margins=True, normalize=index)

#%%
#Correlation - Realtionship between independent variables

data_select=data.select_dtypes(exclude=["object"])
correlation=data_select.corr()
# Loan amount is highly dependent on lender count

#%%
#Data Visualisation

sns.countplot(data['status'])
sns.boxplot(data['loan_amount'])
sns.boxplot(data['lender_count'])
# sns.regplot(data=data,x='loan_amount', y='lender_count')
# sns.countplot('status', 'lender_count', data=data, hue='loan_amount')
# data.groupby('status')['lender_count'].median()

sns.countplot(data['loan_amount'])
sns.distplot(data['loan_amount'])

sns.regplot(x=data['loan_amount'],y =data['lender_count'], scatter=True,fit_reg=False)
# plt.scatter(x=data['loan_amount'],y =data['lender_count'])

sns.countplot(data['lender_count'])
sns.distplot(data['lender_count'])
plt.hist(data['term_in_months'])
sns.distplot(data['term_in_months'])
sns.boxplot(data['term_in_months'])


sum(data['lender_count']<1)
sum(data['lender_count']>355)
sum(data['loan_amount']>20000) #outlier
sum(data['loan_amount']<50)
sum(data['term_in_months']>140)

sns.countplot(data=data,x='currency_policy', hue='status')
pd.crosstab(index = data["currency_policy"],columns = data['status'], margins = True, normalize =  'index')

sns.countplot(data=data,y='country', hue='status')
sns.boxplot(data=data,x='rMPI', y='status')
sns.countplot(data=data,x='activity', hue='status')
sns.countplot(data=data,x='repayment_interval', hue='status')
sns.countplot(data=data,x='original_language', hue='status')
sns.boxplot(data=data,x='repayment_interval', y='loan_amount')
sns.boxplot(data=data,x='original_language', y='loan_amount')
sns.countplot(data=data,x='sector', hue='status')
sns.boxplot(data=data,y='sector', x='loan_amount')
sns.countplot(data=data,y='term_in_months', hue='status')
sns.boxplot(data=data,y='term_in_months', x='status')

sns.countplot(data=data,x='activity', hue='status')

_#%%

cars = data[
        (data.lender_count >= 1) 
      & (data.lender_count <= 355) 
      & (data.loan_amount >= 25) 
      & (data.loan_amount <= 20000)]

# ~60 records are dropped


cars['status'].describe()

# =============================================================================
# Removing insignificant variables
# =============================================================================

col=['distribution_model',]
data=data.drop(columns=col, axis=1)

#%%
# =============================================================================
# Correlation
# =============================================================================

data_select1=data.select_dtypes(exclude=[object])
correlation=data_select1.corr()
round(correlation,3)   
data_select1.corr().loc[:,'status'].abs().sort_values(ascending=False)[1:]                          


#%%
# =============================================================================
# KNN
# =============================================================================

# Reindexing the salary status names to 0,1
data['status']=data['status'].map({'not_funded':0,'funded':1})
print(data['status'])

new_data=pd.get_dummies(data, drop_first=True)

# Storing the column names 
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['status']))
print(features)

# Storing the output values in y
y=new_data['status'].values
print(y)

# Storing the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

# =============================================================================
# KNN
# =============================================================================

# Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 11)  

# Fitting the values for X and Y
KNN_classifier.fit(train_x, train_y) 

# Predicting the test values with model
prediction = KNN_classifier.predict(test_x)

# Performance metric check
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)

# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

print('Misclassified samples: %d' % (test_y != prediction).sum())

"""
Effect of K value on classifier
"""
Misclassified_sample = []
# Calculating error for K values between 1 and 20
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())

print(Misclassified_sample)
# =============================================================================
# END OF SCRIPT
# =============================================================================

# =============================================================================
# LOGISTIC REGRESSION - REMOVING INSIGNIFICANT VARIABLES
# =============================================================================

# Reindexing the salary status names to 0,1
data3['status']=data3['status'].map({'not_funded':0,'funded':1})
print(data4['status'])

cols = ['distribution_model','country']
new_data = data4.drop(cols,axis = 1)

new_data=pd.get_dummies(new_data, drop_first=True)

# Storing the column names 
columns_list2=list(new_data.columns)
print(columns_list2)

# Separating the input names from data
features2=list(set(columns_list2)-set(['SalStat']))
print(features2)

# Storing the output values in y
y2=new_data['status'].values
print(y2)

# Storing the values from input features
x2 = new_data[features2].values
print(x2)

# Splitting the data into train and test
train_x2,test_x2,train_y2,test_y2 = train_test_split(x2,y2,test_size=0.3, random_state=0)

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
print(accuracy_score)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y2 != prediction2).sum())


# =============================================================================
# END OF SCRIPT
# =============================================================================
