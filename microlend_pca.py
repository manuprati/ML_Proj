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

# Importing the following libraries for preprocessing
from sklearn.preprocessing import StandardScaler
# Importing the library for PCA
from sklearn.decomposition import PCA
# Importing the library for Balancing
from sklearn.utils import resample

#%%2
pd.set_option('max_column',100)
data=pd.read_csv('microlending_data.csv',index_col=0)
data.info()

#%%
# =============================================================================
# Cleaning Data
# =============================================================================

#dropping the duplicate records

data2=data.copy()

data2.drop_duplicates(keep='first',inplace=True)
# 1575 duplicate records dropped

#%%

#Filling Missing values

data2.isnull().sum()
# 30 values are missing in borrower gender which can be filled by modal value i.e. female
data2['borrower_genders'].value_counts()
data2['borrower_genders'].fillna(data['borrower_genders'].mode()[0],inplace=True)
print(np.unique(data2['borrower_genders']))


#%% 
print(np.unique(data2['term_in_months']))
data2['term_in_months'].replace('1 Year',12, inplace=True)
data2['term_in_months'].replace('2 Years',24, inplace=True)
data2['term_in_months']=data2['term_in_months'].astype('int64')
data2.describe()
data2.info()
data2['term_in_months'].value_counts()
sns.distplot(data2['term_in_months'],kde=False)
sns.boxplot(data2['term_in_months'])

#%%

data2['lender_count'].value_counts()
sns.distplot(data2['lender_count'],kde=False)
sns.boxplot(data2['lender_count'])

data2.describe()
sum(data2['lender_count']<1)
sum(data2['lender_count']>355)
# lender_count < 1 = 59 which are meaningless because lender_count cannot be less than one so we can remove in data4

# Also lender_count > 355 = 2 which seems ouliers so we can treat them as follows because info about loan amount is also attached.


data2=data2[(data.lender_count >= 1)]

data2['lender_count'] = np.where(data2['lender_count']>=355,
      data2['lender_count'].median(), data2['lender_count'])
#%%

data2['loan_amount'].value_counts()
sns.boxplot(data2['loan_amount'])
sns.distplot(data2['loan_amount'],kde=False)

sum(data2['loan_amount']>12000)
#Although only 4 values more than 12000 in loan amount which are outlier so we cant treat them similarly
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
# data is the cleaned data
# Making two copies of cleaned data
data3=data2.copy()
data4=data3.copy()

data3['status'].value_counts()
# By removing 58 records having lender_count less than one not_funded records have also been reduced by 58
categotical_data = data2.describe(include='O')
categotical_data = categotical_data.drop(['borrower_genders','original_language','distribution_model','country_code','repayment_interval','currency_policy'],axis = 1)

frequencies      = categotical_data.apply(lambda x: x.value_counts()).T.stack()

print(frequencies)

sns.boxplot(x = churn1['TotalCharges'], y = churn1['Churn'])



# Let us impute those missing values using mean based on the output
# varieble 'Churn' – Yes & No

data2.groupby(['status']).mean().groupby('status')['loan_amount'].mean()
data2.groupby(['sector']).mean().groupby('sector')['loan_amount'].mean()
data3.groupby(['activity']).mean().groupby('activity')['loan_amount'].mean()
(data3.groupby(['country']).mean().groupby('country')['loan_amount'].mean()).index()

data2['lender_count'] = data2.groupby('status')['lender_count']\
.transform(lambda x: x.fillna(x.mean()))

churn1['TotalCharges'] = churn1.groupby('Churn')['TotalCharges']\
.transform(lambda x: x.fillna(x.mean()))

churn1.TotalCharges.isnull().sum()

sns.boxplot('sector', 'loan_amount', data=data2)
data2.groupby('sector')['loan_amount'].median()

## people with 35-50 age are more likely to earn > 50000 USD p.a
## people with 25-35 age are more likely to earn <= 50000 USD p.a

sns.boxplot('status', 'lender_count', data=data2)
data2.groupby('status')['lender_count'].median()
data2.groupby('status')['lender_count'].mean()
data2.describe()

#*** Jobtype
JobType     = sns.countplot(y=data2['JobType'],hue = 'SalStat', data=data2)
job_salstat =pd.crosstab(index = data2["JobType"],columns = data2['SalStat'], margins = True, normalize =  'index')  
round(job_salstat*100,1)


sns.countplot(data=data2,x='lender_count', hue='status')
sns.boxplot(data=data2,x='lender_count', y='status')
# Lender count is an important variable as status is dependent gretly on it.


# =============================================================================
#Data Visualisation
# =============================================================================



data3.columns
data3.describe(include='O')

sns.countplot(data3['status'])
pd.crosstab(index=data3['status'], columns='count', normalize=True)
# 87% funded and 13% not funded


data3['activity'].value_counts()
print(np.unique(data3['activity']))
sns.countplot(data=data3,y='activity', hue='status')
activity_status = pd.crosstab(columns=data2['status'], index=data2['activity'], margins=True, normalize='index')
round(activity_status*100,1)

# Activity has 150 value counts like aquaculture, Arts, Well digging which are 100% funded 
# Important variable - except some activity viz. wedding expenses all other funded in great percentage.


data3['country'].value_counts()
sns.countplot(data=data3,y='country', hue='status')
pd.crosstab(columns=data3['status'], index=data3['country'], margins=True, normalize='index')
# Country has 61 value counts like Afghanistan, Belize which are 100% funded 
# Important variable - except some country viz. Brazil and Bolivia of which 29% & 24% not funded.

data3['country_code'].value_counts()
sns.regplot(data=data3,x='country', y='country_code')
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
# Insignicant variable as 29% Portuguess and 26% French were not funded rest are dunded significantly.

data3['repayment_interval'].value_counts()
sns.boxplot(data=data3,x='repayment_interval', y='loan_amount')
sns.countplot(data=data3,y='repayment_interval', hue='status')
pd.crosstab(columns=data3['status'], index=data3['repayment_interval'], margins=True, normalize='index')
# Not significant can be dropped in data4

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
rMPI_Genderpd.crosstab(columns=data3['status'], index=data3['rMPI'], margins=True, normalize='index')
sns.countplot(data=data3,x='rMPI', hue='status')
round(gender_status*100,1)

# Important variable as rMPI value of .005 and .583 were not funded in 39% & 41% cases respectively etc. 


data3.describe()
sns.countplot(data=data3,x='borrower_genders', hue='status')
gender_status=pd.crosstab(columns=data2['borrower_genders'], index=data2['status'], margins=True, normalize='index')
round(gender_status*100,1)
# 64% females were funded but 49% male were not funded
# Hence important variable
#%%
# =============================================================================
# PCA
# =============================================================================
# importing the library of PCA



# importing microlend data set (only most significant columns)
data3 = data3.filter(['loan_amount','term_in_months','rMPI','status'],axis=1)
data3 = data4.copy()

# Reindexing the status names to 0,1
data3['status']=data3['status'].map({'not_funded':0,'funded':1})
print(data3['status'])
data3['status'].value_counts()
data3.describe()

# Balancing the data (status)
# new_data=pd.get_dummies(data3, drop_first=True)
df_majority=data3[data3.status == 1]
df_minority=data3[data3.status == 0]


df_majority_dn=resample(df_majority, replace=False,n_samples=3272, random_state=123)
new=pd.concat([df_minority,df_majority_dn])
new.status.value_counts()


# Information on the data
print (new.info())

new.head()
new.tail()
# Sort the columns of new
input_columns = list(new.iloc[:,:3].columns) #
input_columns.sort()
input_data=new[input_columns]
input_data.head()
input_dataa=input_data.copy()

plt.subplots(figsize=(8,6))
sns.scatterplot(x='term_in_months', y='loan_amount', hue='status', data=new)
# significantly separable 

# sns.scatterplot(x='lender_count', y='loan_amount', hue='status', data=new)
# sns.scatterplot(x='term_in_months', y='lender_count', hue='status', data=new)


# Scaling data using (x-mu)
scaler = StandardScaler(with_std=False)
input_data = scaler.fit_transform(input_data)
input_data = pd.DataFrame(input_data,columns=input_columns)
input_data.head()

# The following code snippet is not working while samples are upscaled 
#%%
u, s, v = np.linalg.svd(input_data) #decomposing using SVD
exp_var=s**2/np.sum(s**2)*100 # Explained variance by each eigen value/PC
exp_var
pc=new[input_columns].dot(v.T) # Rotating and trnsforming from sample space to feature space
pc.columns=['PC1','PC2','PC3']
pc['status']= data4['status']
pc.head()

sns.scatterplot(x='PC1', y='PC2', hue='status', data=pc)
#%%

#END OF SCRIPT



cols = ['lender_count','country','original_language','distribution_model','repayment_interval','currency_policy']
new_data = data3.drop(cols,axis = 1)


# y= data3.filter(['status'],axis=1)
new_data['status']=data3['status'].map({'not_funded':0,'funded':1})
print(new_data['status'])
new_data.status.value_counts()
data5=new_data.copy()
new_data=pd.get_dummies(new_data, drop_first=True)
new_data.columns

df_majority=new_data[new_data.status == 1]
df_minority=new_data[new_data.status == 0]
 
# Balancing the variable status 

df_majority_dn=resample(df_majority, replace=False,n_samples=3272, random_state=123)
new=pd.concat([df_minority,df_majority_dn])
new.status.value_counts()
#new_data['status']=new['status'].map({0:'not_funded',1:'funded'})


# Information on the data
new.status

# Sort the columns of new
input_columns = list(set(new.columns)-set(['status'])) #
input_columns.sort()
input_data=new[input_columns]
input_data.head()
input_dataa=input_data.copy()

plt.subplots(figsize=(8,6))
sns.scatterplot(x='term_in_months', y='loan_amount', hue='status', data=new)
sns.scatterplot(x='rMPI', y='loan_amount', hue='status', data=new)

# Scaling data using (x-mu)
scaler = StandardScaler(with_std=False)
input_data = scaler.fit_transform(input_data)
input_data = pd.DataFrame(input_data,columns=input_columns)
input_data.head()

# The following code snippet will work when variable explorer clears
#%%
u, s, v = np.linalg.svd(input_data) #decomposing using SVD
exp_var=s**2/np.sum(s**2)*100 # Explained variance by each eigen value/PC
exp_var
pc=new[input_columns].dot(v.T) # Rotating and trnsforming from sample space to feature space
pc.columns=[i for i in range(1,229)]
pc['status']= data4['status']
pc.head()


sns.scatterplot(x=1, y=2, hue='status', data=pc)
#%%

# computing covariance using scaled data (renamed the data as 'input_data')
covariance_matrix = input_data.cov()
print (covariance_matrix)

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix.values)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i])for i in range(len(eig_vals))]


#abs - absolute value
eig_pairs.sort(key = lambda x: x[0], reverse=True)# sort eig_pairs in descending order based on the eigen values
#false for ascending order
print('Eigenvalues in descending order:')
# eig_pairs
# setting threshold as '95% variance'
threshold = 0.99999
# Computing number of PCS required to captured specified variance
print('Explained variance in percentage:\n')
cumulative_variance = 0.0
count = 0
eigv_sum = np.sum(eig_vals)
for i,j in enumerate(eig_pairs):
    variance_explained = (j[0]/eigv_sum).real
    print('eigenvalue {}: {}'.format(i+1, variance_explained*100 ))
    cumulative_variance += variance_explained
    count = count+1
    if (cumulative_variance>=threshold):
        break
print('\nCumulative variance=',cumulative_variance*100)

print('Total no. of eig vecs =',len(eig_vecs),'\nselected no. of eig vecs =',count)

# select required PCs based on the count - projection matrix w=d*k
reduced_dimension = np.zeros((len(eig_vecs),count))
for i in range(count):
    reduced_dimension[:,i]= eig_pairs[i][1]
# Projecting the scaled data onto the reduced space (using eigen vectors)
projected_data = input_data.values.dot(reduced_dimension)
projected_dataframe = pd.DataFrame(projected_data,
columns=['PC1','PC2'])
projected_dataframe_with_class_info = pd.concat([projected_dataframe,
data4.status],axis=1)

sns.scatterplot(x='PC1', y='PC2',hue='status', data=projected_dataframe_with_class_info)

# Choosing the extent of variance to be covered by the PCs
PCA_Sklearn = PCA(n_components=2)
# Transforming the iris data input_columns
Projected_data_sklearn= PCA_Sklearn.fit_transform(new.iloc[:,:4])
# Storing the PCs in the data frame
Projected_data_sklearn_df = pd.DataFrame(Projected_data_sklearn, columns=['PC1','PC2'])

# Storing the PCs in the data frame along with class label
Projected_data_sklearn_df_with_class_info=pd.concat([Projected_data_sklearn_df,data4.status],axis=1)
print('Explained variance :\n')
print(PCA_Sklearn.explained_variance_ratio_)

plt.subplots(figsize=(8,6))
sns.scatterplot( x='PC1', y='PC2',hue='status', data=Projected_data_sklearn_df_with_class_info)
plt.show()
#END OF SCRIPT













# importing microlend data set (only most significant columns)
data.info()
#data3 = data4.copy()
data3 = data3.filter(['sector','activity','country','lender_count','loan_amount','term_in_months','rMPI','status'],axis=1)

data3['status']=data3['status'].map({'not_funded':0,'funded':1})
print(data3['status'])
data3['status'].value_counts()
data3.describe()
# Balancing the data (status)
# new_data=pd.get_dummies(data3, drop_first=True)
df_majority=data3[data3.status == 1]
df_minority=data3[data3.status == 0]
from sklearn.utils import resample


df_majority_dn=resample(df_majority, replace=False,n_samples=3272, random_state=123)
new=pd.concat([df_minority,df_majority_dn])
new.status.value_counts()
new = pd.get_dummies(new, drop_first=True)
new.corr()

# Information on the data
print (new.info())

new.head()

# Sort the columns of new
input_columns = list(set(new.columns)-set(['status']))
input_columns.sort()
input_data=new[input_columns]
input_data.head()
input_dataa=input_data.copy()

plt.subplots(figsize=(8,6))
sns.scatterplot(x='term_in_months', y='loan_amount', hue='status', data=new)
sns.scatterplot(x='lender_count', y='loan_amount', hue='status', data=new)
sns.scatterplot(x='term_in_months', y='lender_count', hue='status', data=new)


# Scaling data using (x-mu)
scaler = StandardScaler(with_std=False)
input_data = scaler.fit_transform(input_data)
input_data = pd.DataFrame(input_data,columns=input_columns)
input_data.head()

# The following code snippet is not working on large sample
#%%
u, s, v = np.linalg.svd(input_data) #decomposing using SVD
exp_var=s**2/np.sum(s**2)*100 # Explained variance by each eigen value/PC
pc=new[input_columns].dot(v.T) # Rotating and trnsforming from sample space to feature space
pc.columns=[i for i in range(1,213)]
pc['status']= data4['status']
pc.head()

exp_var
sns.scatterplot(x=1, y=2, hue='status', data=pc)
#%%

# computing covariance using scaled data (renamed the data as 'input_data')
covariance_matrix = input_data.cov()
# print (covariance_matrix)

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix.values)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i])for i in range(len(eig_vals))]
eig_pairs

#abs - absolute value
eig_pairs.sort(key = lambda x: x[0], reverse=True)# sort eig_pairs in descending order based on the eigen values
#false for ascending order
print('Eigenvalues in descending order:')

# setting threshold as '95% variance'
threshold = 0.9999
# Computing number of PCS required to captured specified variance
print('Explained variance in percentage:\n')
cumulative_variance = 0.0
count = 0
eigv_sum = np.sum(eig_vals)
for i,j in enumerate(eig_pairs):
    variance_explained = (j[0]/eigv_sum).real
    print('eigenvalue {}: {}'.format(i+1, variance_explained*100 ))
    cumulative_variance += variance_explained
    count = count+1
    if (cumulative_variance>=threshold):
        break
print('\nCumulative variance=',cumulative_variance*100)

print('Total no. of eig vecs =',len(eig_vecs),'\nselected no. of eig vecs =',count)

# select required PCs based on the count - projection matrix w=d*k
reduced_dimension = np.zeros((len(eig_vecs),count))
for i in range(count):
    reduced_dimension[:,i]= eig_pairs[i][1]
# Projecting the scaled data onto the reduced space (using eigen vectors)
projected_data = input_data.values.dot(reduced_dimension)
projected_dataframe = pd.DataFrame(projected_data,
columns=['PC1','PC2'])
projected_dataframe_with_class_info = pd.concat([projected_dataframe,
data4.status],axis=1)

sns.scatterplot(x='PC1', y='PC2',hue='status', data=projected_dataframe_with_class_info)

# Choosing the extent of variance to be covered by the PCs
PCA_Sklearn = PCA(n_components=2)
# Transforming the iris data input_columns
Projected_data_sklearn= PCA_Sklearn.fit_transform(new)
# Storing the PCs in the data frame
Projected_data_sklearn_df = pd.DataFrame(Projected_data_sklearn, columns=['PC1','PC2'])

# Storing the PCs in the data frame along with class label
Projected_data_sklearn_df_with_class_info=pd.concat([Projected_data_sklearn_df,data4.status],axis=1)
print('Explained variance :\n')
print(PCA_Sklearn.explained_variance_ratio_)

plt.subplots(figsize=(8,6))
sns.scatterplot( x='PC1', y='PC2',hue='status', data=Projected_data_sklearn_df_with_class_info)
plt.show()
#END OF SCRIPT