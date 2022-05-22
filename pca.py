# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:29:45 2020

@author: HP
"""


# Importing pandas to perform operations using DataFrames
import pandas as pd
# Importing numpy to perform Matrix operations
import numpy as np
# Importing matplotlib to plot graphs
import matplotlib.pyplot as plt
import seaborn as sns
# Importing the following libraries for preprocessing
from sklearn.preprocessing import StandardScaler
# Importing the library for PCA
from sklearn.decomposition import PCA

# importing IRIS data set
iris_data = pd.read_csv('Iris_data.csv')

# Information on the data
print (iris_data.info())

iris_data.head()

# dropping column 'Id'
iris_data = iris_data.drop(['Id'],axis=1)
iris_data.head()
##iris_data.iloc[2:3,3:4]
# Sort the columns of X
input_columns = list(iris_data.iloc[:,:4].columns)
input_columns.sort()
input_data=iris_data[input_columns]
input_data.head()
input_dataa=iris_data.copy()
#input_dataa=input_dataa.drop(['Species'],axis=1)
plt.subplots(figsize=(8,6))
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris_data)
plt.show()

# Scaling data using (x-mu)
scaler = StandardScaler(with_std=False)
input_data = scaler.fit_transform(input_data)
input_data = pd.DataFrame(input_data,columns=input_columns)
input_data.head()

u, s, v = np.linalg.svd(input_data) #decomposing using SVD
exp_var=s**2/np.sum(s**2)*100 # Explained variance by each eigen value/PC
pc=iris_data[input_columns].dot(v.T) # Rotating and trnsforming from sample space to feature space
pc.columns=['PC1','PC2','PC3','PC4']
pc['Species']= iris_data['Species']
pc.head()

exp_var
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=pc)

# computing covariance using scaled data (renamed the data as 'input_data')
covariance_matrix = input_data.cov()
print (covariance_matrix)

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix.values)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i])for i in range(len(eig_vals))]
eig_pairs

#abs - absolute value
eig_pairs.sort(key = lambda x: x[0], reverse=True)# sort eig_pairs in descending order based on the eigen values
#false for ascending order
print('Eigenvalues in descending order:')
eig_pairs
# setting threshold as '95% variance'
threshold = 0.95
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
iris_data.Species],axis=1)

sns.scatterplot(x='PC1', y='PC2', hue='Species', data=projected_dataframe_with_class_info)

# Choosing the extent of variance to be covered by the PCs
PCA_Sklearn = PCA(n_components=0.95)
# Transforming the iris data input_columns
Projected_data_sklearn= PCA_Sklearn.fit_transform(iris_data.iloc[:,:4])
# Storing the PCs in the data frame
Projected_data_sklearn_df = pd.DataFrame(Projected_data_sklearn, columns=['PC1','PC2'])

# Storing the PCs in the data frame along with class label
Projected_data_sklearn_df_with_class_info=pd.concat([Projected_data_sklearn_df,iris_data.Species],axis=1)
print('Explained variance :\n')
print(PCA_Sklearn.explained_variance_ratio_)

plt.subplots(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=Projected_data_sklearn_df_with_class_info)
plt.show()
#END OF SCRIPT