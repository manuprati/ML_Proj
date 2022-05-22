# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:10:23 2020

@author: User
"""

# Trip details - Clustering
# We have details of 91 trips taken by different drivers from a cab service company
# The variables shared by the company are- TripID, TripLength, MaxSpeed, MostFreqSpeed,
# TripDuration, Brakes, IdlingTime, Honking
# Analyze the dataset and see whether the data can be separated into different clusters
# Can you identify from the trip details, whether the drive was taken inside the city or on the highway?
# If it is a city drive, can you identify whether it was taken during peak hours or non-peak hours?
# File – tripDetails.xlsx
# Importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data = pd.read_excel(('tripDetails.xlsx'))
data.drop(['TripID'],axis = 1,inplace = True)
data.head()
features = list(data.columns)
print(features)

units = ['kms','kmph','kmph','mins','counts','mins','counts']
feature_units = dict(zip(features,units))
feature_units

for item in features:
 data[item].plot(kind='hist', bins = 15)
 plt.title(item)
 plt.xlabel(feature_units[item])
 plt.show()
 
# From histograms, we observe that the data points are clearly segregated into different groups, with differing
# number of segregations for each feature.
# A look at relationship between different features - correlation

correlation = data.corr()
print(correlation)

sns.heatmap(np.abs(correlation), xticklabels = correlation.columns, yticklabels = correlation.columns)
plt.show()

# From correlation table and correlation heatmap, we see that TripLength, MaxSpeed, MostFreqSpeed are
# highly correlated.
# Visualizing scatter of the data

sns.pairplot(data)
plt.show()

# Observation about scatter
# We see that few clusters are spherically distributed and few are elliptically distributed
# Also there exist different number of clusters ( 2,3,4,5 ) for different pair combination of features
# Few clusters are compact while others are not
# In most of the scatter plots (subplots) above, we see that there are 3 candidate clusters (based on
# compactness and isolation)
# Scaling : Important step in every Machine Learning problem
-
# To avoid giving undue advantage to some features which are expressed in some particular units,
# whose magnitude might be higer than some other feature variable (due to choice of units), scaling all
# features, so that they are numerically of same order of magnitude, is essential.
# We will use standard scaling (xi-u)/sigma.
# Let us help them discover the patterns in the data they have gathered using K-Means clustering
# K-Means Clsutering: ¶
# A technique to partition N observations into K clusters (K≤N) in which each observation belongs to
# cluster with nearest mean
# One of the simplest unsupervised algorithms
# Given N observations (x1,x2,...,xN) , K-means clustering will partition n observations into K (K≤N) sets
# S={s1,...,sk} so as to minimize the within cluster sum of squares (WCSS)
# K-Means Algorithm
# Input: D , k
from sklearn.preprocessing import StandardScaler
import copy as cp

data2 = data.copy()
data2 = StandardScaler().fit_transform(data2.values)
data2 = pd.DataFrame(data2,columns = features)

#     Algorithm:
# Step 1: Randomly choose two points as the cluster centers
# Step 2: Compute the distances and group the closest ones
# Step 3: Compute the new mean and repeat step 2
# Step 4: If change in mean is negligible or no reassignment then stop the process
# Output: Ci - Centroids of k clusters, cluster assignment labels for each
# datapoint
# One Convergence Criteria: No significant decrease in the sum squared error .i.e sum of square of distance
# between each datapoint to its assigned centroid. This is also called inertia

# K-Means using sklearn in python

# Determining number of clusters(K):
# Let us try clustering the data with K-Means for different values of K
# Elbow method – looks at percentage of variance explained as a function of number of clusters
# The point where marginal decrease plateaus is an indicator of the optimal number of clusters
# We will summarize K-Means for different k in an elbow plot below
from sklearn import cluster

distortions = [] # Empty list to store wss
for i in range(1, 11):
 km = cluster.KMeans(n_clusters=i,
 init='k-means++',
 n_init = 10,
 max_iter = 300,
 random_state = 100)
 km.fit(data2.values)
 distortions.append(km.inertia_)
#Plotting the K-means Elbow plot
plt.figure(figsize = (7,7))
plt.plot(range(1,11), distortions, marker='o')
plt.title('ELBOW PLOT')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# Though from elbow plot, we see that k=5 is best number of clusters, we will choose k=3 , because that is
# the point where marginal decrease plateaus.
# We will cluster the data into 3 groups and label the datapoints with their assignment to the clusters.

k = 3
km3 = cluster.KMeans(n_clusters=k,
 init='k-means++',
 n_init = 10,
 max_iter = 300,
 random_state = 100)
km3.fit(data2.values)
labels = km3.labels_
Ccenters = km3.cluster_centers_
data2['labels'] = labels
data2['labels'] = data2['labels'].astype('str')
print(data2['labels'])

sns.pairplot(data2, x_vars = features, y_vars = features, hue='labels', diag_kind='kde')
plt.show()

# We see from pair plot that, for every pair of features, the points have been well clustered into different
# groups. Though isolation and compactness are not observed together in all possible pairs of features.
# Observations:
# Cluster1 is distinguised by comparatively very high values for Brakes,
# IdlingTime, Honking, low MaxSpeed and TripLength
# This is indicative of intercity travel during peak hours
# MaxSpeed, MostFreqSpeed and TripDuration is higher for cluster2 than cluster
# 1 and 3
# Cluster2 is is indicative of highway trips
# Cluster3 is indicative of city trips during non-peak hours }


c_df = pd.concat([data[data2['labels']=='0'].mean(),
 data[data2['labels']=='1'].mean(),
 data[data2['labels']=='2'].mean()],
 axis=1)
c_df.columns = ['cluster1','cluster2','cluster3']

triptype = ['Intercity-Peak hours','Highway','Intercity-Non-peak hours']
data['labels'] = labels
data['labels'] = data['labels'].map({0:triptype[0],1:triptype[1],2:triptype[2]})
print(data.head())

# End