# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Concentration of species for absorbance measurement
import numpy as np
from scipy.io import loadmat
from matplotlib.pyplot import *
import pandas as pd
mat_data = loadmat('data.mat') # load mat-file
wavelengths = mat_data['WAV'][0] # dictionary element with the key = 'WAV' is a list of list.
species=['Cr','Ni','Co'] # Name of the species in the order specified in the data
species_con=pd.DataFrame(np.concatenate([mat_data['PureCrCONC'],mat_data['PureNiCONC'],
 mat_data['PureCoCONC']],axis=1),columns=species)
species_abs=pd.DataFrame(np.concatenate([mat_data['PureCr'],mat_data['PureNi'],
 mat_data['PureCo']],axis=0),
 columns=wavelengths,index=species)
# concentration of each species in the mixtures (each repeated 5 times) - corresponds to Y/output
mixture_con = pd.DataFrame(mat_data['CONC'],columns=species)
mixture_abs = pd.DataFrame(mat_data['DATA'],columns=wavelengths) # absorbance data for each wavelength corresponds to X/input

'''
Function to plot spectra
Arguments: df (pandas.DataFrame), labels (list), header (string), vlines (boolean)
Returns: NoneType
'''
def plot_spectra(df,labels,header,vlines):
 fig, ax = subplots(figsize=(15,4))
 df.plot(ax=ax)
 ax.legend(labels,loc='best');
 ax.grid(True)
 ax.set(title=header,xlabel='Wavelength (nm)',ylabel='Absorbance')
 ax.set_ylim(-0.3,1.3)
 if vlines:
     vlines={408:'b',394:'orange',512:'g',574:'b'}
     for i in vlines:
         ax.axvline(x=i,color=vlines[i]) # drawing a blue vertical line at 408nm
         ax.text(i+1,0.6,str(i),rotation=90)
 # writing the wavelength value corresonding to the vertical line and placing the text vertically rotated
 
 
Maximum absorption of energy are the unique features of each species
Species spectra with vertical lines marking the absorption peaks
# Looping through each species to concatenate species name and its corresponding concentration in mol/litre
labels=[i+'='+str(j) for i,j in zip(species_con.columns,species_con.values.tolist()[0])]
plot_spectra(species_abs.T,labels,'Absorbance spectra of species',False)

plot_spectra(species_abs.T,labels,'Absorbance specra of species',True)

# Maximum absorption of energy are seen at 408 nm and 574nm for Chromium, at 394 nm for Nickel and at 512nm for Cobalt of each species
# Y - concentration of each species in the mixture

mixture_con.shape
mixture_con.head(6)
mixture_abs.shape
mixture_abs.head(6)

'''
Function to get labels for the selected indices
Arguments: df (pandas.DataFrame), ind (list)
Returns: list of strings
'''
def get_labels(df,ind):
 mix_labels=[]
 for i in ind:
     label=''
     for j in range(3):
        label+=str(np.round(df.iloc[i,j],5))+'|'
     mix_labels.append(label)
 return mix_labels

# Plot of absorbance spectra of selected mixtures from 130 samples
sel_ind=[0,5,60,70,100,129]
plot_spectra(mixture_abs.iloc[sel_ind,:].T,get_labels(mixture_con,sel_ind),'Absorbance spectra of mixtures',True)

mix_avg_abs=((mixture_abs + mixture_abs.shift(-1)+mixture_abs.shift(-2)+mixture_abs.shift(-3)+mixture_abs.shift(-4)) / 5)[::5]
mix_uniq_con=mixture_con[::5]
# Plot of the averaged spectra of mixtures for every 5 samples
avg_sel_ind = [0,1,12,14,20,25]
plot_spectra(mix_avg_abs.iloc[avg_sel_ind,:].T,get_labels(mix_uniq_con,avg_sel_ind),
 'Averaged absorbance spectra of mixtures',True)
#Principal Component Regression
mix_avg_abs=((mixture_abs + mixture_abs.shift(-1)+mixture_abs.shift(-2)+mixture_abs.shift(-3)+mixture_abs.shift(-4)) / 5)[::5]
mix_uniq_con=mixture_con[::5]
avg_sel_ind = [0,1,12,14,20,25]
plot_spectra(mix_avg_abs.iloc[avg_sel_ind,:].T,get_labels(mix_uniq_con,avg_sel_ind),
 'Averaged absorbance spectra of mixtures',True)

# Function performing linear regression over training and returns rmse of test predictions
# Function iteratively performing linear regression between y and reduced sets of X in the increasing order of number of PCs over training and returns rmse of test
# predictions for each number of PCs considered.
# Importing model_selection library for using cross_validation
from sklearn import model_selection
# Importing the library for PCA
from sklearn.decomposition import PCA
# Importing the library for Linear Regression
from sklearn.linear_model import LinearRegression 

'''
Function to linearly regress the training samples of X against Y
and to return the list of rmse for each variable in the output (y) of the testing samples
Arguments: X_train (numpy.ndarray),y_train (numpy.ndarray),X_test (numpy.ndarray),y_test (numpy.ndarray)
Returns: rmse (list of arrays)
'''
def linreg(X_train,X_test,y_train,y_test):
 ols = LinearRegression()
 ols.fit(X_train, y_train)
 y_pred = ols.predict(X_test)
 rmse=np.sqrt(((y_test-y_pred)**2).mean(axis=0)) #square root (mean square error)
 return rmse

'''
Function to perform linear regression using leave one out cross validation between Y
and reduced set of X iteratively from 1 principal component to maximum number of principal coomponents specified
Arguments: X (numpy.ndarray), y (numpy.ndarray), nfact (int)
Returns: RMSE_pcr_arr (numpy.ndarray of size nfact,number of variables of y)
'''
def pca_loocv_ols(X,y,nfact):
 RMSE_pcr_lst=list()
 for pc in range(1,nfact,1): # Iterating over number of principal components
     pca = PCA(n_components=pc) # Instantiating pca instance for each number of principal components in the iteration process
     X_red = pca.fit_transform(X)
     rmse_pcr_lst=list()
     lcv = model_selection.LeaveOneOut() # Instantiating an lcv instance
     for tr_idx, tst_idx in lcv.split(X_red): # interating through multiple folds
         X_train, X_test = X_red[tr_idx], X_red[tst_idx] # input X for both train and test
         y_train, y_test = y[tr_idx], y[tst_idx] # output Y for both train and test
         rmse_pcr_lst.append(linreg(X_train,X_test,y_train,y_test)) # Appending rmse for each fold into a list
         #number of validations=26
        #number of PCR loops=25
     RMSE_pcr_lst.append(np.array(rmse_pcr_lst).mean(axis=0))
 return np.array(RMSE_pcr_lst) 
# Function iteratively performing linear regression between y and reduced sets of X in the increasing order of number of PCs over training and returns rmse of test
# predictions for each number of PCs considered.

# Using one of the five measurements (first) for each mixture for modelling and estimating maximum error in the model

mix_sample_abs=mixture_abs[::5] # Sampling every first sample from the 5 experiments of same mixture proportions of species
std_y=np.array(mix_uniq_con).std(axis=0) # estimating standard deviation in the Y corresponding to maximum error in the model
std_y=std_y.reshape(1,-1) # reshaping the vector(Ny,) to array (1,Ny) where Ny is number of variables in output array
std_y

# Performing pcr over sampled spectra of mixtures

RMSE_pcr_samples=np.append(std_y,pca_loocv_ols(np.array(mix_sample_abs),np.array(mix_uniq_con),26),axis=0)
RMSE_pcr_samples.shape

PCR over averaged X samples

RMSE_pcr_avg=np.append(std_y,pca_loocv_ols(np.array(mix_avg_abs),np.array(mix_uniq_con),26),axis=0)
RMSE_pcr_avg.shape

# Plot of RMSE against number of PCs for both sampled and averaged spectra

fig, axs = subplots(1, 2,figsize=(15,4))
axs[0].plot(range(26),RMSE_pcr_samples)
axs[0].set_title('PCR: First Experiment Samples')
axs[1].plot(range(26),RMSE_pcr_avg)
axs[1].set_title('PCR: Averaged Samples')
axs[0].grid(True)
axs[1].grid(True)
for ax in axs.flat:
 ax.set(xlabel='Number of PCs', ylabel='RMSE')
for ax in axs.flat:
 ax.label_outer()
 
 # END OF SCRIPT

