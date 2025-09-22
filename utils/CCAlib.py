"""
@author: Matthijs oude Lohuis
Champalimaud 2023

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.cross_decomposition import CCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA

from sklearn.model_selection import KFold
from scipy.stats import zscore

from loaddata.session_info import load_sessions
from utils.psth import compute_tensor

def CCA_subsample(DATA1,DATA2,nN=None,nK=None,resamples=5,kFold=5,prePCA=None,n_components=1):
    # Data format: 
    #  DATA1 is the source data (number of number of trials x source neurons)
    #  DATA2 is the target data (number of number of trials x source neurons)
    K,N1 = np.shape(DATA1)
    N2 = np.shape(DATA2)[1]
    
    if nN is None:
        nN        = np.min((N1,N2)) #find common minimum number of neurons recorded

    if prePCA and n_components>prePCA:
        n_components = prePCA

    if nK is None:
        nK        = np.floor(K/kFold  * (kFold-1)).astype('int') #find common minimum number of neurons recorded
        # nK        = K #find common minimum number of neurons recorded

    corr_train  = np.full((n_components,resamples,kFold),np.nan)
    corr_test   = np.full((n_components,resamples,kFold),np.nan)

    for iRS in np.arange(resamples):
        randtimepoints = np.random.choice(K,nK,replace=False)
        X = DATA1[np.ix_(randtimepoints,np.random.choice(N1,nN,replace=False))]
        Y = DATA2[np.ix_(randtimepoints,np.random.choice(N2,nN,replace=False))]

        X = zscore(X,axis=0)  #Z score activity for each neuron
        Y = zscore(Y,axis=0)

        if prePCA and nN>prePCA:
            pca         = PCA(n_components=prePCA)
            X           = pca.fit_transform(X)
            Y           = pca.fit_transform(Y)

        model = CCA(n_components = n_components,scale = False, max_iter = 1000)

        #Implementing cross validation
        kf  = KFold(n_splits=kFold, random_state=None,shuffle=True)
        
        # for train_index, test_index in kf.split(X):
        for ikf,(train_index, test_index) in enumerate(kf.split(X)):
            X_train , X_test = X[train_index,:],X[test_index,:]
            Y_train , Y_test = Y[train_index,:],Y[test_index,:]
            
            model.fit(X_train,Y_train)

            # Compute and store canonical correlations for each pair
            X_c, Y_c = model.transform(X_train,Y_train)
            for icomp in range(n_components):
                corr_train[icomp,iRS,ikf] = np.corrcoef(X_c[:,icomp],Y_c[:,icomp], rowvar = False)[0,1]

            X_c, Y_c = model.transform(X_test,Y_test)
            for icomp in range(n_components):
                corr_test[icomp,iRS,ikf] = np.corrcoef(X_c[:,icomp],Y_c[:,icomp], rowvar = False)[0,1]
        
    corr_train  = np.nanmean(corr_train,axis=(1,2))
    corr_test   = np.nanmean(corr_test,axis=(1,2))

    return corr_test,corr_train

def CCA_subsample_1dim(DATA1,DATA2,nN=None,nK=None,resamples=5,kFold=5,prePCA=None):
    # Data format: 
    #  DATA1 is the source data (number of number of trials x source neurons)
    #  DATA2 is the target data (number of number of trials x source neurons)
    K,N1 = np.shape(DATA1)
    N2 = np.shape(DATA2)[1]
    
    if nN is None:
        nN        = np.min((N1,N2)) #find common minimum number of neurons recorded

    if nK is None:
        nK        = np.floor(K/kFold  * (kFold-1)).astype('int') #find common minimum number of neurons recorded

    corr_train = []
    corr_test = []
   
    for iRS in np.arange(resamples):
        # X = DATA1[np.ix_(np.random.choice(N1,nN,replace=False),np.random.choice(K,nK,replace=False))]
        # Y = DATA2[np.ix_(np.random.choice(N2,nN,replace=False),np.random.choice(K,nK,replace=False))]

        randtimepoints = np.random.choice(K,nK,replace=False)
        X = DATA1[np.ix_(randtimepoints,np.random.choice(N1,nN,replace=False))]
        Y = DATA2[np.ix_(randtimepoints,np.random.choice(N2,nN,replace=False))]

        X = zscore(X,axis=1)  #Z score activity for each neuron
        Y = zscore(Y,axis=1)

        if prePCA and nN>prePCA:
            pca         = PCA(n_components=prePCA)
            X           = pca.fit_transform(X)
            Y           = pca.fit_transform(Y)

        model = CCA(n_components = 1,scale = False, max_iter = 1000)

        #Implementing cross validation
        kf  = KFold(n_splits=kFold, random_state=None,shuffle=True)
        
        for train_index, test_index in kf.split(X):
            X_train , X_test = X[train_index,:],X[test_index,:]
            Y_train , Y_test = Y[train_index,:],Y[test_index,:]
            
            model.fit(X_train,Y_train)

            # Compute and store canonical correlations for the first pair
            X_c, Y_c = model.transform(X_train,Y_train)
            corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
            corr_train.append(corr)

            X_c, Y_c = model.transform(X_test,Y_test)
            corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
            corr_test.append(corr)
        
    corr_train  = np.mean(corr_train)
    corr_test   = np.mean(corr_test)

    return corr_test,corr_train

def CCA_subsample_it(DATA1,DATA2,nN=None,nK=None,resamples=5,kFold=5,prePCA=None,n_components=1):
    # Data format: 
    #  DATA1 is the source data (number of number of trials x source neurons)
    #  DATA2 is the target data (number of number of trials x source neurons)
    K,N1 = np.shape(DATA1)
    N2 = np.shape(DATA2)[1]
    
    if nN is None:
        nN        = np.min((N1,N2)) #find common minimum number of neurons recorded

    if prePCA and n_components>prePCA:
        n_components = prePCA

    if nK is None:
        nK        = np.floor(K/kFold  * (kFold-1)).astype('int') #find common minimum number of neurons recorded
        # nK        = K #find common minimum number of neurons recorded

    corr_train  = np.full((n_components,resamples,kFold),np.nan)
    corr_test   = np.full((n_components,resamples,kFold),np.nan)
    
    #1 component only, iteratively fit and remove data along correlated dimension
    model = CCA(n_components = 1,scale = False, max_iter = 1000)

    for iRS in np.arange(resamples):
        randtimepoints = np.random.choice(K,nK,replace=False)
        X = DATA1[np.ix_(randtimepoints,np.random.choice(N1,nN,replace=False))]
        Y = DATA2[np.ix_(randtimepoints,np.random.choice(N2,nN,replace=False))]

        X = zscore(X,axis=0)  #Z score activity for each neuron
        Y = zscore(Y,axis=0)

        if prePCA and nN>prePCA:
            pca         = PCA(n_components=prePCA)
            X           = pca.fit_transform(X)
            Y           = pca.fit_transform(Y)

        #Implementing cross validation
        kf  = KFold(n_splits=kFold, random_state=None,shuffle=True)
        
        # for train_index, test_index in kf.split(X):
        for ikf,(train_index, test_index) in enumerate(kf.split(X)):
            X_train , X_test = X[train_index,:],X[test_index,:]
            Y_train , Y_test = Y[train_index,:],Y[test_index,:]
            
            # X_train_res, X_test_res = X_train.copy(),Y_train.copy()
            # Y_train_res, Y_test_res = Y_train.copy(),Y_train.copy()

            for icomp in range(n_components):

                model.fit(X_train,Y_train)

                # Compute and store canonical correlations for each pair
                X_c, Y_c = model.transform(X_train,Y_train)
                corr_train[icomp,iRS,ikf] = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]

                X_c, Y_c = model.transform(X_test,Y_test)
                corr_test[icomp,iRS,ikf] = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]

                # remove fluctuations along the correlated dimension:
                X_train -= X_train @ model.x_weights_ * model.x_weights_.T
                X_test -= X_test @ model.x_weights_ * model.x_weights_.T
                Y_train -= Y_train @ model.y_weights_ * model.y_weights_.T
                Y_test -= X_test @ model.y_weights_ * model.y_weights_.T

                # X_train2 = X_train - X_train @ model.x_weights_ * model.x_weights_.T

                # plt.imshow(X_train,vmin=-1,vmax=2)
                # plt.imshow(X_train2,vmin=-1,vmax=2)
                # plt.imshow(X_train @ model.x_weights_ * model.x_weights_.T,vmin=-1,vmax=1)

    corr_train  = np.nanmean(corr_train,axis=(1,2))
    corr_test   = np.nanmean(corr_test,axis=(1,2))

    return corr_test,corr_train


def CCA_validate_iterative(X,Y,prePCA=None,n_components=20):
    # Data format: 
    #  DATA1 is the source data (number of number of trials x source neurons)
    #  DATA2 is the target data (number of number of trials x source neurons)

    weights_CCA_full    = np.zeros((n_components,np.shape(X)[1]))
    weights_CCA_it      = np.zeros((n_components,np.shape(X)[1]))

    #1 component only, iteratively fit and remove data along correlated dimension
    model_full = CCA(n_components = n_components,scale = False, max_iter = 1000)
    model_it = CCA(n_components = 1,scale = False, max_iter = 1000)

    X = zscore(X,axis=0)  #Z score activity for each neuron
    Y = zscore(Y,axis=0)

    if prePCA:
        pca         = PCA(n_components=prePCA)
        X           = pca.fit_transform(X)
        Y           = pca.fit_transform(Y)

    model_full.fit(X,Y)

    weights_CCA_full    = model_full.x_weights_

    for icomp in range(n_components):

        model_it.fit(X,Y)

        # Compute and store canonical correlations for each pair
        weights_CCA_it[icomp,:]      = model_it.x_weights_[:,0]

        # remove fluctuations along the correlated dimension:
        X -= X @ model_it.x_weights_ * model_it.x_weights_.T
        Y -= Y @ model_it.y_weights_ * model_it.y_weights_.T

    fig,axes = plt.subplots(1,2,figsize=(4,2.5),sharex=True,sharey=True)
    axes[0].imshow(np.corrcoef(weights_CCA_full),vmin=-1,vmax=1,cmap='bwr')
    axes[1].imshow(np.corrcoef(weights_CCA_it),vmin=-1,vmax=1,cmap='bwr')

    return weights_CCA_full,weights_CCA_it


def CCA_subsample_dynamics_1dim(DATA1,DATA2,nN,nK,resamples=5,kFold=5,prePCA=None):
    # Data format: 
    #  DATA1 is the source data (number of source neurons x number of time points x number of trials)
    #  DATA2 is the target data (number of target neurons x number of time points x number of trials)
    N1,T,K = np.shape(DATA1)
    N2 = np.shape(DATA2)[0]
    
    corr_train = []
    corr_test = []
    
    for iRS in np.arange(resamples):
        X = DATA1[np.ix_(np.random.choice(N1,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]
        Y = DATA2[np.ix_(np.random.choice(N2,nN,replace=False),range(T),np.random.choice(K,nK,replace=False))]

        X = np.reshape(X,(nN,-1),order='F').T #concatenate time bins from each trial and transpose (samples by features now)
        Y = np.reshape(Y,(nN,-1),order='F').T 

        X = zscore(X,axis=0)  #Z score activity for each neuron
        Y = zscore(Y,axis=0)

        if prePCA and nN>prePCA:
            pca         = PCA(n_components=prePCA)
            X           = pca.fit_transform(X)
            Y           = pca.fit_transform(Y)

        model = CCA(n_components = 1,scale = True, max_iter = 1000)

        #Implementing cross validation
        kf  = KFold(n_splits=kFold, random_state=None,shuffle=True)
        
        for train_index, test_index in kf.split(np.arange(nK)):
            
            train_index_bins                = np.full((T,nK),False) #init false array
            test_index_bins                 = np.full((T,nK),False)
            train_index_bins[:,train_index] = True #set all time bins of train trials to true
            test_index_bins[:,test_index]   = True
            train_index_bins                = np.reshape(train_index_bins,-1,order='F') #concatenate time bins from each trial and transpose (samples by features now)
            test_index_bins                 = np.reshape(test_index_bins,-1,order='F')

            #  plt.figure()
            #  plt.plot(train_index_bins[:100])
            #  plt.plot(test_index_bins[:100])
            X_train , X_test = X[train_index_bins,:],X[test_index_bins,:]
            Y_train , Y_test = Y[train_index_bins,:],Y[test_index_bins,:]
            
            model.fit(X_train,Y_train)

            # Compute and store canonical correlations for the first pair
            X_c, Y_c = model.transform(X_train,Y_train)
            corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
            corr_train.append(corr)

            X_c, Y_c = model.transform(X_test,Y_test)
            corr = np.corrcoef(X_c[:,0],Y_c[:,0], rowvar = False)[0,1]
            corr_test.append(corr)
        
    corr_train  = np.mean(corr_train)
    corr_test   = np.mean(corr_test)

    return corr_test,corr_train
