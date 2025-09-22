# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')
from loaddata.get_data_folder import get_local_drive
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import zscore, ttest_rel, ttest_ind

from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning
from utils.plot_lib import * #get all the fixed color schemes
from utils.CCAlib import *
from utils.corr_lib import *
from utils.tuning import compute_tuning_wrapper
from utils.regress_lib import *

# savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\SubspaceOverlap')


#%% 
areas = ['V1','PM','AL','RSP']
nareas = len(areas)

#%% Load example sessions:
session_list        = np.array(['LPE12385_2024_06_13','LPE11998_2024_05_02']) #GN
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list)

# %% 
# sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)
# sessions,nSessions   = filter_sessions(protocols = 'GN',only_all_areas=areas,filter_areas=areas)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=areas,filter_areas=areas)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_areas=areas)

#%%  Load data properly:        
# calciumversion = 'deconv'
calciumversion = 'dF'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)

#%% Cross area subspace difference as a function of dimensionality: 
def cross_area_subspace_wrapper(sessions,arealabels,popsize,maxnoiselevel=20,nmodelfits=10,kfold=5,lam=0,nranks=25,filter_nearby=False):
    """
    Computes the cross area subspace difference as a function of dimensionality.

    Parameters:
    - sessions: list of session objects
    - arealabels: list of area labels
    - popsize: int, number of neurons in a population
    - maxnoiselevel: int, maximum noise level to include in the data
    - nmodelfits: int, number of model fits to do
    - kfold: int, number of folds for cross-validation
    - lam: float, regularization strength for linear regression
    - nranks: int, number of subspace dimensions to test
    - filter_nearby: bool, if True, only includes cells in the data that are close to labeled cells in the same area

    Returns:
    - same_pop_R2, cross_pop_R2: 6D arrays of shape (nranks,nSessions,narealabels,narealabels,narealabels,nmodelfits,kfold)
        cross_pop_R2[i,j,k,m,n,o,p] is the R2 of predicting area n using the predictive subspace of area k to m, 
        while using the i-th rank subspace of area k. This is for session j, resampling o and kfold p
        same_pop_R2[i,j,k,m,n,o,p] is the same but area n is ignored, the test performance on held out data 
        is reported for each area combination k to m
    """

    narealabels         = len(arealabels)
    same_pop_R2         = np.full((nranks,nSessions,narealabels,narealabels,narealabels,nmodelfits,kfold),np.nan)
    cross_pop_R2        = np.full((nranks,nSessions,narealabels,narealabels,narealabels,nmodelfits,kfold),np.nan)

    kf                  = KFold(n_splits=kfold,shuffle=True,random_state=None)
    for ises,ses in tqdm(enumerate(sessions),desc='Cross subspace decoding: ',total=nSessions):
        idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
        K                   = np.sum(idx_T)
        resp                = zscore(sessions[ises].respmat[:,idx_T].T,axis=0,nan_policy='omit')

        if filter_nearby:
            nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
        else: 
            nearfilter      = np.ones(len(sessions[ises].celldata)).astype(bool)

        for imf in range(nmodelfits):
            for iarea in range(narealabels):
                idx_N_i           = np.where(np.all((sessions[ises].celldata['arealabel']==arealabels[iarea],
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,nearfilter	
                                        ),axis=0))[0]
                for jarea in range(narealabels):
                    idx_N_j           = np.where(np.all((sessions[ises].celldata['arealabel']==arealabels[jarea],
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,nearfilter	
                                        ),axis=0))[0]
                    for karea in range(narealabels):
                        idx_N_k           = np.where(np.all((sessions[ises].celldata['arealabel']==arealabels[karea],
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,nearfilter	
                                        ),axis=0))[0]

                        #SUBSAMPLE NEURONS FROM AREAS: SKIP LOOP IF NOT ENOUGH NEURONS
                        if len(idx_N_i)<popsize:
                            continue
                        idx_N_i_sub         = np.random.choice(idx_N_i,popsize,replace=False) #take random subset of neurons
                        
                        if len(np.setdiff1d(idx_N_j,idx_N_i_sub))<popsize:
                            continue
                        idx_N_j_sub         = np.random.choice(np.setdiff1d(idx_N_j,idx_N_i_sub),popsize,replace=False) #take random subset of neurons
                        
                        # if len(np.setdiff1d(idx_N_j,np.concatenate((idx_N_i_sub,idx_N_j_sub))))<popsize:
                        #     continue
                        # idx_N_j_sub2         = np.random.choice(np.setdiff1d(idx_N_j,idx_N_i_sub),popsize,replace=False) #take random subset of neurons
                        
                        if len(np.setdiff1d(idx_N_k,np.concatenate((idx_N_i_sub,idx_N_j_sub))))<popsize:
                            continue
                        idx_N_k_sub         = np.random.choice(np.setdiff1d(idx_N_k,np.concatenate((idx_N_i_sub,idx_N_j_sub))),popsize,replace=False) #take random subset of neurons
                        
                        # Assert that the number of unique values equals the total number of values
                        all_values = np.concatenate([idx_N_i_sub, idx_N_j_sub, idx_N_k_sub])
                        assert len(np.unique(all_values)) == len(all_values), "Arrays contain overlapping values"

                        X,Y,Z = resp[:,idx_N_i_sub],resp[:,idx_N_j_sub],resp[:,idx_N_k_sub]
                        # X,Y,Z,Y2 = resp[:,idx_N_i_sub],resp[:,idx_N_j_sub],resp[:,idx_N_k_sub],resp[:,idx_N_j_sub2]

                        for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                            X_train, X_test = X[idx_train], X[idx_test]
                            Y_train, Y_test = Y[idx_train], Y[idx_test]
                            Z_train, Z_test = Z[idx_train], Z[idx_test]
                            # Y2_train, Y2_test = Y2[idx_train], Y2[idx_test]

                            B_hat_train         = LM(Y_train,X_train, lam=lam)

                            Y_hat_train         = X_train @ B_hat_train

                            # decomposing and low rank approximation of A
                            # U, s, V = linalg.svd(Y_hat_train, full_matrices=False)
                            U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                            for r in range(nranks):
                                B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                            
                                Y_hat_test_rr   = X_test @ B_rrr

                                same_pop_R2[r,ises,iarea,jarea,karea,imf,ikf] = EV(Y_test,Y_hat_test_rr)
        
                                B_hat           = LM(Z_train,X_train @ B_rrr, lam=lam)
        
                                Z_hat_test_rr   = X_test @ B_rrr @ B_hat
                                
                                cross_pop_R2[r,ises,iarea,jarea,karea,imf,ikf] = EV(Z_test,Z_hat_test_rr)

                                # B_hat           = LM(Y2_train,X_train @ B_rrr, lam=lam)
        
                                # Y2_hat_test_rr   = X_test @ B_rrr @ B_hat
                                
                                # subspace_R2[iarea,jarea,karea,2,r,ises,imf,ikf] = EV(Y2_test,Y2_hat_test_rr)
    return same_pop_R2, cross_pop_R2

#%% Cross area subspace difference as a function of dimensionality: 

popsize         = 100
nmodelfits      = 5
kfold           = 2
lam             = 0
nranks          = 50
maxnoiselevel   = 20

# Init output arrays:
subspace_R2         = np.full((3,nranks,nSessions,nmodelfits,kfold),np.nan)

areaX               = 'V1unl'
areaY               = 'PMunl'
areaZ               = 'ALunl'
narealabels         = 1

kf                  = KFold(n_splits=kfold,shuffle=True,random_state=None)
for ises,ses in tqdm(enumerate(sessions),desc='Cross subspace decoding: ',total=nSessions):
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    K                   = np.sum(idx_T)
    resp                = zscore(sessions[ises].respmat[:,idx_T].T,axis=0,nan_policy='omit')

    for imf in range(nmodelfits):
        for iarea in range(narealabels):
            idx_N_i           = np.where(np.all((sessions[ises].celldata['arealabel']==areaX,
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,	
                                    ),axis=0))[0]
            for jarea in range(narealabels):
                idx_N_j           = np.where(np.all((sessions[ises].celldata['arealabel']==areaY,
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,	
                                    ),axis=0))[0]
                for karea in range(narealabels):
                    idx_N_k           = np.where(np.all((sessions[ises].celldata['arealabel']==areaZ,
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,	
                                    ),axis=0))[0]
                    
                    #SUBSAMPLE NEURONS FROM AREAS: SKIP LOOP IF NOT ENOUGH NEURONS
                    if len(idx_N_i)<popsize:
                        continue
                    idx_N_i_sub         = np.random.choice(idx_N_i,popsize,replace=False) #take random subset of neurons
                    if len(np.setdiff1d(idx_N_j,idx_N_i_sub))<popsize:
                        continue
                    idx_N_j_sub         = np.random.choice(np.setdiff1d(idx_N_j,idx_N_i_sub),popsize,replace=False) #take random subset of neurons
                    
                    if len(np.setdiff1d(idx_N_j,np.concatenate((idx_N_i_sub,idx_N_j_sub))))<popsize:
                        continue
                    # idx_N_j_sub2         = np.random.choice(np.setdiff1d(idx_N_j,idx_N_i_sub,idx_N_j_sub),popsize,replace=False) #take random subset of neurons
                    idx_N_j_sub2         = np.random.choice(np.setdiff1d(idx_N_j,np.concatenate((idx_N_i_sub,idx_N_j_sub))),popsize,replace=False) #take random subset of neurons
                    
                    if len(np.setdiff1d(idx_N_k,np.concatenate((idx_N_i_sub,idx_N_j_sub))))<popsize:
                        continue
                    idx_N_k_sub         = np.random.choice(np.setdiff1d(idx_N_k,np.concatenate((idx_N_i_sub,idx_N_j_sub,idx_N_j_sub2))),popsize,replace=False) #take random subset of neurons
                    
                    # Assert that the number of unique values equals the total number of values
                    all_values = np.concatenate([idx_N_i_sub, idx_N_j_sub, idx_N_k_sub, idx_N_j_sub2])
                    assert len(np.unique(all_values)) == len(all_values), "Arrays contain overlapping values"

                    X,Y,Z,Y2 = resp[:,idx_N_i_sub],resp[:,idx_N_j_sub],resp[:,idx_N_k_sub],resp[:,idx_N_j_sub2]

                    for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                        X_train, X_test = X[idx_train], X[idx_test]
                        Y_train, Y_test = Y[idx_train], Y[idx_test]
                        Z_train, Z_test = Z[idx_train], Z[idx_test]
                        Y2_train, Y2_test = Y2[idx_train], Y2[idx_test]

                        B_hat_train         = LM(Y_train,X_train, lam=lam)

                        Y_hat_train         = X_train @ B_hat_train

                        # decomposing and low rank approximation of A
                        # U, s, V = linalg.svd(Y_hat_train, full_matrices=False)
                        U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                        U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                        for r in range(nranks):
                            B_rrr               = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                           
                            Y_hat_test_rr   = X_test @ B_rrr

                            subspace_R2[0,r,ises,imf,ikf] = EV(Y_test,Y_hat_test_rr)
    
                            B_hat           = LM(Z_train,X_train @ B_rrr, lam=lam)
    
                            Z_hat_test_rr   = X_test @ B_rrr @ B_hat
                            
                            subspace_R2[1,r,ises,imf,ikf] = EV(Z_test,Z_hat_test_rr)

                            B_hat           = LM(Y2_train,X_train @ B_rrr, lam=lam)
    
                            Y2_hat_test_rr   = X_test @ B_rrr @ B_hat
                            
                            subspace_R2[2,r,ises,imf,ikf] = EV(Y2_test,Y2_hat_test_rr)

#%% Show results: 
fig,axes = plt.subplots(1,1,figsize=(4,3))
handles = []
ax = axes
meantoplot = np.nanmean(subspace_R2[0,:,:,:,:],axis=(1,2,3))
errortoplot = np.nanstd(subspace_R2[0,:,:,:,:],axis=(1,2,3)) / np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errortoplot,ax=ax,color='g'))

meantoplot = np.nanmean(subspace_R2[1,:,:,:,:],axis=(1,2,3))
errortoplot = np.nanstd(subspace_R2[1,:,:,:,:],axis=(1,2,3))/ np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errortoplot,ax=ax,color='r'))

meantoplot = np.nanmean(subspace_R2[2,:,:,:,:],axis=(1,2,3))
errortoplot = np.nanstd(subspace_R2[2,:,:,:,:],axis=(1,2,3))/ np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errortoplot,ax=ax,color='b'))

ax.legend(handles,('Source to Target (Same neurons)','Source to Cross','Source to Target (Diff neurons)'),
          frameon=False,fontsize=9)
ax.set_xticks(range(0,nranks+1,10))
ax.set_xlabel('Rank')
ax.set_ylabel('CV R2')
sns.despine(top=True,right=True,offset=3,trim=True)
my_savefig(fig,savedir,'RRR_cvR2_CrossVsTarget_%dsessions' % nSessions,formats=['png'])







#%% Cross area subspace predictions: 
arealabels      = np.array(['V1unl', 'PMunl', 'ALunl','RSPunl'])
# arealabels      = np.array(['V1unl', 'PMunl', 'ALunl'])
narealabels     = len(arealabels)

popsize         = 100
maxnoiselevel   = 20
nmodelfits      = 1
kfold           = 5
lam             = 0
nranks          = 25

same_pop_R2, cross_pop_R2     = cross_area_subspace_wrapper(sessions,arealabels,popsize,maxnoiselevel,nmodelfits,kfold,lam=0,nranks=nranks)

# print('Fraction of NaN elements in cross_pop_R2: %.2f' % (np.isnan(cross_pop_R2).sum() / cross_pop_R2.size))

# print('Number of cells in each area for all sessions:')
# for ises, ses in enumerate(sessions):
#     for arealabel in arealabels:
#         print('Session %d: %d in %s' % (ises+1,
#                                         np.sum(ses.celldata['arealabel']==arealabel),
#                                         arealabel))

#%% Performance on trained population always higher than on different neurons
rank        = 5

fig,axes    = plt.subplots(1,2,figsize=(6,3),sharey=False,sharex=False)

ax          = axes[0]
datatoplot  = np.nanmean(same_pop_R2,axis=(2,3,4,5,6))
meantoplot  = np.nanmean(datatoplot,axis=(1)) #one line for every session
errortoplot      = np.nanstd(datatoplot,axis=(1)) / np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errortoplot,ax=ax,color='g'))

datatoplot  = np.nanmean(cross_pop_R2,axis=(2,3,4,5,6))
meantoplot  = np.nanmean(datatoplot,axis=1) #one line for every session
errortoplot      = np.nanstd(datatoplot,axis=1) / np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errortoplot,ax=ax,color='r'))

samedata    = np.nanmean(same_pop_R2,axis=(5,6)).reshape(nranks,-1)
crossdata   = np.nanmean(cross_pop_R2,axis=(5,6)).reshape(nranks,-1)

for r in range(nranks):
    x = samedata[r,:]
    y = crossdata[r,:]
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    t,p = ttest_rel(x[~nas], y[~nas])
    # print('Paired t-test: p=%.3f' % (p))
    if p<0.05:
        ax.text(r,ax.get_ylim()[1]-0.01,'*',fontsize=13,ha='center',va='top') #,color='r'xtoplot,ytoplot,pos=[0.8,0.1])
# ax.axvline(rank,c='k',linestyle='--')
ax.legend(handles,('Same neurons','Diff neurons'),
          frameon=False,fontsize=9)
ax.set_xticks(range(0,nranks+1,5))
ax.set_xlabel('Rank')
ax.set_ylabel('CV R2')

ax = axes[1]
ytoplot     = np.nanmean(cross_pop_R2[rank],axis=(3,4)).flatten() #one point for every session
xtoplot     = np.nanmean(same_pop_R2[rank],axis=(3,4)).flatten()
ax.scatter(xtoplot,ytoplot,c='k',s=7,alpha=0.4)
ax.set_xlim([0,0.35])
ax.set_ylim([0,0.35])
add_paired_ttest_results(ax,xtoplot,ytoplot,pos=[0.8,0.1])
ax.plot([0,1],[0,1],c='k',linestyle='--')
ax.text(0.3,0.9,'rank = %d' % rank,fontsize=9,ha='right',va='top',transform=ax.transAxes)
ax.set_ylabel('Same neurons')
ax.set_xlabel('Different neurons')
ax.set_xticks([0,0.1,0.2,0.3])
ax.set_yticks([0,0.1,0.2,0.3])
sns.despine(top=True,right=True,offset=3)
fig.tight_layout()
# my_savefig(fig,savedir,'RRR_cvR2_DiffSamePopulations_AreaAverage_%dsessions' % (nSessions))


#%% Performance on cross population prediction versus on neurons from the same area:
clrs_conds = ['k','b','r']

# The results for the same population on test trials:
samepopdata = np.nanmean(same_pop_R2,axis=(4,5,6))

# Extract the diagonal entries along the 4th and 5th dimensions, so same area but different neurons
sameareadata = np.nanmean(cross_pop_R2,axis=(5,6))
sameareadata = np.einsum('...ii->...i', sameareadata)

# Extract the off diagonal entries along the 4th and 5th dimensions, so different area, different neurons
diffareadata = np.nanmean(cross_pop_R2,axis=(5,6))
mask = np.ones((4, 4), dtype=bool)
np.fill_diagonal(mask, False)
diffareadata = np.einsum('...ij,ij->...ij', diffareadata, mask)

fig,axes    = plt.subplots(1,1,figsize=(3.5,3),sharey=False,sharex=False)
ax          = axes
handles     = []
datatoplot  = np.nanmean(samepopdata,axis=(2,3))
meantoplot  = np.nanmean(datatoplot,axis=(1)) #one line for every session
errortoplot      = np.nanstd(datatoplot,axis=(1)) / np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errortoplot,ax=ax,color=clrs_conds[0]))

datatoplot  = np.nanmean(sameareadata,axis=(2,3))
meantoplot  = np.nanmean(datatoplot,axis=(1)) #one line for every session
errortoplot      = np.nanstd(datatoplot,axis=(1)) / np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errortoplot,ax=ax,color=clrs_conds[1]))

datatoplot  = np.nanmean(diffareadata,axis=(2,3,4))
meantoplot  = np.nanmean(datatoplot,axis=1) #one line for every session
errortoplot      = np.nanstd(datatoplot,axis=1) / np.sqrt(nSessions)
handles.append(shaded_error(range(nranks),meantoplot,errortoplot,ax=ax,color=clrs_conds[2]))

testdata_samepop    = samepopdata.reshape(nranks,-1)   
testdata_samearea   = sameareadata.reshape(nranks,-1)
testdata_crossarea  = diffareadata.reshape(nranks,-1)

ax.set_ylim([0,0.3])
for r in range(nranks):
    x = testdata_samepop[r,:]
    y = testdata_samearea[r,:]
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    t,p = ttest_rel(x[~nas], y[~nas])
    # print('Paired t-test: p=%.3f' % (p))
    if p<0.05:
        ax.text(r,ax.get_ylim()[1]-0.01,'*',weight='bold',c=clrs_conds[1],fontsize=13,ha='center',va='top') #,color='r'xtoplot,ytoplot,pos=[0.8,0.1])

for r in range(nranks):
    x = testdata_samearea[r,:]
    y = testdata_crossarea[r,:]
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    t,p = ttest_ind(x, y)
    # print('Paired t-test: p=%.3f' % (p))
    if p<0.05:
        ax.text(r,ax.get_ylim()[1]-0.02,'*',weight='bold',c=clrs_conds[2],fontsize=13,ha='center',va='top') #,color='r'xtoplot,ytoplot,pos=[0.8,0.1])

# ax.axvline(rank,c='k',linestyle='--')
ax.legend(handles,('Same pop.','Same area (diff pop.)','Diff area (diff pop.)'),
          frameon=False,fontsize=9,loc='lower right')
ax.set_xticks(range(0,nranks+1,5))
ax.set_xlabel('Rank')
ax.set_ylabel('CV R2')

sns.despine(top=True,right=True,offset=3)
fig.tight_layout()

my_savefig(fig,savedir,'CrossSubspace_R2_diffpops_%dsessions' % (nSessions))

#%% Performance on cross population prediction versus on neurons from the same area:
# # If above the diagonal then predicting neurons from the same area from specific subspace is better
# clrs_areapairs = get_clr_area_labeled(arealabels)
# rank = 5

# fig,axes = plt.subplots(narealabels,narealabels,figsize=(6,6),sharey=True,sharex=True)
# for iarea in range(narealabels):
#     for jarea in range(narealabels):
#         ax          = axes[iarea,jarea]
#         # karea       = 
#         # ytoplot     = np.nanmean(subspace_R2[iarea,jarea,jarea,1,:,:,:],axis=(1,2)) #one point for every session
#         # xtoplot     = np.nanmean(subspace_R2[iarea,jarea,karea,1,:,:,:],axis=(1,2)) #one point for every session
#         # ax.scatter(xtoplot,ytoplot,c='r',s=10)
#         # if jarea==karea and iarea!=jarea:
#             # ax.scatter(xtoplot,ytoplot,c='b',s=10)
#         # kareastoplot = [k for k in range(narealabels) if k not in (iarea,jarea)]
#         kareastoplot = [k for k in range(narealabels)]
#         for karea in kareastoplot:

#             # xtoplot     = np.nanmean(cross_pop_R2[rank,:,iarea,jarea,iarea,:,:],axis=(1,2)) #one point for every session
#             xtoplot     = np.nanmean(cross_pop_R2[rank,:,iarea,jarea,jarea,:,:],axis=(1,2)) #one point for every session
#             ytoplot     = np.nanmean(cross_pop_R2[rank,:,iarea,jarea,karea,:,:],axis=(1,2)) #one point for every session
#             # xtoplot     = np.nanmean(subspace_R2[iarea,jarea,jarea,1,:,:,:],axis=(1,2)) #one point for every session
#             # ytoplot     = np.nanmean(subspace_R2[iarea,jarea,karea,1,:,:,:],axis=(1,2)) #one point for every session
#             ax.scatter(xtoplot,ytoplot,c=clrs_areapairs[karea],s=10)

#         if iarea==0 and jarea==narealabels-1: 
#             h = ax.legend(arealabels[kareastoplot],loc='lower right',frameon=False,fontsize=6)
#             for t,hnd in zip(h.get_texts(),h.legendHandles):
#                 t.set_color(hnd.get_facecolor())
#                 hnd.set_alpha(0)  # Hide the line
#         # add_paired_ttest_results(ax,xtoplot,ytoplot,pos=[0.8,0.1])

#         ax.set_xlim([0,0.4])
#         ax.set_ylim([0,0.4])
#         ax.plot([0,1],[0,1],c='k',linestyle='--')
#         ax.set_title('%s->%s' % (arealabels[iarea],arealabels[jarea]),fontsize=9)
#         # ax.set_ylabel('Same neurons')
#         # ax.set_xlabel('Different neurons')
# # ax.set_xlim([0,np.nanmax(np.nanmean(subspace_R2,axis=(5,6)))*1.1])
# # ax.set_ylim([0,np.nanmax(np.nanmean(subspace_R2,axis=(5,6)))*1.1])
# sns.despine(top=True,right=True,offset=3)
# fig.tight_layout()
# # my_savefig(fig,savedir,'R2_cross_vs_train_diffpops_%dsessions.png' % (nSessions))


#%% 
#          #    ######  ####### #       ####### ######      #####  ####### #       #        #####  
#         # #   #     # #       #       #       #     #    #     # #       #       #       #     # 
#        #   #  #     # #       #       #       #     #    #       #       #       #       #       
#       #     # ######  #####   #       #####   #     #    #       #####   #       #        #####  
#       ####### #     # #       #       #       #     #    #       #       #       #             # 
#       #     # #     # #       #       #       #     #    #     # #       #       #       #     # 
####### #     # ######  ####### ####### ####### ######      #####  ####### ####### #######  #####  

#%% Cross area subspace prediction with labeled cells:
arealabels      = np.array(['V1unl', 'V1lab','PMunl', 'PMlab', 'ALunl', 'RSPunl'])
narealabels     = len(arealabels)

popsize         = 20
maxnoiselevel   = 20
nmodelfits      = 20
kfold           = 5
lam             = 0
nranks          = 19

same_pop_R2, cross_pop_R2     = cross_area_subspace_wrapper(sessions,arealabels,popsize,maxnoiselevel,nmodelfits,kfold,lam=0,nranks=nranks,filter_nearby=True)

#%% 
print('Number of cells in each area for all sessions:')
for ises, ses in enumerate(sessions):
    for arealabel in arealabels:
        idx_N           = np.where(np.all((sessions[ises].celldata['arealabel']==arealabel,
                                        sessions[ises].celldata['noise_level']<maxnoiselevel,filter_nearlabeled(sessions[ises],radius=50)	
                                        ),axis=0))[0]
        print('Session %d: %d in %s' % (ises+1,
                                        len(idx_N),
                                        arealabel))
print('Fraction of NaN elements in cross_pop_R2 due to insufficent population size combination: %.2f' % (np.isnan(cross_pop_R2).sum() / cross_pop_R2.size))

#%% 
same_pop_optimrank = np.full((nSessions,narealabels,narealabels,narealabels),np.nan)	
same_pop_R2_optimrank = np.full((nSessions,narealabels,narealabels,narealabels),np.nan)	
cross_pop_optimrank = np.full((nSessions,narealabels,narealabels,narealabels),np.nan)	
cross_pop_R2_optimrank = np.full((nSessions,narealabels,narealabels,narealabels),np.nan)

for ises in range(nSessions):
    for iarea in range(narealabels):
        for jarea in range(narealabels):
            for karea in range(narealabels):
                if np.all(np.isnan(cross_pop_R2[:,ises,iarea,jarea,karea,:,:])): continue
                same_pop_R2_optimrank[ises,iarea,jarea,karea], \
                same_pop_optimrank[ises,iarea,jarea,karea] = rank_from_R2(
                    same_pop_R2[:,ises,iarea,jarea,karea,:,:].reshape([nranks,nmodelfits*kfold]),
                    nranks,
                    nmodelfits*kfold
                )
                cross_pop_R2_optimrank[ises,iarea,jarea,karea], \
                cross_pop_optimrank[ises,iarea,jarea,karea] = rank_from_R2(
                    cross_pop_R2[:,ises,iarea,jarea,karea,:,:].reshape([nranks,nmodelfits*kfold]),
                    nranks,
                    nmodelfits*kfold)

print('Optimal same pop rank: mean %.2f, std %.2f' % (np.nanmean(same_pop_optimrank),np.nanstd(same_pop_optimrank)))
avg_rank_samepop = int(np.round(np.nanmean(same_pop_optimrank)))                

print('Optimal cross pop rank: mean %.2f, std %.2f' % (np.nanmean(cross_pop_optimrank),np.nanstd(cross_pop_optimrank)))
avg_rank = int(np.round(np.nanmean(cross_pop_optimrank)))

#%% Compare feedforward and feedback V1 and PM labeled hypotheses:
combpairstoplot = np.empty((2,2),dtype=object)

combpairstoplot[0,0] = [['V1unl','PMunl','PMunl'],
              ['V1unl','PMunl','PMlab'],
              ['V1unl','PMlab','PMlab'],
              ['V1unl','PMunl','PMunl']]

combpairstoplot[0,1] = [['V1lab','PMunl','PMunl'],
              ['V1lab','PMlab','PMlab'],
              ['V1lab','PMunl','PMlab'],
              ['V1lab','PMlab','PMunl']]

combpairstoplot[1,0] = [['PMunl','V1unl','V1lab'],
              ['PMunl','V1lab','V1unl'],
              ['PMunl','V1unl','V1lab'],
              ['PMunl','V1lab','V1unl']
              ]

combpairstoplot[1,1] = [
                ['PMlab','V1unl','V1unl'],
              ['PMlab','V1lab','V1lab'],
              ['PMlab','V1unl','V1lab'],
              ['PMlab','V1lab','V1unl']
              ]

clrs        = sns.color_palette('tab10',4)

fig,axes    = plt.subplots(2,2,figsize=(7,6),sharex=True,sharey=True)
for i in range(2):
    for j in range(2): 
        ax = axes[i,j]
        handles = []
        combpairs = combpairstoplot[i,j]
        for ipair in range(len(combpairs)):
            iarea = arealabels.tolist().index(combpairs[ipair][0])
            jarea = arealabels.tolist().index(combpairs[ipair][1])
            karea = arealabels.tolist().index(combpairs[ipair][2])
            datatoplot  = np.nanmean(cross_pop_R2[:,:,iarea,jarea,karea,:,:],axis=(2,3))
            meantoplot  = np.nanmean(datatoplot,axis=1) #one line for every session
            errortoplot = np.nanstd(datatoplot,axis=1) / np.sqrt(nSessions)#one line for every session
            handles.append(shaded_error(range(nranks),meantoplot,errortoplot,ax=ax,color=clrs[ipair]))
        if i==0:
            ax.set_title('Feedforward')
        elif i==1:
            ax.set_title('Feedback')
            ax.set_xlabel('Rank')
        ax.legend(handles,combpairs,loc='lower right',frameon=False,fontsize=8)
        ax.set_xlim([0,nranks])
        if j==0: 
            ax.set_ylabel('R2')

sns.despine(top=True,right=True,offset=3)
fig.tight_layout()

my_savefig(fig,savedir,'CrossSubspace_R2_V1PM_Labeled_%dsessions' % nSessions,formats=['png'])

#%% Generalization to other cross target populations:
# The idea is to compare the R2 performance when predicting a population that it was not trained on
# Normalize the performance by the performance on the population it was trained on
# If the performance is higher, the model is better at generalizing to a population it was not trained on
# idx 1: source population
# idx 2: target population trained to predict
# idx 3: population to generalize to within the same area

combpairstoplot = np.array([['V1unl','PMunl','PMlab'],
              ['V1lab','PMunl','PMlab'],
              ['PMunl','V1unl','V1lab'],
              ['PMlab','V1unl','V1lab'],
              ['V1unl','PMlab','PMunl'],
              ['V1lab','PMlab','PMunl'],
              ['PMunl','V1lab','V1unl'],
              ['PMlab','V1lab','V1unl']
              ])
ncombpairstoplot = len(combpairstoplot)

idx_unl_to_lab = np.array([0,1,2,3]) #the pairs that generalize from unlabeled to labeled
idx_lab_to_unl = np.array([4,5,6,7]) #vice versa

#%% Make a generalization index: 
# Divide the performance by the performance on a different population, but from the same type it was trained on
# rankversion = 'optimrank'
rankversion = 'avgrank'

if rankversion == 'optimrank':
    cross_R2_norm      = copy.deepcopy(cross_pop_R2_optimrank)
elif rankversion == 'avgrank':
    cross_R2_norm      = np.nanmean(cross_pop_R2[avg_rank],axis=(-1,-2))

for iarea in range(narealabels):
    for jarea in range(narealabels):
        cross_R2_norm[:,iarea,jarea,:] = cross_R2_norm[:,iarea,jarea,:] / cross_R2_norm[:,iarea,jarea,jarea][:,np.newaxis]

#%% Plot the results
clrs        = sns.color_palette('tab10',8)

sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":8})   

fig,axes = plt.subplots(2,4,figsize=(5,4),sharex=False,sharey=True)

axes = axes.flatten()

for ipair in range(len(combpairstoplot)):
    ax = axes[ipair]
    iarea = arealabels.tolist().index(combpairstoplot[ipair][0])
    jarea = arealabels.tolist().index(combpairstoplot[ipair][1])
    # karea = arealabels.tolist().index(combpairstoplot[ipair][2])
    kareas = [arealabels.tolist().index(x) for x in combpairstoplot[ipair,[1,2]]]
    # kareas = np.sort([arealabels.tolist().index(x) for x in combpairstoplot[ipair,[1,2]]])

    datatoplot      = cross_R2_norm[:,iarea,jarea,kareas]
    meantoplot      = np.nanmean(datatoplot,axis=0) #one line for every session
    errortoplot     = np.nanstd(datatoplot,axis=0)# / np.sqrt(nSessions)#one line for every session
    for ial in range(2):
        ax.scatter(np.ones(nSessions)*ial-0.2,datatoplot[:,ial],8,color='k',alpha=0.5)
    ax.plot(datatoplot.T,linewidth=0.5,color='k',alpha=0.5)
    ax.errorbar(range(2),meantoplot,errortoplot,color=clrs[ipair],
                alpha=1,marker='o',markersize=8,linestyle='none')
    if ipair == 0:
        ax.set_ylabel('Subspace Generalization index')

    # datatoplot      = cross_R2_norm[:,iarea,jarea,:]
    # meantoplot      = np.nanmean(datatoplot,axis=0) #one line for every session
    # errortoplot     = np.nanstd(datatoplot,axis=0)# / np.sqrt(nSessions)#one line for every session
    # for ial in range(len(arealabels)):
    #     ax.scatter(np.ones(nSessions)*ial-0.2,datatoplot[:,ial],8,color='k',alpha=0.5)
    # ax.errorbar(range(narealabels),meantoplot,errortoplot,color=clrs[ipair],
    #             alpha=1,marker='o',markersize=8,linestyle='none')
    
    ax.set_title('->'.join(combpairstoplot[ipair][:2]),fontsize=9)
    # ax.set_xticks(range(narealabels),arealabels,rotation=45,fontsize=6)
    ax.set_xticks(range(2),arealabels[kareas],rotation=0,fontsize=6)
    ax.set_ylim([0,2])
    ax.axhline(1,linestyle='--',color='k',alpha=0.5)

sns.despine(top=True,right=True,offset=3)
fig.tight_layout()
my_savefig(fig,savedir,'Generalization_V1PM_AllCross_%dsessions_%s' % (nSessions,rankversion))

#%% Test generalization across multiple combinations of feedforward and feedback neurons: 
data_generalization = np.full((nSessions,len(combpairstoplot)),np.nan)
for ipair,pair in enumerate(combpairstoplot):
    iarea = arealabels.tolist().index(pair[0])
    jarea = arealabels.tolist().index(pair[1])
    karea = arealabels.tolist().index(pair[2])
    data_generalization[:,ipair] = cross_R2_norm[:,iarea,jarea,karea]

fig,axes = plt.subplots(1,1,figsize=(1.6,3),sharex=True,sharey=True)
ax = axes
ax.scatter(np.zeros(nSessions*len(idx_unl_to_lab)),data_generalization[:,idx_unl_to_lab].flatten(),8,color='k',alpha=0.5)
ax.errorbar(0.2,np.nanmean(data_generalization[:,idx_unl_to_lab]),
            np.nanstd(data_generalization[:,idx_unl_to_lab]) / np.sqrt(nSessions),
            color='b',marker='o',alpha=1,markerfacecolor='none',markersize=8)

ax.scatter(np.ones(nSessions*len(idx_lab_to_unl)),data_generalization[:,idx_lab_to_unl].flatten(),8,color='k',alpha=0.5)
ax.errorbar(1.2,np.nanmean(data_generalization[:,idx_lab_to_unl]),
            np.nanstd(data_generalization[:,idx_lab_to_unl]) / np.sqrt(nSessions),
            markerfacecolor='none',color='g',marker='o',alpha=1,markersize=8)

#Stats:
xdata = data_generalization[:,idx_unl_to_lab].flatten()
ydata = data_generalization[:,idx_lab_to_unl].flatten()

mask = ~np.isnan(xdata) & ~np.isnan(ydata) #filter either nan in xdata or ydata
xdata = xdata[mask]
ydata = ydata[mask]
T,p  = ttest_rel(xdata,ydata)
add_stat_annotation(ax, 0, 1, 1.85, p, h=0.05)

#Make up:
ax.set_ylabel('Subspace Generalization index')
# ax.set_title('Subspace Generalization index')
ax.set_xticks([0,1],['Unl->\nLab','Lab->\nUnl'],rotation=0,fontsize=8)
ax.set_ylim([0,2])
ax.set_yticks([0,0.5,1,1.5,2],['0','0.5','1.0','1.5','2.0'],fontsize=9)
ax.set_xlim([-0.5,1.5])
ax.axhline(1,linestyle='--',color='k',alpha=0.5)
sns.despine(top=True,right=True,offset=3,trim=True)

my_savefig(fig,savedir,'GeneralizationIndex_V1PM_Cross_%dsessions_%s' % (nSessions,rankversion))


#%% Define function to plot these scatters with paired comparisons
def scatter_cross_pop(ax,arealabels,data,combpairstoplot,color='k'):

    idx_x = [arealabels.tolist().index(x) for x in combpairstoplot[0]]
    xdata = data[:,idx_x[0],idx_x[1],idx_x[2]]
    idx_y = [arealabels.tolist().index(x) for x in combpairstoplot[1]]
    ydata = data[:,idx_y[0],idx_y[1],idx_y[2]]
    # np.clip(xdata,0,0.2,out=xdata)
    # np.clip(ydata,0,0.2,out=ydata)
    h = ax.scatter(xdata,ydata,11,color=color,alpha=0.5)
    ax.plot([0,1],[0,1],linestyle='--',color='k',alpha=0.5)
    ax.set_xlim([0,my_ceil(np.nanmax(np.concatenate([xdata,ydata])),1)])
    ax.set_ylim([0,my_ceil(np.nanmax(np.concatenate([xdata,ydata])),1)])
    # ax.set_xlim([0,0.21])
    # ax.set_ylim([0,0.21])
    ax_nticks(ax,3)
    add_paired_ttest_results(ax, xdata,ydata,pos=[0.8,0.1],fontsize=10)

    return h
    
def plot_paired_combpairs_ALRSP(ax,arealabels,data,combpairstoplot1,combpairstoplot2,
                                combpairstoplot3,combpairstoplot4):

    fig,axes = plt.subplots(1,2,figsize=(5,2.8),sharex=True,sharey=True)
    ax = axes[0]

    idx_x = [arealabels.tolist().index(x) for x in combpairstoplot1[0]]
    xdata1 = data[:,idx_x[0],idx_x[1],idx_x[2]]
    idx_y = [arealabels.tolist().index(x) for x in combpairstoplot1[1]]
    ydata1 = data[:,idx_y[0],idx_y[1],idx_y[2]]
    idx_x = [arealabels.tolist().index(x) for x in combpairstoplot2[0]]
    xdata2 = data[:,idx_x[0],idx_x[1],idx_x[2]]
    idx_y = [arealabels.tolist().index(x) for x in combpairstoplot2[1]]
    ydata2 = data[:,idx_y[0],idx_y[1],idx_y[2]]

    ax.scatter(xdata1,ydata1,11,color='b',alpha=0.5)
    ax.scatter(xdata2,ydata2,11,color='g',alpha=0.5)
    ax.plot([0,1],[0,1],linestyle='--',color='k',alpha=0.5)
    ax_nticks(ax,3)
    add_paired_ttest_results(ax,np.concatenate((xdata1,xdata2)),np.concatenate((ydata1,ydata2)),pos=[0.8,0.1],fontsize=10)

    ax.set_title('Feedforward',fontsize=9)
    ax.legend(labels=['AL','RSP'],frameon=False,fontsize=11,loc='upper left')
    my_legend_strip(ax)

    ax.set_xlabel('%s->%s' % (combpairstoplot1[0][0],combpairstoplot1[0][1]))
    ax.set_ylabel('%s->%s' % (combpairstoplot1[1][0],combpairstoplot1[1][1]))

    ax = axes[1]

    idx_x = [arealabels.tolist().index(x) for x in combpairstoplot3[0]]
    xdata3 = data[:,idx_x[0],idx_x[1],idx_x[2]]
    idx_y = [arealabels.tolist().index(x) for x in combpairstoplot3[1]]
    ydata3 = data[:,idx_y[0],idx_y[1],idx_y[2]]
    idx_x = [arealabels.tolist().index(x) for x in combpairstoplot4[0]]
    xdata4 = data[:,idx_x[0],idx_x[1],idx_x[2]]
    idx_y = [arealabels.tolist().index(x) for x in combpairstoplot4[1]]
    ydata4 = data[:,idx_y[0],idx_y[1],idx_y[2]]

    ax.scatter(xdata3,ydata3,11,color='b',alpha=0.5)
    ax.scatter(xdata4,ydata4,11,color='g',alpha=0.5)
    ax.plot([0,1],[0,1],linestyle='--',color='k',alpha=0.5)
    add_paired_ttest_results(ax,np.concatenate((xdata3,xdata4)),np.concatenate((ydata3,ydata4)),pos=[0.8,0.1],fontsize=10)

    ax.set_title('Feedback',fontsize=9)
    ax.set_xlabel('%s->%s' % (combpairstoplot3[0][0],combpairstoplot3[0][1]))
    ax.set_ylabel('%s->%s' % (combpairstoplot3[1][0],combpairstoplot3[1][1]))
    allmax = np.nanmax(np.concatenate([xdata1,ydata1,xdata2,ydata2,xdata3,ydata3,xdata4,ydata4]))
    ax.set_xlim([0,my_ceil(allmax,1)])
    ax.set_ylim([0,my_ceil(allmax,1)])
    ax.set_xlim([0,0.25])
    ax.set_ylim([0,0.25])
    ax_nticks(ax,5)

    plt.suptitle('Generalization across areas',fontsize=9)
    sns.despine(top=True,right=True,offset=3,trim=True)
    plt.tight_layout()
    return fig

#%% 
rankversion = 'optimrank'
data    = copy.deepcopy(cross_pop_R2_optimrank)
rankversion = 'avgrank'
data    = np.nanmean(cross_pop_R2[avg_rank],axis=(-1,-2))

#%% Generalization to other target populations:
# The idea is to compare the R2 performance when predicting neurons from the same populations
# but that it was not trained on
# Are labeled neurons better in generalizing?
# If the performance is higher, generalizing to a population it was not trained on is better
# idx 1: source population
# idx 2: target population trained to predict
# idx 3: population to generalize to (same as idx 2)

fig,axes = plt.subplots(1,2,figsize=(5,2.8),sharex=True,sharey=True)
ax = axes[0]
combpairstoplot = np.array([['V1unl','PMunl','PMunl'],
                            ['V1lab','PMunl','PMunl']])
scatter_cross_pop(ax,arealabels,data,combpairstoplot,color='b')
ax.set_title('Feedforward',fontsize=9)
ax.set_xlabel('%s->%s' % (combpairstoplot[0][0],combpairstoplot[0][1]))
ax.set_ylabel('%s->%s' % (combpairstoplot[1][0],combpairstoplot[1][1]))

ax = axes[1]
combpairstoplot = np.array([['PMunl','V1unl','V1unl'],
                            ['PMlab','V1unl','V1unl']])
scatter_cross_pop(ax,arealabels,data,combpairstoplot,color='g')
ax.set_title('Feedback',fontsize=9)
ax.set_xlabel('%s->%s' % (combpairstoplot[0][0],combpairstoplot[0][1]))
ax.set_ylabel('%s->%s' % (combpairstoplot[1][0],combpairstoplot[1][1]))
  
plt.suptitle('Generalization within area',fontsize=9)
sns.despine(top=True,right=True,offset=3,trim=True)
plt.tight_layout()
my_savefig(fig,savedir,'Generalization_V1PM_Labeled_Same_%dsessions_%s' % (nSessions,rankversion),formats=['png'])


#%% Generalization to other target populations:
# The idea is to compare the R2 performance when predicting neurons from the same populations
# but that it was not trained on
# Are labeled neurons better in generalizing?

combpairstoplot1 = np.array([['V1unl','PMunl','ALunl'],
                            ['V1lab','PMunl','ALunl']])
combpairstoplot2 = np.array([['V1unl','PMunl','RSPunl'],
                            ['V1lab','PMunl','RSPunl']])

combpairstoplot3 = np.array([['PMunl','V1unl','ALunl'],
                            ['PMlab','V1unl','ALunl']])
combpairstoplot4 = np.array([['PMunl','V1unl','RSPunl'],
                            ['PMlab','V1unl','RSPunl']])

fig = plot_paired_combpairs_ALRSP(ax,arealabels,data,combpairstoplot1,combpairstoplot2,
                                combpairstoplot3,combpairstoplot4)

# my_savefig(fig,savedir,'Generalization_cross_area_labeledV1PM_%dsessions_%s' % (nSessions,rankversion))
my_savefig(fig,savedir,'Generalization_cross_area_labeledSourceV1PM_%dsessions_%s' % (nSessions,rankversion),formats=['png'])


#%% Generalization to other target populations:
# The idea is to compare the R2 performance when predicting neurons from the same populations
# but that it was not trained on
# is predicting labeled neurons better in generalizing?

combpairstoplot1 = np.array([['V1unl','PMunl','ALunl'],
                            ['V1unl','PMlab','ALunl']])
combpairstoplot2 = np.array([['V1unl','PMunl','RSPunl'],
                            ['V1unl','PMlab','RSPunl']])

combpairstoplot3 = np.array([['PMunl','V1unl','ALunl'],
                            ['PMunl','V1lab','ALunl']])
combpairstoplot4 = np.array([['PMunl','V1unl','RSPunl'],
                            ['PMunl','V1lab','RSPunl']])

fig = plot_paired_combpairs_ALRSP(ax,arealabels,data,combpairstoplot1,combpairstoplot2,
                                combpairstoplot3,combpairstoplot4)

my_savefig(fig,savedir,'Generalization_cross_area_labeledTargetV1PM_%dsessions_%s' % (nSessions,rankversion),formats=['png'])


#%% 














#%% Clustering of nodes in 2D space: 

rankversion = 'optimrank'
fig,axes = plt.subplots(2,2,figsize=(6,6))
ax = axes[0,0]
data = np.nanmean(same_pop_R2_optimrank,axis=(0,3)) #average across sessions, and karea cross areas (not used for same pop data)
assert(data.shape[0] == data.shape[1]),'not square matrix'

ax.imshow(data,cmap='viridis',vmin=np.min(data),vmax=np.max(data))
ax.set_xticks(range(len(arealabels)),arealabels,rotation=45,fontsize=6)
ax.set_yticks(range(len(arealabels)),arealabels,fontsize=6)
ax.set_xlabel('Target area')
ax.set_ylabel('Source area')
ax.set_title('Test R2',fontsize=12)

from sklearn.manifold import TSNE
data = zscore(data)
tsne = TSNE(n_components=2,verbose=1,perplexity=3)
data_tsne = tsne.fit_transform(data)

ax = axes[0,1]
ax.scatter(data_tsne[:,0],data_tsne[:,1],25,color='k',alpha=0.5)
for iarea,area in enumerate(arealabels):
    ax.text(data_tsne[iarea,0],data_tsne[iarea,1]+0.3,area,ha='center',va='center',fontsize=10)

ax = axes[1,0]
data = np.nanmean(same_pop_optimrank,axis=(0,3)) #average across sessions, and karea cross areas (not used for same pop data)
ax.imshow(data,cmap='viridis',vmin=np.min(data),vmax=np.max(data))
ax.set_xticks(range(len(arealabels)),arealabels,rotation=45,fontsize=6)
ax.set_yticks(range(len(arealabels)),arealabels,fontsize=6)
ax.set_xlabel('Target area')
ax.set_ylabel('Source area')
ax.set_title('Optimal Rank',fontsize=12)

from sklearn.manifold import TSNE
data = zscore(data)
tsne = TSNE(n_components=2,verbose=1,perplexity=3)
data_tsne = tsne.fit_transform(data)

ax = axes[1,1]
ax.scatter(data_tsne[:,0],data_tsne[:,1],25,color='k',alpha=0.5)
for iarea,area in enumerate(arealabels):
    ax.text(data_tsne[iarea,0],data_tsne[iarea,1]+0.3,area,ha='center',va='center',fontsize=10)

# data = zscore(data)
# pca = PCA(n_components=2)
# pca.fit(data)

# data_pca = pca.transform(data)

# ax = axes[1,1]
# ax.scatter(data_pca[:,0],data_pca[:,1],25,color='k',alpha=0.5)
# for iarea,area in enumerate(arealabels):
#     ax.text(data_pca[iarea,0],data_pca[iarea,1]+0.3,area,ha='center',va='center',fontsize=10)
#     # ax.scatter(X, Y, c=clrs_areas[iarea], marker = '.',s=10, alpha=0.25)
# ax.set_title('Area locations in PC space',fontsize=12)
ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')

sns.despine(top=True,right=True,offset=3,trim=True)
plt.tight_layout()

my_savefig(fig,savedir,'CrossArea_RRR_Subspace_Clustering_%dsessions_%s' % (nSessions,rankversion))
# fig.savefig(os.path.join(savedir,'PCA_TuningCurve_HITMISS_%dsessions.png') % (nSessions), format='png')


#%%  

#%% Using trial averaged or using timepoint fluctuations:

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=['V1','PM','AL'],filter_areas=['V1','PM','AL'])

#%%  Load data properly:        
calciumversion = 'dF'
# calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)

    # [sessions[ises].tensor,t_axis]     = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
    #                              t_pre, t_post, binsize,method='nearby')
    

#%%
print('Number of cells in AL per session:')
for ises in range(nSessions):
    print('%d: %d' % (ises,np.sum(np.all((sessions[ises].celldata['roi_name']=='AL',
                                           sessions[ises].celldata['noise_level']<20),axis=0))))


#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
areacombs  = [['V1','PM','AL'],
              ['V1','AL','PM'],
              ['PM','V1','AL'],
              ['PM','AL','V1'],
              ['AL','V1','PM'],
              ['AL','PM','V1']]

nareacombs     = len(areacombs)

Nsub                = 50
kfold               = 5
nmodelfits          = 10
lam                 = 0

rank                = 5
nstims              = 16

R2_cv               = np.full((nareacombs,2,2,nstims,nSessions,nmodelfits,kfold),np.nan)

kf                  = KFold(n_splits=kfold,shuffle=True,random_state=None)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model'):    # iterate over sessions
    # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
    for istim,stim in enumerate([0]): # loop over orientations 
        # idx_T               = ses.trialdata['stimCond']==stim
        idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
        for icomb, (areax,areay,areaz) in enumerate(areacombs):

            idx_areax           = np.where(np.all((ses.celldata['roi_name']==areax,
                                    ses.celldata['noise_level']<20),axis=0))[0]
            idx_areay           = np.where(np.all((ses.celldata['roi_name']==areay,
                                    ses.celldata['noise_level']<20),axis=0))[0]
            idx_areaz           = np.where(np.all((ses.celldata['roi_name']==areaz,
                                    ses.celldata['noise_level']<20),axis=0))[0]
            
            if len(idx_areax)>=Nsub and len(idx_areay)>=Nsub and len(idx_areaz)>=Nsub:
                for irbh,regress_out_behavior in enumerate([False,True]):
                    respmat                 = ses.respmat[:,idx_T].T
                    
                    if regress_out_behavior:
                        X       = np.stack((sessions[ises].respmat_videome[idx_T],
                        sessions[ises].respmat_runspeed[idx_T],
                        sessions[ises].respmat_pupilarea[idx_T],
                        sessions[ises].respmat_pupilx[idx_T],
                        sessions[ises].respmat_pupily[idx_T]),axis=1)
                        X       = np.column_stack((X,sessions[ises].respmat_videopc[:,idx_T].T))
                        X       = zscore(X,axis=0,nan_policy='omit')

                        si      = SimpleImputer()
                        X       = si.fit_transform(X)

                        respmat,_  = regress_out_behavior_modulation(ses,X,respmat,rank=10,lam=0,perCond=False)

                    for imf in range(nmodelfits):
                        idx_areax_sub       = np.random.choice(idx_areax,Nsub,replace=False)
                        idx_areay_sub       = np.random.choice(idx_areay,Nsub,replace=False)
                        idx_areaz_sub       = np.random.choice(idx_areaz,Nsub,replace=False)
                    
                        X                   = respmat[:,idx_areax_sub]
                        Y                   = respmat[:,idx_areay_sub]
                        Z                   = respmat[:,idx_areaz_sub]
                        
                        X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
                        Y                   = zscore(Y,axis=0)
                        Z                   = zscore(Z,axis=0)

                        for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                            X_train, X_test = X[idx_train], X[idx_test]
                            Y_train, Y_test = Y[idx_train], Y[idx_test]
                            Z_train, Z_test = Z[idx_train], Z[idx_test]

                            B_hat_train         = LM(Y_train,X_train, lam=lam)

                            Y_hat_train         = X_train @ B_hat_train

                            # decomposing and low rank approximation of A
                            U, s, V = linalg.svd(Y_hat_train, full_matrices=False)

                            S = linalg.diagsvd(s,U.shape[0],s.shape[0])

                            # for r in range(nranks):
                            B_rrr               = B_hat_train @ V[:rank,:].T @ V[:rank,:] #project beta coeff into low rank subspace
                                # Y_hat_rr_test       = X_test @ B_rrr #project test data onto low rank predictive subspace
                                # R2_cv_folds[r,ikf] = EV(Y_test,Y_hat_rr_test)

                            Y_hat_test_rr = X_test @ B_rrr

                            # R2_cv[iapl,r,iori,ises,i,ikf] = EV(Y_test,Y_hat_test_rr)
                            R2_cv[icomb,0,irbh,istim,ises,imf,ikf] = EV(Y_test,Y_hat_test_rr)

                            B_hat         = LM(Z_train,X_train @ B_rrr, lam=lam)

                            Z_hat_test_rr = X_test @ B_rrr @ B_hat
                            
                            R2_cv[icomb,1,irbh,istim,ises,imf,ikf] = EV(Z_test,Z_hat_test_rr)

#%% Show the data: R2
# tempdata = np.nanmean(R2_cv,axis=(2,4,5)) #if cross-validated: average across orientations, model samples and kfolds
tempdata = np.nanmean(R2_cv,axis=(2,3,5,6)) #if cross-validated: average across orientations, model samples and kfolds
tempdata = np.nanmean(R2_cv,axis=(3,5,6)) #if cross-validated: average across orientations, model samples and kfolds
fig, axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
clrs_combs = sns.color_palette('colorblind',len(areacombs))

ax = axes[0]

handles = []
for icomb, (areax,areay,areaz) in enumerate(areacombs):
    # ax = axes[np.array(iapl>3,dtype=int)]
    ax.scatter(tempdata[icomb,0,0,:],tempdata[icomb,1,0,:],s=15,color=clrs_combs[icomb],alpha=1)
    # ax.scatter(tempdata[icomb,0],tempdata[icomb,1,0,:],s=15,color=clrs_combs[icomb],alpha=1)
handles = [plt.Line2D([0], [0], marker='o', color='w', label=areacombs[icomb],
                         markerfacecolor=clrs_combs[icomb], markersize=5) for icomb in range(len(areacombs))]
ax.legend(handles=handles, loc='upper left',frameon=False,fontsize=6)
ax.plot([0,0.2],[0,0.2],linestyle='--',color='k',alpha=0.5)
ax.set_ylabel('R2 (XsubY->Z)')
ax.set_xlabel('R2 (X->Y)')

ax = axes[1]
for icomb, (areax,areay,areaz) in enumerate(areacombs):
    # ax.scatter(tempdata[icomb,0,1,:],tempdata[icomb,1,1,:],s=15,color=clrs_combs[icomb],alpha=1)
    ax.scatter(tempdata[icomb,0,1,:],tempdata[icomb,1,1,:],s=15,color=clrs_combs[icomb],alpha=1)

# ax.set_xlim([0,0.2])
# ax.set_ylim([0,0.2])
ax.set_xlabel('R2 (X->Y)')
ax.set_xticks([0,0.1,0.2])
ax.set_yticks([0,0.1,0.2])
ax.plot([0,0.2],[0,0.2],linestyle='--',color='k',alpha=0.5)
sns.despine(top=True,right=True,offset=3)

my_savefig(fig,savedir,'RRR_cvR2_V1PMAL_Cross_RegressBehav_%dsessions' % (nSessions))
# plt.savefig(os.path.join(savedir,'RRR_cvR2_V1PMAL_Cross_RegressBehav_%dsessions' % (nSessions)),
#                         bbox_inches='tight')
