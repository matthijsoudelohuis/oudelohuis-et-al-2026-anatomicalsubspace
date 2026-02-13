# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import zscore,wilcoxon,ttest_rel

from loaddata.session_info import filter_sessions,load_sessions, report_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.plot_lib import * #get all the fixed color schemes
from utils.regress_lib import *
from utils.RRRlib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\Validation\\')


#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,filter_noiselevel=True)

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])
report_sessions(sessions)

#%%  Load data properly:        
calciumversion = 'dF'
for ises in range(nSessions):
    sessions[ises].load_tensor(load_calciumdata=True,calciumversion=calciumversion,keepraw=False)

t_axis = sessions[0].t_axis

# #%%  Load data properly:        
# calciumversion = 'dF'
# # calciumversion = 'deconv'
# for ises in range(nSessions):
#     sessions[ises].load_respmat(calciumversion=calciumversion)

#%% 
####### #     #    #    #     # ######  #       #######    ######     #    #     # #    # 
#        #   #    # #   ##   ## #     # #       #          #     #   # #   ##    # #   #  
#         # #    #   #  # # # # #     # #       #          #     #  #   #  # #   # #  #   
#####      #    #     # #  #  # ######  #       #####      ######  #     # #  #  # ###    
#         # #   ####### #     # #       #       #          #   #   ####### #   # # #  #   
#        #   #  #     # #     # #       #       #          #    #  #     # #    ## #   #  
####### #     # #     # #     # #       ####### #######    #     # #     # #     # #    # 

#%% Show RRR example: 
nranks              = 40
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
nsampleneurons      = 25
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
R2_ranks            = np.full((nranks,nmodelfits,kfold),np.nan)

ses                 = sessions[0]
idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                        ses.celldata['noise_level']<20),axis=0))[0]
idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                        ses.celldata['noise_level']<20),axis=0))[0]

idx_T               = ses.trialdata['stimCond']==0

#on tensor during the response:
X                   = ses.tensor[np.ix_(idx_areax,idx_T,idx_resp)]
Y                   = ses.tensor[np.ix_(idx_areay,idx_T,idx_resp)]

#subtract mean response across trials:
X                   -= np.mean(X,axis=1,keepdims=True)
Y                   -= np.mean(Y,axis=1,keepdims=True)

# reshape to time points x neurons
X                   = X.reshape(len(idx_areax),-1).T
Y                   = Y.reshape(len(idx_areay),-1).T

ev,rank,R2_ranks = RRR_wrapper(Y, X, nN=nsampleneurons,nM=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% 
R2_ranks_r = np.reshape(R2_ranks,(nranks,nmodelfits*kfold))

#%% Plotting, show performance across ranks:
xticks = [0,10,20,30,40]
fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
handles = []
handles.append(shaded_error(range(nranks),R2_ranks_r.T,error='sem',color='k',alpha=0.3,ax=ax))
# ax.text(rank,ev,'|',color='b',fontsize=15,horizontalalignment='center',verticalalignment='bottom')
ax.axhline(y=ev,color='k',linestyle='--',alpha=0.5) #ax.text(rank,ev,'|',color='b',fontsize=15,horizontalalignment='center',verticalalignment='bottom')
ax.axvline(x=rank,color='k',linestyle='--',alpha=0.5) #ax.text(rank,ev,'|',color='b',fontsize=15,horizontalalignment='center',verticalalignment='bottom')
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.set_xticks(xticks)
ax.set_xlabel('Rank')
ax.set_ylabel('Performance ($R^2$)')
sns.despine(fig=fig,top=True,right=True,trim=True)
my_savefig(fig,savedir,'RRR_rank_ev_procedure_%d' % nsampleneurons,formats=['png'])

#%%
from scipy.stats import pearsonr

# Y: shape (n_timepoints, n_neurons, n_trials)
def split_half_reliability(Y):
    n_trials = Y.shape[2]
    idx = np.random.permutation(n_trials)
    half = n_trials // 2
    Y1 = np.mean(Y[:, :, idx[:half]], axis=2)
    Y2 = np.mean(Y[:, :, idx[half:]], axis=2)
    
    # Flatten over time and neurons for population-level reliability
    r, _ = pearsonr(Y1.flatten(), Y2.flatten())
    
    # Spearman–Brown correction
    r_sb = (2 * r) / (1 + r)
    R2_ceiling = r_sb ** 2
    return R2_ceiling

#%%
from scipy.stats import pearsonr

def noise_ceiling_split_half(Y, n_splits=100, random_state=None):
    """
    Estimate noise ceiling (split-half reliability) for neural data.

    Parameters
    ----------
    Y : array, shape (n_neurons, n_trials, n_timepoints)
        Neural activity data.
    n_splits : int, optional
        Number of random trial splits to average over.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    R2_ceiling_mean : float
        Estimated mean noise ceiling (R²) across random splits.
    R2_ceiling_all : np.ndarray
        All R² ceiling estimates from each split.
    """

    rng = np.random.default_rng(random_state)
    n_neurons, n_trials, n_timepoints = Y.shape
    R2_ceiling_all = np.zeros(n_splits)

    for i in range(n_splits):
        # Randomly split trials into two halves
        perm = rng.permutation(n_trials)
        half = n_trials // 2
        idx1, idx2 = perm[:half], perm[half:]
        
        # Average activity across trials within each half
        Y1 = np.mean(Y[:, idx1, :], axis=1)  # shape (n_neurons, n_timepoints)
        Y2 = np.mean(Y[:, idx2, :], axis=1)

        # Flatten across neurons and timepoints for population-level correlation
        r, _ = pearsonr(Y1.flatten(), Y2.flatten())

        # Spearman–Brown correction
        r_sb = (2 * r) / (1 + r) if r < 1 else 1.0
        R2_ceiling_all[i] = r_sb ** 2

    return R2_ceiling_all.mean(), R2_ceiling_all

#%% 
def noise_ceiling_residuals(Y_resid, conds=None, n_splits=200, random_state=None, return_all=False):
    """
    Estimate noise ceiling (R^2) for trial-to-trial residuals.

    Parameters
    ----------
    Y_resid : array_like
        Residual activity, shape (n_neurons, n_trials, n_timepoints).
        These should be single-trial residuals (mean stimulus response removed).
    conds : array_like or None, optional
        Condition labels for each trial, shape (n_trials,). If provided,
        random splits are performed *within each condition* to avoid mixing different stimuli.
    n_splits : int
        Number of random splits to average over.
    random_state : int or None
        RNG seed.
    return_all : bool
        If True, also return array of R^2 estimates for each split.

    Returns
    -------
    R2_ceiling_mean : float
        Mean estimated noise ceiling (R^2) across splits.
    R2_ceiling_all : np.ndarray (optional)
        All R^2 estimates (length n_splits).
    """
    rng = np.random.default_rng(random_state)
    Y = np.asarray(Y_resid)
    n_neurons, n_trials, n_timepoints = Y.shape

    # helper to split indices within a group
    def split_indices(indices):
        perm = rng.permutation(indices)
        half = len(indices) // 2
        # if odd, leave one out (could also assign ceil/ floor)
        return perm[:half], perm[half:half+half]

    R2_vals = np.zeros(n_splits)

    # If no condition labels, treat all trials as one condition
    if conds is None:
        conds = np.zeros(n_trials, dtype=int)

    conds = np.asarray(conds)
    unique_conds = np.unique(conds)

    for s in range(n_splits):
        # collect averaged halves over conditions
        Y1_parts = []
        Y2_parts = []

        for c in unique_conds:
            idx = np.where(conds == c)[0]
            if len(idx) < 2:
                # can't split this condition; skip it
                continue
            idx1, idx2 = split_indices(idx)
            if len(idx1) == 0 or len(idx2) == 0:
                # not enough trials in this condition to form halves; skip
                continue
            # average across trials in each half -> shape (n_neurons, n_timepoints)
            Y1_c = np.mean(Y[:, idx1, :], axis=1)
            Y2_c = np.mean(Y[:, idx2, :], axis=1)
            Y1_parts.append(Y1_c)
            Y2_parts.append(Y2_c)

        if len(Y1_parts) == 0:
            raise ValueError("Not enough trials to split in any condition. Need >=2 trials for at least one condition.")

        # concatenate across conditions along time axis to keep neuron alignment:
        # result shape (n_neurons, n_timepoints * n_conditions_used)
        Y1_cat = np.concatenate(Y1_parts, axis=1)
        Y2_cat = np.concatenate(Y2_parts, axis=1)

        # Flatten over neurons and time to compute population-level correlation
        vec1 = Y1_cat.ravel()
        vec2 = Y2_cat.ravel()

        # if constant vectors, pearsonr will fail; guard:
        if np.std(vec1) == 0 or np.std(vec2) == 0:
            r = 0.0
        else:
            r, _ = pearsonr(vec1, vec2)

        # Spearman-Brown correction (avoid division-by-zero)
        if r >= 1.0:
            r_sb = 1.0
        else:
            r_sb = (2 * r) / (1 + r)

        R2_vals[s] = r_sb ** 2

    if return_all:
        return R2_vals.mean(), R2_vals
    else:
        return R2_vals.mean()
    
#%%

split_half_reliability(Y[:20,:,:])
split_half_reliability(Y[:20,:,:])

noise_ceiling_split_half(Y[:20,:,:],n_splits=2)

Y = Y[:20,:,:]
Y_resid = Y[:20,:,:]

#%%

noise_ceiling_residuals(Y_resid=Y[:20,:,:], conds=None, n_splits=2)

#%% 
   #     #####  ######  #######  #####   #####     ####### ### #     # ####### 
  # #   #     # #     # #     # #     # #     #       #     #  ##   ## #       
 #   #  #       #     # #     # #       #             #     #  # # # # #       
#     # #       ######  #     #  #####   #####        #     #  #  #  # #####   
####### #       #   #   #     #       #       #       #     #  #     # #       
#     # #     # #    #  #     # #     # #     #       #     #  #     # #       
#     #  #####  #     # #######  #####   #####        #    ### #     # ####### 

#%% RRR across time relative to stimulus onset:
lam                 = 0
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 10 #number of times new neurons are resampled - many for final run
kfold               = 5
maxnoiselevel       = 20
nStim               = 8

ntimebins           = len(t_axis)
nsampleneurons      = 100

arealabelpairs      = ['V1-PM','PM-V1']
narealabelpairs     = len(arealabelpairs)
clrs_arealabelpairs = get_clr_area_pairs(arealabelpairs)

R2_cv               = np.full((narealabelpairs,ntimebins,nSessions,nStim),np.nan)
optim_rank          = np.full((narealabelpairs,ntimebins,nSessions,nStim),np.nan)
R2_ranks            = np.full((narealabelpairs,ntimebins,nSessions,nStim,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model across time relative to stimulus onset'):
    for iapl, arealabelpair in enumerate(arealabelpairs):
        ses.trialdata['stimCond'] = np.mod(ses.trialdata['stimCond'],8)
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['roi_name']==alx,
                                ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==aly,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                ),axis=0))[0]

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim

            for ibin in range(ntimebins):
                X       = ses.tensor[np.ix_(idx_areax,idx_T,[ibin])].squeeze().T
                Y       = ses.tensor[np.ix_(idx_areay,idx_T,[ibin])].squeeze().T

                X       = X[~np.isnan(X).any(axis=1),:]
                Y       = Y[~np.isnan(Y).any(axis=1),:]

                R2_cv[iapl,ibin,ises,istim],optim_rank[iapl,ibin,ises,istim],R2_ranks[iapl,ibin,ises,istim,:,:,:]      = RRR_wrapper(
                                                                        Y, X, nN=nsampleneurons,lam=lam,
                                                                       nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))


#%% Average across FF and FB and across folds: 
R2_toplot            = np.nanmean(R2_cv,axis=(0,3))
rank_toplot          = np.nanmean(optim_rank,axis=(0,3))

#%% Plotting, show performance across time:
t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
handles = []
# for iapl, arealabelpair in enumerate(arealabelpairs):
handles.append(shaded_error(t_axis,R2_toplot[:,:].T,error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
# ax.legend(handles=handles,labels=arealabelpairs,loc='best',fontsize=8)
# my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('RRR R2')
sns.despine(fig=fig,top=True,right=True,trim=True)
my_savefig(fig,savedir,'RRR_perf_across_time',formats=['png'])


#%% 
######  ####### ######      #####  ### ####### ####### 
#     # #     # #     #    #     #  #       #  #       
#     # #     # #     #    #        #      #   #       
######  #     # ######      #####   #     #    #####   
#       #     # #                #  #    #     #       
#       #     # #          #     #  #   #      #       
#       ####### #           #####  ### ####### ####### 

#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons
popsizes            = np.array([5,10,20,50,100,200,500])
# popsizes            = np.array([5,10,20])
npopsizes           = len(popsizes)
nranks              = 40
nStim               = 16
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
R2_cv               = np.full((nSessions,nStim,npopsizes,npopsizes),np.nan)
optim_rank          = np.full((nSessions,nStim,npopsizes,npopsizes),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
    for istim,stim in enumerate([0]): # loop over orientations 
        idx_T               = ses.trialdata['stimCond']==stim

        #on tensor during the response:
        X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
        Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
        
        #subtract mean response across trials:
        X                   -= np.mean(X,axis=1,keepdims=True)
        Y                   -= np.mean(Y,axis=1,keepdims=True)

        # reshape to neurons x time points
        X                   = X.reshape(len(idx_areax),-1).T
        Y                   = Y.reshape(len(idx_areay),-1).T

        for ixpop,xpop in enumerate(popsizes):
            for iypop,ypop in enumerate(popsizes):
                if len(idx_areax)>=xpop and len(idx_areay)>=ypop:
                    R2_cv[ises,istim,ixpop,iypop],optim_rank[ises,istim,ixpop,iypop],_ = RRR_wrapper(Y, X, nN=xpop,nM=ypop,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot R2 for different number of V1 and PM neurons
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["r","g"], N=256) 
# cmap=LinearSegmentedColormap.from_list('rp',["green","white","purple"], N=256) 
cmap = sns.color_palette('magma', as_cmap=True)
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
ax = axes[0]
# s = ax.imshow(np.nanmean(R2_cv,axis=(0,1)).T,cmap=cmap,vmin=0,vmax=np.nanmax(np.nanmean(R2_cv,axis=(0,1))),origin='lower')
sns.heatmap(np.nanmean(R2_cv,axis=(0,1)).T,cmap=cmap,vmin=0,vmax=np.nanmax(np.nanmean(R2_cv,axis=(0,1))),annot=True,ax=ax,
            xticklabels=popsizes,yticklabels=popsizes,linewidths=0.1,cbar=False,fmt="1.2f",
            annot_kws={'size': 8},square=True)
ax.set_title('Performance ($R^2$)')
ax.set_xlabel('# source neurons')
ax.set_ylabel('# target neurons')

# Does the dimensionality increase with increasing number of neurons?
ax = axes[1]
sns.heatmap(np.nanmean(optim_rank,axis=(0,1)).T,cmap=cmap,vmin=0,vmax=np.nanmax(np.nanmean(optim_rank,axis=(0,1))),annot=True,ax=ax,
            xticklabels=popsizes,yticklabels=popsizes,linewidths=0.1,cbar=False,fmt="2.0f",
            annot_kws={'size': 8},square=True)
ax.invert_yaxis()
ax.set_xlabel('# source neurons')
ax.set_title('Rank')
sns.despine(top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,savedir,'R2_RRR_Tensor_Rank_PopSize_V1PM_%dsessions' % nSessions,formats=['png'])



#%% 
#     # ### ####### #     # ### #     #    #     #  #####        #     #####  ######  #######  #####   #####  
#  #  #  #     #    #     #  #  ##    #    #     # #     #      # #   #     # #     # #     # #     # #     # 
#  #  #  #     #    #     #  #  # #   #    #     # #           #   #  #       #     # #     # #       #       
#  #  #  #     #    #######  #  #  #  #    #     #  #####     #     # #       ######  #     #  #####   #####  
#  #  #  #     #    #     #  #  #   # #     #   #        #    ####### #       #   #   #     #       #       # 
#  #  #  #     #    #     #  #  #    ##      # #   #     #    #     # #     # #    #  #     # #     # #     # 
 ## ##  ###    #    #     # ### #     #       #     #####     #     #  #####  #     # #######  #####   #####  

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealabelpairs  = ['V1-V1',
                    'PM-PM',
                    'V1-PM',
                    'PM-V1'
                    ]

clrs_arealabelpairs = get_clr_area_pairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

nsampleneurons      = 100
nranks              = 25
nmodelfits          = 50 #number of times new neurons are resampled 
nStim               = 16
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]

R2_cv               = np.full((narealabelpairs,nSessions,nStim),np.nan)
optim_rank          = np.full((narealabelpairs,nSessions,nStim),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    if np.sum((ses.celldata['roi_name']=='V1') & (ses.celldata['noise_level']<20))<(nsampleneurons*2):
        continue
    
    if np.sum((ses.celldata['roi_name']=='PM') & (ses.celldata['noise_level']<20))<(nsampleneurons*2):
        continue

    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['roi_name']==alx,
                                ses.celldata['noise_level']<20
                                ),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==aly,
                                ses.celldata['noise_level']<20
                                ),axis=0))[0]
    
        if np.any(np.intersect1d(idx_areax,idx_areay)): #if interactions within one population:
            if not np.array_equal(idx_areax, idx_areay): 
                print('Arealabelpair %s has partly overlapping neurons'%arealabelpair)
            idx_areax, idx_areay = np.array_split(np.random.permutation(idx_areax), 2)

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        # for istim,stim in enumerate([0]): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim

            #on tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            #subtract mean response across trials:
            X                   -= np.mean(X,axis=1,keepdims=True)
            Y                   -= np.mean(Y,axis=1,keepdims=True)

            # reshape to time points x neurons
            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
            R2_cv[iapl,ises,istim],optim_rank[iapl,ises,istim],_  = RRR_wrapper(Y, X, nN=nsampleneurons,nranks=nranks,nmodelfits=nmodelfits)

#%% Plotting:
clr = clrs_arealabelpairs[0]
R2_toplot = np.reshape(R2_cv,(narealabelpairs,nSessions*nStim))
rank_toplot = np.reshape(optim_rank,(narealabelpairs,nSessions*nStim))
# R2_toplot = np.nanmean(R2_cv,axis=2)
# rank_toplot = np.nanmean(optim_rank,axis=2)

fig,axes = plt.subplots(1,2,figsize=(5,2.5))

clrs = get_clr_areas(['V1','PM'])
ax = axes[0]
comps = [[0,3],[1,2]]
for icomp,comp in enumerate(comps):
    ax.scatter(R2_toplot[comp[0],:],R2_toplot[comp[1],:],s=120,edgecolor='w',marker='.',color=clrs[icomp])
    # ax.scatter(R2_cv[comp[0],:],R2_cv[comp[1],:],s=80,alpha=0.8,marker='.',color=clrs[icomp])
    ax_nticks(ax,3)
    ax.set_xlabel('Within')
    ax.set_ylabel('Across')
    ax.set_title('R2')

    _,pval = ttest_rel(R2_toplot[comp[0],:],R2_toplot[comp[1],:],nan_policy='omit')
    ax.text(0.6,0.1*(1+icomp),'%sp=%1.2e' % (get_sig_asterisks(pval),pval),transform=ax.transAxes,fontsize=10,color=clrs[icomp])
ax.legend(['V1 (V1->V1 vs PM->V1)','PM (V1->PM vs PM->PM)'],frameon=False,fontsize=7,loc='upper left')
# ax.legend(['V1','PM'],frameon=False,fontsize=10)
my_legend_strip(ax)
ax.set_xlim([0,0.2])
ax.set_ylim([0,0.2])
ax.plot([0,0.2],[0,0.2],':',color='grey',linewidth=1)

ax = axes[1]
for icomp,comp in enumerate(comps):
    ax.scatter(rank_toplot[comp[0],:],rank_toplot[comp[1],:],s=120,edgecolor='w',marker='.',color=clrs[icomp])
    ax_nticks(ax,3)
    ax.set_xlabel('Within')
    ax.set_title('Rank')
    _,pval = ttest_rel(rank_toplot[comp[0],:],rank_toplot[comp[1],:],nan_policy='omit')
    ax.text(0.6,0.1*(1+icomp),'%sp=%1.2e' % (get_sig_asterisks(pval),pval),transform=ax.transAxes,fontsize=10,color=clrs[icomp])
# ax.legend(['V1','PM'],frameon=False,fontsize=10)
ax.legend(['V1 (V1->V1 vs PM->V1)','PM (V1->PM vs PM->PM)'],frameon=False,fontsize=7,loc='upper left')
my_legend_strip(ax)
ax.set_xlim([0,15])
ax.set_ylim([0,15])
ax.plot([0,25],[0,25],':',color='grey',linewidth=1)
# ax.legend(['V1 (V1->V1 vs PM->V1','PM (PM->PM vs V1->PM'],frameon=False,fontsize=8)

sns.despine(offset=3,top=True,right=True)
plt.tight_layout()
# my_savefig(fig,savedir,'RRR_R2Rank_WithinVSAcross_%dneurons' % nsampleneurons,formats=['png'])


#%%






#%% 
######  ####### ######  ######  #######  #####     #    ####### ####### ######  
#     # #       #     # #     # #       #     #   # #      #    #       #     # 
#     # #       #     # #     # #       #        #   #     #    #       #     # 
#     # #####   ######  ######  #####   #       #     #    #    #####   #     # 
#     # #       #       #   #   #       #       #######    #    #       #     # 
#     # #       #       #    #  #       #     # #     #    #    #       #     # 
######  ####### #       #     # #######  #####  #     #    #    ####### ######  



#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons
popsizes            = np.array([5,10,20,50,100,200,500])
# popsizes            = np.array([5,10,20])
npopsizes           = len(popsizes)
nranks              = 40
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
R2_cv               = np.full((nSessions,npopsizes,npopsizes),np.nan)
optim_rank          = np.full((nSessions,npopsizes,npopsizes),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    for ixpop,xpop in enumerate(popsizes):
        for iypop,ypop in enumerate(popsizes):
            if len(idx_areax)>=xpop and len(idx_areay)>=ypop:
                R2_cv[ises,ixpop,iypop],optim_rank[ises,ixpop,iypop] = RRR_wrapper(Y, X, nN=xpop,nM=ypop,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)


#%% Plot R2 for different number of V1 and PM neurons
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["r","g"], N=256) 
cmap = sns.color_palette('magma', as_cmap=True)
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
ax = axes[0]
s = ax.imshow(np.nanmean(R2_cv,axis=0).T,cmap=cmap,vmin=0,vmax=np.nanmax(np.nanmean(R2_cv,axis=0)),origin='lower')
ax.set_xticks(range(npopsizes),labels=popsizes)
ax.set_yticks(range(npopsizes),labels=popsizes)
fig.colorbar(s, ax=ax,shrink=0.3,orientation='vertical')
ax.set_title('R2')
ax.set_xlabel('# source neurons')
ax.set_ylabel('# target neurons')

# Does the dimensionality increase with increasing number of neurons?
ax = axes[1]
s = ax.imshow(np.nanmean(optim_rank,axis=0).T,cmap=cmap,vmin=0,vmax=20,origin='lower')
ax.set_xticks(range(npopsizes),labels=popsizes)
ax.set_yticks(range(npopsizes),labels=popsizes)
fig.colorbar(s, ax=ax,shrink=0.3,orientation='vertical')
ax.set_xlabel('# source neurons')
# ax.set_ylabel('# target neurons')
ax.set_title('Rank')

sns.despine(top=True,right=True,offset=3)

plt.tight_layout()
my_savefig(fig,savedir,'R2_RRR_Rank_PopSize_Both_V1PM_%dsessions.png' % nSessions,formats=['png'])


#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons
popsizes            = np.array([5,10,20,50,100,200,500])
# popsizes            = np.array([5,10,20,50,100])
npopsizes           = len(popsizes)
nranks              = 50
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
R2_cv               = np.full((nSessions,npopsizes),np.nan)
optim_rank          = np.full((nSessions,npopsizes),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    for ipop,pop in enumerate(popsizes):
        if len(idx_areax)>=pop and len(idx_areay)>=pop:
            R2_cv[ises,ipop],optim_rank[ises,ipop]             = RRR_wrapper(Y, X, nN=pop,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)


#%% Plot R2 for different number of V1 and PM neurons
clrs_popsizes = sns.color_palette("rocket",len(popsizes))

fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=True)
ax = axes[0]
ax.scatter(popsizes, np.nanmean(R2_cv,axis=0), marker='o', color=clrs_popsizes)
# ax.plot(popsizes, np.nanmean(R2_cv,axis=0),color='k', linewidth=2)
shaded_error(popsizes,R2_cv,center='mean',error='sem',color='k',ax=ax)
ax.set_ylim([0,0.3])
# ax.set_xticks(popsizes)
ax.axhline(y=0,color='k',linestyle='--')

ax.set_xlabel('Population size')
ax.set_ylabel('RRR R2')
ax_nticks(ax,4)
# ax.set_xscale('log')

# Does the dimensionality increase with increasing number of neurons?
ax = axes[1]
ax.scatter(popsizes, np.nanmean(optim_rank,axis=0), marker='o', color=clrs_popsizes)
shaded_error(popsizes,optim_rank,center='mean',error='sem',color='k',ax=ax)

ax.plot(popsizes,popsizes**0.5,color='r',linestyle='--',linewidth=1)
ax.text(100,13,'$n^{1/2}$',color='r',fontsize=12)
ax.plot(popsizes,popsizes**0.3333,color='g',linestyle='--',linewidth=1)
ax.text(100,2,'$n^{1/3}$',color='g',fontsize=12)

ax.set_ylim([0,20])
ax.set_ylabel('Dimensionality')
ax.set_xlabel('Population size')
ax.set_xticks(popsizes[::2])
ax.set_xticks([10,100,200,500])

sns.despine(top=True,right=True,offset=3)

plt.tight_layout()
my_savefig(fig,savedir,'R2_RRR_Rank_PopSize_V1PM_%dsessions.png' % nSessions,formats=['png'])



#%% Show RRR performance as a function of the number of trials:
nsampleneurons  = 500
ntrials         = np.array([10,20,50,100,200,400,600,1000,2000,3200])
# ntrials         = np.array([50,100,200,400,1000])
ntrialsubsets   = len(ntrials)
kfold           = 5
nranks          = 50
lam             = 1
R2_cv           = np.full((nSessions,ntrialsubsets,nranks,kfold),np.nan)

kf              = KFold(n_splits=kfold,shuffle=True)

for ises,ses in enumerate(sessions):
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]

    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:

        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)

        for intrials,nt in enumerate(ntrials):
            idx_T           = np.random.choice(ses.respmat.shape[1],nt,replace=False)

            X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
            Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
            
            X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
            Y                   = zscore(Y,axis=0)

            for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                
                X_train, X_test     = X[idx_train], X[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                B_hat_train         = LM(Y_train,X_train, lam=lam)

                Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = linalg.svd(Y_hat_train)
                S = linalg.diagsvd(s,U.shape[0],s.shape[0])

                for r in range(nranks):
                    Y_hat_rr_test       = X_test @ B_hat_train @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace

                    R2_cv[ises,intrials,r,ikf] = EV(Y_test,Y_hat_rr_test)


#%% 
R2data = np.full((nSessions,ntrialsubsets),np.nan)
rankdata = np.full((nSessions,ntrialsubsets),np.nan)

for ises,ses in enumerate(sessions):
    for intrials,nt in enumerate(ntrials):
        R2data[ises,intrials],rankdata[ises,intrials] = rank_from_R2(R2_cv[ises,intrials,:,:],nranks,kfold)

#%% plot the results:
lambdacolors = sns.color_palette('magma',ntrialsubsets)

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax  = axes[0]
for intrials,nt in enumerate(ntrials):
    tempdata = np.nanmean(R2_cv[:,intrials,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[intrials],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('R2')
ax.set_ylim([-0.05,0.5])
ax.legend(ntrials,frameon=False,ncol=2,fontsize=8,title='number of trials')

ax = axes[1]
for ises,ses in enumerate(sessions):
    ax.plot(ntrials,rankdata[ises,:],color='grey',linewidth=1)
ax.plot(ntrials,np.nanmean(rankdata,axis=0),color='k',linewidth=1.5)
ax.scatter(ntrials,np.nanmean(rankdata,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('number of trials')
# ax.set_xscale('log')
ax.set_ylabel('estimated rank')
ax.set_ylim([0,np.max(rankdata)+1])

ax = axes[2]
for ises,ses in enumerate(sessions):
    ax.plot(ntrials,R2data[ises,:],color='grey',linewidth=1)
ax.plot(ntrials,np.nanmean(R2data,axis=0),color='k',linewidth=1.5)
ax.scatter(ntrials,np.nanmean(R2data,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('number of trials')
# ax.set_xscale('log')
ax.set_ylabel('R2 at optimal rank')
ax.set_ylim([0,np.max(R2data)+.05])
plt.tight_layout()
plt.savefig(os.path.join(savedir,'RRR_nTrials_lam%d_Rank_%dneurons.png' % (lam,nsampleneurons)), format = 'png')


#%%  Within to across area dimensionality comparison: 

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\WithinAcross\\')


#%% 
min_cells = 1500
sessions,nSessions   = filter_sessions(protocols = 'GR',min_cells=min_cells,filter_areas=['V1','PM'])

#%% 
print('Number of cells in V1 and in PM:')
for ises,ses in enumerate(sessions):
    print('Session %d: %d cells in V1, %d cells in PM' % (ises,np.sum(ses.celldata['roi_name']=='V1'),np.sum(ses.celldata['roi_name']=='PM')))

#%%  Load data properly:        
calciumversion = 'dF'
# calciumversion = 'deconv'

for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=False, load_calciumdata=True,load_videodata=False,
                                calciumversion=calciumversion,keepraw=False)

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealabelpairs  = ['V1-V1',
                    'PM-PM',
                    'V1-PM',
                    'PM-V1'
                    ]
alp_withinacross = ['Within',
                   'Within',
                   'Across',
                   'Across'] 

clrs_arealabelpairs = get_clr_area_pairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

lam                 = 0
nsampleneurons      = 100
nranks              = 50
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5

R2_cv               = np.full((narealabelpairs,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,nSessions),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    if np.sum((ses.celldata['roi_name']=='V1') & (ses.celldata['noise_level']<20))<(nsampleneurons*2):
        continue
    
    if np.sum((ses.celldata['roi_name']=='PM') & (ses.celldata['noise_level']<20))<(nsampleneurons*2):
        continue

    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['roi_name']==alx,
                                ses.celldata['noise_level']<20
                                ),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==aly,
                                ses.celldata['noise_level']<20
                                ),axis=0))[0]
    
        if np.any(np.intersect1d(idx_areax,idx_areay)): #if interactions within one population:
            if not np.array_equal(idx_areax, idx_areay): 
                print('Arealabelpair %s has partly overlapping neurons'%arealabelpair)
            idx_areax, idx_areay = np.array_split(np.random.permutation(idx_areax), 2)

        X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T

        R2_cv[iapl,ises],optim_rank[iapl,ises],_  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plotting:
clr = clrs_arealabelpairs[0]

fig,axes = plt.subplots(1,2,figsize=(4.5,2.5))

clrs = get_clr_areas(['V1','PM'])
ax = axes[0]
comps = [[0,3],[1,2]]
for icomp,comp in enumerate(comps):
    ax.scatter(R2_cv[comp[0],:],R2_cv[comp[1],:],s=120,edgecolor='w',marker='.',color=clrs[icomp])
    # ax.scatter(R2_cv[comp[0],:],R2_cv[comp[1],:],s=80,alpha=0.8,marker='.',color=clrs[icomp])
    ax_nticks(ax,3)
    ax.set_xlabel('Within')
    ax.set_ylabel('Across')
    ax.set_title('R2')

    _,pval = ttest_rel(R2_cv[comp[0],:],R2_cv[comp[1],:],nan_policy='omit')
    ax.text(0.6,0.1*(1+icomp),'%sp=%.3f' % (get_sig_asterisks(pval),pval),transform=ax.transAxes,fontsize=10,color=clrs[icomp])
ax.legend(['V1','PM'],frameon=False,fontsize=10)
my_legend_strip(ax)
ax.set_xlim([0,0.4])
ax.set_ylim([0,0.4])
ax.plot([0,0.4],[0,0.4],':',color='grey',linewidth=1)

ax = axes[1]
for icomp,comp in enumerate(comps):
    ax.scatter(optim_rank[comp[0],:],optim_rank[comp[1],:],s=120,edgecolor='w',marker='.',color=clrs[icomp])
    ax_nticks(ax,3)
    ax.set_xlabel('Within')
    ax.set_title('Rank')
    _,pval = ttest_rel(optim_rank[comp[0],:],optim_rank[comp[1],:],nan_policy='omit')
    ax.text(0.6,0.1*(1+icomp),'%sp=%.3f' % (get_sig_asterisks(pval),pval),transform=ax.transAxes,fontsize=10,color=clrs[icomp])
ax.legend(['V1','PM'],frameon=False,fontsize=10)
my_legend_strip(ax)
ax.set_xlim([0,25])
ax.set_ylim([0,25])
ax.plot([0,25],[0,25],':',color='grey',linewidth=1)
# ax.legend(['V1 (V1->V1 vs PM->V1','PM (PM->PM vs V1->PM'],frameon=False,fontsize=8)

sns.despine(offset=3,top=True,right=True)
plt.tight_layout()

my_savefig(fig,savedir,'RRR_R2Rank_WithinVSAcross_%dneurons' % nsampleneurons,formats=['png'])






#%% 

# DEPRECATED:





#%% get optimal lambda
nsampleneurons  = 500
lambdas         = np.logspace(-6, 5, 10)
# lambdas         = np.array([0,0.01,0.1,1])
nlambdas        = len(lambdas)
kfold           = 5
nranks          = 50

R2_cv           = np.full((nSessions,nlambdas,nranks,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different labmdas'):
    idx_T               = ses.trialdata['Orientation']==0
    # idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]

    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:

        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)

        X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
        
        X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
        Y                   = zscore(Y,axis=0)

        # Explanation of steps
        # X is of shape K x N (samples by features), Y is of shape K x M
        # K is the number of samples, N is the number of neurons in area 1,
        # M is the number of neurons in area 2

        # multiple linear regression, B_hat is of shape N x M:
        # B_hat               = LM(Y,X, lam=lam) 
        #RRR: do SVD decomp of Y_hat, 
        # U is of shape K x r, S is of shape r x r, V is of shape r x M
        # Y_hat_rr,U,S,V     = RRR(Y, X, B_hat, r) 

        for ilam,lam in enumerate(lambdas):
            # cross-validation version
            # R2_cv   = np.zeros(kfold)
            kf          = KFold(n_splits=kfold,shuffle=True)
            for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                
                X_train, X_test     = X[idx_train], X[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                B_hat_train         = LM(Y_train,X_train, lam=lam)

                Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = linalg.svd(Y_hat_train)
                S = linalg.diagsvd(s,U.shape[0],s.shape[0])

                for r in range(nranks):
                    Y_hat_rr_test       = X_test @ B_hat_train @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace

                    R2_cv[ises,ilam,r,ikf] = EV(Y_test,Y_hat_rr_test)


#%% plot the results for lam = 0
lambdacolors = sns.color_palette('magma',nlambdas)

fig,ax = plt.subplots(1,1,figsize=(3,3))
for ilam,lam in enumerate([lambdas[0]]):
    tempdata = np.nanmean(R2_cv[:,ilam,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[ilam],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('Test R2')
ax.set_xticks(range(nranks+5)[::5])
sns.despine(ax=ax,top=True,right=True,trim=True)
# ax.legend(my_ceil(np.log10(lambdas),1),frameon=False,ncol=2,title='lambda (10^x)')

#%% Compute optimal rank:
R2data = np.full((nSessions,nlambdas),np.nan)
rankdata = np.full((nSessions,nlambdas),np.nan)
for ises,ses in enumerate(sessions):
    if np.sum(np.isnan(R2_cv[ises,:,:,:]))>0:
        continue
    for ilam,lam in enumerate(lambdas):
        R2data[ises,ilam],rankdata[ises,ilam] = rank_from_R2(R2_cv[ises,ilam,:,:],nranks,kfold)

#%% plot the results:
lambdacolors = sns.color_palette('magma',nlambdas)

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax  = axes[0]
for ilam,lam in enumerate(lambdas):
    tempdata = np.nanmean(R2_cv[:,ilam,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[ilam],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('R2')
ax.legend(my_ceil(np.log10(lambdas),1),frameon=False,ncol=2,title='lambda (10^x)')

ax = axes[1]
for ises,ses in enumerate(sessions):
    ax.plot(lambdas,rankdata[ises,:],color='grey',linewidth=1)
ax.plot(lambdas,np.nanmean(rankdata,axis=0),color='k',linewidth=1.5)
ax.scatter(lambdas,np.nanmean(rankdata,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('lambda')
ax.set_xscale('log')
ax.set_ylabel('estimated rank')
ax.set_ylim([0,np.nanmax(rankdata)+1])

ax = axes[2]
for ises,ses in enumerate(sessions):
    ax.plot(lambdas,R2data[ises,:],color='grey',linewidth=1)
ax.plot(lambdas,np.nanmean(R2data,axis=0),color='k',linewidth=1.5)
ax.scatter(lambdas,np.nanmean(R2data,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('lambda')
ax.set_xscale('log')
ax.set_ylabel('R2 at optimal rank')
ax.set_ylim([0,np.nanmax(R2data)+.05])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'RRR_perOri_Lam_Rank_%dneurons.png' % nsampleneurons), format = 'png')
# plt.savefig(os.path.join(savedir,'RRR_Lam_Rank_%dneurons.png' % nsampleneurons), format = 'png')
my_savefig(fig,savedir,'RRR_Lam_Rank_%dneurons.png' % nsampleneurons,formats=['png'])
# my_savefig(fig,savedir,'RRR_perOri_Lam_Rank_%dneurons.png' % nsampleneurons,formats=['png'])



#%% get optimal pre pCA
nsampleneurons  = 500
PCAdims         = np.array([1,2,5,10,20,50,100])
# lambdas         = np.array([0,0.01,0.1,1])
nPCAdims        = len(PCAdims)
kfold           = 5
nranks          = 50
lam             = 5000
R2_cv           = np.full((nSessions,nPCAdims,nranks,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different pre PCA dims'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]

    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:

        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)

        X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
        
        X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
        Y                   = zscore(Y,axis=0)

        # Explanation of steps
        # X is of shape K x N (samples by features), Y is of shape K x M
        # K is the number of samples, N is the number of neurons in area 1,
        # M is the number of neurons in area 2

        # multiple linear regression, B_hat is of shape N x M:
        # B_hat               = LM(Y,X, lam=lam) 
        #RRR: do SVD decomp of Y_hat, 
        # U is of shape K x r, S is of shape r x r, V is of shape r x M
        # Y_hat_rr,U,S,V     = RRR(Y, X, B_hat, r) 

        for ipc,PCAdim in enumerate(PCAdims):
            # cross-validation version
            # R2_cv   = np.zeros(kfold)

            Xmodel      = PCA(n_components=PCAdim)
            Xpca        = Xmodel.fit_transform(X)
            Ymodel      = PCA(n_components=PCAdim)
            Ypca        = Ymodel.fit_transform(Y)

            kf          = KFold(n_splits=kfold,shuffle=True)

            for ikf, (idx_train, idx_test) in enumerate(kf.split(Xpca)):
                
                X_train, X_test     = Xpca[idx_train], Xpca[idx_test]
                Y_train, Y_test     = Ypca[idx_train], Ypca[idx_test]

                B_hat_train         = LM(Y_train,X_train, lam=lam)

                Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = linalg.svd(Y_hat_train)
                S = linalg.diagsvd(s,U.shape[0],s.shape[0])

                for r in range(nranks):
                    Y_hat_rr_test       = X_test @ B_hat_train @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace

                    R2_cv[ises,ipc,r,ikf] = EV(Y_test,Y_hat_rr_test) * Ymodel.explained_variance_ratio_.sum()


#%% 
R2data = np.full((nSessions,nPCAdims),np.nan)
rankdata = np.full((nSessions,nPCAdims),np.nan)

for ises,ses in enumerate(sessions):
    for ipc,PCAdim in enumerate(PCAdims):
        R2data[ises,ipc],rankdata[ises,ipc] = rank_from_R2(R2_cv[ises,ipc,:,:],nranks,kfold)

#%% plot the results:
lambdacolors = sns.color_palette('magma',nPCAdims)

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax  = axes[0]
for ilam,lam in enumerate(PCAdims):
    tempdata = np.nanmean(R2_cv[:,ilam,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[ilam],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('R2')

ax = axes[1]
for ises,ses in enumerate(sessions):
    ax.plot(PCAdims,rankdata[ises,:],color='grey',linewidth=1)
ax.plot(PCAdims,np.nanmean(rankdata,axis=0),color='k',linewidth=1.5)
ax.scatter(PCAdims,np.nanmean(rankdata,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('PCAdim')
ax.set_xscale('log')
ax.set_ylabel('estimated rank')
ax.set_ylim([0,np.max(rankdata)+1])

ax = axes[2]
for ises,ses in enumerate(sessions):
    ax.plot(PCAdims,R2data[ises,:],color='grey',linewidth=1)
ax.plot(PCAdims,np.nanmean(R2data,axis=0),color='k',linewidth=1.5)
ax.scatter(PCAdims,np.nanmean(R2data,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('PCAdim')
ax.set_xscale('log')
ax.set_ylabel('R2 at optimal rank')
ax.set_ylim([0,np.max(R2data)+.05])
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'RRR_PrePCA_lam5000_Rank_%dneurons.png' % nsampleneurons), format = 'png')
# plt.savefig(os.path.join(savedir,'RRR_PrePCA_lam0_Rank_%dneurons.png' % nsampleneurons), format = 'png')
plt.savefig(os.path.join(savedir,'RRR_PrePCA_lam0_perOri_Rank_%dneurons.png' % nsampleneurons), format = 'png')




#%%%


#%%

for ses in sessions:
    ses.respmat = np.nanmean(ses.tensor[:,:,idx_resp],axis=2)

#%%
from utils.tuning import compute_tuning_wrapper

sessions = compute_tuning_wrapper(sessions)

#%% Do RRR of V1 and PM labeled and unlabeled neurons
kfold           = 5
nmodelfits      = 2
nsampleneurons  = 50
nStim           = 16
idx_resp        = np.where((t_axis>=0) & (t_axis<=1.5))[0]

percs           = np.linspace(0,100,10)
cellfield       = 'event_rate'
cellfield       = 'OSI'
# cellfield       = 'tuning_var'
# cellfield       = 'noise_level'
data            = np.full((len(percs),nSessions,nStim),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different  sizes'):
    for iperc,perc in enumerate(percs):
        idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                                ses.celldata[cellfield]>np.percentile(ses.celldata[cellfield],perc),	
                                ),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                                ses.celldata[cellfield]>np.percentile(ses.celldata[cellfield],perc),	
                                ),axis=0))[0]

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
            continue

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim

            #on tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            #subtract mean response across trials:
            X                   -= np.mean(X,axis=1,keepdims=True)
            Y                   -= np.mean(Y,axis=1,keepdims=True)

            # reshape to neurons x time points
            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            #zscore
            X                   = zscore(X,axis=0,nan_policy='omit')  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0,nan_policy='omit')

            #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
            data[iperc,ises,istim],_,_ = RRR_wrapper(Y, X, 
                            nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)
            # print(R2_cv)

#%%

plt.plot(percs,np.nanmean(data,axis=(1,2)),color='k',linewidth=2)

