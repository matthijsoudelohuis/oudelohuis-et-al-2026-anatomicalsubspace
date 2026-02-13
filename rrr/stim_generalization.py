# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.linalg import subspace_angles as subangle

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.plot_lib import * #get all the fixed color schemes
# from utils.corr_lib import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.psth import compute_tensor
from params import load_params

params = load_params()
savedir = os.path.join(params['savedir'],'RRR','Validation')

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches


#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE09830_2023_04_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR
session_list        = np.array([['LPE09830_2023_04_10']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],min_lab_cells_V1=20,min_lab_cells_PM=20)
report_sessions(sessions)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%%  Load data properly:        
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
# t_pre       = -1    #pre s
# t_post      = 1.9     #post s
# binsize     = 0.2
vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
                            ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

behavfields = np.array(['runspeed','diffrunspeed'])

t_pre       = -1         #pre s
t_post      = 2.17        #post s
binsize     = 1/5.35

for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=params['calciumversion'])
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 method='nearby')
    
    [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    #Subsample behavioral data 10 times before binning:
    sessions[ises].behaviordata.drop('session_id',axis=1,inplace=True)
    sessions[ises].behaviordata = sessions[ises].behaviordata.groupby(sessions[ises].behaviordata.index // 10).mean()
    sessions[ises].behaviordata['diffrunspeed'] = np.diff(sessions[ises].behaviordata['runspeed'],prepend=0)
    [sessions[ises].tensor_run,t_axis] = compute_tensor(sessions[ises].behaviordata[behavfields], sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    
    delattr(sessions[ises],'calciumdata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'videodata')

#%%
for ises in range(nSessions):
    sessions[ises].respmat = np.nanmean(sessions[ises].tensor[:,:,t_axis>0],axis=(2))
    sessions[ises].respmat_videome = np.nanmean(sessions[ises].tensor_vid[np.ix_([0],range(sessions[ises].tensor_vid.shape[1]),t_axis>0)],axis=(2)).squeeze()
    # sessions[ises].respmat_videome = np.nansum(sessions[ises].respmat_videome,axis=0)
    # sessions[ises].respmat_runspeed = np.nanmean(sessions[ises].tensor_run[0,:,t_axis>0],axis=(1)).squeeze()
    sessions[ises].respmat_runspeed = np.nanmean(sessions[ises].tensor_run[np.ix_([0],range(sessions[ises].tensor_run.shape[1]),t_axis>0)],axis=(2)).squeeze()

for ises in range(nSessions):
    sessions[ises].respmat_videome -= np.nanmin(sessions[ises].respmat_videome,keepdims=True)
    sessions[ises].respmat_videome /= np.nanmax(sessions[ises].respmat_videome,keepdims=True)

#%% Subtracting mean response across trials for each stimulus condition
for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Subtracting mean response across trials'):
    N = len(sessions[ises].celldata)
    idx_resp = t_axis>0
    for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        idx_T               = sessions[ises].trialdata['stimCond']==stim

        #on tensor during the response:
        sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)] -= np.nanmean(sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)],axis=1,keepdims=True)
    
    idx_resp = t_axis<0
    for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        idx_T               = np.concatenate([[0],sessions[ises].trialdata['stimCond'][:-1]])==stim

        #on tensor during the response:
        sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)] -= np.nanmean(sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)],axis=1,keepdims=True)

#%% 
 #####  ####### ### #     #     #####  #     # ######   #####  ######     #     #####  ####### 
#     #    #     #  ##   ##    #     # #     # #     # #     # #     #   # #   #     # #       
#          #     #  # # # #    #       #     # #     # #       #     #  #   #  #       #       
 #####     #     #  #  #  #     #####  #     # ######   #####  ######  #     # #       #####   
      #    #     #  #     #          # #     # #     #       # #       ####### #       #       
#     #    #     #  #     #    #     # #     # #     # #     # #       #     # #     # #       
 #####     #    ### #     #     #####   #####  ######   #####  #       #     #  #####  ####### 

#%%
# celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

# sns.histplot(data=celldata,x='noise_level',hue='arealabel',stat='probability',common_norm=False,cumulative=True,
#              bins=np.linspace(0,100,500),element='step',fill=False)

#%% Parameters for RRR for size-matched populations of V1 and PM labeled and unlabeled neurons
lam                 = 0
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 10 #number of times new neurons are resampled - many for final run
kfold               = 5
maxnoiselevel       = 20
nStim               = 16

# idx_resp            = np.where((t_axis>=0) & (t_axis<=1))[0]
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
# idx_resp            = np.where((t_axis>=0.5) & (t_axis<=1.5))[0]
ntimebins           = len(idx_resp)
minsampleneurons    = 10
filter_nearby       = True

dataversions        = np.array(['original','behavout'])
# dataversions        = np.array(['original','','neuralout'])
dataversions        = np.array(['original'])
nversions           = len(dataversions) #0: original,1:behavior out, 2:neural out
rank_behavout       = 5
fixed_rank          = None

#%%

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Deleting data'):
    if hasattr(ses,'tensor_neuralout'):
        delattr(sessions[ises],'tensor_neuralout')
    if hasattr(ses,'tensor_behavout'):
        delattr(sessions[ises],'tensor_behavout')

#%% Do RRR of V1 and PM labeled and unlabeled neurons

# Main analysis: identify pairs of populations and do crossvalidated RRR between these. 
# Compute crossvalidated EV for each rank, identify optimal rank, and store ev at this rank
# Do this for many pairs and subsequent analyses focus on different comparissons
# Additionally, do this for different versions of the data (original, behavior out, neural out)
# Behavior out: remove activity explained by behavioral variables
# Neural out: remove activity explained by simultaneous AL + RSP data 
# (comparisons with AL and RSP in arealabelpairs with neural_out thus don't make sense)

arealabelpairs  = [
                    'V1unl-PMunl', #Feedforward pairs
                    'PMunl-V1unl', #Feedback pairs
                    ]

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

nsampleneurons      = 25
nmodelfits          = 5 #number of times new neurons are resampled - many for final run
fixed_rank          = 3

R2_cv               = np.full((narealabelpairs,3,nSessions,nStim,nStim),np.nan)
optim_rank          = np.full((narealabelpairs,3,nSessions,nStim,nStim),np.nan)
R2_ranks            = np.full((narealabelpairs,3,nSessions,nStim,nStim,nranks,nmodelfits,kfold),np.nan)
subspace_weights    = np.full((narealabelpairs,3,nSessions,nStim,nsampleneurons,nsampleneurons,nmodelfits,kfold),np.nan)
subspace_angle      = np.full((narealabelpairs,3,nSessions,nStim,nStim,fixed_rank,nmodelfits,kfold),np.nan)

stim = np.unique(sessions[0].trialdata['stimCond'])
stimlabels = np.unique(sessions[0].trialdata['Orientation'])

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different populations'):
for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting RRR model for different populations'):
    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=30)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    if not hasattr(ses,'tensor_behavout') and 'behavout' in dataversions: 
        sessions[ises].tensor_behavout = copy.copy(sessions[ises].tensor)
        #Get behavioral matrix:
        B                   = np.concatenate((sessions[ises].tensor_vid,
                                sessions[ises].tensor_run),axis=0)
        
        for area in ['V1','PM']:
            idx_N    = np.where(np.all((ses.celldata['roi_name']==area,
                                        idx_nearby,
                                        ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]

            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations
                idx_T               = sessions[ises].trialdata['stimCond']==stim

                Bstim                   = B[:,idx_T,:].reshape(np.shape(B)[0],-1).T
                Bstim                   = zscore(Bstim,axis=0,nan_policy='omit')
                Bstim                   = Bstim[:,~np.all(np.isnan(Bstim),axis=0)]

                tempdata            = sessions[ises].tensor[np.ix_(idx_N,idx_T,np.arange(len(t_axis)))]
                N,K,T = np.shape(tempdata)
                Y_r = np.reshape(tempdata,(N,K*T),order='C').T
                Y_orig,Y_hat,Y_out  = regress_out_cv(X=Bstim,Y=Y_r,rank=np.min([rank_behavout,len(idx_N)-1]),
                                                    lam=0,kfold=5)
                # print(area,EV(Y_orig,Y_hat))
                sessions[ises].tensor_behavout[np.ix_(idx_N,idx_T,np.arange(len(t_axis)))] = np.reshape(Y_out.T,(N,K,T),order='C')

    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]

        if np.any(np.intersect1d(idx_areax,idx_areay)): #if interactions within one population:
            if not np.array_equal(idx_areax, idx_areay): 
                print('Arealabelpair %s has partly overlapping neurons'%arealabelpair)
            idx_areax, idx_areay = np.array_split(np.random.permutation(idx_areax), 2)

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons: #skip exec if not enough neurons in one of the populations
            continue
        
        for istim,stimI in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            for jstim,stimJ in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        # for istim,stimI in enumerate(np.unique(ses.trialdata['stimCond'][0])): # loop over orientations 
            # for jstim,stimJ in enumerate(np.unique(ses.trialdata['stimCond'][0])): # loop over orientations 
                idx_T_i               = ses.trialdata['stimCond']==stimI
                idx_T_j               = ses.trialdata['stimCond']==stimJ
                
                if 'original' in dataversions:
                    #Get neural data from training trial type:
                    X_i                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T_i,idx_resp)]
                    Y_i                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T_i,idx_resp)]
                    
                    # reshape to neurons x time points
                    X_i                   = X_i.reshape(len(idx_areax),-1).T
                    Y_i                   = Y_i.reshape(len(idx_areay),-1).T

                    #Get neural data from test trial type:
                    X_j                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T_j,idx_resp)]
                    Y_j                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T_j,idx_resp)]
                    
                    # reshape to neurons x time points
                    X_j                   = X_j.reshape(len(idx_areax),-1).T
                    Y_j                   = Y_j.reshape(len(idx_areay),-1).T

                    R2_cv_folds = np.full((nranks,nmodelfits,kfold),np.nan)

                    kf      = KFold(n_splits=kfold,shuffle=True)

                    # Data format: 
                    K,N     = np.shape(X_i)
                    M       = np.shape(Y_i)[1]

                    nN  = nsampleneurons or min(M,N)
                    nM  = nsampleneurons or nsampleneurons or min(M,N)
                    nK  = K
                    for imf in range(nmodelfits): #loop over model fits, randomly subsample neurons
                        idx_areax_sub           = np.random.choice(N,nN,replace=False)
                        idx_areay_sub           = np.random.choice(M,nM,replace=False)

                        X_i_sub                   = X_i[:,idx_areax_sub]
                        Y_i_sub                   = Y_i[:,idx_areay_sub]
                        
                        X_i_sub                   = zscore(X_i_sub,axis=0)  #Z score activity for each neuron across trials/timepoints
                        Y_i_sub                   = zscore(Y_i_sub,axis=0)
                    
                        X_j_sub                   = X_j[:,idx_areax_sub]
                        Y_j_sub                   = Y_j[:,idx_areay_sub]
                        
                        X_j_sub                   = zscore(X_j_sub,axis=0)  #Z score activity for each neuron across trials/timepoints
                        Y_j_sub                   = zscore(Y_j_sub,axis=0)
                        # print('hello')

                        for ikf, (idx_train, idx_test) in enumerate(kf.split(X_sub)):
                            
                            X_train, X_test     = X_i_sub[idx_train], X_j_sub[idx_test]
                            Y_train, Y_test     = Y_i_sub[idx_train], Y_j_sub[idx_test]

                            B_hat_train         = LM(Y_train,X_train, lam=lam)

                            Y_hat_train         = X_train @ B_hat_train

                            # decomposing and low rank approximation of A
                            U, s, V = svds(Y_hat_train,k=np.min((nranks,nN,nM))-1,which='LM')
                            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                            S = linalg.diagsvd(s,U.shape[0],s.shape[0])
                            
                            for r in range(nranks):
                                B_rrr               = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace

                                Y_hat_rr_test       = X_test @ B_rrr #project test data onto low rank predictive subspace

                                R2_cv_folds[r,imf,ikf] = EV(Y_test,Y_hat_rr_test)
                        
                        subspace_weights[iapl,0,ises,istim,:,:,imf,ikf] = B_hat_train @ V[:fixed_rank,:].T @ V[:fixed_rank,:] 

                    if fixed_rank is not None:
                        rank = fixed_rank
                        repmean = np.nanmean(R2_cv_folds[rank,:,:])
                    else:
                        repmean,rank = rank_from_R2(R2_cv_folds.reshape([nranks,nmodelfits*kfold]),nranks,nmodelfits*kfold)

                    #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
                    R2_cv[iapl,0,ises,istim,jstim] = repmean
                    optim_rank[iapl,0,ises,istim,jstim] = rank
                    R2_ranks[iapl,0,ises,istim,jstim,:,:,:] = R2_cv_folds


#%% Plot R2 for different crosscombinations of stimuli: 
cmap = sns.color_palette('magma', as_cmap=True)
fig,axes = plt.subplots(1,1,figsize=(6*cm,6*cm),sharey=True,sharex=True)
ax = axes
plotdata = np.nanmean(R2_cv[:,0,:,:,:],axis=(0,1))
sns.heatmap(plotdata,cmap=cmap,vmin=0,vmax=my_ceil(np.nanmax(plotdata),2),annot=False,ax=ax,square=True,
            xticklabels=stimlabels,yticklabels=stimlabels,cbar=True,
            cbar_kws={'shrink': 0.6, 'ticks': [0, my_ceil(np.nanmax(plotdata),2)],'label': 'R2'})
ax.set_title('Performance ($R^2$)')
ax.invert_yaxis()
ax.set_xlabel('Stimulus direction (train)')
ax.set_ylabel('Stimulus direction (test)')

# Does the dimensionality change?
# ax = axes[1]
# plotdata = np.nanmean(optim_rank[:,0,:,:,:],axis=(0,1))
# sns.heatmap(plotdata,cmap=cmap,vmin=0,vmax=my_ceil(np.nanmax(plotdata)),annot=False,ax=ax,
#             xticklabels=stimlabels,yticklabels=stimlabels,cbar=True,
#             square=True)
# ax.invert_yaxis()
# ax.set_xlabel('Stimulus direction (train)')
# ax.set_ylabel('Stimulus direction (test)')
# ax.set_title('Rank')
# sns.despine(top=True,right=True,offset=0)
ax.set_xticks(stim[::2]+0.5,stimlabels[::2],rotation=45)
ax.set_yticks(stim[::2]+0.5,stimlabels[::2])
plt.tight_layout()
my_savefig(fig,savedir,'Cross_Stimulus_R2_RRR_V1PM_%dsessions' % nSessions)

#%% 
nsampleneurons = 100
fixed_rank = 3
subspace_weights    = np.full((narealabelpairs,3,nSessions,nStim,nsampleneurons,nsampleneurons,nmodelfits),np.nan)
subspace_angle      = np.full((narealabelpairs,3,nSessions,nStim,nStim,fixed_rank,nmodelfits),np.nan)

for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting RRR model for different populations'):
    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=30)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)
   
    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]

        if np.any(np.intersect1d(idx_areax,idx_areay)): #if interactions within one population:
            if not np.array_equal(idx_areax, idx_areay): 
                print('Arealabelpair %s has partly overlapping neurons'%arealabelpair)
            idx_areax, idx_areay = np.array_split(np.random.permutation(idx_areax), 2)

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons: #skip exec if not enough neurons in one of the populations
            continue

        # Data format: 
        N     = len(idx_areax)
        M     = len(idx_areay)
        
        for imf in range(nmodelfits): #loop over model fits, randomly subsample neurons
            idx_areax_sub           = np.random.choice(N,nsampleneurons,replace=False)
            idx_areay_sub           = np.random.choice(M,nsampleneurons,replace=False)

            for istim,stimI in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
                idx_T_i               = ses.trialdata['stimCond']==stimI
                
                if 'original' in dataversions:
                    #Get neural data from training trial type:
                    X_i                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T_i,idx_resp)]
                    Y_i                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T_i,idx_resp)]
                    
                    # reshape to neurons x time points
                    X_i                   = X_i.reshape(len(idx_areax),-1).T
                    Y_i                   = Y_i.reshape(len(idx_areay),-1).T

                    X_i_sub               = X_i[:,idx_areax_sub]
                    Y_i_sub               = Y_i[:,idx_areay_sub]
                    
                    X_i_sub               = zscore(X_i_sub,axis=0)  #Z score activity for each neuron across trials/timepoints
                    Y_i_sub               = zscore(Y_i_sub,axis=0)
                    
                    B_hat                 = LM(Y_i_sub,X_i_sub, lam=lam)

                    Y_hat                 = X_i_sub @ B_hat

                    # decomposing and low rank approximation of A
                    U, s, V = svds(Y_hat,k=np.min((nranks,nN,nM))-1,which='LM')
                    U, s, V = U[:, ::-1], s[::-1], V[::-1, :]
                    
                    subspace_weights[iapl,0,ises,istim,:,:,imf] = B_hat @ V[:fixed_rank,:].T @ V[:fixed_rank,:] 

#%%
for iapl in range(narealabelpairs):
     for ises in range(nSessions):
        for istim in range(nStim):
            for jstim in range(nStim):
                for imf in range(nmodelfits):
                    for ikf in range(kfold):
                        subspace_angle[iapl,0,ises,istim,jstim,:,imf] = subangle(subspace_weights[iapl,0,ises,istim,:,:,imf],subspace_weights[iapl,0,ises,jstim,:,:,imf])
subspace_angle = np.degrees(subspace_angle)
subspace_angle = np.flip(subspace_angle,axis=5)

#%% Plot principle angle for different crosscombinations of stimuli: 
cmap = sns.color_palette('magma', as_cmap=True)
fig,axes = plt.subplots(1,fixed_rank,figsize=(fixed_rank*5*cm,5*cm),sharey=False,sharex=False)
for irank in range(fixed_rank):
    ax = axes[irank]
    plotdata = np.nanmean(subspace_angle[:,0,:,:,:,irank,:],axis=(0,1,-1))
    sns.heatmap(plotdata,cmap=cmap,vmin=0,vmax=90,annot=False,ax=ax,square=True,
                xticklabels=[],yticklabels=[],
                cbar=True,cbar_kws={'shrink': 0.7, 'ticks': [0, 45, 90],'label': 'degrees'})
    ax.set_title('Principle angle (rank %d)' % irank)
    ax.invert_yaxis()
    if irank==0: 
        ax.set_ylabel('Stimulus direction')
    ax.set_xticks(stim[::2]+0.5,['{:.0f}'.format(i) for i in stimlabels[::2]],rotation=45)
    ax.set_xlabel('Stimulus direction')
axes[0].set_yticks(stim[::2]+0.5,stimlabels[::2])
plt.tight_layout()
my_savefig(fig,savedir,'Principle_Angle_V1PM_%dsessions' % nSessions)
