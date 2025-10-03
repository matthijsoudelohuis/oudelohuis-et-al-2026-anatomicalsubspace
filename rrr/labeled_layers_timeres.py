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
from scipy.signal import medfilt
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from skimage.measure import block_reduce
from tqdm import tqdm

from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plot_lib import * #get all the fixed color schemes
from utils.tuning import compute_tuning_wrapper
from utils.regress_lib import *
from utils.RRRlib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\TimeResolution\\')

#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)


#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])
# areas = ['V1','PM','AL','RSP']
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=areas,filter_areas=areas)
report_sessions(sessions)

#%%  Load data properly:        
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
# t_pre       = -1    #pre s
# t_post      = 1.9     #post s
# binsize     = 0.2
calciumversion = 'dF'
vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
                            ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

behavfields = np.array(['runspeed','diffrunspeed'])

for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion)
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 method='nearby')
    t_pre       = t_axis[0]         #pre s
    t_post      = t_axis[-1]        #post s
    binsize     = np.diff(t_axis)[0]
    [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    
    sessions[ises].behaviordata.drop('session_id',axis=1,inplace=True)
    sessions[ises].behaviordata = sessions[ises].behaviordata.groupby(sessions[ises].behaviordata.index // 10).mean()
    sessions[ises].behaviordata['diffrunspeed'] = np.diff(sessions[ises].behaviordata['runspeed'],prepend=0)
    [sessions[ises].tensor_run,t_axis] = compute_tensor(sessions[ises].behaviordata[behavfields], sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    delattr(sessions[ises],'calciumdata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'videodata')

# #%%  Load data properly:        
# calciumversion = 'dF'
# for ises in range(nSessions):
#     sessions[ises].load_tensor(load_calciumdata=True,calciumversion=calciumversion,keepraw=False)
# t_axis = sessions[0].t_axis

#%%  #assign arealayerlabel
for ses in sessions:
    ses.reset_layer(splitdepth=250)
    ses.celldata['arealayer'] = ses.celldata['roi_name'] + ses.celldata['layer']
    #assign arealayerlabel
    ses.celldata['arealayerlabel'] = ses.celldata['arealabel'] + ses.celldata['layer'] 


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
    
    # plt.imshow(np.nanmean(sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)],axis=1),aspect='auto',vmin=-0.1,vmax=0.1)

# plt.imshow(np.nanmean(sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)],axis=1),aspect='auto',vmin=-0.1,vmax=0.1)
# plt.imshow(np.nanmean(sessions[ises].tensor[:,idx_T,:],axis=1),aspect='auto',vmin=-0.1,vmax=0.1)

#%% ########################## RRR across time relative to stimulus onset: ###### 
# Main analysis: identify pairs of populations and do crossvalidated RRR between these. 
# Compute crossvalidated EV for each rank, identify optimal rank, and store ev at this rank
# Do this for many pairs and subsequent analyses focus on different comparissons
# Additionally, do this for different versions of the data (original, behavior out, neural out)
# Behavior out: remove activity explained by behavioral variables
# Neural out: remove activity explained by simultaneous AL + RSP data 
# (comparisons with AL and RSP in arealabelpairs with neural_out thus don't make sense)

#%% Parameters for RRR for size-matched populations of V1 and PM labeled and unlabeled neurons
nranks              = 10 #number of ranks of RRR to be evaluated
nmodelfits          = 5 #number of times new neurons are resampled - many for final run
maxnoiselevel       = 20
nStim               = 16
kfold               = 5

minsampleneurons    = 10
filter_nearby       = True

# idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
idx_resp            = np.where((t_axis>=-10) & (t_axis<=10))[0]
ntimebins           = len(idx_resp)

dataversions        = ['original','behavout','neuralout']
nversions           = len(dataversions) #0: original,1:behavior out, 2:neural out
rank_behavout       = 5
rank_neuralout      = 5

#%% Do RRR of V1 and PM labeled and unlabeled neurons
# arealabelpairs  = [
#                     'V1unl-PMunl', #Feedforward pairs
#                     'V1lab-PMunl',

#                     'PMunl-V1unl', #Feedback pairs
#                     'PMlab-V1unl',

#                     # Comparing looped pairs:
#                     'V1unl-PMlab',
#                     'V1lab-PMlab',
#                     'PMunl-V1lab',
#                     'PMlab-V1lab',
#                     ]

arealabelpairs  = np.array([
                    'V1unlL2/3-PMunlL2/3', #V1 L2/3 to other layers
                    'V1labL2/3-PMunlL2/3',

                    'V1unlL2/3-PMlabL2/3',
                    'V1labL2/3-PMlabL2/3',

                    'V1unlL2/3-PMunlL5',
                    'V1labL2/3-PMunlL5',

                    'V1unlL2/3-PMlabL5',
                    'V1labL2/3-PMlabL5',

                    'PMunlL2/3-V1unlL2/3', #PM L2/3 to other layers
                    'PMlabL2/3-V1unlL2/3',

                    'PMunlL2/3-V1labL2/3',
                    'PMlabL2/3-V1labL2/3',

                    'PMunlL2/3-PMunlL5',
                    'PMlabL2/3-PMunlL5',

                    'PMunlL2/3-PMlabL5',
                    'PMlabL2/3-PMlabL5',

                    'PMunlL5-V1unlL2/3', #PM L5 to other layers
                    'PMlabL5-V1unlL2/3',

                    'PMunlL5-V1labL2/3',
                    'PMlabL5-V1labL2/3',

                    'PMunlL5-PMunlL2/3',
                    'PMlabL5-PMunlL2/3',

                    'PMunlL5-PMlabL2/3',
                    'PMlabL5-PMlabL2/3',
                   ])


narealabelpairs     = len(arealabelpairs)
clrs_arealabelpairs = sns.color_palette("tab10", narealabelpairs)

R2_cv               = np.full((narealabelpairs,nversions,ntimebins,nSessions,nStim),np.nan)
optim_rank          = np.full((narealabelpairs,nversions,ntimebins,nSessions,nStim),np.nan)
# R2_ranks            = np.full((narealabelpairs,nversions,ntimebins,nSessions,nStim,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR models'):
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting RRR models'):
    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=30)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    if not hasattr(ses,'tensor_behavout'): 
        #Get behavioral matrix: 
        B                   = np.concatenate((sessions[ises].tensor_vid,
                                sessions[ises].tensor_run),axis=0)
        B                   = B.reshape(np.shape(B)[0],-1).T
        B                   = zscore(B,axis=0,nan_policy='omit')
        
        N,K,T = np.shape(sessions[ises].tensor)
        Y_r = np.reshape(sessions[ises].tensor,(N,K*T),order='C').T
        _,_,Y_r,_,ev  = regress_out_behavior_modulation(ses,X=B,Y=Y_r,rank=rank_behavout,lam=0)
        sessions[ises].tensor_behavout = np.reshape(Y_r.T,(N,K,T),order='C')

    # plt.imshow(np.nanmean(sessions[ises].tensor,axis=2),aspect='auto',vmin=-0.1,vmax=0.1)
    # plt.imshow(np.nanmean(sessions[ises].tensor_behavout,axis=2),aspect='auto',vmin=-0.1,vmax=0.1)

    if not hasattr(ses,'tensor_neuralout'): 
        idx_ALRSP   = np.where(np.all((np.isin(ses.celldata['roi_name'],['AL','RSP']),
                                    ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]

        idx_V1PM    = np.where(np.all((np.isin(ses.celldata['roi_name'],['V1','PM']),
                                    ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
        
        if len(idx_ALRSP)>nsampleneurons:
            sessions[ises].tensor_neuralout = np.full(np.shape(sessions[ises].tensor),np.nan)
            
            X                   = sessions[ises].tensor[idx_ALRSP,:,:]
            X_r = np.reshape(X,(len(idx_ALRSP),K*T),order='C').T
            
            Y                   = sessions[ises].tensor[idx_V1PM,:,:]
            Y_r = np.reshape(Y,(len(idx_V1PM),K*T),order='C').T

            _,_,Y_neuralout,_,ev  = regress_out_behavior_modulation(ses,X=X_r,Y=Y_r,rank=rank_neuralout,lam=0)
            # print(ev)
            # Y_neuralout = np.reshape(Y_r.T,(len(idx_V1PM),K*T),order='C')
            sessions[ises].tensor_neuralout[idx_V1PM,:,:] = np.reshape(Y_neuralout.T,(len(idx_V1PM),K,T),order='C')

    for iapl, arealabelpair in enumerate(arealabelpairs):
        idx = int((iapl//2)*2+1) #get the index of the arealabelpair with the labeled cells, because always the lower numbered one
        nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealayerlabel']==i,
                                                 ses.celldata['noise_level']<maxnoiselevel,
                                                 idx_nearby),axis=0)) for i in arealabelpairs[idx].split('-')],
                                    )
        if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
            continue

        # nsampleneurons      = np.min([nsampleneurons,200*ntimebins*0.2])
        alx,aly             = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['arealayerlabel']==alx,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealayerlabel']==aly,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                # idx_nearby
                                ),axis=0))[0]

        if np.any(np.intersect1d(idx_areax,idx_areay)): #if interactions within one population:
            if not np.array_equal(idx_areax, idx_areay): 
                print('Arealabelpair %s has partly overlapping neurons'%arealabelpair)
            idx_areax, idx_areay = np.array_split(np.random.permutation(idx_areax), 2)

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons: #skip exec if not enough neurons in one of the populations
            continue

        # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        for istim,stim in enumerate([0]): # loop over orientations 

            idx_T               = ses.trialdata['stimCond']==stim

            #Get the activity for the two subsets of neurons for this stimulus condition:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]

            #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
            R2_cv[iapl,0,:,ises,istim],optim_rank[iapl,0,:,ises,istim],_  = RRR_wrapper_tensor(Y, X, 
                            nN=nsampleneurons,nK=None,nranks=nranks,nmodelfits=nmodelfits)

            #Get the activity for the two subsets of neurons for this stimulus condition:
            X                   = sessions[ises].tensor_behavout[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor_behavout[np.ix_(idx_areay,idx_T,idx_resp)]

            R2_cv[iapl,1,:,ises,istim],optim_rank[iapl,1,:,ises,istim],_  = RRR_wrapper_tensor(Y, X, 
                            nN=nsampleneurons,nK=None,nranks=nranks,nmodelfits=nmodelfits,fixed_rank=int(optim_rank[iapl,0,:,ises,istim].mean()))

            idx_ALRSP   = np.where(np.all((np.isin(ses.celldata['roi_name'],['AL','RSP']),
                                        ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
            
            if len(idx_ALRSP)>nsampleneurons: 
                #Get the activity for the two subsets of neurons for this stimulus condition:
                X                   = sessions[ises].tensor_neuralout[np.ix_(idx_areax,idx_T,idx_resp)]
                Y                   = sessions[ises].tensor_neuralout[np.ix_(idx_areay,idx_T,idx_resp)]

                R2_cv[iapl,2,:,ises,istim],optim_rank[iapl,2,:,ises,istim],_  = RRR_wrapper_tensor(Y,X,
                                nN=nsampleneurons,nK=None,nranks=nranks,nmodelfits=nmodelfits,fixed_rank=int(optim_rank[iapl,0,:,ises,istim].mean()))

            # #Get behavioral matrix: 
            # B                   = np.concatenate((sessions[ises].tensor_vid[np.ix_(range(np.shape(sessions[ises].tensor_vid)[0]),idx_T,idx_resp)],
            #                         sessions[ises].tensor_run[np.ix_(range(np.shape(sessions[ises].tensor_run)[0]),idx_T,idx_resp)]),axis=0)
            # B                   = B.reshape(np.shape(B)[0],-1).T
            # B                   = zscore(B,axis=0,nan_policy='omit')
            # si                  = SimpleImputer()
            # B                   = si.fit_transform(B)
            
            # X_r = np.reshape(X,(len(idx_areax),np.sum(idx_T)*ntimebins),order='C').T
            # _,_,X_behavout,_,ev  = regress_out_behavior_modulation(ses,X=B,Y=X_r,rank=rank_behavout,lam=0)
            # X_behavout = np.reshape(X_behavout.T,(len(idx_areax),np.sum(idx_T),ntimebins),order='C')

            # # plt.imshow(np.nanmean(X,axis=0),aspect='auto',vmin=-0.1,vmax=0.1)
            # # plt.imshow(np.nanmean(X_behavout,axis=0),aspect='auto',vmin=-0.1,vmax=0.1)
            
            # # plt.imshow(np.nanmean(X,axis=2),aspect='auto',vmin=-0.1,vmax=0.1)
            # # plt.imshow(np.nanmean(X_behavout,axis=2),aspect='auto',vmin=-0.1,vmax=0.1)

            # Y_r = np.reshape(Y,(len(idx_areay),np.sum(idx_T)*ntimebins),order='C').T
            # _,_,Y_behavout,_,ev  = regress_out_behavior_modulation(ses,X=B,Y=Y_r,rank=rank_behavout,lam=0)
            # # print(ev)
            # Y_behavout = np.reshape(Y_behavout.T,(len(idx_areay),np.sum(idx_T),ntimebins),order='C')

            # R2_cv[iapl,1,:,ises,istim],optim_rank[iapl,1,:,ises,istim],_  = RRR_wrapper_tensor(Y_behavout, X_behavout, 
            #                 nN=nsampleneurons,nK=None,nranks=nranks,nmodelfits=nmodelfits,fixed_rank=int(optim_rank[iapl,0,:,ises,istim].mean()))

            # idx_ALRSP   = np.where(np.all((np.isin(ses.celldata['roi_name'],['AL','RSP']),
            # # idx_ALRSP   = np.where(np.all((np.isin(ses.celldata['roi_name'],['PM','RSP']),
            #                             ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
            # if len(idx_ALRSP)>nsampleneurons: 
            #     Z                   = sessions[ises].tensor[np.ix_(idx_ALRSP,idx_T,idx_resp)]
                
            #     Z_r = np.reshape(Z,(len(idx_ALRSP),np.sum(idx_T)*ntimebins),order='C').T

            #     X_r = np.reshape(X,(len(idx_areax),np.sum(idx_T)*ntimebins),order='C').T
            #     _,_,X_neuralout,_,ev  = regress_out_behavior_modulation(ses,X=Z_r,Y=X_r,rank=rank_neuralout,lam=0)
            #     # print(ev)
            #     X_neuralout = np.reshape(X_neuralout.T,(len(idx_areax),np.sum(idx_T),ntimebins),order='C')

            #     Y_r = np.reshape(Y,(len(idx_areay),np.sum(idx_T)*ntimebins),order='C').T
            #     _,_,Y_neuralout,_,ev  = regress_out_behavior_modulation(ses,X=Z_r,Y=Y_r,rank=rank_behavout,lam=0)
            #     # print(ev)
            #     Y_neuralout = np.reshape(Y_neuralout.T,(len(idx_areay),np.sum(idx_T),ntimebins),order='C')

            #     R2_cv[iapl,2,:,ises,istim],optim_rank[iapl,2,:,ises,istim],_  = RRR_wrapper_tensor(Y_neuralout, X_neuralout, 
            #                 nN=nsampleneurons,nK=None,nranks=nranks,nmodelfits=nmodelfits,fixed_rank=int(optim_rank[iapl,0,:,ises,istim].mean()))

# #%% Show figure for each of the arealabelpairs and each of the dataversions
# for iversion, version in enumerate(dataversions):
#     #Reshape stim x sessions:
#     R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
#     optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
#     R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))

#     for idx in np.array([[0,1],[2,3]]):
#         clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
#         fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
#         # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

#%% Plotting:
idx_version = 0

clr = clrs_arealabelpairs[0]
R2_toplot = np.reshape(R2_cv[:,idx_version,:,:,:],(narealabelpairs,ntimebins,nSessions*nStim))
rank_toplot = np.reshape(optim_rank[:,idx_version,:,:,:],(narealabelpairs,ntimebins,nSessions*nStim))

t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(3,4,figsize=(13,10),sharex=True,sharey=False)

plotcontrasts = np.array([[[0,1],[2,3],[4,5],[6,7]],
                         [[8,9],[10,11],[12,13],[14,15]],
                         [[16,17],[18,19],[20,21],[22,23]]])

nR,nC,_ = np.shape(plotcontrasts)

clrs        = ['grey','red']

for iR in range(nR):
    for iC in range(nC):
        plotcontrast = plotcontrasts[iR,iC]
        ax = axes[iR,iC]
        handles = []
        for iapl,apl in enumerate(plotcontrast):
        # for iapl, arealabelpair in enumerate(arealabelpairs[idx]):
            # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealayerlabels[iapl],label=arealabelpair)
            # handles.append(shaded_error(t_axis,R2_toplot[iapl,:,:],error='sem',color=clrs_arealabelpairs[apl],alpha=0.3,ax=ax))
            handles.append(shaded_error(t_axis,R2_toplot[apl,:,:].T,error='sem',color=clrs[iapl],alpha=0.3,ax=ax))

        ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
        thickness = ax.get_ylim()[1]/20
        ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
        ax.legend(handles=handles,labels=list(arealabelpairs[plotcontrast]),loc='best',fontsize=8)
        my_legend_strip(ax)
        ax_nticks(ax,3)
        ax.set_xticks(t_ticks)
        ax.set_xticklabels(t_ticks)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('R2')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,savedir,'RRR_V1PM_GR_across_time',formats=['png'])
# my_savefig(fig,savedir,'RRR_V1PM_layer_labeled_GR_across_time',formats=['png'])

#%% Plotting:
idx_version = 0

clr = clrs_arealabelpairs[0]
R2_toplot = np.reshape(R2_cv[:,idx_version,:,:,:],(narealabelpairs,ntimebins,nSessions*nStim))
rank_toplot = np.reshape(optim_rank[:,idx_version,:,:,:],(narealabelpairs,ntimebins,nSessions*nStim))

# R2_toplot = np.nanmean(R2_cv[:,idx_version,:,:,:],axis=3)
# rank_toplot = np.nanmean(optim_rank[:,idx_version,:,:,:],(narealabelpairs,ntimebins,nSessions*nStim))

t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(3,4,figsize=(13,10),sharex=True,sharey=True)
# fig,axes = plt.subplots(3,4,figsize=(13,10),sharex=False,sharey=False)

plotcontrasts = np.array([[[0,1],[2,3],[4,5],[6,7]],
                         [[8,9],[10,11],[12,13],[14,15]],
                         [[16,17],[18,19],[20,21],[22,23]]])

nR,nC,_ = np.shape(plotcontrasts)

clrs        = ['grey','red']

for iR in range(nR):
    for iC in range(nC):
        plotcontrast = plotcontrasts[iR,iC]
        ax = axes[iR,iC]
        
        handles = []
        # clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
        datatoplot = R2_toplot[plotcontrast[1],:,:] - R2_toplot[plotcontrast[0],:,:]
        # datatoplot = (R2_toplot[plotcontrast[1],:,:] / R2_toplot[plotcontrast[0],:,:])*100-100
        handles.append(shaded_error(t_axis,datatoplot.T,error='sem',color=clrs[iapl],alpha=0.3,ax=ax))

        ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
        thickness = ax.get_ylim()[1]/20
        ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
        ax.legend(handles=handles,labels=list(arealabelpairs[plotcontrast]),loc='best',fontsize=8)
        my_legend_strip(ax)
        ax_nticks(ax,3)
        ax.set_xticks(t_ticks)
        ax.set_xticklabels(t_ticks)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('R2')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,savedir,'RRR_V1PM_GR_across_time',formats=['png'])
my_savefig(fig,savedir,'RRR_V1PM_diff_layer_labeled_GR_across_time',formats=['png'])

#%% 
# print(np.nanmean(R2_cv))
# R2_cv               = np.nanmean(R2_cv,axis=1)
# optim_rank          = np.nanmean(optim_rank,axis=1)
# R2_ranks            = np.nanmean(R2_ranks,axis=1)

# #%% Plotting:
# t_ticks = np.array([-1,0,1,2])
# fig,axes = plt.subplots(2,2,figsize=(5.5,5.5))
# ax = axes[0,0]
# handles = []
# for iapl, arealabelpair in enumerate(arealabelpairs[:4]):
#     # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
#     handles.append(shaded_error(t_axis,R2_cv[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
# ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
# thickness = ax.get_ylim()[1]/20
# ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
# ax.legend(handles=handles,labels=arealabelpairs,loc='best',fontsize=8)
# my_legend_strip(ax)
# ax_nticks(ax,3)
# ax.set_xticks(t_ticks)
# ax.set_xticklabels(t_ticks)
# ax.set_xlabel('Time (sec)')
# ax.set_ylabel('R2')

# ax = axes[1,0]
# handles = []
# for iapl, arealabelpair in enumerate(arealabelpairs[4:]):
#     # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
#     handles.append(shaded_error(t_axis,R2_cv[:,iapl+4,:],error='sem',color=clrs_arealabelpairs[iapl+4],alpha=0.3,ax=ax))
# ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
# thickness = ax.get_ylim()[1]/20
# ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
# ax.legend(handles=handles,labels=arealabelpairs[4:],loc='best',fontsize=8)
# my_legend_strip(ax)
# ax_nticks(ax,3)
# ax.set_xticks(t_ticks)
# ax.set_xticklabels(t_ticks)
# ax.set_xlabel('Time (sec)')
# ax.set_ylabel('R2')

# ax = axes[0,1]
# handles = []
# for iapl, arealabelpair in enumerate(arealabelpairs[:4]):
#     # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
#     handles.append(shaded_error(t_axis,optim_rank[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
# ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
# thickness = ax.get_ylim()[1]/20
# ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
# ax.legend(handles=handles,labels=arealabelpairs,loc='best',fontsize=8)
# my_legend_strip(ax)
# ax_nticks(ax,3)
# ax.set_xticks(t_ticks)
# ax.set_xticklabels(t_ticks)
# ax.set_xlabel('Time (sec)')
# ax.set_ylabel('Optimal rank')

# ax = axes[1,1]
# handles = []
# for iapl, arealabelpair in enumerate(arealabelpairs[4:]):
#     # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
#     handles.append(shaded_error(t_axis,optim_rank[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl+4],alpha=0.3,ax=ax))
# ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
# thickness = ax.get_ylim()[1]/20
# ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
# ax.legend(handles=handles,labels=arealabelpairs[4:],loc='best',fontsize=8)
# my_legend_strip(ax)
# ax_nticks(ax,3)
# ax.set_xticks(t_ticks)
# ax.set_xticklabels(t_ticks)
# ax.set_xlabel('Time (sec)')
# ax.set_ylabel('Optimal rank')

# sns.despine(fig=fig, top=True, right=True, offset = 3)
# plt.tight_layout()
# # my_savefig(fig,savedir,'RRR_V1PM_labeled_timeres')
# # my_savefig(fig,savedir,'RRR_V1PM_labeled_GR_across_time',formats=['png'])
# my_savefig(fig,savedir,'RRR_V1PM_labeled_GR_perOri_across_time',formats=['png'])


# #%% Plotting:
# t_ticks = np.array([-1,0,1,2])
# fig,axes = plt.subplots(1,2,figsize=(6,2.5))
# ax = axes[0]
# handles = []
# for iapl, arealabelpair in enumerate(arealabelpairs):
#     # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
#     handles.append(shaded_error(t_axis,R2_cv[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
# ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
# thickness = ax.get_ylim()[1]/20
# ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
# ax.legend(handles=handles,labels=arealabelpairs,loc='upper left',fontsize=8)
# my_legend_strip(ax)
# ax_nticks(ax,3)
# ax.set_xticks(t_ticks)
# ax.set_xticklabels(t_ticks)
# ax.set_xlabel('Time (sec)')
# ax.set_ylabel('R2')

# ax = axes[1]
# handles = []
# for iapl, arealabelpair in enumerate(arealabelpairs):
#     # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
#     handles.append(shaded_error(t_axis,optim_rank[:,iapl,:],error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
# ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
# thickness = ax.get_ylim()[1]/20
# ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
# ax.legend(handles=handles,labels=arealabelpairs,loc='upper left',fontsize=8)
# my_legend_strip(ax)
# ax_nticks(ax,3)
# ax.set_xticks(t_ticks)
# ax.set_xticklabels(t_ticks)
# ax.set_xlabel('Time (sec)')
# ax.set_ylabel('Optimal rank')

# sns.despine(fig=fig, top=True, right=True, offset = 3)
# plt.tight_layout()
# my_savefig(fig,savedir,'RRR_V1PM_GR_across_time',formats=['png'])
