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
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy import stats
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.plot_lib import * #get all the fixed color schemes
# from utils.corr_lib import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.pair_lib import value_matching
from utils.psth import compute_tensor
from params import load_params

params = load_params()
savedir = os.path.join(params['savedir'],'RRR','Labeling')

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


#%%  Load data properly:        
# ## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
# ## Parameters for temporal binning
# t_pre       = -1    #pre s
# t_post      = 1.9     #post s
# binsize     = 0.2
# vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
#                             ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

# behavfields = np.array(['runspeed','diffrunspeed'])

# for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
#     sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
#                                 calciumversion=params['calciumversion'])
#     [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
#                                  t_pre, t_post, method='binmean',binsize=binsize)
#     [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
#                                  t_pre, t_post, method='binmean',binsize=binsize)
#     sessions[ises].behaviordata['diffrunspeed'] = np.diff(sessions[ises].behaviordata['runspeed'],prepend=0)
#     [sessions[ises].tensor_run,t_axis] = compute_tensor(sessions[ises].behaviordata[behavfields], sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'], 
#                                  t_pre, t_post, method='binmean',binsize=binsize)
#     delattr(sessions[ises],'calciumdata')
#     delattr(sessions[ises],'behaviordata')
#     delattr(sessions[ises],'videodata')

# #%%  Load data properly:        
# for ises in range(nSessions):
#     sessions[ises].load_tensor(load_calciumdata=True,calciumversion=params['calciumversion'],keepraw=False)

# t_axis = sessions[0].t_axis

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

#%% Test wrapper function: (should be fast, just one model fit etc. explained variance around 0.05-0.07)
nsampleneurons  = 20
nranks          = 25
nmodelfits      = 1 #number of times new neurons are resampled 
kfold           = 5
R2_cv           = np.full((nSessions),np.nan)
optim_rank      = np.full((nSessions),np.nan)
idx_resp        = np.where((t_axis>=0) & (t_axis<=1.5))[0]

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<20),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<20),axis=0))[0]
    
    #on tensor during the response:
    X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)].reshape(len(idx_areax),-1).T
    Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)].reshape(len(idx_areay),-1).T

    #on averaged response:
    # X                   = np.nanmean(sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)],axis=2).T
    # Y                   = np.nanmean(sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)],axis=2).T

    R2_cv[ises],optim_rank[ises],_      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))

#%% 
######  ######  ######     #          #    ######     #     #  #####     #     # #     # #       
#     # #     # #     #    #         # #   #     #    #     # #     #    #     # ##    # #       
#     # #     # #     #    #        #   #  #     #    #     # #          #     # # #   # #       
######  ######  ######     #       #     # ######     #     #  #####     #     # #  #  # #       
#   #   #   #   #   #      #       ####### #     #     #   #        #    #     # #   # # #       
#    #  #    #  #    #     #       #     # #     #      # #   #     #    #     # #    ## #       
#     # #     # #     #    ####### #     # ######        #     #####      #####  #     # ####### 

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

dataversions        = np.array(['original','behavout','neuralout'])
dataversions        = np.array(['original','behavout'])
# dataversions        = np.array(['original','','neuralout'])
# dataversions        = np.array(['original'])
# dataversions        = np.array(['neuralout'])
nversions           = len(dataversions) #0: original,1:behavior out, 2:neural out
rank_behavout       = 5
rank_neuralout      = 5
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
                    'V1lab-PMunl',

                    'PMunl-V1unl', #Feedback pairs
                    'PMlab-V1unl',

                    # Comparing looped pairs:
                    'V1unl-PMlab',
                    'V1lab-PMlab',
                    'PMunl-V1lab',
                    'PMlab-V1lab',

                    #Feedforward selectivity
                    'V1unl-ALunl', #to AL
                    'V1lab-ALunl',
                    'V1unl-RSPunl', #to RSP
                    'V1lab-RSPunl',

                    #Feedback selectivity
                    'PMunl-ALunl', #to AL
                    'PMlab-ALunl',
                    'PMunl-RSPunl',#to RSP
                    'PMlab-RSPunl'
                    ]

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

R2_cv               = np.full((narealabelpairs,3,nSessions,nStim),np.nan)
optim_rank          = np.full((narealabelpairs,3,nSessions,nStim),np.nan)
R2_ranks            = np.full((narealabelpairs,3,nSessions,nStim,nranks,nmodelfits,kfold),np.nan)

nmatchbins          = 10
valuematching       = None
# valuematching       = 'noise_level'
# valuematching       = 'event_rate'
# valuematching       = 'skew'
# valuematching       = 'meanF'
sampleN = np.zeros((nSessions))

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different populations'):
for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting RRR model for different populations'):
    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=30)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    #take the smallest sample size
    allpops             = np.array([i.split('-') for i in arealabelpairs])
    sourcepops,targetpops          = allpops[:,0],allpops[:,1]
    # nsampleneurons      = np.min(np.concatenate(([np.sum(np.all((ses.celldata['arealabel']==i,
    #                                              ses.celldata['noise_level']<maxnoiselevel,
    #                                              idx_nearby),axis=0)) for i in sourcepops],
    #                                 [np.sum(np.all((ses.celldata['arealabel']==i,
    #                                              ses.celldata['noise_level']<maxnoiselevel
    #                                              ),axis=0)) for i in targetpops])))

    nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
                                                 ses.celldata['noise_level']<maxnoiselevel,
                                                 idx_nearby),axis=0)) for i in sourcepops],
                                  )
    sampleN[ises] = nsampleneurons

    # if not hasattr(ses,'tensor_behavout') and 'behavout' in dataversions: 
    #     sessions[ises].tensor_behavout = copy.copy(sessions[ises].tensor)
    #     #Get behavioral matrix:
    #     B                   = np.concatenate((sessions[ises].tensor_vid,
    #                             sessions[ises].tensor_run),axis=0)
    #     B                   = B.reshape(np.shape(B)[0],-1).T
    #     B                   = zscore(B,axis=0,nan_policy='omit')
    #     B                   = B[:,~np.all(np.isnan(B),axis=0)]
        
    #     for area in ['V1','PM']:
    #         idx_N    = np.where(np.all((ses.celldata['roi_name']==area,
    #                                     idx_nearby,
    #                                     ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
           
    #         N,K,T = np.shape(sessions[ises].tensor[idx_N])
    #         Y_r = np.reshape(sessions[ises].tensor[idx_N],(N,K*T),order='C').T
    #         Y_orig,Y_hat,Y_out  = regress_out_cv(X=B,Y=Y_r,rank=np.min([rank_behavout,len(idx_N)-1]),
    #                                              lam=0,kfold=5)
    #         # print(area,EV(Y_orig,Y_hat))
    #         sessions[ises].tensor_behavout[idx_N,:,:] = np.reshape(Y_out.T,(N,K,T),order='C')

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

    # plt.imshow(np.nanmean(sessions[ises].tensor,axis=2),aspect='auto',vmin=-0.1,vmax=0.1)
    # plt.imshow(np.nanmean(sessions[ises].tensor_behavout,axis=2),aspect='auto',vmin=-0.1,vmax=0.1)

    # if not hasattr(ses,'tensor_neuralout') and 'neuralout' in dataversions: 
    if 'neuralout' in dataversions: 
        idx_ALRSP   = np.where(np.all((np.isin(ses.celldata['roi_name'],['AL','RSP']),
                                    ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
        if len(idx_ALRSP)>nsampleneurons:
            sessions[ises].tensor_neuralout = copy.copy(sessions[ises].tensor)
            _,K,T = np.shape(sessions[ises].tensor)
            X                   = sessions[ises].tensor[idx_ALRSP,:,:]
            X_r = np.reshape(X,(len(idx_ALRSP),K*T),order='C').T
            
            for area in ['V1','PM']:
                idx_N    = np.where(np.all((ses.celldata['roi_name']==area,
                                            idx_nearby,
                                            ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
                if len(idx_N)>nsampleneurons:
                    Y                   = sessions[ises].tensor[idx_N,:,:]
                    Y_r = np.reshape(Y,(len(idx_N),K*T),order='C').T

                    # _,_,Y_neuralout,_,ev  = regress_out_behavior_modulation(ses,X=X_r,Y=Y_r,rank=np.min([rank_neuralout,len(idx_N)]),lam=0)
                    # print(ev)
                    # sessions[ises].tensor_neuralout[idx_N,:,:] = np.reshape(Y_neuralout.T,(len(idx_N),K,T),order='C')
                    
                    Y_orig,Y_hat,Y_out  = regress_out_cv(X=X_r,Y=Y_r,rank=np.min([rank_neuralout,len(idx_N)-1]),
                                                    lam=0,kfold=5)
                    # print(area,EV(Y_orig,Y_hat))
                    sessions[ises].tensor_neuralout[idx_N,:,:] = np.reshape(Y_out.T,(len(idx_N),K,T),order='C')

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

        if valuematching is not None: #balance populations by some metric from celldata:
            #Get value to match from celldata:
            values      = sessions[ises].celldata[valuematching].to_numpy()
            idx_joint   = np.concatenate((idx_areax,idx_areay))
            group       = np.concatenate((np.zeros(len(idx_areax)),np.ones(len(idx_areay))))
            idx_sub     = value_matching(idx_joint,group,values[idx_joint],bins=nmatchbins,showFig=False)
            idx_areax   = np.intersect1d(idx_areax,idx_sub) #recover subset from idx_joint
            idx_areay   = np.intersect1d(idx_areay,idx_sub)

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons: #skip exec if not enough neurons in one of the populations
            continue
        
        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim

            # idx_T         = np.all((
            #                         ses.respmat_videome < params['maxvideome'],
            #                         ses.respmat_runspeed < params['maxrunspeed'],
            #                         ses.trialdata['stimCond']==stim),axis=0)
            
            if 'original' in dataversions:
                X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
                Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
                
                # reshape to neurons x time points
                X                   = X.reshape(len(idx_areax),-1).T
                Y                   = Y.reshape(len(idx_areay),-1).T

                #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
                R2_cv[iapl,0,ises,istim],optim_rank[iapl,0,ises,istim],R2_ranks[iapl,0,ises,istim,:,:,:]  = RRR_wrapper(Y, X, 
                                nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits,fixed_rank=fixed_rank)

            if 'behavout' in dataversions:
                X                   = sessions[ises].tensor_behavout[np.ix_(idx_areax,idx_T,idx_resp)]
                Y                   = sessions[ises].tensor_behavout[np.ix_(idx_areay,idx_T,idx_resp)]
                
                # reshape to neurons x time points
                X                   = X.reshape(len(idx_areax),-1).T
                Y                   = Y.reshape(len(idx_areay),-1).T

                #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
                R2_cv[iapl,1,ises,istim],optim_rank[iapl,1,ises,istim],R2_ranks[iapl,1,ises,istim,:,:,:]  = RRR_wrapper(Y, X, 
                                nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits,fixed_rank=fixed_rank)

            if 'neuralout' in dataversions and hasattr(ses,'tensor_neuralout') and iapl<8:
                X                   = sessions[ises].tensor_neuralout[np.ix_(idx_areax,idx_T,idx_resp)]
                Y                   = sessions[ises].tensor_neuralout[np.ix_(idx_areay,idx_T,idx_resp)]
                
                # reshape to neurons x time points
                X                   = X.reshape(len(idx_areax),-1).T
                Y                   = Y.reshape(len(idx_areay),-1).T

                #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
                R2_cv[iapl,2,ises,istim],optim_rank[iapl,2,ises,istim],R2_ranks[iapl,2,ises,istim,:,:,:]  = RRR_wrapper(Y, X, 
                                nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits,fixed_rank=fixed_rank)

            # #Get behavioral matrix: 
            # B                   = np.concatenate((sessions[ises].tensor_vid[np.ix_(range(np.shape(sessions[ises].tensor_vid)[0]),idx_T,idx_resp)],
            #                         sessions[ises].tensor_run[np.ix_(range(np.shape(sessions[ises].tensor_run)[0]),idx_T,idx_resp)]),axis=0)
            # B                   = B.reshape(np.shape(B)[0],-1).T
            # B                   = zscore(B,axis=0,nan_policy='omit')

            # si                  = SimpleImputer()
            # B                   = si.fit_transform(B)
            # _,_,X_behavout,_,_  = regress_out_behavior_modulation(ses,X=B,Y=X,rank=rank_behavout,lam=0)

            # _,_,Y_behavout,_,_  = regress_out_behavior_modulation(ses,X=B,Y=Y,rank=rank_behavout,lam=0)

            # R2_cv[iapl,1,ises,istim],optim_rank[iapl,1,ises,istim],R2_ranks[iapl,1,ises,istim,:,:,:]  = RRR_wrapper(Y_behavout, X_behavout, 
            #                 nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

            # idx_ALRSP   = np.where(np.all((np.isin(ses.celldata['roi_name'],['AL','RSP']),
            #                             ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
            # if len(idx_ALRSP)>nsampleneurons: 
            #     Z                   = sessions[ises].tensor[np.ix_(idx_ALRSP,idx_T,idx_resp)]
            #     Z                   -= np.mean(Z,axis=1,keepdims=True)
            #     Z                   = Z.reshape(len(idx_ALRSP),-1).T
            #     Z                   = zscore(Z,axis=0,nan_policy='omit')  #Z score activity for each neuron

            #     _,_,X_neuralout,_,_  = regress_out_behavior_modulation(ses,X=Z,Y=X,rank=rank_neuralout,lam=0)

            #     _,_,Y_neuralout,_,_  = regress_out_behavior_modulation(ses,X=Z,Y=Y,rank=rank_neuralout,lam=0)
                
            #     R2_cv[iapl,2,ises,istim],optim_rank[iapl,2,ises,istim],R2_ranks[iapl,2,ises,istim,:,:,:]  = RRR_wrapper(Y_neuralout, X_neuralout, 
            #                 nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print('Population size: %d (%d-%d)' % (np.mean(sampleN[sampleN>minsampleneurons]),np.min(sampleN[sampleN>minsampleneurons]),np.max(sampleN[sampleN>minsampleneurons])))

#%% 

#          #    ######     #     #  #####     #     # #     # #       
#         # #   #     #    #     # #     #    #     # ##    # #       
#        #   #  #     #    #     # #          #     # # #   # #       
#       #     # ######     #     #  #####     #     # #  #  # #       
#       ####### #     #     #   #        #    #     # #   # # #       
#       #     # #     #      # #   #     #    #     # #    ## #       
####### #     # ######        #     #####      #####  #     # ####### 


#%% Show figure for each of the arealabelpairs and each of the dataversions
for iversion, version in enumerate(dataversions):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))
    # R2_data                 = np.nanmean(R2_cv[:,iversion,:,:],axis=2)
    # optim_rank_data                 = np.nanmean(optim_rank[:,iversion,:,:],axis=2)
    if np.any(~np.isnan(R2_data)):
        for idx in np.array([[0,1],[2,3]]):
            clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
            fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
            # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

#%% Reshape data to stretch stim x sessions
nversions = 3
R2_cv_2         = np.reshape(R2_cv,(narealabelpairs,nversions,nSessions*nStim))
optim_rank_2    = np.reshape(optim_rank,(narealabelpairs,nversions,nSessions*nStim))
R2_ranks_2      = np.reshape(R2_ranks,(narealabelpairs,nversions,nSessions*nStim,nranks,nmodelfits,kfold))

# R2_cv_2[optim_rank_2<2] = np.nan

# R2_cv_2         = np.nanmean(R2_cv,axis=3)
# optim_rank_2    = np.nanmean(optim_rank,axis=3)
# R2_ranks_2      = np.reshape(R2_ranks,(narealabelpairs,nversions,nSessions*nStim,nranks,nmodelfits,kfold))

# R2_cv_2[optim_rank_2<2] = np.nan

# R2_cv_2 = np.nanmean(R2_ranks[:,:,:,:,4,:,:],axis=(-1,-2,-3))
# rank = 5
# R2_cv_2         = np.reshape(np.nanmean(R2_ranks[:,:,:,:,rank,:,:],axis=(-1,-2)),(narealabelpairs,nversions,nSessions*nStim))
# optim_rank_2     = np.reshape(np.nanmean(optim_rank[:,:,:,:,rank,:,:],axis=(-1,-2)),(narealabelpairs,nversions,nSessions*nStim))
# R2_ranks_2      = np.reshape(R2_ranks,(narealabelpairs,nversions,nSessions*nStim,nranks,nmodelfits,kfold))

#%% Define the ratio of R2 between V1PM and V1ND
ratiodata_FF       = (R2_cv_2[1] / R2_cv_2[0])*100-100 #
ratiodata_FB       = (R2_cv_2[3] / R2_cv_2[2])*100-100


#%% Plot the R2 for each of the arealabelpairs and each of the dataversions
#Residual variance explained goes down with behavior or brainwide activity regressed out: 
nversions = 3
fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5*cm,4*cm))
ax = axes[0]
# ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[0],axis=1),yerr=np.nanstd(R2_cv_2[0],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[0])
# ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[0],axis=1),yerr=np.nanstd(R2_cv_2[0],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[0])
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[[0,1]],axis=(0,2)),yerr=np.nanstd(R2_cv_2[[0,1]],axis=(0,2))/np.sqrt(nSessions),color='k')
ax.set_ylabel("Performance $R^2$")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FF')

ax = axes[1]
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[[2,3]],axis=(0,2)),yerr=np.nanstd(R2_cv_2[[2,3]],axis=(0,2))/np.sqrt(nSessions),color='k')
# ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[2],axis=1),yerr=np.nanstd(R2_cv_2[2],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[2])
# ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[3],axis=1),yerr=np.nanstd(R2_cv_2[3],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[3])
ax.axhline(y=0,color='k',linestyle='--')
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_title('FB')

plt.tight_layout()
sns.despine(fig=fig,trim=True)
# my_savefig(fig,savedir,'RRR_cvR2_FF_FB_diffversions_%dsessions' % (nSessions))

#%% Plot the R2 for each of the arealabelpairs and each of the dataversions
# #Residual variance explained goes down with behavior or brainwide activity regressed out: 

fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5,4))
ax = axes[0]
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[0],axis=1),yerr=np.nanstd(R2_cv_2[0],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[0])
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[1],axis=1),yerr=np.nanstd(R2_cv_2[1],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[1])
ax.set_ylabel("Performance $R^2$")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FF')

ax = axes[1]
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[2],axis=1),yerr=np.nanstd(R2_cv_2[2],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[2])
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[3],axis=1),yerr=np.nanstd(R2_cv_2[3],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[3])
ax.axhline(y=0,color='k',linestyle='--')
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_title('FB')

plt.tight_layout()
sns.despine(fig=fig,trim=True)
# my_savefig(fig,savedir,'RRR_cvR2_FF_FB_labeled_diffversions_%dsessions' % (nSessions))

#%% Plot the R2 for each of the arealabelpairs and each of the dataversions
#Residual variance explained goes down with behavior or brainwide activity regressed out: 

fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5,4))
ax = axes[0]
ax.errorbar(x=range(nversions),y=np.nanmean(optim_rank_2[0],axis=1),yerr=np.nanstd(optim_rank_2[0],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[0])
ax.errorbar(x=range(nversions),y=np.nanmean(optim_rank_2[1],axis=1),yerr=np.nanstd(optim_rank_2[1],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[1])
ax.set_ylabel("Rank")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FF')

ax = axes[1]
ax.errorbar(x=range(nversions),y=np.nanmean(optim_rank_2[2],axis=1),yerr=np.nanstd(optim_rank_2[2],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[2])
ax.errorbar(x=range(nversions),y=np.nanmean(optim_rank_2[3],axis=1),yerr=np.nanstd(optim_rank_2[3],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[3])
ax.axhline(y=0,color='k',linestyle='--')
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_title('FB')

plt.tight_layout()
sns.despine(fig=fig,trim=True)
# my_savefig(fig,savedir,'RRR_rank_FF_FB_diffversions_%dsessions' % (nSessions))


#%% Make the figure: 
fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5,4))
ax = axes[0]
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FF,axis=1),yerr=np.nanstd(ratiodata_FF,axis=1)/np.sqrt(np.shape(ratiodata_FF)[1]))
ax.set_ylabel("Relative performance in %\n$V1_{PM}$ vs. $V1_{ND}$")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FF')

for it,(ix,iy) in enumerate(np.array([[0,1],[0,2],[1,2]])):
    h,p = stats.ttest_rel(ratiodata_FF[ix],ratiodata_FF[iy],nan_policy='omit')
    add_stat_annotation(ax, ix,iy, 30+it*2, p, h=1)

ax = axes[1]
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FB,axis=1),yerr=np.nanstd(ratiodata_FB,axis=1)/np.sqrt(np.shape(ratiodata_FF)[1]))
for it,(ix,iy) in enumerate(np.array([[0,1],[0,2],[1,2]])):
    h,p = stats.ttest_rel(ratiodata_FB[ix],ratiodata_FB[iy],nan_policy='omit')
    add_stat_annotation(ax, ix,iy, 30+it*2, p, h=1)

ax.axhline(y=0,color='k',linestyle='--')
ax.set_ylabel("Relative performance in %\n$PM_{V1}$ vs $PM_{ND}$")
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_title('FB')

plt.tight_layout()
sns.despine(fig=fig,trim=True)
# my_savefig(fig,savedir,'RRR_cvR2_ratio_FF_FB_diffversions_%dsessions' % (nSessions))


#%% 

#######    #    ######   #####  ####### #######       #    ######  #######    #    
   #      # #   #     # #     # #          #         # #   #     # #         # #   
   #     #   #  #     # #       #          #        #   #  #     # #        #   #  
   #    #     # ######  #  #### #####      #       #     # ######  #####   #     # 
   #    ####### #   #   #     # #          #       ####### #   #   #       ####### 
   #    #     # #    #  #     # #          #       #     # #    #  #       #     # 
   #    #     # #     #  #####  #######    #       #     # #     # ####### #     # 

 #####  ######  #######  #####  ### ####### ###  #####  ### ####### #     # 
#     # #     # #       #     #  #  #        #  #     #  #     #     #   #  
#       #     # #       #        #  #        #  #        #     #      # #   
 #####  ######  #####   #        #  #####    #  #        #     #       #    
      # #       #       #        #  #        #  #        #     #       #    
#     # #       #       #     #  #  #        #  #     #  #     #       #    
 #####  #       #######  #####  ### #       ###  #####  ###    #       #                       



#%% 
idx_FF = np.array([[0,1],[8,9],[10,11]])
idx_FB = np.array([[2,3],[12,13],[14,15]])

#%% 
for iversion, version in enumerate(dataversions):
# for iversion, version in enumerate(dataversions[:1]):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))

    for idx in idx_FF:
        if np.any(~np.isnan(R2_data)):
            clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
            fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
            # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

#%% 
for iversion, version in enumerate(dataversions):
# for iversion, version in enumerate(dataversions[:1]):
# for iversion, version in enumerate(dataversions[2:]):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))

    for idx in idx_FB:
        if np.any(~np.isnan(R2_data)):
            clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
            fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
            # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))


#%% Define the ratio of R2 between V1PM and V1ND
ratiodata_FF       = (R2_cv_2[9] / R2_cv_2[8])*100-100 #
ratiodata_FB       = (R2_cv_2[13] / R2_cv_2[12])*100-100

# ratiodata_FF       = (R2_cv_2[11] / R2_cv_2[10])*100-100 #
# ratiodata_FB       = (R2_cv_2[15] / R2_cv_2[14])*100-100

#%% Make the figure: 
fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5,4))
ax = axes[0]
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FF,axis=1),yerr=np.nanstd(ratiodata_FF,axis=1)/np.sqrt(np.shape(ratiodata_FF)[1]))
ax.set_ylabel("Relative performance in %\n$V1_{PM}$ vs. $V1_{ND}$")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('V1-->AL')

for it in range(nversions):
    h,p = stats.ttest_1samp(ratiodata_FF[it],0,nan_policy='omit')
    add_stat_annotation(ax, it,it,np.nanmean(ratiodata_FF,axis=1)[it]+5, p, h=None)

for it,(ix,iy) in enumerate(np.array([[0,1],[0,2],[1,2]])):
    h,p = stats.ttest_rel(ratiodata_FF[ix],ratiodata_FF[iy],nan_policy='omit')
    # add_stat_annotation(ax, ix,iy, 80+it*10, p, h=None)

ax = axes[1]
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FB,axis=1),yerr=np.nanstd(ratiodata_FB,axis=1)/np.sqrt(np.shape(ratiodata_FF)[1]))
for it in range(nversions):
    h,p = stats.ttest_1samp(ratiodata_FB[it],0,nan_policy='omit')
    add_stat_annotation(ax, it,it,np.nanmean(ratiodata_FB,axis=1)[it]+5, p, h=None)

for it,(ix,iy) in enumerate(np.array([[0,1],[0,2],[1,2]])):
    h,p = stats.ttest_rel(ratiodata_FB[ix],ratiodata_FB[iy],nan_policy='omit')
    # add_stat_annotation(ax, ix,iy, 80+it*10, p, h=None)

ax.axhline(y=0,color='k',linestyle='--')
ax.set_ylabel("Relative performance in %\n$PM_{V1}$ vs $PM_{ND}$")
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_title('FB')

plt.tight_layout()
sns.despine(fig=fig,trim=True)
# my_savefig(fig,savedir,'RRR_cvR2_ratio_FF_FB_AL_diffversions_%dsessions' % (nSessions))

#%% 

#       ####### ####### ######  ####### ######  
#       #     # #     # #     # #       #     # 
#       #     # #     # #     # #       #     # 
#       #     # #     # ######  #####   #     # 
#       #     # #     # #       #       #     # 
#       #     # #     # #       #       #     # 
####### ####### ####### #       ####### ######  

#%% 
idx_FF = np.array([[0,5]])
idx_FB = np.array([[2,7]])

#%% 
for iversion, version in enumerate(dataversions):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))
    
    # R2_data                 = np.nanmean(R2_cv[:,iversion,:,:],axis=2)
    # optim_rank_data         = np.nanmean(optim_rank[:,iversion,:,:],axis=2)

    for idx in idx_FF:
        if np.any(~np.isnan(R2_data)):
            clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
            fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
            # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

#%% 
for iversion, version in enumerate(dataversions):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))

    for idx in idx_FB:
        if np.any(~np.isnan(R2_data)):
            clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
            fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
            # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))


#%% Define the ratio of R2 between V1PM and V1ND
ratiodata_FF       = (R2_cv_2[5] / R2_cv_2[0])*100-100 #
ratiodata_FB       = (R2_cv_2[7] / R2_cv_2[2])*100-100

# ratiodata_FF       = (R2_cv_2[8] - R2_cv_2[9])*100-100 #
# ratiodata_FB       = (R2_cv_2[12] -  R2_cv_2[13])*100-100

#%% Make the figure: 
fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5,4))
ax = axes[0]
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FF,axis=1),yerr=np.nanstd(ratiodata_FF,axis=1)/np.sqrt(np.shape(ratiodata_FF)[1]))
ax.set_ylabel("Relative performance in %\n$V1_{PM}$ vs. $V1_{ND}$")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FF')

for it in range(nversions):
    h,p = stats.ttest_1samp(ratiodata_FF[it],0,nan_policy='omit')
    add_stat_annotation(ax, it,it,np.nanmean(ratiodata_FF,axis=1)[it]+5, p, h=None)

for it,(ix,iy) in enumerate(np.array([[0,1],[0,2],[1,2]])):
    h,p = stats.ttest_rel(ratiodata_FF[ix],ratiodata_FF[iy],nan_policy='omit')
    add_stat_annotation(ax, ix,iy, 80+it*10, p, h=None)

ax = axes[1]
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FB,axis=1),yerr=np.nanstd(ratiodata_FB,axis=1)/np.sqrt(np.shape(ratiodata_FF)[1]))
for it in range(nversions):
    h,p = stats.ttest_1samp(ratiodata_FB[it],0,nan_policy='omit')
    add_stat_annotation(ax, it,it,np.nanmean(ratiodata_FB,axis=1)[it]+5, p, h=None)

for it,(ix,iy) in enumerate(np.array([[0,1],[0,2],[1,2]])):
    h,p = stats.ttest_rel(ratiodata_FB[ix],ratiodata_FB[iy],nan_policy='omit')
    add_stat_annotation(ax, ix,iy, 80+it*10, p, h=None)

ax.axhline(y=0,color='k',linestyle='--')
ax.set_ylabel("Relative performance in %\n$PM_{V1}$ vs $PM_{ND}$")
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_title('FB')

plt.tight_layout()
sns.despine(fig=fig,trim=True)


#%% Average across stim:
# R2_cv_2         = np.nanmean(R2_cv,axis=2)
# optim_rank_2    = np.nanmean(optim_rank,axis=2)
# R2_ranks_2      = np.nanmean(R2_ranks,axis=2)

# #%% Reshape stim x sessions:
# R2_cv_2         = np.reshape(R2_cv,(narealabelpairs,nSessions*nStim))
# optim_rank_2    = np.reshape(optim_rank,(narealabelpairs,nSessions*nStim))
# R2_ranks_2      = np.reshape(R2_ranks,(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))

# R2_cv_2[R2_cv_2==0] = np.nan

# #%% Plot the R2 performance and number of dimensions per area pair
# # fig         = plot_RRR_R2_arealabels(R2_cv,optim_rank,R2_ranks,arealabelpairs,clrs_arealabelpairs)
# # my_savefig(fig,savedir,'RRR_cvR2_RegressOutBehavior_V1PM_LabUnl_%dsessions' % nSessions)
# # fig         = plot_RRR_R2_arealabels(R2_cv[:2],optim_rank[:2],R2_ranks[:2],arealabelpairs[:2],clrs_arealabelpairs[:2])
# # fig         = plot_RRR_R2_arealabels(R2_cv[2:4],optim_rank[2:4],R2_ranks[2:4],arealabelpairs[2:4],clrs_arealabelpairs[2:4])
# # fig         = plot_RRR_R2_arealabels(R2_cv[[1,3]],optim_rank[[1,3]],R2_ranks[[0,3]],arealabelpairs[[0,3]],clrs_arealabelpairs[2:4])

# #Same target population:
# # for idx in np.array([[0,1],[2,3],[4,5],[6,7]]):
# for idx in np.array([[0,1],[2,3]]):
#     clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
#     # fig         = plot_RRR_R2_arealabels_paired(R2_cv[idx],optim_rank[idx],R2_ranks[idx],np.array(arealabelpairs)[idx],clrs)
#     # fig         = plot_RRR_R2_arealabels_paired(R2_cv[idx],optim_rank[idx],R2_ranks[idx],np.array(arealabelpairs)[idx],clrs)
#     fig         = plot_RRR_R2_arealabels_paired(R2_cv_2[idx],optim_rank_2[idx],R2_ranks_2[idx],np.array(arealabelpairs)[idx],clrs)
#     # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

# # for idx in np.array([[0,1],[2,3],[4,5],[6,7]]):
#     # mean,sd = np.nanmean(R2_cv[idx[1]] / R2_cv[idx[0]])*100-100,np.nanstd(R2_cv[idx[1]] / R2_cv[idx[0]])
#     # mean,sd = np.nanmean(R2_cv_2[idx[1]] / R2_cv_2[idx[0]])*100-100,np.nanstd(R2_cv_2[idx[1]] / R2_cv_2[idx[0]])*100
#     mean,sd = np.nanmean(R2_cv_2[idx[1]]) / np.nanmean(R2_cv_2[idx[0]])*100-100,np.nanstd(R2_cv_2[idx[1]] / R2_cv_2[idx[0]])*100
#     print('%s vs %s: %2.1f %% +/- %2.1f' % (arealabelpairs[idx[1]],arealabelpairs[idx[0]],mean,sd))

# #Different target population:
# for idx in np.array([[0,3],[4,7]]):
#     clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
#     # fig         = plot_RRR_R2_arealabels_paired(R2_cv[idx],optim_rank[idx],R2_ranks[idx],np.array(arealabelpairs)[idx],clrs)
#     fig         = plot_RRR_R2_arealabels_paired(R2_cv_2[idx],optim_rank_2[idx],R2_ranks_2[idx],np.array(arealabelpairs)[idx],clrs=)
#     my_savefig(fig,savedir,'RRR_cvR2_diffTarget_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

# for idx in np.array([[0,3],[4,7]]):
#     # mean,sd = np.nanmean(R2_cv[idx[1]] / R2_cv[idx[0]])*100-100,np.nanstd(R2_cv[idx[1]] / R2_cv[idx[0]])
#     mean,sd = np.nanmean(R2_cv_2[idx[1]]) / np.nanmean(R2_cv_2[idx[0]])*100-100,np.nanstd(R2_cv_2[idx[1]] / R2_cv_2[idx[0]])*100
#     print('%s vs %s: %2.1f %% +/- %2.1f' % (arealabelpairs[idx[1]],arealabelpairs[idx[0]],mean,sd))

# #%% AL and RSP sessions only:
# normalize   = False
# for idx in np.array([[0,1],[2,3],[4,5]]):
#     clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
#     fig         = plot_RRR_R2_arealabels_paired(R2_cv[idx],optim_rank[idx],R2_ranks[idx],np.array(arealabelpairs)[idx],clrs,normalize=normalize)
#     # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%d_ALRSP_sessions' % (arealabelpairs[idx[1]],version,nSessions))

# for idx in np.array([[0,1],[2,3],[4,5]]):
#     mean,sd = np.nanmean(R2_cv[idx[1]] / R2_cv[idx[0]])*100-100,np.nanstd(R2_cv[idx[1]] / R2_cv[idx[0]])
#     print('%s vs %s: %2.1f %% +/- %2.1f' % (arealabelpairs[idx[1]],arealabelpairs[idx[0]],mean,sd))

#%% LAB and UNL in same population

#%% Parameters for RRR for size-matched populations of V1 and PM labeled and unlabeled neurons
lam                 = 0
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 10 #number of times new neurons are resampled - many for final run
kfold               = 5
maxnoiselevel       = 20
nStim               = 16

# idx_resp            = np.where((t_axis>=0) & (t_axis<=1))[0]
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
ntimebins           = len(idx_resp)
minsampleneurons    = 10
filter_nearby       = True

dataversions        = np.array(['original','behavout','neuralout'])
dataversions        = np.array(['original','behavout'])
# dataversions        = np.array(['original','','neuralout'])
dataversions        = np.array(['original'])
# dataversions        = np.array(['neuralout'])
nversions           = len(dataversions) #0: original,1:behavior out, 2:neural out
rank_behavout       = 5
rank_neuralout      = 5
fixed_rank          = None

#%%

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Deleting data'):
    if hasattr(ses,'tensor_neuralout'):
        delattr(sessions[ises],'tensor_neuralout')
    if hasattr(ses,'tensor_behavout'):
        delattr(sessions[ises],'tensor_behavout')


#%% Do RRR of V1 and PM labeled and unlabeled neurons

sourcearealabelpairs = ['V1unl','V1lab']
targetarealabelpair = 'PMunl'

clrs_arealabelpairs = get_clr_area_labeled(sourcearealabelpairs)
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 50
nranks              = 5
nmodelfits          = 20
R2_neuron           = np.full((narealabelpairs,Nsub,nSessions,nStim,nranks,kfold,nmodelfits),np.nan)
# R2_cv               = np.full((narealabelpairs,3,nSessions,nStim),np.nan)
# optim_rank          = np.full((narealabelpairs,3,nSessions,nStim),np.nan)
# R2_ranks            = np.full((narealabelpairs,3,nSessions,nStim,nranks,nmodelfits,kfold),np.nan)
# R2_ranks            = np.full((narealabelpairs,3,nSessions,nStim,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different populations'):
    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=30)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    idx_areax1          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[0],
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    idx_areax2          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[1],
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    idx_areay       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpair,
                                            ses.celldata['noise_level']<maxnoiselevel,	
                                            idx_nearby),axis=0))[0]

    if len(idx_areax1)<Nsub or len(idx_areax2)<Nsub or len(idx_areay)<Nsub: #skip exec if not enough neurons in one of the populations
        continue

    for imf in range(nmodelfits):
        idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
        idx_areax2_sub       = np.random.choice(idx_areax2,Nsub,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,Nsub,replace=False)

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim

            X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
            X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

            # reshape to neurons x time points
            X1                  = X1.reshape(len(idx_areax1_sub),-1).T
            X2                  = X2.reshape(len(idx_areax2_sub),-1).T
            Y                   = Y.reshape(len(idx_areay_sub),-1).T

            X1  = zscore(X1,axis=0)
            X2  = zscore(X2,axis=0)
            Y   = zscore(Y,axis=0)

            # OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS    
            R2_kfold    = np.zeros((kfold))
            kf          = KFold(n_splits=kfold)
            for ikf, (idx_train, idx_test) in enumerate(kf.split(X1)):
                X1_train, X1_test     = X1[idx_train], X1[idx_test]
                X2_train, X2_test     = X2[idx_train], X2[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                #RRR X1 to Y
                B_hat_train         = LM(Y_train,X1_train, lam=lam)
                Y_hat_train         = X1_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                for r in range(nranks):
                    B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                    # B_rrr           = B_hat_train @ V[:r+1,:].T @ V[:r+1,:] #project beta coeff into low rank subspace
                    # B_rrr           = B_hat_train @ V[r,:][np.newaxis,:].T @ V[r,:][np.newaxis,:] #project beta coeff into low rank subspace
                    Y_hat_test_rr   = X1_test @ B_rrr
                    # R2_neuron = np.full((narealabelpairs,Nsub,nSessions,nStim,rank_behavout,kfold,nmodelfits),np.nan)
                    R2_neuron[0,:,ises,istim,r,ikf,imf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')

                #RRR X2 to Y
                B_hat_train         = LM(Y_train,X2_train, lam=lam)
                Y_hat_train         = X2_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                for r in range(nranks):
                    B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                    # B_rrr           = B_hat_train @ V[:r+1,:].T @ V[:r+1,:] #project beta coeff into low rank subspace
                    # B_rrr           = B_hat_train @ V[r,:][np.newaxis,:].T @ V[r,:][np.newaxis,:] #project beta coeff into low rank subspace
                    Y_hat_test_rr   = X2_test @ B_rrr
                    # R2_neuron = np.full((narealabelpairs,Nsub,nSessions,nStim,rank_behavout,kfold,nmodelfits),np.nan)
                    R2_neuron[1,:,ises,istim,r,ikf,imf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')

# print('Population size: %d (%d-%d)' % (np.mean(sampleN[sampleN>minsampleneurons]),np.min(sampleN[sampleN>minsampleneurons]),np.max(sampleN[sampleN>minsampleneurons])))

#%% Show the correlation between R2 predicted by labeled and unlabeled neurons:
examplerank = 0
# xdata = R2_neuron[0,:,:,:,examplerank,:,:].flatten()
# ydata = R2_neuron[1,:,:,:,examplerank,:,:].flatten()
# xdata = np.nanmean(R2_neuron[0,:,:,:,:,:,:],axis=2) #average across stimuli
# ydata = np.nanmean(R2_neuron[1,:,:,:,:,:,:],axis=2)

xdata = np.nanmean(R2_neuron[0,:,:,:,:,:,:],axis=(2,4)) #average across stimuli
ydata = np.nanmean(R2_neuron[1,:,:,:,:,:,:],axis=(2,4))

xdata = np.diff(xdata,axis=2) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
ydata = np.diff(ydata,axis=2)

plotlims = [-0.1,my_ceil(np.nanmax([xdata,ydata]),1)]

clrs = sns.color_palette('viridis',nranks)
fig,ax = plt.subplots(1,nranks-1,figsize=(nranks*3.5*cm,4*cm),sharey=True,sharex=True)
for r in range(nranks-1):
    xdatatoplot = xdata[:,:,r,:].flatten()
    ydatatoplot = ydata[:,:,r,:].flatten()
    # xdatatoplot = xdata[:,:,r,:,:].flatten()
    # ydatatoplot = ydata[:,:,r,:,:].flatten()
    ax[r].set_title('Rank %d' % (r+1))
    if r==0:
        ax[r].set_xlabel('R2 unl ')
        ax[r].set_ylabel('R2 lab ')
    ax[r].scatter(xdatatoplot,ydatatoplot,alpha=0.3,color=clrs[r],s=5)
    ax[r].set_xlim(plotlims)
    ax[r].set_ylim(plotlims)
    ax[r].set_xticks(np.arange(plotlims[0],plotlims[1]+0.01,0.1))
    ax[r].set_yticks(np.arange(plotlims[0],plotlims[1]+0.01,0.1))
    # ax_nticks(ax[r],5)
    ax[r].plot(plotlims,plotlims,'k--',linewidth=1,color='grey')
    xdatatoplot = xdatatoplot[~np.isnan(ydatatoplot)]
    ydatatoplot = ydatatoplot[~np.isnan(ydatatoplot)]
    print(np.corrcoef(xdatatoplot,ydatatoplot)[0,1])
plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'RRR_unique_cvR2_V1lab_V1unl_PMunl_%dneurons' % Nsub)

# my_savefig(fig,savedir,'RRR_cvR2_V1unl_V

# #%% Show the correlation between R2 predicted by labeled and unlabeled neurons:

# # xdata = np.nanmean(R2_neuron[0,:,:,:,:,rank_behavout-1,:,:],axis=(0,4,5,6)).flatten()
# # ydata = np.nanmean(R2_neuron[1,:,:,:,:,rank_behavout-1,:,:],axis=(0,4,5,6)).flatten()
# examplerank = 0
# # xdata = R2_neuron[0,:,:,:,examplerank,:,:].flatten()
# # ydata = R2_neuron[1,:,:,:,examplerank,:,:].flatten()

# xdata = R2_neuron[0,:,:,:,:,:,:]
# ydata = R2_neuron[1,:,:,:,:,:,:]
# clrs = sns.color_palette('viridis',nranks)
# fig,ax = plt.subplots(1,nranks,figsize=(nranks*4,4),sharey=True)
# for r in range(nranks):
#     xdatatoplot = xdata[:,:,:,r,:,:].flatten()
#     ydatatoplot = ydata[:,:,:,r,:,:].flatten()
#     ax[r].set_title('Rank %d' % r)
#     ax[r].set_xlabel('R2 unlabeled neurons')
#     ax[r].set_ylabel('R2 labeled neurons')
#     ax[r].scatter(xdatatoplot,ydatatoplot,alpha=0.1,color=clrs[r],s=5)
#     ax[r].set_xlim([-0.1,0.6])
#     ax[r].set_ylim([-0.1,0.6])
#     ax[r].plot([-0.1,0.6],[-0.1,0.6],'k--')
#     print(np.corrcoef(xdatatoplot,ydatatoplot)[0,1])
# plt.tight_layout()
# sns.despine(fig=fig,trim=True,top=True,right=True)
# # my_savefig(fig,savedir,'RRR_cvR2_V1unl_V

#%% 
sourcearealabelpairs = ['V1unl','V1lab']
targetarealabelpair = 'PMunl'

clrs_arealabelpairs = get_clr_area_labeled(sourcearealabelpairs)
narealabelpairs     = len(sourcearealabelpairs)

#%% Show latents: 
ises = 1
stim = 5
Nsub = 50
rank = 3
idx_resp            = np.where((t_axis>=0) & (t_axis<=2))[0]

ses = sessions[ises]

idx_areax1          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[0],
                            ses.celldata['noise_level']<maxnoiselevel,	
                            ),axis=0))[0]
idx_areax2          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[1],
                            ses.celldata['noise_level']<maxnoiselevel,	
                            ),axis=0))[0]
idx_areay       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpair,
                                        ses.celldata['noise_level']<maxnoiselevel,	
                                        ),axis=0))[0]
print(len(idx_areax2))

np.random.seed(0)

idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
idx_areax2_sub       = np.random.choice(idx_areax2,Nsub,replace=False)
idx_areay_sub        = np.random.choice(idx_areay,Nsub*2,replace=False)

idx_T               = ses.trialdata['stimCond']==stim

X1                  = ses.tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
X2                  = ses.tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
Y                   = ses.tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

# reshape to neurons x time points
X1                  = X1.reshape(len(idx_areax1_sub),-1).T
X2                  = X2.reshape(len(idx_areax2_sub),-1).T
Y                   = Y.reshape(len(idx_areay_sub),-1).T

X1  = zscore(X1,axis=0)
X2  = zscore(X2,axis=0)
Y   = zscore(Y,axis=0)

X = np.concatenate((X1,X2),axis=1)

#RRR X to Y
B_hat         = LM(Y,X, lam=lam)
Y_hat         = X @ B_hat

# decomposing and low rank approximation of A
U, s, V = svds(Y_hat,k=nranks,which='LM')
U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

B_rrr           = B_hat @ V[:rank,:].T @ V[:rank,:] #project beta coeff into low rank subspace
Y_hat_test_rr   = X @ B_rrr

# How much of the variance in the source area is aligned with the predictive subspace:
Ub, sb, Vb = svds(B_rrr,k=rank,which='LM')
# Ub, sb, Vb = svds(B_hat,k=rank,which='LM')
# print(EV(X_test, X_test @ Ub @ Ub.T))

X_1 = copy.deepcopy(X)
X_1[:,Nsub:] = 0
# plt.imshow(X_1,vmin=-3,vmax=3,aspect='auto')
Z_1 = X_1 @ Ub 
# Y_hat_test_rr   = X_1 @ B_rrr

X_2 = copy.deepcopy(X)
X_2[:,:Nsub] = 0
# plt.imshow(X_2,vmin=-3,vmax=3,aspect='auto')
# Y_hat_test_rr   = X_2 @ B_rrr
Z_2 = X_2 @ Ub

#%% Little bit of smoothing:
kernel_size = 3
kernel = np.ones(kernel_size) / kernel_size
for r in range(rank):
    Z_1[:,r] = np.convolve(Z_1[:,r], kernel, mode='same')
    Z_2[:,r] = np.convolve(Z_2[:,r], kernel, mode='same')

#%% Plot excerpt of latent dimensions over time
clrs = sns.color_palette('viridis',rank)
fig,axes = plt.subplots(rank,1,figsize=(9*cm,rank*3*cm),sharex=True)

starttimepoint_idx  = 100
ntimebins           = 250
idx_K = np.arange(starttimepoint_idx,starttimepoint_idx+ntimebins)
for r in range(rank):
    ax = axes[r]
    ax.plot(Z_1[idx_K,r],color=clrs_arealabelpairs[0],alpha=1,label='V1 unlabeled')
    ax.plot(Z_2[idx_K,r],color=clrs_arealabelpairs[1],alpha=1,label='V1 labeled')
    ax.set_ylabel('Latent dim %d' % (r+1))
    if r==0:
        ax.legend(frameon=False)
        my_legend_strip(ax)
    ax.set_title('Latent %d' % (r+1))
    ax.axis('off')
    if r==0:
        ax.add_artist(AnchoredSizeBar(ax.transData, 10*ses.sessiondata['fs'][0],
                        "10 Sec", loc=4, frameon=False))
plt.tight_layout()
my_savefig(fig,savedir,'Example_Latents_Joint_V1lab_V1unl_PMunl_%dneurons' % Nsub)

#%% Do RRR of V1 and PM labeled and unlabeled neurons simultaneously

Nsub                = 50
nranks              = 10
nmodelfits          = 10
R2_cv               = np.full((3,nSessions,nStim),np.nan)
optim_rank          = np.full((3,nSessions,nStim),np.nan)
R2_ranks            = np.full((3,nSessions,nStim,nranks,kfold,nmodelfits),np.nan)
R2_ranks_neurons    = np.full((3,Nsub*2,nSessions,nStim,nranks,kfold,nmodelfits),np.nan)
source_dim          = np.full((3,nSessions,nStim,nmodelfits),np.nan)
R2_sourcealigned    = np.full((3,nSessions,nStim,nranks,kfold,nmodelfits),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different populations'):
    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=30)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    idx_areax1          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[0],
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    idx_areax2          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[1],
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    idx_areay       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpair,
                                            ses.celldata['noise_level']<maxnoiselevel,	
                                            idx_nearby),axis=0))[0]

    if len(idx_areax1)<Nsub or len(idx_areax2)<Nsub or len(idx_areay)<Nsub: #skip exec if not enough neurons in one of the populations
        continue

    for imf in range(nmodelfits):
        idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
        idx_areax2_sub       = np.random.choice(idx_areax2,Nsub,replace=False)
        idx_areay_sub        = np.random.choice(idx_areay,Nsub*2,replace=False)

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim

            X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
            X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

            # reshape to neurons x time points
            X1                  = X1.reshape(len(idx_areax1_sub),-1).T
            X2                  = X2.reshape(len(idx_areax2_sub),-1).T
            Y                   = Y.reshape(len(idx_areay_sub),-1).T

            X1  = zscore(X1,axis=0)
            X2  = zscore(X2,axis=0)
            Y   = zscore(Y,axis=0)

            X = np.concatenate((X1,X2),axis=1)

            source_dim[0,ises,istim,imf] = estimate_dimensionality(X,method='participation_ratio')
            source_dim[1,ises,istim,imf] = estimate_dimensionality(X1,method='participation_ratio')
            source_dim[2,ises,istim,imf] = estimate_dimensionality(X2,method='participation_ratio')

            # OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS    
            R2_kfold    = np.zeros((kfold))
            kf          = KFold(n_splits=kfold)
            for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                X_train, X_test     = X[idx_train], X[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                #RRR X to Y
                B_hat_train         = LM(Y_train,X_train, lam=lam)
                Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                for r in range(nranks):
                    B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                    Y_hat_test_rr   = X_test @ B_rrr

                    # How much of the variance in the source area is aligned with the predictive subspace:
                    # Ub, sb, Vb = svds(B_hat_train,k=r+1,which='LM')
                    # Ub, sb, Vb = svds(B_rrr,k=r+1,which='LM')
                    # print(EV(X_test, X_test @ Ub @ Ub.T))

                    R2_ranks[0,ises,istim,r,ikf,imf] = EV(Y_test,Y_hat_test_rr)
                    R2_ranks_neurons[0,:,ises,istim,r,ikf,imf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')
                    
                    X_test_1 = copy.deepcopy(X_test)
                    X_test_1[:,Nsub:] = 0
                    Y_hat_test_rr   = X_test_1 @ B_rrr

                    R2_ranks[1,ises,istim,r,ikf,imf] = EV(Y_test,Y_hat_test_rr)
                    R2_ranks_neurons[1,:,ises,istim,r,ikf,imf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')

                    X_test_2 = copy.deepcopy(X_test)
                    X_test_2[:,:Nsub] = 0
                    Y_hat_test_rr   = X_test_2 @ B_rrr

                    R2_ranks[2,ises,istim,r,ikf,imf] = EV(Y_test,Y_hat_test_rr)
                    R2_ranks_neurons[2,:,ises,istim,r,ikf,imf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')

                    # How much of the variance in the source area is aligned with the predictive subspace:
                    Ub, sb, Vb = svds(B_rrr,k=r+1,which='LM')
                    R2_sourcealigned[0,ises,istim,r,ikf,imf] = EV(X_test, X_test @ Ub @ Ub.T)
                    R2_sourcealigned[1,ises,istim,r,ikf,imf] = EV(X_test_1, X_test_1 @ Ub @ Ub.T)
                    R2_sourcealigned[2,ises,istim,r,ikf,imf] = EV(X_test_2, X_test_2 @ Ub @ Ub.T)

#%%
for ises in range(nSessions):
    if np.any(~np.isnan(R2_ranks[0,ises,:,:,:,:])):
        for istim in range(nStim):
            if fixed_rank is not None:
                rank = fixed_rank
                R2_cv[0,ises,istim] = np.nanmean(R2_ranks[0,ises,istim,rank,:,:])
                R2_cv[1,ises,istim] = np.nanmean(R2_ranks[1,ises,istim,rank,:,:])
                R2_cv[2,ises,istim] = np.nanmean(R2_ranks[2,ises,istim,rank,:,:])
            else:
                R2_cv[0,ises,istim],optim_rank[0,ises,istim] = rank_from_R2(R2_ranks[0,ises,istim,:,:,:].reshape([nranks,nmodelfits*kfold]),nranks,nmodelfits*kfold)
                R2_cv[1,ises,istim],optim_rank[1,ises,istim] = rank_from_R2(R2_ranks[1,ises,istim,:,:,:].reshape([nranks,nmodelfits*kfold]),nranks,nmodelfits*kfold)
                R2_cv[2,ises,istim],optim_rank[2,ises,istim] = rank_from_R2(R2_ranks[2,ises,istim,:,:,:].reshape([nranks,nmodelfits*kfold]),nranks,nmodelfits*kfold)

#%% 
fig, axes = plt.subplots(1,2,figsize=(8*cm,4*cm))
ax = axes[0]
ax.plot(range(nranks),np.nanmean(R2_ranks[0],axis=(0,1,3,4)),label='All neurons',color='grey')
ax.plot(np.nanmean(R2_ranks[1],axis=(0,1,3,4)),label=sourcearealabelpairs[0],color=clrs_arealabelpairs[0])
ax.plot(np.nanmean(R2_ranks[2],axis=(0,1,3,4)),label=sourcearealabelpairs[1],color=clrs_arealabelpairs[1])
leg = ax.legend(frameon=False,fontsize=6)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Cross-validated R2')

plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True)
ax = axes[1]

#%% How much of the variance in the source area is aligned with the predictive subspace:
fig, axes = plt.subplots(1,2,figsize=(8*cm,4*cm))
ax = axes[0]
# ax.plot(range(nranks),np.nanmean(R2_ranks[0],axis=(0,1,3,4)),label='All neurons',color='grey')
ax.plot(np.nanmean(R2_sourcealigned[1],axis=(0,1,3,4)),label=sourcearealabelpairs[0],color=clrs_arealabelpairs[0])
ax.plot(np.nanmean(R2_sourcealigned[2],axis=(0,1,3,4)),label=sourcearealabelpairs[1],color=clrs_arealabelpairs[1])
leg = ax.legend(frameon=False,fontsize=6)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Aligned source variance R2')

plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True)
# ax = axes[1]
my_savefig(fig,savedir,'RRR_source_aligned_R2_V1lab_V1unl_PMunl_%dneurons' % Nsub)


#%% Show the correlation between R2 predicted by labeled and unlabeled neurons:
xdata = np.nanmean(R2_ranks_neurons[1,:,:,:,:,:,:],axis=(2,4)) #average across stimuli/kfolds
ydata = np.nanmean(R2_ranks_neurons[2,:,:,:,:,:,:],axis=(2,4))

xdata = np.diff(xdata,axis=2) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
ydata = np.diff(ydata,axis=2)

plotlims = [-0.1,my_ceil(np.nanmax([xdata,ydata]),1)]

clrs = sns.color_palette('viridis',nranks)
fig,ax = plt.subplots(1,nranks-1,figsize=(nranks*3.2*cm,4*cm),sharey=True,sharex=True)
for r in range(nranks-1):
    xdatatoplot = xdata[:,:,r,:].flatten()
    ydatatoplot = ydata[:,:,r,:].flatten()
    # xdatatoplot = xdata[:,:,r,:,:].flatten()
    # ydatatoplot = ydata[:,:,r,:,:].flatten()
    ax[r].set_title('Rank %d' % (r+1))
    if r==0:
        ax[r].set_xlabel('R2 unl ')
        ax[r].set_ylabel('R2 lab ')
    ax[r].scatter(xdatatoplot,ydatatoplot,alpha=0.3,color=clrs[r],s=5)
    ax[r].set_xlim(plotlims)
    ax[r].set_ylim(plotlims)
    ax[r].set_xticks(np.arange(plotlims[0],plotlims[1]+0.01,0.1))
    ax[r].set_yticks(np.arange(plotlims[0],plotlims[1]+0.01,0.1))
    # ax_nticks(ax[r],5)
    ax[r].plot(plotlims,plotlims,'--',linewidth=1,color='grey')
    xdatatoplot = xdatatoplot[~np.isnan(ydatatoplot)]
    ydatatoplot = ydatatoplot[~np.isnan(ydatatoplot)]
    print(np.corrcoef(xdatatoplot,ydatatoplot)[0,1])
plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'RRR_unique_cvR2_V1lab_V1unl_PMunl_%dneurons' % Nsub)
