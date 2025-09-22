# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import zscore

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.plot_lib import * #get all the fixed color schemes
# from utils.corr_lib import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.pair_lib import value_matching
from utils.psth import compute_tensor

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\Labeling')

#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)


#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])
report_sessions(sessions)

#%%  Load data properly:        
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 1.9     #post s
binsize     = 0.2
calciumversion = 'dF'
vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
                            ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

behavfields = np.array(['runspeed','diffrunspeed'])

for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion)
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
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

#%% Test wrapper function: (should be fast, just one model fit etc. EV around 0.05-0.07)
nsampleneurons  = 20
nranks          = 25
nmodelfits      = 1 #number of times new neurons are resampled 
kfold           = 5
R2_cv           = np.full((nSessions),np.nan)
optim_rank      = np.full((nSessions),np.nan)
idx_resp        = np.where((t_axis>=0) & (t_axis<=1))[0]

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

#%% Parameters for RRR for size-matched populations of V1 and PM labeled and unlabeled neurons
lam                 = 0
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 1 #number of times new neurons are resampled - many for final run
kfold               = 5
maxnoiselevel       = 20
nStim               = 16

# idx_resp            = np.where((t_axis>=0) & (t_axis<=1))[0]
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
# idx_resp            = np.where((t_axis>=0.5) & (t_axis<=1.5))[0]
ntimebins           = len(idx_resp)
minsampleneurons    = 10
filter_nearby       = True

dataversions        = ['original','behavout','neuralout']
nversions           = len(dataversions) #0: original,1:behavior out, 2:neural out
rank_behavout       = 5
rank_neuralout      = 5

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

# R2_cv               = np.full((narealabelpairs,nSessions),np.nan)
# optim_rank          = np.full((narealabelpairs,nSessions),np.nan)
# R2_ranks            = np.full((narealabelpairs,nSessions,nranks,nmodelfits,kfold),np.nan)

# R2_cv               = np.full((narealabelpairs,nSessions,nStim),np.nan)
# optim_rank          = np.full((narealabelpairs,nSessions,nStim),np.nan)
# R2_ranks            = np.full((narealabelpairs,nSessions,nStim,nranks,nmodelfits,kfold),np.nan)

R2_cv               = np.full((narealabelpairs,nversions,nSessions,nStim),np.nan)
optim_rank          = np.full((narealabelpairs,nversions,nSessions,nStim),np.nan)
R2_ranks            = np.full((narealabelpairs,nversions,nSessions,nStim,nranks,nmodelfits,kfold),np.nan)


nmatchbins          = 10
valuematching       = None
# valuematching       = 'noise_level'
# valuematching       = 'event_rate'
# valuematching       = 'skew'
# valuematching       = 'meanF'

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    # idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)
    # idx_T               = ses.trialdata['stimCond']==0

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

    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby
                                ),axis=0))[0]

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
            R2_cv[iapl,0,ises,istim],optim_rank[iapl,0,ises,istim],R2_ranks[iapl,0,ises,istim,:,:,:]  = RRR_wrapper(Y, X, 
                            nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

            #Get behavioral matrix: 
            B                   = np.concatenate((sessions[ises].tensor_vid[np.ix_(range(np.shape(sessions[ises].tensor_vid)[0]),idx_T,idx_resp)],
                                    sessions[ises].tensor_run[np.ix_(range(np.shape(sessions[ises].tensor_run)[0]),idx_T,idx_resp)]),axis=0)
            B                   = B.reshape(np.shape(B)[0],-1).T
            B                   = zscore(B,axis=0,nan_policy='omit')

            si                  = SimpleImputer()
            B                   = si.fit_transform(B)
            _,_,X_behavout,_,_  = regress_out_behavior_modulation(ses,X=B,Y=X,rank=rank_behavout,lam=0)

            _,_,Y_behavout,_,_  = regress_out_behavior_modulation(ses,X=B,Y=Y,rank=rank_behavout,lam=0)

            R2_cv[iapl,1,ises,istim],optim_rank[iapl,1,ises,istim],R2_ranks[iapl,1,ises,istim,:,:,:]  = RRR_wrapper(Y_behavout, X_behavout, 
                            nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

            idx_ALRSP   = np.where(np.all((np.isin(ses.celldata['roi_name'],['AL','RSP']),
                                        ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
            if len(idx_ALRSP)>nsampleneurons: 
                Z                   = sessions[ises].tensor[np.ix_(idx_ALRSP,idx_T,idx_resp)]
                Z                   -= np.mean(Z,axis=1,keepdims=True)
                Z                   = Z.reshape(len(idx_ALRSP),-1).T
                Z                   = zscore(Z,axis=0,nan_policy='omit')  #Z score activity for each neuron

                _,_,X_neuralout,_,_  = regress_out_behavior_modulation(ses,X=Z,Y=X,rank=rank_neuralout,lam=0)

                _,_,Y_neuralout,_,_  = regress_out_behavior_modulation(ses,X=Z,Y=Y,rank=rank_neuralout,lam=0)
                
                R2_cv[iapl,2,ises,istim],optim_rank[iapl,2,ises,istim],R2_ranks[iapl,2,ises,istim,:,:,:]  = RRR_wrapper(Y_behavout, X_behavout, 
                            nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% 


#          #    ######     #     #  #####     #     # #     # #       
#         # #   #     #    #     # #     #    #     # ##    # #       
#        #   #  #     #    #     # #          #     # # #   # #       
#       #     # ######     #     #  #####     #     # #  #  # #       
#       ####### #     #     #   #        #    #     # #   # # #       
#       #     # #     #      # #   #     #    #     # #    ## #       
####### #     # ######        #     #####      #####  #     # ####### 

#%% 
for iversion, version in enumerate(dataversions):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))

    # R2_data[R2_data==0]     = np.nan
    for idx in np.array([[0,1],[2,3]]):
        clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
        fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
        # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

#%% Define the ratio of R2 between V1PM and V1ND
R2_cv_2         = np.reshape(R2_cv,(narealabelpairs,nversions,nSessions*nStim))
optim_rank_2    = np.reshape(optim_rank,(narealabelpairs,nversions,nSessions*nStim))
R2_ranks_2      = np.reshape(R2_ranks,(narealabelpairs,nversions,nSessions*nStim,nranks,nmodelfits,kfold))

ratiodata_FF       = (R2_cv_2[1] / R2_cv_2[0])*100-100 #
ratiodata_FB       = (R2_cv_2[3] / R2_cv_2[2])*100-100

#%% Make the figure: 
fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5,4))
ax = axes[0]
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FF,axis=1),yerr=np.nanstd(ratiodata_FF,axis=1)/np.sqrt(nSessions*nStim))
ax.set_ylabel("RRR predictive ratio $V1_{PM}$ / $V1_{ND}$")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FF')

ax = axes[1]
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FB,axis=1),yerr=np.nanstd(ratiodata_FB,axis=1)/np.sqrt(nSessions*nStim))
ax.axhline(y=0,color='k',linestyle='--')
ax.set_ylabel("RRR predictive ratio $PM_{V1}$ / $PM_{ND}$")
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_title('FB')

plt.tight_layout()
sns.despine(fig=fig,trim=True)







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
# for iversion, version in enumerate(dataversions):
for iversion, version in enumerate(dataversions[:2]):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))

    # R2_data[R2_data==0]     = np.nan
    for idx in idx_FF:
        clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
        fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
        # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

#%% 
# for iversion, version in enumerate(dataversions):
for iversion, version in enumerate(dataversions[:2]):
# for iversion, version in enumerate(dataversions[2:]):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))

    # R2_data[R2_data==0]     = np.nan
    for idx in idx_FB:
        clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
        fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
        # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))





#%% 

#       ####### ####### ######  ####### ######  
#       #     # #     # #     # #       #     # 
#       #     # #     # #     # #       #     # 
#       #     # #     # ######  #####   #     # 
#       #     # #     # #       #       #     # 
#       #     # #     # #       #       #     # 
####### ####### ####### #       ####### ######  


#%% 
idx_FF = np.array([[0,1],[4,5]])
idx_FB = np.array([[2,3],[6,7]])

#%% 
for iversion, version in enumerate(dataversions):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))

    # R2_data[R2_data==0]     = np.nan
    for idx in idx_FF:
        clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
        fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
        # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

#%% 
for iversion, version in enumerate(dataversions):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,kfold))

    # R2_data[R2_data==0]     = np.nan
    for idx in idx_FB:
        clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
        fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
        # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))






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

