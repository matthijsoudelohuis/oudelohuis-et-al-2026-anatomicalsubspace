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

#%%  #assign arealayerlabel
for ses in sessions:
    ses.reset_layer(splitdepth=300)
    ses.celldata['arealayer'] = ses.celldata['roi_name'] + ses.celldata['layer']
    #assign arealayerlabel
    ses.celldata['arealayerlabel'] = ses.celldata['arealabel'] + ses.celldata['layer'] 

#%% #############################################################################

# RRR


#%% RRR across time relative to stimulus onset:
nranks              = 20
nmodelfits          = 50 #number of times new neurons are resampled 
kfold               = 5
lam                 = 0
nsampleneurons      = 25
filter_nearby       = True


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
                    ]


clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

ntimebins           = len(t_axis)
R2_cv               = np.full((nSessions,narealabelpairs,ntimebins),np.nan)
optim_rank          = np.full((nSessions,narealabelpairs,ntimebins),np.nan)
R2_ranks            = np.full((nSessions,narealabelpairs,ntimebins,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)

    for iapl, arealabelpair in enumerate(arealabelpairs):
        alx,aly = arealabelpair.split('-')
        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=30)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
            continue

        for ibin in range(ntimebins):
            X = ses.tensor[np.ix_(idx_areax,idx_T,[ibin])].squeeze().T
            Y = ses.tensor[np.ix_(idx_areay,idx_T,[ibin])].squeeze().T

            R2_cv[ises,iapl,ibin],optim_rank[ises,iapl,ibin],R2_ranks[ises,iapl,ibin,:,:,:]      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))


#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealayerpairs  = ['V1L2/3-PML2/3',
                   'V1L2/3-PML5',
                   'PML2/3-V1L2/3',
                   'PML5-V1L2/3',
                   'PML2/3-PML5',
                   'PML5-PML2/3'                   
                   ]

arealayerpairs  = ['V1unlL2/3-PMunlL2/3',
                   'V1labL2/3-PMunlL2/3',
                    'V1unlL2/3-PMlabL2/3',
                   'V1labL2/3-PMlabL2/3',
                   ]

# arealayerlabels = ['V1unlL2/3',
#                     'V1labL2/3',
#                     'PMunlL2/3',
#                     'PMlabL2/3',
#                     'PMunlL5',
#                     'PMlabL5',
#                    ]

#%% RRR across time relative to stimulus onset:
nranks              = 20
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
lam                 = 0

arealayerpairs  = ['V1unlL2/3-PMunlL2/3',
                   'V1labL2/3-PMunlL2/3',
                    'V1unlL2/3-PMlabL2/3',
                   'V1labL2/3-PMlabL2/3',
                   'V1unlL2/3-PMunlL5',
                   'V1labL2/3-PMunlL5',
                    'V1unlL2/3-PMlabL5',
                   'V1labL2/3-PMlabL5',
                   ]

arealayerpairs  = ['PMunlL2/3-V1unlL2/3',
                   'PMunlL2/3-V1labL2/3',
                    'PMlabL2/3-V1unlL2/3',
                   'PMlabL2/3-V1labL2/3',
                   'PMunlL5-V1unlL2/3',
                   'PMunlL5-V1labL2/3',
                    'PMlabL5-V1unlL2/3',
                   'PMlabL5-V1labL2/3',
                   ]

narealayerpairs     = len(arealayerpairs)

nsampleneurons      = 20
filter_nearby       = True

# arealabelpairs      = ['V1unl-PMunl',
#                         'PMunl-V1unl',
#                         ]
# nsampleneurons      = 200
# filter_nearby       = False

# clrs_arealabelpairs = get_clr_area_labelpairs(arealayerpairs)
clrs_arealayerlabels = sns.color_palette('tab10',narealayerpairs)


ntimebins           = len(t_axis)
R2_cv               = np.full((nSessions,narealayerpairs,ntimebins),np.nan)
optim_rank          = np.full((nSessions,narealayerpairs,ntimebins),np.nan)
R2_ranks            = np.full((nSessions,narealayerpairs,ntimebins,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
    idx_T               = np.ones(len(ses.trialdata['stimCond']),dtype=bool)

    for iapl, arealayerpair in enumerate(arealayerpairs):
        alx,aly = arealayerpair.split('-')
        if filter_nearby:
            idx_nearby  = filter_nearlabeled(ses,radius=50)
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealayerlabel']==alx,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealayerlabel']==aly,
                                ses.celldata['noise_level']<20,	
                                idx_nearby),axis=0))[0]

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
            continue

        for ibin in range(ntimebins):
            X = ses.tensor[np.ix_(idx_areax,idx_T,[ibin])].squeeze().T
            Y = ses.tensor[np.ix_(idx_areay,idx_T,[ibin])].squeeze().T

            R2_cv[ises,iapl,ibin],optim_rank[ises,iapl,ibin],R2_ranks[ises,iapl,ibin,:,:,:]      = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))

#%% Plotting:
t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(2,2,figsize=(5.5,5.5))
ax = axes[0,0]
handles = []
for iapl, arealabelpair in enumerate(arealayerpairs[:4]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealayerlabels[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,R2_cv[:,iapl,:],error='sem',color=clrs_arealayerlabels[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealayerpairs,loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('R2')

ax = axes[1,0]
handles = []
for iapl, arealabelpair in enumerate(arealayerpairs[4:]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealayerlabels[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,R2_cv[:,iapl+4,:],error='sem',color=clrs_arealayerlabels[iapl+4],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealayerpairs[4:],loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('R2')

ax = axes[0,1]
handles = []
for iapl, arealabelpair in enumerate(arealayerpairs[:4]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealayerlabels[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,optim_rank[:,iapl,:],error='sem',color=clrs_arealayerlabels[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealayerpairs,loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Optimal rank')

ax = axes[1,1]
handles = []
for iapl, arealabelpair in enumerate(arealayerpairs[4:]):
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealayerlabels[iapl],label=arealabelpair)
    handles.append(shaded_error(t_axis,optim_rank[:,iapl,:],error='sem',color=clrs_arealayerlabels[iapl+4],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealayerpairs[4:],loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Optimal rank')

sns.despine(fig=fig, top=True, right=True, offset = 3)
plt.tight_layout()
# my_savefig(fig,savedir,'RRR_V1PM_labeled_timeres')
# my_savefig(fig,savedir,'RRR_V1PM_labeled_GR_across_time',formats=['png'])
my_savefig(fig,savedir,'RRR_V1PM_layer_labeled_GR_across_time',formats=['png'])





# #%% RRR across time relative to stimulus onset:
# nranks              = 20
# nmodelfits          = 1 #number of times new neurons are resampled 
# kfold               = 5
# lam                 = 500

# arealabelpairs      = ['V1unl-PMunl',
#                         'V1unl-PMlab',
#                         'V1lab-PMunl',
#                         'V1lab-PMlab',
#                         'PMunl-V1unl',
#                         'PMunl-V1lab',
#                         'PMlab-V1unl',
#                         'PMlab-V1lab'
#                         ]
# nsampleneurons      = 25
# filter_nearby       = True

# arealabelpairs      = ['V1unl-PMunl',
#                         'PMunl-V1unl',
#                         ]
# nsampleneurons      = 50
# filter_nearby       = False

# clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
# narealabelpairs     = len(arealabelpairs)

# oris                = np.unique(np.mod(sessions[0].trialdata['Orientation'],180))
# nOris               = len(oris)
# ntimebins           = len(t_axis)
# R2_cv               = np.full((nSessions,nOris,narealabelpairs,ntimebins),np.nan)
# optim_rank          = np.full((nSessions,nOris,narealabelpairs,ntimebins),np.nan)
# R2_ranks            = np.full((nSessions,nOris,narealabelpairs,ntimebins,nranks,nmodelfits,kfold),np.nan)

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for session:'):
#     ses.trialdata['Orientation'] = np.mod(ses.trialdata['Orientation'],180)
#     for iapl, arealabelpair in enumerate(arealabelpairs):
#         alx,aly = arealabelpair.split('-')
#         if filter_nearby:
#             idx_nearby  = filter_nearlabeled(ses,radius=50)
#         else:
#             idx_nearby = np.ones(len(ses.celldata),dtype=bool)

#         idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
#                                 ses.celldata['noise_level']<20,	
#                                 idx_nearby),axis=0))[0]
#         idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
#                                 ses.celldata['noise_level']<20,	
#                                 idx_nearby),axis=0))[0]

#         if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
#             continue

#         for iori,ori in enumerate(oris):
#             idx_T               = ses.trialdata['Orientation']==ori
#             for ibin in range(ntimebins):
#                 X = ses.tensor[np.ix_(idx_areax,idx_T,[ibin])].squeeze().T
#                 Y = ses.tensor[np.ix_(idx_areay,idx_T,[ibin])].squeeze().T

#                 R2_cv[ises,iori,iapl,ibin],optim_rank[ises,iori,iapl,ibin],R2_ranks[ises,iori,iapl,ibin,:,:,:]      = RRR_wrapper(
#                                                                         Y, X, nN=nsampleneurons,lam=lam,
#                                                                        nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

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
