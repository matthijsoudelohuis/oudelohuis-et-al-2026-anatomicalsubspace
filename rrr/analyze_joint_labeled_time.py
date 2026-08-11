# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pickle
from matplotlib.patches import Rectangle

from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.params import load_params
from utils.corr_lib import filter_sharednan
from utils.RRRlib import *

params = load_params()
figdir = os.path.join(params['figdir'],'RRR','Labeling','Time')
resultdir = params['resultdir']

#%%  
version = 'FF_original'
FF_filename = 'RRR_time_Joint_labeled_FF_original_2026-08-05_17-21-26'
# FF_filename = 'RRR_time_Joint_labeled_FF_original_2026-08-04_18-30-56'
# FF_filename = 'RRR_time_Joint_labeled_FB_original_2026-08-04_23-40-52'

version = 'FB_original'
FB_filename = 'RRR_time_Joint_labeled_FB_original_2026-08-07_15-04-21'
# FB_filename = 'RRR_time_Joint_labeled_FF_original_2026-08-04_15-42-10'
# FB_filename = 'RRR_time_Joint_labeled_FB_original_2026-08-04_19-19-22'

#%% Load the data:
data = np.load(os.path.join(resultdir,FF_filename + '.npz'),allow_pickle=True)
for key in data.keys():
    exec(key+'_FF=data[key]')

with open(os.path.join(resultdir,FF_filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

data = np.load(os.path.join(resultdir,FB_filename + '.npz'),allow_pickle=True)
for key in data.keys():
    exec(key+'_FB=data[key]')

with open(os.path.join(resultdir,FB_filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

nmodelfits = params['nmodelfits']
Nsub = params['Nsub']

clrs_arealabelpairs = ['grey','grey','red']
narealabelpairs = 3

#%% Find best rank and cvR2 at this rank:
fixed_rank = None
for ises in tqdm(range(params['nSessions'])):
    for istim in range(params['nStim']):
        for isubspace in range(params['nsubspaces']):
            for isubpop in range(4):
                for t in range(params['nT']):
                    if not np.isnan(R2_ranks_FF[isubpop,isubspace,ises,istim]).all():
                        R2_cv_FF[isubpop,isubspace,ises,istim,t],optim_rank_FF[isubpop,isubspace,ises,istim,t] = rank_from_R2(R2_ranks_FF[isubpop,isubspace,ises,istim,t,:,:,:].reshape([params['nranks'],params['nmodelfits']*params['kfold']]),params['nranks'],params['nmodelfits']*params['kfold'])
                    if not np.isnan(R2_ranks_FB[isubpop,isubspace,ises,istim]).all():
                        R2_cv_FB[isubpop,isubspace,ises,istim,t],optim_rank_FB[isubpop,isubspace,ises,istim,t] = rank_from_R2(R2_ranks_FB[isubpop,isubspace,ises,istim,t,:,:,:].reshape([params['nranks'],params['nmodelfits']*params['kfold']]),params['nranks'],params['nmodelfits']*params['kfold'])

# plt.hist(optim_rank_FF.flatten(),bins=np.arange(0,16))

#%%
# fixed_rank = 4
# R2_cv_FF = np.nanmean(R2_ranks_FF[:,:,:,:,:,fixed_rank,:,:],axis=(-1,-2))
# R2_cv_FB = np.nanmean(R2_ranks_FB[:,:,:,:,:,fixed_rank,:,:],axis=(-1,-2))

                    # if np.any(~np.isnan(R2_ranks[isubpop,isubspace,ises])):
                    #     if fixed_rank is not None:
                    #         # rank = fixed_rank
                    #         R2_cv[isubpop,isubspace,ises,istim,t] = np.nanmean(R2_ranks[isubpop,isubspace,ises,istim,t,fixed_rank,:,:])
                    #     else:
                    #         if not np.isnan(R2_ranks[isubpop,isubspace,ises,istim]).all():
                    #             R2_cv[isubpop,isubspace,ises,istim,t],optim_rank[isubpop,isubspace,ises,istim] = rank_from_R2(R2_ranks[isubpop,isubspace,ises,istim,t,:,:,:].reshape([params['nranks'],params['nmodelfits']*params['kfold']]),params['nranks'],params['nmodelfits']*params['kfold'])
# plt.hist(R2_cv.flatten())

#%% Show behavior and non-behavior-related variability over time: 
t_ticks = np.array([-1,0,1,2])
ymin = -0.0005
iapl = 0
subspacelabels = np.array(['Full','Behav','Non-behav'])
clrs_subspaces = get_clr_subspaces(subspacelabels)

fig,axes = plt.subplots(1,2,figsize=(2*3*cm,1*3*cm),sharex=True,sharey=True)
for idirec,(direc,data,alps) in enumerate(zip(['FF','FB'],[R2_cv_FF,R2_cv_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
    ax = axes[idirec]
    handles = []
    # for isubspace in range(0,3):
    for isubspace,subspace in enumerate(range(1,3)):
        R2_toplot = np.nanmean(data[1:],axis=0)
        R2_toplot = np.reshape(R2_toplot[subspace,:,:,:],(params['nSessions']*params['nStim'],params['nT']))
        handles.append(shaded_error(params['t_axis'],R2_toplot,error='sem',center='mean',
                                color=clrs_subspaces[subspace],alpha=0.3,ax=ax,linewidth=1,label=subspacelabels[subspace]))

    ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
    thickness = ax.get_ylim()[1]/15
    ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
    # ax.legend(handles=handles,labels=list(subspacelabels),loc='best')
    ax.legend(handles=handles,loc='best')
    my_legend_strip(ax)
    ax_nticks(ax,3)
    ax.set_xticks(t_ticks)
    ax.set_xlim([-1,1.9])
    ax.set_xticklabels(t_ticks)
    ax.set_xlabel('time (s)')
    ax.axhline(0,linestyle=':',color='grey')
    if idirec == 0:
        ax.set_ylabel('perf. within\n subspace (R$^{2}$)')
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,figdir,'RRR_joint_time_eachsubspace')

#%% Show an example session labeled and unlabeled: 
ises = 10
isubspace = 2

ises = 0
isubspace = 1

plotcontrast = np.array([1,3])
clrs = ['grey','red']

R2_toplot = np.reshape(R2_cv_FF[:,isubspace,ises,:,:],(narealabelpairs+1,params['nStim'],params['nT']))
t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
handles = []

for iapl,apl in enumerate(plotcontrast):
    handles.append(shaded_error(params['t_axis'],R2_toplot[apl,:,:],error='sem',color=clrs[iapl],alpha=0.3,ax=ax))

ymin = 0.00
ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/15
ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=list(arealabeled_to_figlabels(sourcearealabelpairs_FF[plotcontrast-1])),loc='best')
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xlim([-1,1.9])
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (s)')
ax.set_ylabel('R$^{2}$')
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,figdir,'RRR_joint_time_subspace%d_%s_example_session_%d' % (isubspace,version, ises))

#%% Show all sessions:
plotcontrast = np.array([1,3])
# plotcontrast = np.array([2,3])
clrs = ['grey','red']
t_ticks = np.array([-1,0,1,2])
ymin = 0
isubspace = 1
fig,axes = plt.subplots(params['nSessions'],2,figsize=(2*3*cm,params['nSessions']*3*cm),sharex=True,sharey=True)
for ises in range(params['nSessions']):
    for idirec,(direc,data,alps) in enumerate(zip(['FF','FB'],[R2_cv_FF,R2_cv_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
        R2_toplot = np.reshape(data[:,isubspace,ises,:,:],(narealabelpairs+1,params['nStim'],params['nT']))
        ax = axes[ises,idirec]
        handles = []
        for iapl,apl in enumerate(plotcontrast):
            handles.append(shaded_error(params['t_axis'],R2_toplot[apl,:,:],error='sem',
                                        color=clrs[iapl],alpha=0.3,ax=ax,linewidth=1))

        ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
        thickness = ax.get_ylim()[1]/15
        ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
        ax.legend(handles=handles,labels=list(arealabeled_to_figlabels(sourcearealabelpairs_FF[plotcontrast-1])),loc='best')
        my_legend_strip(ax)
        ax_nticks(ax,3)
        ax.set_xticks(t_ticks)
        ax.set_xticklabels(t_ticks)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('R$^{2}$')
sns.despine(fig=fig, top=True, right=True, offset = 2)
# my_savefig(fig,figdir,'RRR_joint_time_fullsubspace_%s_example_session_%d' % (version, ises))

#%% Plotting the mean across time across sessions: 
t_ticks = np.array([-1,0,1,2])
# fig,axes = plt.subplots(1,2,figsize=(8*cm,3.5*cm))
# ymin = 0.01
# ymin = 0.0
subspacelabels = np.array(['full','behav','non-behav'])
fig,axes = plt.subplots(3,2,figsize=(8*cm,8*cm),sharex=True)
for idirec,(direc,data,alps) in enumerate(zip(['FF','FB'],[R2_cv_FF,R2_cv_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
    for isubspace in range(params['nsubspaces']):
        ax = axes[isubspace,idirec]
        R2_toplot = np.reshape(data,(narealabelpairs+1,params['nsubspaces'],params['nSessions']*params['nStim'],params['nT']))

        handles = []

        for iapl,apl in enumerate(plotcontrast):
            handles.append(shaded_error(params['t_axis'],R2_toplot[apl,isubspace,:,:],error='sem',color=clrs[iapl],alpha=0.3,ax=ax))
            # handles.append(shaded_error(params['t_axis'],R2_toplot[apl][isubspace],error='sem',color=clrs[iapl],alpha=0.3,ax=ax))

        # ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],3)])
        thickness = ax.get_ylim()[1]/15
        ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
        if isubspace==0:
            ax.legend(handles=handles,labels=list(arealabeled_to_figlabels(alps[plotcontrast-1])),loc='best')
            # ax.legend(handles=handles,loc='best')
            my_legend_strip(ax)
        ax_nticks(ax,3)
        ax.set_xticks(t_ticks)
        ax.set_xticklabels(t_ticks)
        ax.set_xlabel('time (s)')
        if idirec==0: 
            ax.set_ylabel('performance (R$^{2}$)')
        ax.set_title(direc + ' ' + subspacelabels[isubspace])
        ax.set_xlim([-1,1.9])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 2)
# my_savefig(fig,figdir,'RRR_joint_time_Raw_FF_FB' )
# my_savefig(fig,figdir,'labeled_RRR_time_Raw_FF_FB_behavsubspace' )

#%% Get the R2 ratio data:
# # plotcontrasts   = np.array([[1,2],[1,3]])
# plotcontrasts   = np.array([[2,1],[1,2],[2,3],[1,3]])
# # noise_constant  = 1e-4
# noise_constant  = 0
# # clipval         = 1e-4
# # clipval         = -np.inf
# clipval         = 0

# R2_ratiodata    = np.full((2,2,len(plotcontrasts),params['nSessions']*params['nStim'],params['nT']),np.nan)

# for idirec,[direc,data,alps] in enumerate(zip(['FF','FB'],[R2_cv_FF,R2_cv_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
# # for idirec,[direc,data,alps] in enumerate(zip(['FF','FB'],[R2_ranks_FF,R2_ranks_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
#     R2_toplot = np.reshape(data,(narealabelpairs+1,params['nsubspaces'],params['nSessions']*params['nStim'],params['nT']))
#     R2_toplot[R2_toplot<=0] = np.nan
#     for isubspace in range(2):
#         # R2_toplot = np.clip(R2_toplot,clipval,np.inf) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
#         for iplotcontrast,plotcontrast in enumerate(plotcontrasts):
#             R2_ratiodata[idirec,isubspace,iplotcontrast,:,:] = (R2_toplot[plotcontrast[1],isubspace,:,:]+noise_constant) / (R2_toplot[plotcontrast[0],isubspace,:,:]+noise_constant) #add a small constant to avoid division by zero
#             # R2_ratiodata[idirec,iranks,iplotcontrast,:,:] = (R2_toplot[plotcontrast[1],:,:]+noise_constant) - (R2_toplot[plotcontrast[0],:,:]+noise_constant) #add a small constant to avoid division by zero
#             # R2_ratio[R2_ratio<0.5] = np.nan

# # R2_ratiodata = np.concatenate((np.nanmean(R2_ratiodata[:,:,:2],axis=2,keepdims=True),
#                             #    np.nanmean(R2_ratiodata[:,:,2:],axis=2,keepdims=True)),axis=2)
# # R2_ratiodata[np.isinf(R2_ratiodata)] == np.nan
# R2_ratiodata = np.concatenate((np.nanmean(R2_ratiodata[:,:,:2],axis=2,keepdims=True),
#                                np.nanmean(R2_ratiodata[:,:,2:],axis=2,keepdims=True)),axis=2)
# # R2_ratiodata[np.isnan(R2_ratiodata)] == 0

#%% Get the R2 ratio data:
plotcontrasts   = np.array([[1,2],[1,3]]) #doesn't matter much which contrast is used of course
# plotcontrasts   = np.array([[2,1],[2,3]])
subspaces       = np.array([1,2]) #select behav and non-behav (not full)

noise_constant  = 0
clipval         = 1e-3
# clipval         = 0.0005
R2_ratiodata    = np.full((2,2,len(plotcontrasts),params['nSessions']*params['nStim'],params['nT']),np.nan)

for idirec,[direc,data,alps] in enumerate(zip(['FF','FB'],[R2_cv_FF,R2_cv_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
    for isubspace,subspace in enumerate(subspaces):
        for iplotcontrast,plotcontrast in enumerate(plotcontrasts):
            R2_toplot = np.reshape(data[:,subspace,:,:,:],(narealabelpairs+1,params['nSessions']*params['nStim'],params['nT']))
            R2_toplot = np.clip(R2_toplot,clipval,np.inf) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
            R2_ratiodata[idirec,isubspace,iplotcontrast,:,:] = (R2_toplot[plotcontrast[1],:,:]+noise_constant) / (R2_toplot[plotcontrast[0],:,:]+noise_constant) #add a small constant to avoid division by zero

# plotcontrasts   = np.array([[2,1],[1,2],[2,3],[1,3]])
# R2_ratiodata = np.concatenate((np.nanmean(R2_ratiodata[:,:,:2],axis=2,keepdims=True),
#                                np.nanmean(R2_ratiodata[:,:,2:],axis=2,keepdims=True)),axis=2)

#%% Plot the ratio across time across sessions: 
ymin = 0.9
minymax = 1.6
clrs = ['grey','red']
plotcontrasts = np.array([[1,2],[1,3]])

thickness = 0.05

twin_iti = np.array([-1,0])
twin_resp = np.array([0,1])
# twin_iti = np.array([-0.5,0])
idx_iti = (params['t_axis']>=twin_iti[0]) & (params['t_axis']<=twin_iti[1])
idx_resp = (params['t_axis']>=twin_resp[0]) & (params['t_axis']<=twin_resp[1])
patchalpha  = 0.2
patchcolors = ['grey','blue']

fig,axes = plt.subplots(2,2,figsize=(8*cm,7*cm),sharex=False,sharey='row')
for idirec,[direc,data,alps] in enumerate(zip(['FF','FB'],[R2_ranks_FF,R2_ranks_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
    for isubspace,subspacelabel in enumerate(subspacelabels[1:]):
        ax = axes[isubspace,idirec]
        if direc == 'FF': 
            figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
        elif direc == 'FB': 
            figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']

        handles = []
        for iplotcontrast,plotcontrast in enumerate(plotcontrasts):
            # handles.append(shaded_error(params['t_axis'],R2_ratiodata[idirec,isubspace,iplotcontrast,:,:],error='ci95',color=clrs[iplotcontrast],alpha=0.3,ax=ax))
            handles.append(shaded_error(params['t_axis'],R2_ratiodata[idirec,isubspace,iplotcontrast,:,:],center='mean',
                                        error='ci95',color=clrs[iplotcontrast],alpha=0.3,ax=ax,linewidth=1))
        
        # add_paired_ttest_results(ax,np.nanmean(R2_ratiodata[idirec,isubspace,0,:,idx_iti],axis=0),
        #                          np.nanmean(R2_ratiodata[idirec,isubspace,0,:,idx_resp],axis=0),pos=[0.4,0.9],color=clrs[0])  
        # add_paired_ttest_results(ax,np.nanmean(R2_ratiodata[idirec,isubspace,1,:,idx_iti],axis=0),
        #                          np.nanmean(R2_ratiodata[idirec,isubspace,1,:,idx_resp],axis=0),pos=[0.4,0.8],color=clrs[1])

        add_paired_wilcoxon_results(ax,np.nanmean(R2_ratiodata[idirec,isubspace,0,:,idx_iti],axis=0),
                                 np.nanmean(R2_ratiodata[idirec,isubspace,0,:,idx_resp],axis=0),pos=[0.4,0.9],color=clrs[0],fontsize=6)  
        add_paired_wilcoxon_results(ax,np.nanmean(R2_ratiodata[idirec,isubspace,1,:,idx_iti],axis=0),
                                 np.nanmean(R2_ratiodata[idirec,isubspace,1,:,idx_resp],axis=0),pos=[0.4,0.8],color=clrs[1],fontsize=6)

        thickness = ax.get_ylim()[1]/25
        ax.axhline(y=1,color='grey',linestyle='--',linewidth=0.8)
        ax.legend(handles=handles,labels=figlabels,loc='best')
        my_legend_strip(ax)
        ax.set_xticks(t_ticks)
        ax.set_xticklabels(t_ticks)
        ax.set_xlabel('time (s)')
        if idirec==0:
            ax.set_ylabel('performance ratio')
        # ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
        ax.set_ylim([ymin,my_ceil(np.max([minymax,ax.get_ylim()[1]]),2)])
        ax.tick_params(axis='y', which='major', labelleft=True)     
        ax.set_xlim([-1,1.9])
        rect = Rectangle((twin_iti[0],ymin),twin_iti[1]-twin_iti[0],np.diff(ax.get_ylim())[0],fc=patchcolors[0],
                         ec='none',alpha=patchalpha)
        ax.add_patch(rect)
        rect = Rectangle((twin_resp[0],ymin),twin_resp[1]-twin_resp[0],np.diff(ax.get_ylim())[0],fc=patchcolors[1],
                         ec='none',alpha=patchalpha)
        ax.add_patch(rect)

        # rect = Rectangle((twin_iti[0],ymin),twin_iti[1]-twin_iti[0],np.diff(ax.get_ylim())[0],fc='none',ec='black',alpha=0.5,lw=0.8,linestyle='--')
        # ax.add_patch(rect)
        # rect = Rectangle((twin_resp[0],ymin),twin_resp[1]-twin_resp[0],np.diff(ax.get_ylim())[0],fc='none',ec='blue',lw=0.8,linestyle='--')
        # ax.add_patch(rect)

        ax.set_title(direc + '-' + subspacelabel)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 2, trim=False)
my_savefig(fig,figdir,'RRR_joint_time_ratio_FF_FB_splitsubspaces')

#%% Relationship between FF vs FB ratios
# R2_ratiodata is of shape (2,2,2,ndatasets,nT), 
# where the first dimension is the direction (FF,FB) and the second dimension is the subspace (behav/nonbehav)
# and the third dimension is the contrast between labeled and unlabeled

ndatasets           = params['nSessions'] * params['nStim']
nsubspaces          = len(subspaces)
nplotcontrasts      = len(plotcontrasts)
ndirections         = 2

#Compute crosscorrelation function
ts = params['t_axis']
dt = np.median(np.diff(ts))
max_lag_secs = 2.0
max_lag_samps = int(np.round(max_lag_secs / dt))
lags = np.arange(-max_lag_samps, max_lag_samps + 1) * dt
nlags = len(lags)

def normalized_crosscorr(x,y,max_lag_samps):
    """Per-row (per-dataset) normalized cross-correlation.
    x,y: (ndatasets,ntimepoints). Returns (ndatasets,nlags)."""
    ndatasets   = x.shape[0]
    nlags       = 2*max_lag_samps + 1
    cc          = np.full((ndatasets,nlags),np.nan)
    for i in range(ndatasets):
        xi,yi = x[i,:],y[i,:]
        if np.any(np.isnan(xi)) or np.any(np.isnan(yi)):
            continue #skip datasets with missing samples in this window
        xi = xi - np.mean(xi)
        yi = yi - np.mean(yi)
        denom = np.sqrt(np.sum(xi**2) * np.sum(yi**2))
        if denom == 0:
            continue
        full    = np.correlate(xi,yi,mode='full') / denom
        center  = len(full) // 2
        cc[i,:] = full[center-max_lag_samps:center+max_lag_samps+1]
    return cc

#crosscorrmat: dataset axis first so it can be fed straight into shaded_error(), as with R2_ratiodata
crosscorrmat = np.full((nplotcontrasts,nsubspaces,nsubspaces,ndatasets,nlags),np.nan)

for idirec in range(ndirections):
    for iplotcontrast in range(nplotcontrasts):
        for isubspace in range(nsubspaces):
            for jsubspace in range(nsubspaces):
                x = R2_ratiodata[0,isubspace,iplotcontrast,:,:]
                y = R2_ratiodata[1,jsubspace,iplotcontrast,:,:]
                crosscorrmat[iplotcontrast,isubspace,jsubspace,:,:] = normalized_crosscorr(x,y,max_lag_samps)

# Example: plot mean +/- ci95 cross-correlation across datasets for one combination
fig,axes = plt.subplots(nsubspaces,nsubspaces,figsize=(7*cm,6.5*cm))
for isubspace in range(nsubspaces):
    for jsubspace in range(nsubspaces):
        for iplotcontrast in range(nplotcontrasts):
        # for iplotcontrast in [1]:
            ax = axes[isubspace,jsubspace]
            shaded_error(lags,crosscorrmat[iplotcontrast,isubspace,jsubspace,:,:],
                        center='mean',error='ci95',color=clrs[iplotcontrast],ax=ax,linewidth=0.8)
            ax.axvline(x=0,color='grey',linestyle='--',linewidth=0.8)
            ax.set_xlabel('lag (s)')
            ax.set_ylabel('cross-correlation')
            ax.axhline(0,linestyle=':',color='grey',linewidth=0.5)
            ax.text(0,0.9,'FF:%s<-\nleads' % subspacelabels[isubspace+1],transform=ax.transAxes,fontsize=4,horizontalalignment='left')
            ax.text(1,0.9,'->FB:%s\nleads' % subspacelabels[jsubspace+1],transform=ax.transAxes,fontsize=4,horizontalalignment='right')
            # ax.set_title('FF:%s, FB:%s' % (subspacelabels[isubspace+1],subspacelabels[jsubspace+1]),fontsize=6)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 2, trim=False)
# my_savefig(fig,figdir,'RRR_joint_time_ratio_FF_FB_splitsubspaces')
print('%1.2f sessions with simultaneous data' % (np.sum(~np.any(np.isnan(crosscorrmat),axis=(0,1,2,4)))/params['nStim']))
my_savefig(fig,figdir,'perf_ratio_crosscorr_FF_FB_subspaces')

# #%% Relationship between subspace ratios:
# # R2_ratiodata is of shape (2,2,2,ndatasets,nT), 
# # where the first dimension is the direction (FF,FB) and the second dimension is the subspace (behav/nonbehav)
# # and the third dimension is the contrast between labeled and unlabeled

# ndatasets = params['nSessions'] * params['nStim']
# nsubspaces          = len(subspaces)
# nplotcontrasts      = len(plotcontrasts)
# ndirections         = 2
# # sigmat = np.full((2,2),np.nan)

# #Compute crosscorrelation function
# ts = params['t_axis']
# dt = np.median(np.diff(ts))
# max_lag_secs = 2.0
# max_lag_samps = int(np.round(max_lag_secs / dt))
# lags = np.arange(-max_lag_samps, max_lag_samps + 1) * dt
# nlags = len(lags)

# def normalized_crosscorr(x,y,max_lag_samps):
#     """Per-row (per-dataset) normalized cross-correlation.
#     x,y: (ndatasets,ntimepoints). Returns (ndatasets,nlags)."""
#     ndatasets   = x.shape[0]
#     nlags       = 2*max_lag_samps + 1
#     cc          = np.full((ndatasets,nlags),np.nan)
#     for i in range(ndatasets):
#         xi,yi = x[i,:],y[i,:]
#         if np.any(np.isnan(xi)) or np.any(np.isnan(yi)):
#             continue #skip datasets with missing samples in this window
#         xi = xi - np.mean(xi)
#         yi = yi - np.mean(yi)
#         denom = np.sqrt(np.sum(xi**2) * np.sum(yi**2))
#         if denom == 0:
#             continue
#         full    = np.correlate(xi,yi,mode='full') / denom
#         center  = len(full) // 2
#         cc[i,:] = full[center-max_lag_samps:center+max_lag_samps+1]
#     return cc

# #crosscorrmat: dataset axis first so it can be fed straight into shaded_error(), as with R2_ratiodata
# crosscorrmat = np.full((ndirections,nplotcontrasts,nsubspaces,nsubspaces,ndatasets,nlags),np.nan)

# for idirec in range(ndirections):
#     for iplotcontrast in range(nplotcontrasts):
#         for isubspace in range(nsubspaces):
#             for jsubspace in range(nsubspaces):
#                 x = R2_ratiodata[idirec,isubspace,iplotcontrast,:,:]
#                 y = R2_ratiodata[idirec,jsubspace,iplotcontrast,:,:]
#                 crosscorrmat[idirec,iplotcontrast,isubspace,jsubspace,:,:] = normalized_crosscorr(x,y,max_lag_samps)

# idirec = 0
# # Example: plot mean +/- ci95 cross-correlation across datasets for one combination
# fig,axes = plt.subplots(nsubspaces,nsubspaces,figsize=(7*cm,7*cm))
# for isubspace in range(nsubspaces):
#     for jsubspace in range(nsubspaces):
#         for iplotcontrast in range(nplotcontrasts):
#             ax = axes[isubspace,jsubspace]
#             shaded_error(lags,crosscorrmat[idirec,iplotcontrast,isubspace,jsubspace,:,:],
#                         center='mean',error='ci95',color=clrs[iplotcontrast],ax=ax,linewidth=0.8)
#             shaded_error(lags,crosscorrmat[idirec,iplotcontrast,isubspace,jsubspace,:,:],
#                         center='mean',error='ci95',color=clrs[iplotcontrast],ax=ax,linewidth=0.8)
#             ax.axvline(x=0,color='grey',linestyle='--',linewidth=0.8)
#             ax.set_title('%s,%s' % (subspacelabels[isubspace+1],subspacelabels[jsubspace+1]),fontsize=6)
#             ax.set_xlabel('lag (s)')
#             ax.set_ylabel('cross-correlation')
#             ax.set_xlabel(f'lag (s), + means {subspacelabels[jsubspace+1]} leads')
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True, offset = 2, trim=False)
# # print('negative lag is x leading, y following. Positive lag is the opposite. X is always')
# # my_savefig(fig,figdir,'RRR_joint_time_ratio_FF_crosscorrelations_subspaces')

# idirec = 1
# # Example: plot mean +/- ci95 cross-correlation across datasets for one combination
# fig,axes = plt.subplots(nsubspaces,nsubspaces,figsize=(7*cm,7*cm))
# for isubspace in range(nsubspaces):
#     for jsubspace in range(nsubspaces):
#         for iplotcontrast in range(nplotcontrasts):
#             ax = axes[isubspace,jsubspace]
#             shaded_error(lags,crosscorrmat[idirec,iplotcontrast,isubspace,jsubspace,:,:],
#                         center='mean',error='ci95',color=clrs[iplotcontrast],ax=ax,linewidth=0.8)
#             shaded_error(lags,crosscorrmat[idirec,iplotcontrast,isubspace,jsubspace,:,:],
#                         center='mean',error='ci95',color=clrs[iplotcontrast],ax=ax,linewidth=0.8)
#             ax.axvline(x=0,color='grey',linestyle='--',linewidth=0.8)
#             ax.set_title('X:%s,Y:%s' % (subspacelabels[isubspace+1],subspacelabels[jsubspace+1]),fontsize=6)
#             ax.set_xlabel('lag (s)')
#             ax.set_ylabel('cross-correlation')
#             ax.set_xlabel(f'lag (s), + means X leads')
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True, offset = 2, trim=False)
# # my_savefig(fig,figdir,'RRR_joint_time_ratio_FB_crosscorrelations_subspaces')
