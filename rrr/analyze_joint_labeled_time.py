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
# subspacelabels = np.array(['Full','Behav','Stim'])
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

    ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],3)])
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
sns.despine(fig=fig, top=True, right=True, offset = 2)
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
sns.despine(fig=fig, top=True, right=True, offset = 2)
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
plotcontrasts   = np.array([[1,2],[1,3]]) #doesn't matter much which contrast is used of course
# plotcontrasts   = np.array([[2,1],[2,3]])
subspaces       = np.array([1,2]) #select behav and non-behav (not full)

clipval         = 1e-3
# clipval         = 1e-4
R2_ratiodata    = np.full((2,2,len(plotcontrasts),params['nSessions']*params['nStim'],params['nT']),np.nan)

for idirec,[direc,data,alps] in enumerate(zip(['FF','FB'],[R2_cv_FF,R2_cv_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
    for isubspace,subspace in enumerate(subspaces):
        for iplotcontrast,plotcontrast in enumerate(plotcontrasts):
            R2_toplot = np.reshape(data[:,subspace,:,:,:],(narealabelpairs+1,params['nSessions']*params['nStim'],params['nT']))
            R2_toplot = np.clip(R2_toplot,clipval,np.inf) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
            R2_ratiodata[idirec,isubspace,iplotcontrast,:,:] = (R2_toplot[plotcontrast[1],:,:]) / (R2_toplot[plotcontrast[0],:,:]) #add a small constant to avoid division by zero

# plotcontrasts   = np.array([[2,1],[1,2],[2,3],[1,3]])
# R2_ratiodata = np.concatenate((np.nanmean(R2_ratiodata[:,:,:2],axis=2,keepdims=True),
#                                np.nanmean(R2_ratiodata[:,:,2:],axis=2,keepdims=True)),axis=2)

#%% Plot the ratio across time across sessions: 
ymin = 0.9
minymax = 1.6
clrs = ['grey','red']
plotcontrasts = np.array([[1,2],[1,3]])

thickness = 0.05
ntests = 2

twin_pre = np.array([-1,0])
twin_post = np.array([0,1])
# twin_pre = np.array([-0.5,0])
idx_pre = (params['t_axis']>=twin_pre[0]) & (params['t_axis']<=twin_pre[1])
idx_post = (params['t_axis']>=twin_post[0]) & (params['t_axis']<=twin_post[1])
patchalpha  = 0.2
patchcolors = ['grey','blue']

fig,axes = plt.subplots(2,2,figsize=(8*cm,7*cm),sharex=False,sharey='col')
for idirec,[direc,data,alps] in enumerate(zip(['FF','FB'],[R2_ranks_FF,R2_ranks_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
    for isubspace,subspacelabel in enumerate(subspacelabels[1:]):
        ax = axes[idirec,isubspace]
        if direc == 'FF': 
            figlabels = ['V1$_{ND2}$/V1$_{ND1}$','V1$_{PM}$/V1$_{ND1}$']
        elif direc == 'FB': 
            figlabels = ['PM$_{ND2}$/PM$_{ND1}$','PM$_{V1}$/PM$_{ND1}$']

        handles = []
        for iplotcontrast,plotcontrast in enumerate(plotcontrasts):
            # handles.append(shaded_error(params['t_axis'],R2_ratiodata[idirec,isubspace,iplotcontrast,:,:],error='ci95',color=clrs[iplotcontrast],alpha=0.3,ax=ax))
            handles.append(shaded_error(params['t_axis'],R2_ratiodata[idirec,isubspace,iplotcontrast,:,:],center='mean',
                                        error='ci95',color=clrs[iplotcontrast],alpha=0.3,ax=ax,linewidth=1))
        
            x,y = filter_sharednan(np.nanmean(R2_ratiodata[idirec,isubspace,iplotcontrast,:,idx_pre],axis=0),np.nanmean(R2_ratiodata[idirec,isubspace,iplotcontrast,:,idx_post],axis=0))
            pval = np.clip(wilcoxon(x,y)[1]*ntests,0,1)
            print('%s, %s, %s vs %s, pre vs post, p=%1.3f' % (direc,subspacelabel,alps[plotcontrast[0]-1],alps[plotcontrast[1]-1],pval))
            add_stat_annotation(ax,-0.5,0.5,1.2+iplotcontrast*0.1,pval,color=clrs[iplotcontrast],h=0,fontsize=6)

        # x,y = filter_sharednan(np.nanmean(R2_ratiodata[idirec,isubspace,1,:,idx_pre],axis=0),np.nanmean(R2_ratiodata[idirec,isubspace,1,:,idx_post],axis=0))
        # pval= wilcoxon(x,y)[1]*ntests
        # add_stat_annotation(ax,-0.5,0.5,1.3,pval,color=clrs[1],h=0,fontsize=6)

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
        rect = Rectangle((twin_pre[0],ymin),twin_pre[1]-twin_pre[0],np.diff(ax.get_ylim())[0],fc=patchcolors[0],
                         ec='none',alpha=patchalpha)
        ax.add_patch(rect)
        rect = Rectangle((twin_post[0],ymin),twin_post[1]-twin_post[0],np.diff(ax.get_ylim())[0],fc=patchcolors[1],
                         ec='none',alpha=patchalpha)
        ax.add_patch(rect)

        # rect = Rectangle((twin_pre[0],ymin),twin_pre[1]-twin_pre[0],np.diff(ax.get_ylim())[0],fc='none',ec='black',alpha=0.5,lw=0.8,linestyle='--')
        # ax.add_patch(rect)
        # rect = Rectangle((twin_post[0],ymin),twin_post[1]-twin_post[0],np.diff(ax.get_ylim())[0],fc='none',ec='blue',lw=0.8,linestyle='--')
        # ax.add_patch(rect)

        ax.set_title(direc + '-' + subspacelabel)
print('Wilcoxon signed rank test, Bonferroni-corrected')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 2, trim=False)
# my_savefig(fig,figdir,'RRR_joint_time_ratio_FF_FB_splitsubspaces')

#%% Relationship between FF vs FB ratios
# R2_ratiodata is of shape (2,2,2,ndatasets,nT), 
# where the first dimension is the direction (FF,FB) and the second dimension is the subspace (behav/nonbehav)
# and the third dimension is the contrast between labeled and unlabeled

subspacelabels = np.array(['Full','Behav','Stim'])
#Find the timepoint of max ratio for FF and FB stimulus related:
idx_time = np.logical_and(params['t_axis']>-.8,params['t_axis']<1.9)
isubspace = 1 #stim-related
iplotcontrast = 1 #lab/unl
# np.argmax(tempdata,axis=2)
clrs = ['purple','green']
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
for idirec, direc in enumerate(['FF','FB']):
    tempdata = R2_ratiodata[idirec,isubspace,iplotcontrast,:,:]
    tempdata = tempdata[:,idx_time]
    # tempdata[np.all(tempdata<=1.5,axis=1),:] = np.nan
    tempdata[tempdata==1] = np.nan
    tempdata = tempdata[~np.all(np.isnan(tempdata),axis=1),:]
    tpoints = params['t_axis'][idx_time]
    ax.hist(tpoints[np.nanargmax(tempdata,axis=1)],color=clrs[idirec],alpha=0.5,bins=params['t_axis']+np.diff(params['t_axis']).mean())

ax.legend(['FF','FB'],frameon=False)
ax.set_ylabel('# datasets')
ax.set_xlabel('time (s)')
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
sns.despine(fig=fig, top=True, right=True, offset = 2, trim=False)
my_savefig(fig,figdir,'Peak_perf_ratio_stim_FF_FB')

#%%
ndatasets           = params['nSessions'] * params['nStim']
nsubspaces          = len(subspaces)
nplotcontrasts      = 2
ndirections         = 2

#compute cross-covariance between datasets for different lags and different timepoints relative to stimulus onset:

ts = params['t_axis']
ntimepoints = len(ts)
dt = np.median(np.diff(ts))
max_lag_secs = 2.0
max_lag_samps = int(np.round(max_lag_secs / dt))
lags = np.arange(-max_lag_samps, max_lag_samps + 1)
tlags = np.arange(-max_lag_samps, max_lag_samps + 1) * dt
nlags = len(lags)
lagticks = np.arange(-2,2.5,1)

crosscovdata = np.full((nsubspaces,nplotcontrasts,ntimepoints,nlags),np.nan)
crosscorrdata = np.full((nsubspaces,nplotcontrasts,ntimepoints,nlags),np.nan)

for isubspace in range(nsubspaces):
    for iplotcontrast in range(nplotcontrasts):
        for it,t in enumerate(ts):
            for ilag,lag in enumerate(lags):
                # print(t,lag)
                # print(it,lag)
                if (it+lag)>0 and (it+lag)<ntimepoints:
                    x = R2_ratiodata[0,isubspace,iplotcontrast,:,it]
                    y = R2_ratiodata[1,isubspace,iplotcontrast,:,it+lag]
                    x,y = filter_sharednan(x,y)
                    crosscovdata[isubspace,iplotcontrast,it,ilag] = np.cov([x,y])[0,1]
                    crosscorrdata[isubspace,iplotcontrast,it,ilag] = np.corrcoef([x,y])[0,1]

#%%
contrastlabels = ['unl/unl','lab/unl']
fig,axes = plt.subplots(2,2,figsize=(8*cm,8*cm))
vmin,vmax = np.nanpercentile(crosscovdata,[2,99])
for isubspace in range(nsubspaces):
    for iplotcontrast in range(nplotcontrasts):
        ax = axes[isubspace,iplotcontrast]
        ax.pcolor(tlags,ts,crosscovdata[isubspace,iplotcontrast],vmin=vmin,vmax=vmax,cmap='magma')
        # ax.pcolor(tlags,ts,crosscorrdata[isubspace,iplotcontrast],vmin=vmin,vmax=vmax,cmap='magma')

        ax.set_xlabel('lag (s)')
        ax.set_xticks(lagticks)
        ax.set_xticklabels(lagticks)
        ax.set_ylabel('time rel. to stim onset (s)')
        ax.set_yticks(t_ticks)
        ax.set_yticklabels(t_ticks)
        ax.set_title('%s-%s' % (subspacelabels[isubspace+1],contrastlabels[iplotcontrast]))
        ax.axvline(0,color='grey',linestyle=':',linewidth=0.8)

sns.despine(fig=fig, top=True, right=True, offset = 2, trim=False)
my_savefig(fig,figdir,'Crosscovariance_FF_FB_splitsubspaces_time_lag_heatmaps')
# my_savefig(fig,figdir,'Crosscorr_FF_FB_splitsubspaces_time_lag_heatmaps')

#%%  
fig,axes = plt.subplots(1,1,figsize=(6*cm,4*cm))
ax = axes
idx_time = np.logical_and(params['t_axis']>=0,params['t_axis']<1)
# idx_time = np.logical_and(params['t_axis']>=-1,params['t_axis']<0)
ls = [':','-']
ylim = [-0.15,0.7]
for isubspace in range(nsubspaces):
    for iplotcontrast in range(nplotcontrasts):

        ax.plot(tlags,np.nanmean(crosscovdata[isubspace,iplotcontrast,idx_time,:],axis=0),
                color=clrs_subspaces[isubspace+1],linestyle=ls[iplotcontrast])
ax.legend(['Behav - unl/unl','Behav - lab/unl','Stim - unl/unl','Stim - lab/unl'],reverse=True,bbox_to_anchor=(1.05,0.8))
ax.text(0.1,0.9,'FB leads',transform=ax.transAxes,fontsize=5,horizontalalignment='left')
ax.text(0.9,0.9,'FF leads',transform=ax.transAxes,fontsize=5,horizontalalignment='right')

ax.axvline(0,color='grey',linestyle=':',linewidth=0.8)
ax.set_xlabel('lag (s)')
ax.set_xticks(lagticks)
ax.set_xticklabels(lagticks)
ax.set_ylim(ylim)
ax.set_ylabel('crosscovariance\nacross datasets')
sns.despine(fig=fig, top=True, right=True, offset = 2, trim=False)
my_savefig(fig,figdir,'Crosscovariance_FF_FB_splitsubspaces_crosscov_avg_resp_respwindow')
# my_savefig(fig,figdir,'Crosscovariance_FF_FB_splitsubspaces_crosscov_avg_resp_baseline')

