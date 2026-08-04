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
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy import stats
import pickle
from statsmodels.stats.anova import AnovaRM
from matplotlib.patches import Rectangle

from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.RRRlib import *
from utils.regress_lib import *
from utils.params import load_params
from utils.corr_lib import filter_sharednan

params = load_params()
figdir = os.path.join(params['figdir'],'RRR','Labeling','Time')
resultdir = params['resultdir']

#%%  
version = 'FF_original'
FF_filename = 'RRR_time_Joint_labeled_FF_original_2026-08-04_15-42-10'

version = 'FB_original'
# FB_filename = 'RRR_time_Joint_labeled_FB_original_2026-04-02_22-50-15'
FB_filename = 'RRR_time_Joint_labeled_FF_original_2026-08-04_15-42-10'

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


#%% Show an example session:
ises = 0
# ises = 13
# ises = 1
plotcontrast = np.array([1,3])
clrs = ['grey','red']

R2_toplot = np.reshape(R2_cv_FF[:,0,ises,:,:],(narealabelpairs+1,params['nStim'],params['nT']))
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
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (s)')
ax.set_ylabel('R$^{2}$')
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,figdir,'RRR_joint_time_fullsubspace_%s_example_session_%d' % (version, ises))

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
        ax.set_ylabel('performance (R$^{2}$)')
        ax.set_title(direc + ' ' + subspacelabels[isubspace])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 2)
# my_savefig(fig,figdir,'RRR_joint_time_Raw_FF_FB' )
# my_savefig(fig,figdir,'labeled_RRR_time_Raw_FF_FB_behavsubspace' )

# #%% Plot the ratio across time across sessions: 
# # All ranks averaged
# plotcontrasts = np.array([[1,2],[1,3]])
# noise_constant = 1e-5
# ymin = 0.9

# for direc,data,alps in zip(['FF','FB'],[R2_cv_FF,R2_cv_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB]):

#     R2_toplot = np.reshape(data,(narealabelpairs+1,params['nSessions']*params['nStim'],params['nT']))
#     R2_toplot = np.clip(R2_toplot,np.nanpercentile(R2_toplot,1),np.nanpercentile(R2_toplot,99)) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)

#     if params['direction'] == 'FF': 
#         figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
#     elif params['direction'] == 'FB': 
#         figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']
#     clrs = ['grey','red']

#     fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
#     ax = axes
#     handles = []
#     for iplotcontrast,plotcontrast in enumerate(plotcontrasts):
#         R2_ratio = (R2_toplot[plotcontrast[1],:,:]+noise_constant) / (R2_toplot[plotcontrast[0],:,:]+noise_constant) #add a small constant to avoid division by zero
#         # R2_ratio = (R2_toplot[plotcontrast[1],:,:]+noise_constant) - (R2_toplot[plotcontrast[0],:,:]+noise_constant) #add a small constant to avoid division by zero
#         handles.append(shaded_error(params['t_axis'],R2_ratio,error='ci95',color=clrs[iplotcontrast],alpha=0.3,ax=ax))
#     ax.axhline(y=1,color='grey',linestyle='--')
#     ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
#     thickness = ax.get_ylim()[1]/15
#     ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
#     ax.legend(handles=handles,labels=figlabels,loc='best')
#     my_legend_strip(ax)
#     ax.set_xlim([-1,2])
#     ax_nticks(ax,3)
#     ax.set_xticks(t_ticks)
#     ax.set_xticklabels(t_ticks)
#     ax.set_xlabel('Time (s)')
#     ax.set_ylabel('R$^{2}$ ratio')
#     sns.despine(fig=fig, top=True, right=True, offset = 3)
# # my_savefig(fig,figdir,'RRR_joint_time_ratio_%s' % (version))

#%% Get the R2 ratio data:
plotcontrasts   = np.array([[1,2],[1,3]])
plotcontrasts   = np.array([[2,1],[1,2],[2,3],[1,3]])
# plotcontrasts   = np.array([[1,2],[1,2],[2,3],[1,3]])
# noise_constant  = 1e-4
noise_constant  = 0
# clipval         = 1e-4
clipval         = -np.inf

R2_ratiodata    = np.full((2,2,len(plotcontrasts),params['nSessions']*params['nStim'],params['nT']),np.nan)

for idirec,[direc,data,alps] in enumerate(zip(['FF','FB'],[R2_cv_FF,R2_cv_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
# for idirec,[direc,data,alps] in enumerate(zip(['FF','FB'],[R2_ranks_FF,R2_ranks_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
    # idx_ses = np.arange(params['nSessions'])
    # idx_ses = np.delete(idx_ses,13)
    # data = data[:,idx_ses]
    R2_toplot = np.reshape(data,(narealabelpairs+1,params['nsubspaces'],params['nSessions']*params['nStim'],params['nT']))

    for isubspace in range(2):
    # for iranks,rankstoaverage in enumerate(ranksplit):
        # R2_toplot = data
        # R2_toplot = np.diff(data,axis=4) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
        # R2_toplot = np.diff(data,axis=4) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
        # R2_toplot = np.nanmean(R2_toplot[:,:,:,:,rankstoaverage],axis=(4,5,6)) #average across ranks selected
        # R2_toplot = np.clip(R2_toplot,clipval,np.inf) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
        R2_toplot = np.clip(R2_toplot,clipval,np.inf) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
        # R2_toplot[R2_toplot < 0] = np.nan

        for iplotcontrast,plotcontrast in enumerate(plotcontrasts):
            R2_ratiodata[idirec,isubspace,iplotcontrast,:,:] = (R2_toplot[plotcontrast[1],isubspace+1,:,:]+noise_constant) / (R2_toplot[plotcontrast[0],isubspace,:,:]+noise_constant) #add a small constant to avoid division by zero
            # R2_ratiodata[idirec,iranks,iplotcontrast,:,:] = (R2_toplot[plotcontrast[1],:,:]+noise_constant) - (R2_toplot[plotcontrast[0],:,:]+noise_constant) #add a small constant to avoid division by zero
            # R2_ratio[R2_ratio<0.5] = np.nan

R2_ratiodata = np.concatenate((np.nanmean(R2_ratiodata[:,:,:2],axis=2,keepdims=True),
                               np.nanmean(R2_ratiodata[:,:,2:],axis=2,keepdims=True)),axis=2)

#%% Plot the ratio across time across sessions: 
# noise_constant = 0
ymin = 0.9
# clipval = 1e-4
# clipval = 1e-9
clrs = ['grey','red']
plotcontrasts = np.array([[1,2],[1,3]])

thickness = 0.05

twin_iti = np.array([-1,0])
idx_iti = (params['t_axis']>=twin_iti[0]) & (params['t_axis']<=twin_iti[1])
twin_resp = np.array([0,1.25])
idx_resp = (params['t_axis']>=twin_resp[0]) & (params['t_axis']<=twin_resp[1])

tracedata = np.full((2,2,2,(params['nSessions']-1)*params['nStim'],params['nT']),np.nan)

fig,axes = plt.subplots(2,2,figsize=(10*cm,9*cm),sharex=True,sharey=False)
# for i,[direc,data,alps] in enumerate(zip(['FF','FB'],[R2_cv_FF,R2_cv_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
for idirec,[direc,data,alps] in enumerate(zip(['FF','FB'],[R2_ranks_FF,R2_ranks_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
    for isubspace,subspacelabel in enumerate(subspacelabels[1:]):
        ax = axes[isubspace,idirec]
        if direc == 'FF': 
            figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
        elif direc == 'FB': 
            figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']

        handles = []
        for iplotcontrast,plotcontrast in enumerate(plotcontrasts):
            # tracedata[idirec,iranks,iplotcontrast,:,:] = R2_ratio
            handles.append(shaded_error(params['t_axis'],R2_ratiodata[idirec,isubspace,iplotcontrast,:,:],error='ci95',color=clrs[iplotcontrast],alpha=0.3,ax=ax))
        
        # add_paired_ttest_results(ax,np.nanmean(R2_ratio[:,idx_iti],axis=1),np.nanmean(R2_ratio[:,idx_resp],axis=1),pos=[0.4,0.95])
        add_paired_ttest_results(ax,np.nanmean(R2_ratiodata[idirec,isubspace,0,:,idx_iti],axis=0),
                                 np.nanmean(R2_ratiodata[idirec,isubspace,0,:,idx_resp],axis=0),pos=[0.4,0.9],color=clrs[0])  
        add_paired_ttest_results(ax,np.nanmean(R2_ratiodata[idirec,isubspace,1,:,idx_iti],axis=0),
                                 np.nanmean(R2_ratiodata[idirec,isubspace,1,:,idx_resp],axis=0),pos=[0.4,0.8],color=clrs[1])

        thickness = ax.get_ylim()[1]/25
        ax.axhline(y=1,color='grey',linestyle='--')
        ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
        ax.legend(handles=handles,labels=figlabels,loc='best')
        my_legend_strip(ax)
        ax_nticks(ax,3)
        ax.set_xticks(t_ticks)
        ax.set_xticklabels(t_ticks)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('R$^{2}$ ratio')
        # ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
        ax.set_ylim([ymin,my_ceil(np.max([1.5,ax.get_ylim()[1]]),2)])
        ax.set_xlim([-1,1.8])

        rect = Rectangle((twin_iti[0],ymin),twin_iti[1]-twin_iti[0],np.diff(ax.get_ylim())[0],fc='none',ec='black',alpha=0.5,lw=0.8,linestyle='--')
        ax.add_patch(rect)
        rect = Rectangle((twin_resp[0],ymin),twin_resp[1]-twin_resp[0],np.diff(ax.get_ylim())[0],fc='none',ec='blue',lw=0.8,linestyle='--')
        ax.add_patch(rect)

        ax.set_title(direc + '-' + subspacelabel)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 2, trim=False)
# my_savefig(fig,figdir,'RRR_joint_time_ratio_FF_FB_splitsubspaces')

#%% Relationship between  FF vs FB ratios
# Tracedata is of shape (2,2,2,nSessions-1,nT), 
# where the first dimension is the direction (FF,FB) and the second dimension is the rank (0,1) or (2,3,4)
# and the third dimension is the contrast between labeled and unlabeled for example
iplotcontrast = 1
corrmat = np.full((2,2),np.nan)
sigmat = np.full((2,2),np.nan)
for iranks,irankstoaverage in enumerate(ranksplit):
    for jranks,jrankstoaverage in enumerate(ranksplit):
        corrmat[iranks,jranks],sigmat[iranks,jranks] = stats.spearmanr(R2_ratiodata[0,iranks,iplotcontrast].flatten(),
                                                                       R2_ratiodata[1,jranks,iplotcontrast].flatten(),nan_policy='omit')
        
        datatocorr = R2_ratiodata.copy()
        datatocorr -= np.nanmean(datatocorr,axis=4,keepdims=True)
        corrmat[iranks,jranks],sigmat[iranks,jranks] = stats.spearmanr(datatocorr[0,iranks,iplotcontrast].flatten(),
                                                                       datatocorr[1,jranks,iplotcontrast].flatten(),nan_policy='omit')
        
        # xdata = R2_ratiodata[0,iranks,iplotcontrast].flatten()
        # ydata = R2_ratiodata[1,jranks,iplotcontrast].flatten()
        # xdata,ydata = filter_sharednan(xdata,ydata)
        # corrmat[iranks,jranks],sigmat[iranks,jranks] = stats.pearsonr(xdata,ydata)

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
im = sns.heatmap(corrmat,annot=True,ax=ax,cmap='bwr',vmin=-.6,vmax=.6,cbar_kws={'ticks':[-0.6,-.3,0,0.3,0.6]})
for i in range(2):
    for j in range(2):
        if sigmat[i,j] < 0.05:
            im.text(j+0.5,i+0.3,get_sig_asterisks(sigmat[i,j]),ha='center',va='center',color='k',fontsize=10)
# ax.set_xticks(np.arange(2))
# ax.set_yticks(np.arange(2))
ax.set_xticklabels(ranksplitlabels)
ax.set_yticklabels(ranksplitlabels)
ax.set_xlabel('Feedforward')
ax.set_ylabel('Feedback')
# my_savefig(fig,figdir,'RRR_joint_time_tracedata_corrmat_plotcontrast_%d' % iplotcontrast)

