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

from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.RRRlib import *
from utils.regress_lib import *
from utils.params import load_params
from utils.corr_lib import filter_sharednan

params = load_params()
figdir = os.path.join(params['figdir'],'RRR','Labeling','FeedForward')
# figdir = os.path.join(params['figdir'],'RRR','Labeling','Feedback')
resultdir = params['resultdir']

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches


#%%  
version = 'FF_original'
filename = 'RRR_Separate_labeled_FF_original_2026-05-21_17-17-36'
exampleses = 12

# version = 'FB_original'
# filename = 'RRR_Separate_labeled_FB_original_2026-05-22_13-16-01'
# exampleses = 1

version = 'FF_AL_original'
filename = 'RRR_Separate_labeled_FF_AL_original_2026-05-22_14-30-47'
exampleses = 4

# version = 'FB_AL_original'
# filename = 'RRR_Separate_labeled_FB_AL_original_2026-05-22_14-45-21'

#%% Load the data:
data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)
for key in data.keys():
    exec(key+'=data[key]')
    # print(key)

with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

# data = np.load(os.path.join(resultdir,FB_filename + '.npz'),allow_pickle=True)
# for key in data.keys():
#     exec(key+'_FB=data[key]')

# with open(os.path.join(resultdir,FB_filename + '_params' + '.txt'), "rb") as myFile:
#     params = pickle.load(myFile)

nmodelfits = params['nmodelfits']

#%% Show an example session:
clrs_arealabelpairs = ['grey','red']
nrankstoplot = 10
narealabelpairs = 2
fig, axes = plt.subplots(1,1,figsize=(5*cm,4.5*cm))
ax = axes
handles = []
for iapl,apl in enumerate(sourcearealabelpairs):
    ymeantoplot = np.nanmean(R2_ranks[iapl][exampleses],axis=(0,2,3))
    yerrortoplot = np.nanstd(R2_ranks[iapl][exampleses],axis=(0,2,3)) / np.sqrt(params['nmodelfits'])
    handles.append(shaded_error(np.arange(params['nranks']),ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[iapl],alpha=0.3))
    meanrank = np.nanmean(optim_rank[iapl][exampleses])
    meanr2 = np.nanmean(R2_cv[iapl][exampleses])
    ax.plot(meanrank,meanr2+0.005,color=clrs_arealabelpairs[iapl],marker='v',markersize=5)

leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Cross-validated R2')
ax.set_xlim([0,nrankstoplot])
# ax.set_xticks([0,1,5,10])
ax.set_xticks([1,4,7,10])
plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
# my_savefig(fig,figdir,'RRR_separate_cvR2_labunl_%s_ExampleSesion' % params['direction'])

#%% Show the mean across sessions:
xposrank = 10
idxs = np.array([0,1])
meanranks = np.nanmean(optim_rank,axis=(-1,-2))
meanR2 = np.nanmean(R2_cv,axis=(-1,-2))
R2_rank_datatoplot = np.nanmean(R2_ranks,axis=(4,5))

idx_ses = np.all(~np.isnan(R2_ranks),axis=(0,2,3,4))
# R2_ranks[:,~idx_ses,:,:,:] = np.nan
# R2_cv[:,~idx_ses] = np.nan
# optim_rank[:,~idx_ses] = np.nan

fig, axes = plt.subplots(1,1,figsize=(5*cm,4.5*cm))
ax = axes
handles = []
# R2_rank_datatoplot = R2_rank_datatoplot[:,~np.any(np.isnan(R2_rank_datatoplot),axis=(0,2,3)),:,:] #keep only the arealabelpairs that have data

ydata = R2_rank_datatoplot[idxs[0]]
ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
                            color=clrs_arealabelpairs[idxs[0]-1],alpha=0.3))
ydata = R2_rank_datatoplot[idxs[1]]
# ydata = np.nanmean(R2_rank_datatoplot[idxs[1]])
ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
                            color=clrs_arealabelpairs[idxs[1]-1],alpha=0.3))
for idx in idxs:
    ax.plot(meanranks[idx],meanR2[idx]+0.005,color=clrs_arealabelpairs[idx-1],marker='v',markersize=5)

leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs[idxs-1]),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Cross-validated R2')

x = optim_rank[idxs[0],:]
y = optim_rank[idxs[1],:]
nas = np.logical_or(np.isnan(x), np.isnan(y))
t,p = ttest_rel(x[~nas], y[~nas])
print('Paired t-test (Rank): p=%.3f' % (p))
ax.plot(meanranks[idxs],np.repeat(np.nanmean(meanR2[idxs]),2)+0.007,linestyle='-',color='k',linewidth=2)
ax.text(np.nanmean(meanranks),np.nanmean(meanR2[idxs])+0.009,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

x = R2_cv[idxs[0],:]
y = R2_cv[idxs[1],:]
nas = np.logical_or(np.isnan(x), np.isnan(y))
t,p = ttest_rel(x[~nas], y[~nas])
print('Paired t-test (R2): p=%.3f' % (p))
ax.plot([xposrank,xposrank],meanR2[idxs],linestyle='-',color='k',linewidth=2)
ax.text(xposrank+0.5,np.nanmean(meanR2[idxs])+0.005,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

ax.set_xticks(np.arange(params['nranks'])[::3]+1)
ax.set_xlim([0,nrankstoplot])

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
# my_savefig(fig,figdir,'RRR_joint_cvR2_labunl_%s_%dsessions' % (version,params['nSessions']))

#%% Show figure for each of the arealabelpairs and each of the dataversions
#Reshape stim x sessions:
R2_data                 = np.reshape(R2_cv,(narealabelpairs,params['nSessions']*params['nStim']))
optim_rank_data         = np.reshape(optim_rank,(narealabelpairs,params['nSessions']*params['nStim']))
R2_ranks_data           = np.reshape(R2_ranks,(narealabelpairs,params['nSessions']*params['nStim'],params['nranks'],nmodelfits,params['kfold']))

clrs        = ['grey','red']
fig         = plot_RRR_R2_arealabels_paired(R2_data,optim_rank_data,R2_ranks_data,np.array(sourcearealabelpairs_FF),clrs)
my_savefig(fig,figdir,'RRR_cvR2_%s_%dsessions' % (version,params['nSessions']))


#%%


# FEEDBACK

#%% Show an example session:
clrs_arealabelpairs = ['grey','red']
nrankstoplot = 10
narealabelpairs = 2
fig, axes = plt.subplots(1,1,figsize=(5*cm,4.5*cm))
ax = axes
# FB_exampleses = 1
handles = []
for iapl,apl in enumerate(sourcearealabelpairs_FB):
    ymeantoplot = np.nanmean(R2_ranks_FB[iapl][FB_exampleses],axis=(0,2,3))
    yerrortoplot = np.nanstd(R2_ranks_FB[iapl][FB_exampleses],axis=(0,2,3)) / np.sqrt(params['nmodelfits'])
    handles.append(shaded_error(np.arange(params['nranks']),ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[iapl],alpha=0.3))
    meanrank = np.nanmean(optim_rank_FB[iapl][FB_exampleses])
    meanr2 = np.nanmean(R2_cv_FB[iapl][FB_exampleses])
    ax.plot(meanrank,meanr2+0.005,color=clrs_arealabelpairs[iapl],marker='v',markersize=5)

leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs_FB),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Cross-validated R2')
ax.set_xlim([0,nrankstoplot])
# ax.set_xticks([0,1,5,10])
ax.set_xticks([1,4,7,10])
plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
# my_savefig(fig,figdir,'RRR_separate_cvR2_labunl_FB_ExampleSesion')

#%% Show figure for each of the arealabelpairs and each of the dataversions
#Reshape stim x sessions:
R2_data                 = np.reshape(R2_cv_FB,(narealabelpairs,params['nSessions']*params['nStim']))
optim_rank_data         = np.reshape(optim_rank_FB,(narealabelpairs,params['nSessions']*params['nStim']))
R2_ranks_data           = np.reshape(R2_ranks_FB,(narealabelpairs,params['nSessions']*params['nStim'],params['nranks'],nmodelfits,params['kfold']))

clrs        = ['grey','red']
fig         = plot_RRR_R2_arealabels_paired(R2_data,optim_rank_data,R2_ranks_data,np.array(sourcearealabelpairs_FB),clrs)
my_savefig(fig,figdir,'RRR_cvR2_FB_%dsessions' % (params['nSessions']))
