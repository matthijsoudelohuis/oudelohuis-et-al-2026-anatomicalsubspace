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
figdir = os.path.join(params['figdir'],'RRR','Labeling','Behavior')
resultdir = params['resultdir']

#%% Load the data:
version = 'FF'
filename_FF = 'RRR_Joint_labeled_behavsubspace_FF_2026-08-03_15-15-45'
# filename_FF = 'RRR_Joint_labeled_behavsubspace_FF_2026-08-03_15-15-45'
filename_FF = 'RRR_Joint_labeled_behavsubspace_FF_2026-08-03_16-11-31'
version = 'FB'
filename_FB = 'RRR_Joint_labeled_behavsubspace_FB_2026-08-03_17-20-57'

#%% Load the FF data:
data = np.load(os.path.join(resultdir,filename_FF + '.npz'),allow_pickle=True)

for key in data.keys():
    exec(key+'_FF=data[key]')

with open(os.path.join(resultdir,filename_FF + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

nmodelfits = params['nmodelfits']
Nsub = params['Nsub']

#%% Load the FF data:
data = np.load(os.path.join(resultdir,filename_FB + '.npz'),allow_pickle=True)

for key in data.keys():
    exec(key+'_FB=data[key]')

with open(os.path.join(resultdir,filename_FB + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

nmodelfits = params['nmodelfits']
Nsub = params['Nsub']

#%% FEEDFORWARD:

#%% Show an example FF session:
subspacelabels = np.array(['Full','Behav','Non-behav'])
clrs_subspaces = get_clr_subspaces(subspacelabels)
lines_subspaces = ['-','--',':']
ises = 0
nrankstoplot = 6
fig, axes = plt.subplots(1,1,figsize=(4.5*cm,3.6*cm),sharex=True,sharey=True)
ax = axes
handles = []
for isubspace in range(3):
    ymeantoplot = np.nanmean(R2_ranks_FF[0][isubspace][ises],axis=(0,2,3))
    yerrortoplot = np.nanstd(R2_ranks_FF[0][isubspace][ises],axis=(0,2,3)) / np.sqrt(nmodelfits)
    handles.append(shaded_error(np.arange(params['nranks']),ymeantoplot,yerrortoplot,ax=ax,
                                color=clrs_subspaces[isubspace],alpha=0.3,linewidth=1))
    # handles.append(shaded_error(np.arange(params['nranks']),ymeantoplot,yerrortoplot,ax=ax,
                                # color='k',linestyle=lines_subspaces[isubspace],alpha=0.3))

leg = ax.legend(handles,subspacelabels,frameon=False)
my_legend_strip(ax)
ax_nticks(ax,4)
ax.set_xlabel('rank')
ax.set_ylabel(r'performance (R$^{2}$)')
ax.set_xticks(np.arange(1,20)[::2])
ax.set_xlim([0,nrankstoplot])
ax.set_ylim([0,ax.get_ylim()[1]])
sns.despine(fig=fig,trim=False,top=True,right=True,offset=2)
my_savefig(fig,figdir,'perf_behavsubspace_rank_FF_exampleSession_%s' % sessions[ises].session_id)

#%% Show an example FB session:
ises = 2
fig, axes = plt.subplots(1,1,figsize=(4.5*cm,3.6*cm),sharex=True,sharey=True)
ax = axes
handles = []
for isubspace in range(3):
    ymeantoplot = np.nanmean(R2_ranks_FB[0][isubspace][ises],axis=(0,2,3))
    yerrortoplot = np.nanstd(R2_ranks_FB[0][isubspace][ises],axis=(0,2,3)) / np.sqrt(nmodelfits)
    handles.append(shaded_error(np.arange(params['nranks']),ymeantoplot,yerrortoplot,ax=ax,
                                color=clrs_subspaces[isubspace],alpha=0.3))
    # handles.append(shaded_error(np.arange(params['nranks']),ymeantoplot,yerrortoplot,ax=ax,
    #                             color='k',linestyle=lines_subspaces[isubspace],alpha=0.3))

leg = ax.legend(handles,subspacelabels,frameon=False,bbox_to_anchor=(0.8,0.8))
# my_legend_strip(ax)
ax.set_xlabel('rank')
ax.set_ylabel(r'performance (R$^{2}$)')
ax.set_xticks(np.arange(1,20)[::2])
ax.set_xlim([0,nrankstoplot])
ax.set_ylim([0,ax.get_ylim()[1]])
sns.despine(fig=fig,trim=False,top=True,right=True,offset=2)

#%% 
R2_ranks_FF_all = np.nanmean(R2_ranks_FF[0][0], axis=(0,1,3,4))
R2_ranks_FF_sum = np.nanmean(R2_ranks_FF[0][1], axis=(0,1,3,4)) + np.nanmean(R2_ranks_FF[0][2], axis=(0,1,3,4)) 

fig, axes = plt.subplots(1,2,figsize=(8.5*cm,3.6*cm),sharex=True,sharey=True)
ax = axes[0]
ax.plot(np.arange(params['nranks'])+1,R2_ranks_FF_all,color='k')
ax.plot(np.arange(params['nranks'])+1,R2_ranks_FF_sum,color='blue',linestyle='--')
ax.set_xlabel('Rank')

R2_ranks_FB_all = np.nanmean(R2_ranks_FB[0][0], axis=(0,1,3,4))
R2_ranks_FB_sum = np.nanmean(R2_ranks_FB[0][1], axis=(0,1,3,4)) + np.nanmean(R2_ranks_FB[0][2], axis=(0,1,3,4))

ax = axes[1]
ax.plot(np.arange(params['nranks'])+1,R2_ranks_FB_all,color='k')
ax.plot(np.arange(params['nranks'])+1,R2_ranks_FB_sum,color='blue',linestyle='--')

# leg = ax.legend(handles,subspacelabels[1:],frameon=False)
# my_legend_strip(ax)
ax.set_ylabel(r'R$^{2}$')
ax.set_xlim([1,nrankstoplot])
sns.despine(fig=fig,trim=False,top=True,right=True,offset=2)

#%% 
nrankstoplot = 6
tempdata = np.diff(np.nanmean(R2_ranks_FF[0],(-1,-2)), axis=-1)
tempdata[tempdata<0] = np.nan
fracdata_behav = (tempdata[1] / (tempdata[1] + tempdata[2])).reshape(np.shape(R2_ranks_FF)[2]*params['nStim'],params['nranks']-1)
fracdata_nobeh = (tempdata[2] / (tempdata[1] + tempdata[2])).reshape(np.shape(R2_ranks_FF)[2]*params['nStim'],params['nranks']-1)

fracdata_behav = fracdata_behav[:,:nrankstoplot]
fracdata_nobeh = fracdata_nobeh[:,:nrankstoplot]
fig, axes = plt.subplots(1,1,figsize=(4.5*cm,3.6*cm),sharex=True,sharey=True)
handles = []
ax = axes
handles.append(shaded_error(np.arange(nrankstoplot)+1,y=fracdata_behav,ax=ax,color=clrs_subspaces[1],alpha=0.2,linewidth=1))
handles.append(shaded_error(np.arange(nrankstoplot)+1,y=fracdata_nobeh,ax=ax,color=clrs_subspaces[2],alpha=0.2,linewidth=1))

tempdata = np.diff(np.nanmean(R2_ranks_FB[0],(-1,-2)), axis=-1)
tempdata[tempdata<0] = np.nan
fracdata_behav = (tempdata[1] / (tempdata[1] + tempdata[2])).reshape(np.shape(R2_ranks_FB)[2]*params['nStim'],params['nranks']-1)
fracdata_nobeh = (tempdata[2] / (tempdata[1] + tempdata[2])).reshape(np.shape(R2_ranks_FB)[2]*params['nStim'],params['nranks']-1)
fracdata_behav = fracdata_behav[:,:nrankstoplot]
fracdata_nobeh = fracdata_nobeh[:,:nrankstoplot]

handles.append(shaded_error(np.arange(nrankstoplot)+1,y=fracdata_behav,ax=ax,color=clrs_subspaces[1],alpha=0.2,linestyle='--',linewidth=1))
handles.append(shaded_error(np.arange(nrankstoplot)+1,y=fracdata_nobeh,ax=ax,color=clrs_subspaces[2],alpha=0.2,linestyle='--',linewidth=1))
ax.axhline(y=1,color='k',linestyle='--',linewidth=1)
ax.set_xlabel('dimension')
leg = ax.legend(handles,subspacelabels[1:],frameon=False)
# my_legend_strip(ax)
ax.set_ylabel(r'relative R$^{2}$')
ax.set_xticks(np.arange(1,20)[::2])
ax.set_xlim([0.75,nrankstoplot])
ax.set_ylim([0,1])
sns.despine(fig=fig,trim=False,top=True,right=True,offset=2)
my_savefig(fig,figdir,'frac_perf_behavsubspace')

#%% Show an example session per population:
clrs_arealabelpairs = ['grey','grey','red']
narealabelpairs = 3
subspacelabels = np.array(['Full','Behav','Non-behav'])
ises = 1
fig, axes = plt.subplots(1,3,figsize=(9*cm,3.6*cm),sharex=True,sharey=True)
for isubspace in range(3):
    handles = []
    ax = axes[isubspace]
    for iapl,apl in enumerate(sourcearealabelpairs_FF):
        ymeantoplot = np.nanmean(R2_ranks_FF[iapl+1][isubspace][ises],axis=(0,2,3))
        yerrortoplot = np.nanstd(R2_ranks_FF[iapl+1][isubspace][ises],axis=(0,2,3)) / np.sqrt(nmodelfits)
        handles.append(shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[iapl],alpha=0.3))

    leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs_FF),frameon=False)
    my_legend_strip(ax)
    ax.set_xlabel('Rank')
    if isubspace == 0: 
        ax.set_ylabel(r'R$^{2}$')
    ax.set_title(subspacelabels[isubspace])
ax.set_xlim([1,nrankstoplot])
ax.set_xticks(np.arange(nrankstoplot)+1)
plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
# my_savefig(fig,figdir,'RRR_joint_behavsubspace_%s_ExampleSession' % (version))


#%% 
fig, axes = plt.subplots(1,2,figsize=(7.1*cm,3.9*cm),sharex=True,sharey=False)
contrasts   = np.array([[3,1],[2,1]])
# contrasts   = np.array([[3,2],[1,2]])
clrs        = ['red','grey']

ax = axes[0]
handles = []
for icontrast,contrast in enumerate(contrasts):
    for isubspace in range(1,3):
        ratiodata = (R2_cv_FF[contrast[0]][isubspace] / R2_cv_FF[contrast[1]][isubspace]).flatten()
        handle = ax.errorbar(x=isubspace-1,y=np.nanmean(ratiodata),yerr=np.nanstd(ratiodata)/np.sqrt(len(ratiodata)),
                             color=clrs[icontrast],capsize=2,elinewidth=1,marker='o',markersize=4)[0]
        if isubspace == 0:
            handles.append(handle)
ax.axhline(y=1,color='k',linestyle='--',linewidth=1)
ax.legend(handles,['V1$_{PM}$/V1$_{ND}$','V1$_{ND}$/V1$_{ND}$'],frameon=False)
my_legend_strip(ax)
ax.set_title('FF')
ax.set_ylabel(r'performance ratio')
ax_nticks(ax,4)
ax.set_xticks([0,1],subspacelabels[1:])

ax = axes[1]
handles = []
for icontrast,contrast in enumerate(contrasts):
    for isubspace in range(1,3):
        ratiodata = (R2_cv_FB[contrast[0]][isubspace] / R2_cv_FB[contrast[1]][isubspace]).flatten() 
        handle = ax.errorbar(x=isubspace-1,y=np.nanmean(ratiodata),yerr=np.nanstd(ratiodata)/np.sqrt(len(ratiodata)),
                             color=clrs[icontrast],capsize=2,elinewidth=1,marker='o',markersize=4)[0]
        if isubspace == 0:
            handles.append(handle)
ax.axhline(y=1,color='k',linestyle='--',linewidth=1)
ax.legend(handles,['PM$_{V1}$/PM$_{ND}$','PM$_{ND}$/PM$_{ND}$'],frameon=False)
my_legend_strip(ax)
ax_nticks(ax,4)
ax.set_xlim([-0.3,1.3])
ax.set_title('FB')
ax.set_ylabel(r'performance ratio')
# ax.set_ylim([0,1])
plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True,offset=2)
my_savefig(fig,figdir,'perf_ratio_labunl_behavsubspace_%dsessions' % params['nSessions'])

