# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

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
from params import load_params
from utils.corr_lib import filter_sharednan

params = load_params()
figdir = os.path.join(params['figdir'],'RRR','Labeling','Behavior')
resultdir = params['resultdir']

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% Load the data:
version = 'FF'
filename_FF = 'RRR_Joint_labeled_FF_movesplit_2026-04-21_23-07-50'

version = 'FB'
filename_FB = 'RRR_Joint_labeled_FB_movesplit_2026-04-22_17-49-35'

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

#%% Show an example session:
clrs_arealabelpairs = ['grey','grey','red']
narealabelpairs = 3
statelabels = np.array(['Still','Moving'])
ises = 4
fig, axes = plt.subplots(1,2,figsize=(9*cm,3.6*cm),sharex=True,sharey=True)
for istate in range(2):
    handles = []
    ax = axes[istate]
    for iapl,apl in enumerate(sourcearealabelpairs_FF):
        ymeantoplot = np.nanmean(R2_ranks_FF[iapl+1][istate][ises],axis=(0,2,3))
        yerrortoplot = np.nanstd(R2_ranks_FF[iapl+1][istate][ises],axis=(0,2,3)) / np.sqrt(nmodelfits)
        handles.append(shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[iapl],alpha=0.3))

    leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs_FF),frameon=False)
    my_legend_strip(ax)
    ax.set_xlabel('Rank')
    if istate == 0: 
        ax.set_ylabel(r'R$^{2}$')
    ax.set_title(statelabels[istate])

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
# my_savefig(fig,figdir,'RRR_joint_cvR2_labunl_%s_ExampleSesion' % (version))


#%% Show the mean across sessions:
clrs_arealabelpairs = ['grey','grey','red']

nrankstoplot = 15
xposrank = 14
idxs = np.array([1,3])
meanranks = np.nanmean(optim_rank_FF,axis=(-1,-2))
meanR2 = np.nanmean(R2_cv_FF,axis=(-1,-2))

fig, axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharex=True,sharey=True)
for istate in range(2):
    ax = axes[istate]

    handles = []
    ydata = np.nanmean(R2_ranks_FF[idxs[0]][istate],axis=(3,4))
    ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
    handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
                                color=clrs_arealabelpairs[idxs[0]-1],alpha=0.3))
    ydata = np.nanmean(R2_ranks_FF[idxs[1]][istate],axis=(3,4))
    ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
    handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
                                color=clrs_arealabelpairs[idxs[1]-1],alpha=0.3))
    for idx in idxs:
        ax.plot(meanranks[idx][istate],meanR2[idx][istate]+0.005,color=clrs_arealabelpairs[idx-1],marker='v',markersize=5)
    leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs_FF[idxs-1]),frameon=False)
    my_legend_strip(ax)
    ax.set_xlabel('Rank')
    if istate == 0: 
        ax.set_ylabel(r'R$^{2}$')
    ax.set_title(statelabels[istate])
    x = optim_rank_FF[idxs[0],istate].flatten()
    y = optim_rank_FF[idxs[1],istate].flatten()
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    t,p = ttest_rel(x[~nas], y[~nas])
    print('Paired t-test (Rank): p=%.3f' % (p))
    ax.plot([np.nanmean(x),np.nanmean(y)],np.repeat(np.nanmean(meanR2[idxs][istate]),2)+0.007,linestyle='-',color='k',linewidth=2)
    ax.text(np.nanmean([x,y]),np.nanmean(meanR2[idxs][istate])+0.009,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

    x = R2_cv_FF[idxs[0],istate].flatten()
    y = R2_cv_FF[idxs[1],istate].flatten()
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    t,p = ttest_rel(x[~nas], y[~nas])
    print('Paired t-test (R2): p=%.3f' % (p))
    ax.plot([xposrank,xposrank],[np.nanmean(x),np.nanmean(y)],linestyle='-',color='k',linewidth=2)
    ax.text(xposrank+0.5,np.nanmean([x,y])+0.005,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

    ax.set_xticks(np.arange(params['nranks'])[::3]+1)
    ax.set_xlim([0,nrankstoplot])

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
my_savefig(fig,figdir,'RRR_joint_R2_MoveSplit_labunl_FF_%dsessions' % (params['nSessions']))


#%% Show figure for each of the arealabelpairs and each of the dataversions
#Reshape stim x sessions:
R2_data                 = np.reshape(R2_cv_FF,(narealabelpairs+1,2,params['nSessions']*params['nStim']))
optim_rank_data         = np.reshape(optim_rank_FF,(narealabelpairs+1,2,params['nSessions']*params['nStim']))
R2_ranks_data           = np.reshape(R2_ranks_FF,(narealabelpairs+1,2,params['nSessions']*params['nStim'],params['nranks'],nmodelfits,params['kfold']))
if np.any(~np.isnan(R2_data)):
    for istate in range(2):
        for idx in np.array([[1,3]]):
        # clrs        = ['grey',get_clr_area_labeled([sourcearealabelpairs[idx[1]].split('-')[0]])]
            clrs        = ['grey','red']
            fig         = plot_RRR_R2_arealabels_paired(R2_data[idx][:,istate],optim_rank_data[idx][:,istate],R2_ranks_data[idx][:,istate],np.array(sourcearealabelpairs_FF)[idx-1],clrs)
        # my_savefig(fig,figdir,'RRR_cvR2_%s_%s_%dsessions' % (sourcearealabelpairs[idx[1]-1],version,params['nSessions']))

#%% FEEDBACK: 

#%% Show an example session:
clrs_arealabelpairs = ['grey','grey','red']
narealabelpairs = 3
statelabels = np.array(['Still','Moving'])
ises = 4
fig, axes = plt.subplots(1,2,figsize=(9*cm,3.6*cm),sharex=True,sharey=True)
for istate in range(2):
    handles = []
    ax = axes[istate]
    for iapl,apl in enumerate(sourcearealabelpairs_FB):
        ymeantoplot = np.nanmean(R2_ranks_FB[iapl+1][istate][ises],axis=(0,2,3))
        yerrortoplot = np.nanstd(R2_ranks_FB[iapl+1][istate][ises],axis=(0,2,3)) / np.sqrt(nmodelfits)
        handles.append(shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[iapl],alpha=0.3))

    leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs_FB),frameon=False)
    my_legend_strip(ax)
    ax.set_xlabel('Rank')
    if istate == 0: 
        ax.set_ylabel(r'R$^{2}$')
    ax.set_title(statelabels[istate])

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
# my_savefig(fig,figdir,'RRR_joint_cvR2_labunl_%s_ExampleSesion' % (version))


#%% Show the mean across sessions:
clrs_arealabelpairs = ['grey','grey','red']

nrankstoplot = 15
xposrank = 14
idxs = np.array([1,3])
meanranks = np.nanmean(optim_rank_FB,axis=(-1,-2))
meanR2 = np.nanmean(R2_cv_FB,axis=(-1,-2))

fig, axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharex=True,sharey=True)
for istate in range(2):
    ax = axes[istate]

    handles = []
    ydata = np.nanmean(R2_ranks_FB[idxs[0]][istate],axis=(3,4))
    ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
    handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
                                color=clrs_arealabelpairs[idxs[0]-1],alpha=0.3))
    ydata = np.nanmean(R2_ranks_FB[idxs[1]][istate],axis=(3,4))
    ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
    handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
                                color=clrs_arealabelpairs[idxs[1]-1],alpha=0.3))
    for idx in idxs:
        ax.plot(meanranks[idx][istate],meanR2[idx][istate]+0.005,color=clrs_arealabelpairs[idx-1],marker='v',markersize=5)
    leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs_FB[idxs-1]),frameon=False)
    my_legend_strip(ax)
    ax.set_xlabel('Rank')
    if istate == 0: 
        ax.set_ylabel(r'R$^{2}$')
    ax.set_title(statelabels[istate])
    x = optim_rank_FB[idxs[0],istate].flatten()
    y = optim_rank_FB[idxs[1],istate].flatten()
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    t,p = ttest_rel(x[~nas], y[~nas])
    print('Paired t-test (Rank): p=%.3f' % (p))
    ax.plot([np.nanmean(x),np.nanmean(y)],np.repeat(np.nanmean(meanR2[idxs][istate]),2)+0.007,linestyle='-',color='k',linewidth=2)
    ax.text(np.nanmean([x,y]),np.nanmean(meanR2[idxs][istate])+0.009,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

    x = R2_cv_FB[idxs[0],istate].flatten()
    y = R2_cv_FB[idxs[1],istate].flatten()
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    t,p = ttest_rel(x[~nas], y[~nas])
    print('Paired t-test (R2): p=%.3f' % (p))
    ax.plot([xposrank,xposrank],[np.nanmean(x),np.nanmean(y)],linestyle='-',color='k',linewidth=2)
    ax.text(xposrank+0.5,np.nanmean([x,y])+0.005,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

    ax.set_xticks(np.arange(params['nranks'])[::3]+1)
    ax.set_xlim([0,nrankstoplot])

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
my_savefig(fig,figdir,'RRR_joint_R2_MoveSplit_labunl_FB_%dsessions' % (params['nSessions']))


#%% Show figure for each of the arealabelpairs and each of the dataversions
#Reshape stim x sessions:
R2_data                 = np.reshape(R2_cv_FB,(narealabelpairs+1,2,params['nSessions']*params['nStim']))
optim_rank_data         = np.reshape(optim_rank_FB,(narealabelpairs+1,2,params['nSessions']*params['nStim']))
R2_ranks_data           = np.reshape(R2_ranks_FB,(narealabelpairs+1,2,params['nSessions']*params['nStim'],params['nranks'],nmodelfits,params['kfold']))
if np.any(~np.isnan(R2_data)):
    for istate in range(2):
        for idx in np.array([[1,3]]):
        # clrs        = ['grey',get_clr_area_labeled([sourcearealabelpairs[idx[1]].split('-')[0]])]
            clrs        = ['grey','red']
            fig         = plot_RRR_R2_arealabels_paired(R2_data[idx][:,istate],optim_rank_data[idx][:,istate],R2_ranks_data[idx][:,istate],np.array(sourcearealabelpairs_FF)[idx-1],clrs)
        # my_savefig(fig,figdir,'RRR_cvR2_%s_%s_%dsessions' % (sourcearealabelpairs[idx[1]-1],version,params['nSessions']))



#%% Identify which dimensions are particularly enhanced in labeled cells:
data = np.nanmean(R2_ranks,axis=(5)) #average across kfolds
data = np.diff(data,axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

diffmetric = 'ratio' #'difference'
# diffmetric = 'difference' #'difference'
noise_constant = 1e-3
# noise_constant = 0
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm),sharey=True,sharex=True)
ax = axes
handles = []
pthr = 0.05 / (params['nranks']-1) #Bonferroni correction for multiple comparisons across ranks
if diffmetric == 'ratio':
    # ymeantoplot = np.nanmean(data[2],axis=(0,1,3)) / (np.nanmean(data[1],axis=(0,1,3))+1e-3)
    # yerrortoplot = np.nanstd(data[2],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
    ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)
    yerrortoplot = (np.nanstd(data[2],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1],axis=(0,1,3))+noise_constant) / np.sqrt(params['nSessions']*nmodelfits)
# 
    # ydata = (np.nanmean(data[2],axis=(3))+noise_constant) / (np.nanmean(data[1],axis=(3))+noise_constant)
    # ymeantoplot = np.nanmean(ydata,axis=(0,1))
    # yerrortoplot = np.nanstd(ydata,axis=(0,1)) / np.sqrt(params['nSessions']*params['nStim'])

elif diffmetric == 'difference':
    ymeantoplot = np.nanmean(data[2] - data[1],axis=(0,1,3))
    yerrortoplot = np.nanstd(data[2] - data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
handles.append(shaded_error(np.arange(params['nranks']-1)+1,ymeantoplot,yerrortoplot,ax=ax,color='black',alpha=0.3))

for r in range(nrankstoplot):
    # ydata = (data[2,:,:,r]+noise_constant) /  (data[1,:,:,r]+noise_constant)
    # ydata = (np.nanmean(data[2,:,:,r],axis=2)+noise_constant) /  (np.nanmean(data[1,:,:,r],axis=2)+noise_constant)
    ydata = (np.nanmean(data[2,:,:,r],axis=2)) /  (np.nanmean(data[1,:,:,r],axis=2))
    ydata = ydata.flatten()

    # ydata = np.nanmean(ydata,axis=2).flatten()
    h,p = stats.ttest_1samp(ydata,1,nan_policy='omit')
    if p<pthr:
        print('Rank %d is significantly enhanced in unlabeled cells (p=%.3f)' % (r+1,p))
        ax.text(r,1.05,'*',ha='center',va='bottom',color='black',fontsize=10)
    # ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)

if diffmetric == 'ratio':
    ymeantoplot = (np.nanmean(data[3],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)
    yerrortoplot = (np.nanstd(data[3],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1],axis=(0,1,3))+noise_constant) / np.sqrt(params['nSessions']*nmodelfits)
    # ymeantoplot = np.nanmean(data[3],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3))
    # yerrortoplot = np.nanstd(data[3],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)

    # ydata = (np.nanmean(data[3],axis=(3))+noise_constant) / (np.nanmean(data[1],axis=(3))+noise_constant)
    # ymeantoplot = np.nanmean(ydata,axis=(0,1))
    # yerrortoplot = np.nanstd(ydata,axis=(0,1)) / np.sqrt(params['nSessions']*params['nStim'])

elif diffmetric == 'difference':
    ymeantoplot = np.nanmean(data[3] - data[1],axis=(0,1,3))
    yerrortoplot = np.nanstd(data[3] - data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
handles.append(shaded_error(np.arange(params['nranks']-1)+1,ymeantoplot,yerrortoplot,ax=ax,color='red',alpha=0.3))

for r in range(nrankstoplot):
    # ydata = (data[3,:,:,r]+noise_constant) /  (data[1,:,:,r]+noise_constant)
    # ydata = data[3,:,:,r] /  data[1,:,:,r]
    # ydata = np.nanmean(ydata,axis=2).flatten()
    ydata = (np.nanmean(data[3,:,:,r],axis=2)) /  (np.nanmean(data[1,:,:,r],axis=2))

    # ydata = (np.nanmean(data[3,:,:,r],axis=2)+noise_constant) /  (np.nanmean(data[1,:,:,r],axis=2)+noise_constant)
    # ydata = (np.nanmean(data[3,:,:,r],axis=2)) /  (np.nanmean(data[1,:,:,r],axis=2))
    ydata = ydata.flatten()

    h,p = stats.ttest_1samp(ydata,1,nan_policy='omit')
    if p<pthr:
        print('Rank %d is significantly enhanced in labeled cells (p=%.3f)' % (r+1,p))
        ax.text(r+1,1.2,'*',ha='center',va='bottom',color='red',fontsize=10)
    # ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)

# ax.legend(handles,['V1$_{ND}$/V1$_{ND}$','V1$_{PM}$/V1$_{ND}$'],frameon=False)
ax.legend(handles,['PM$_{ND}$/PM$_{ND}$','PM$_{V1}$/PM$_{ND}$'],frameon=False)
my_legend_strip(ax)
ax_nticks(ax,4)
ax.set_xticks(np.arange(nrankstoplot)[::3]+1)
ax.set_xlim([1,nrankstoplot])
ax.set_ylim([0.9,1.25])
ax.set_xlabel('dimension')
ax.set_ylabel('R$^{2}$ %s' % diffmetric)
if diffmetric == 'ratio':
    ax.axhline(y=1,color='grey',linestyle='--')
elif diffmetric == 'difference':
    ax.axhline(y=0,color='grey',linestyle='--')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
# my_savefig(fig,figdir,'RRR_R2_%s_rank_noiseconstant_%s_%dsessions' % (diffmetric,version,params['nSessions']))
# my_savefig(fig,figdir,'RRR_unique_cvR2_V1lab_V1unl_V1unl_%dneurons' % Nsub)

