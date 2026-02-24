# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os

from rrr.RRR_V1PM_labeling import R2_ranks
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy import stats
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pickle
from statsmodels.stats.multitest import multipletests

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
# from utils.corr_lib import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.pair_lib import value_matching
from utils.psth import compute_tensor
from params import load_params
from utils.corr_lib import filter_sharednan

params = load_params()
figdir = os.path.join(params['figdir'],'RRR','Labeling','Looped')
resultdir = params['resultdir']

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% Load the data:
version = 'FF_original'
filename = 'RRR_Joint_looped_FF_original_2026-02-23_23-26-57'

version = 'FF_behavout'
filename = 'RRR_Joint_looped_FF_behavout_2026-02-24_07-30-21'

# version = 'FB_original'
# filename = 'RRR_Joint_looped_FB_original_2026-02-24_09-26-47'

version = 'FB_behavout'
filename = 'RRR_Joint_looped_FB_behavout_2026-02-24_10-57-25'

#%% Save the data:
data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)

for key in data.keys():
    if key not in ['R2_ranks_neurons']:
        print(key)  
        exec(key+'=data[key]')

with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

params['multcomp_method'] = 'holm'

#%% Show an example session:
clrs_arealabelpairs = np.array([['#7D7D7D','#D100EB'],
                       ['#EB5200', '#EA0101']])
nsourcearealabelpairs = len(sourcearealabelpairs)
ntargetarealabelpairs = len(targetarealabelpairs)

fig, axes = plt.subplots(1,1,figsize=(6*cm,5*cm))
ax = axes
ises = 13
handles = []
labels = []
for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        ymeantoplot = np.nanmean(R2_ranks[isa][ita][ises],axis=(0,2,3))
        yerrortoplot = np.nanstd(R2_ranks[isa][ita][ises],axis=(0,2,3)) / np.sqrt(params['nmodelfits'])
        handles.append(shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[isa,ita],alpha=0.3))
        labels.append(arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea))

leg = ax.legend(handles,labels,frameon=False, reverse=True)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Cross-validated R2')
ax.set_xticks(np.arange(params['nranks'])[::3]+1)
ax.set_title('Example session')
plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
my_savefig(fig,figdir,'RRR_joint_looped_cvR2_ranks_%s_ExampleSesion' % (version))

#%% Show for all sessions:
fig, axes = plt.subplots(1,1,figsize=(6*cm,5*cm))
ax = axes
handles = []
labels = []
for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        ymeantoplot = np.nanmean(R2_ranks[isa][ita],axis=(0,1,3,4))
        yerrortoplot = np.nanstd(R2_ranks[isa][ita],axis=(0,1,3,4)) / np.sqrt(params['nSessions']*params['nStim'])
        handles.append(shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[isa,ita],alpha=0.3))
        labels.append(arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea))

leg = ax.legend(handles,labels,frameon=False, reverse=True)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Cross-validated R2')
ax.set_xticks(np.arange(params['nranks'])[::3]+1)

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
my_savefig(fig,figdir,'RRR_joint_looped_cvR2_ranks_%s_%dsessions' % (version,params['nSessions']))


#%% Show Perofmrance R2 across stim and sessions:
fig, axes = plt.subplots(1,1,figsize=(6*cm,5*cm))
ax = axes
handles = []
labels = []

data = np.full((nsourcearealabelpairs*ntargetarealabelpairs,params['nSessions']*params['nStim']),np.nan)
labels = np.full((nsourcearealabelpairs*ntargetarealabelpairs),'',dtype=object)
for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        data[isa*ntargetarealabelpairs+ita] = R2_cv[isa][ita].flatten()
        labels[isa*ntargetarealabelpairs+ita] = arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea)

ax.plot(np.arange(4),data,color='black',marker='o',linestyle='-',alpha=0.1,markersize=2)
ax.errorbar(np.arange(4),np.nanmean(data,axis=1),yerr=np.nanstd(data,axis=1)/np.sqrt(params['nSessions']*params['nStim']),
            color='black',marker='',linestyle='-',alpha=1,markersize=8)
for i in range(4):
    ax.plot(i,np.nanmean(data,axis=1)[i],color=clrs_arealabelpairs.flatten()[i],marker='o',linestyle='',alpha=1,markersize=8)

ax.set_ylabel('Cross-validated R2')
ax.set_ylim([0,my_ceil(np.nanmax(data),2)])

# Perform pairwise t-tests between the groups and multipletests correction for multiple comparisons:
df = pd.DataFrame(data.T,columns=labels)
from itertools import combinations
group_labels = df.columns
combinations = list(combinations(group_labels, 2))
pvalues = np.full((len(combinations)),np.nan)
xlocs = np.full((len(combinations),2),np.nan)

for icomb, comb in enumerate(combinations):
    group1 = df[comb[0]]
    group2 = df[comb[1]]
    t_stat, p_value = stats.ttest_rel(group1, group2, nan_policy='omit')
    
    pvalues[icomb] = p_value
    xlocs[icomb] = [np.where(labels == comb[0])[0][0], np.where(labels == comb[1])[0][0]]
pvalues_corrected = multipletests(pvalues, method=params['multcomp_method'])[1]

for icomb, comb in enumerate(combinations):
    if pvalues_corrected[icomb] < 0.05:
        print(f"Comparison: {comb[0]} vs {comb[1]}, p-value: {pvalues[icomb]:.4f}")
        x1 = np.where(labels == comb[0])[0][0]
        x2 = np.where(labels == comb[1])[0][0]
        add_stat_annotation(ax, x1, x2, np.nanpercentile(data,90) + icomb*0.0015, pvalues[icomb], h=0)
        
ax_nticks(ax,4)
plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True,offset=3)
ax.set_xticks(np.arange(4),labels,rotation=45,ha='right')
my_savefig(fig,figdir,'RRR_joint_looped_cvR2_%s_%dsessions' % (version,params['nSessions']))

#%% Show optimal rank across stim and sessions:
fig, axes = plt.subplots(1,1,figsize=(6*cm,5*cm))
ax = axes
handles = []
labels = []

data = np.full((nsourcearealabelpairs*ntargetarealabelpairs,params['nSessions']*params['nStim']),np.nan)
labels = np.full((nsourcearealabelpairs*ntargetarealabelpairs),'',dtype=object)
for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        data[isa*ntargetarealabelpairs+ita] = optim_rank[isa][ita].flatten()
        labels[isa*ntargetarealabelpairs+ita] = arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea)

ax.plot(np.arange(4),data,color='black',marker='o',linestyle='-',alpha=0.1,markersize=2)
ax.errorbar(np.arange(4),np.nanmean(data,axis=1),yerr=np.nanstd(data,axis=1)/np.sqrt(params['nSessions']*params['nStim']),
            color='black',marker='',linestyle='-',alpha=1,markersize=8)
for i in range(4):
    ax.plot(i,np.nanmean(data,axis=1)[i],color=clrs_arealabelpairs.flatten()[i],marker='o',linestyle='',alpha=1,markersize=8)
ax.set_ylabel('Optimal rank')
ax.set_ylim([0,9])
ax_nticks(ax,4)
ax.set_yticks(np.arange(0,10,2))

# Perform pairwise t-tests between the groups and multipletests correction for multiple comparisons:
df = pd.DataFrame(data.T,columns=labels)
from itertools import combinations
group_labels = df.columns
combinations = list(combinations(group_labels, 2))
pvalues = np.full((len(combinations)),np.nan)
xlocs = np.full((len(combinations),2),np.nan)

for icomb, comb in enumerate(combinations):
    group1 = df[comb[0]]
    group2 = df[comb[1]]
    t_stat, p_value = stats.ttest_rel(group1, group2, nan_policy='omit')
    
    pvalues[icomb] = p_value
    xlocs[icomb] = [np.where(labels == comb[0])[0][0], np.where(labels == comb[1])[0][0]]
pvalues_corrected = multipletests(pvalues, method=params['multcomp_method'])[1]

for icomb, comb in enumerate(combinations):
    if pvalues_corrected[icomb] < 0.05:
        print(f"Comparison: {comb[0]} vs {comb[1]}, p-value: {pvalues[icomb]:.4f}")
        x1 = np.where(labels == comb[0])[0][0]
        x2 = np.where(labels == comb[1])[0][0]
        add_stat_annotation(ax, x1, x2,8 + icomb*0.2, pvalues[icomb], h=0)

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True,offset=3)
ax.set_xticks(np.arange(4),labels,rotation=45,ha='right')
my_savefig(fig,figdir,'RRR_joint_looped_optimrank_%s_%dsessions' % (version,params['nSessions']))


#%% Identify which dimensions are particularly enhanced in labeled cells:
data = np.nanmean(R2_ranks,axis=(5,6)) #average across kfolds
# data = np.nanmean(np.clip(R2_ranks,0,1),axis=(6)) #average across kfolds
data = np.diff(data,axis=4) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

# np.nanpercentile(R2_ranks,[0,100])

diffmetric = 'ratio' #'difference'
noise_constant = 1e-3
# noise_constant = 0
nrankstoplot = 12
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm),sharey=True,sharex=True)
ax = axes
handles = []
pthr = 0.05 / (params['nranks']-1) #Bonferroni correction for multiple comparisons across ranks

for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        datatoplot = data[isa][ita] #shape (nSessions,nStim,nModelFits,nRanks,nKfolds)
        datatoplot = datatoplot / data[0][0]
        datatoplot = datatoplot+noise_constant / data[0][0]+noise_constant

        # ymeantoplot = (np.nanmean(datatoplot,axis=(0,1,3))) 
        # yerrortoplot = (np.nanstd(datatoplot,axis=(0,1,3))) / np.sqrt(params['nSessions']*params['nStim'])
        
        # datatoplot = np.nanmean(data[isa][ita],axis=-1) #shape (nSessions,nStim,nModelFits,nRanks,nKfolds)
        # datatoplot = datatoplot / np.nanmean(data[0][0],axis=-1)

        ymeantoplot = (np.nanmean(datatoplot,axis=(0,1))) 
        yerrortoplot = (np.nanstd(datatoplot,axis=(0,1))) / np.sqrt(params['nSessions']*params['nStim'])
        
        # datatoplot = np.nanmean(data[isa][ita],axis=(0,1,3)) + noise_constant #shape (nSessions,nStim,nModelFits,nRanks,nKfolds)
        # normalization = np.nanmean(data[0][0],axis=(0,1,3)) + noise_constant
        # datatoplot = datatoplot / normalization

        # ymeantoplot = datatoplot
        # yerrortoplot = (np.nanstd(datatoplot,axis=(0,1,3))) / np.sqrt(params['nSessions']*params['nStim'])
        # yerrortoplot = datatoplot 
        # ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)
        # yerrortoplot = (np.nanstd(data[2],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1],axis=(0,1,3))+noise_constant) / np.sqrt(params['nSessions']*params['nStim'])
    # 
        
        
        
        handles.append(shaded_error(np.arange(params['nranks']-1)+1,ymeantoplot,yerrortoplot,ax=ax,
                                    color=clrs_arealabelpairs[isa,ita],alpha=0.3))


    # ydata = (n

# if diffmetric == 'ratio':
#     # ymeantoplot = np.nanmean(data[2],axis=(0,1,3)) / (np.nanmean(data[1],axis=(0,1,3))+1e-3)
#     # yerrortoplot = np.nanstd(data[2],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*params['nStim'])
#     ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)
#     yerrortoplot = (np.nanstd(data[2],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1],axis=(0,1,3))+noise_constant) / np.sqrt(params['nSessions']*params['nStim'])
# # 
#     # ydata = (np.nanmean(data[2],axis=(3))+noise_constant) / (np.nanmean(data[1],axis=(3))+noise_constant)
#     # ymeantoplot = np.nanmean(ydata,axis=(0,1))
#     # yerrortoplot = np.nanstd(ydata,axis=(0,1)) / np.sqrt(params['nSessions']*params['nStim'])

# elif diffmetric == 'difference':
#     ymeantoplot = np.nanmean(data[2] - data[1],axis=(0,1,3))
#     yerrortoplot = np.nanstd(data[2] - data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*params['nStim'])
# handles.append(shaded_error(np.arange(params['nranks']-1)+1,ymeantoplot,yerrortoplot,ax=ax,color='black',alpha=0.3))

# for r in range(nrankstoplot):
#     # ydata = (data[2,:,:,r]+noise_constant) /  (data[1,:,:,r]+noise_constant)
#     # ydata = (np.nanmean(data[2,:,:,r],axis=2)+noise_constant) /  (np.nanmean(data[1,:,:,r],axis=2)+noise_constant)
#     ydata = (np.nanmean(data[2,:,:,r],axis=2)) /  (np.nanmean(data[1,:,:,r],axis=2))
#     ydata = ydata.flatten()

#     # ydata = np.nanmean(ydata,axis=2).flatten()
#     h,p = stats.ttest_1samp(ydata,1,nan_policy='omit')
#     if p<pthr:
#         print('Rank %d is significantly enhanced in unlabeled cells (p=%.3f)' % (r+1,p))
#         ax.text(r,1.05,'*',ha='center',va='bottom',color='black',fontsize=10)
#     # ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)

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


