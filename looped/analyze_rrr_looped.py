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
from statsmodels.stats.multitest import multipletests

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from looped.run_RRR_joint_looped_time import R2_ranks
from utils.plot_lib import * #get all the fixed color schemes
# from utils.corr_lib import *
# from utils.RRRlib import *
from utils.regress_lib import *
from utils.params import load_params

params = load_params()
figdir = os.path.join(params['figdir'],'RRR','Labeling','Looped')
resultdir = params['resultdir']

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% Load the data:
version = 'FF_original'
filename = 'RRR_Joint_looped_FF_original_2026-02-23_23-26-57'

# version = 'FF_behavout'
# filename = 'RRR_Joint_looped_FF_behavout_2026-02-24_07-30-21'

# version = 'FB_original'
# filename = 'RRR_Joint_looped_FB_original_2026-02-24_09-26-47'

# version = 'FB_behavout'
# filename = 'RRR_Joint_looped_FB_behavout_2026-02-24_10-57-25'

#%% Load the data:
data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)

for key in data.keys():
    if key not in ['R2_ranks_neurons','weights_in']:
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
nrankstoplot = 10
fig, axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
ises = 13
ises = 9
handles = []
labels = []
for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        ymeantoplot = np.nanmean(R2_ranks[isa][ita][ises],axis=(0,2,3))
        yerrortoplot = np.nanstd(R2_ranks[isa][ita][ises],axis=(0,2,3)) / np.sqrt(params['nmodelfits'])
        handles.append(shaded_error(np.arange(params['nranks']),ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[isa,ita],alpha=0.3))
        labels.append(arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea))

leg = ax.legend(handles,labels,frameon=False, reverse=True,fontsize=5)
my_legend_strip(ax)
ax.set_xlabel('rank')
ax.set_ylabel('performance')
ax.set_xticks(np.arange(params['nranks'])[::3]+1)
ax.set_title('Example session')
ax.set_xlim([0,nrankstoplot])
plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
# my_savefig(fig,figdir,'RRR_joint_looped_cvR2_ranks_%s_ExampleSession_%d' % (version, ises))

#%% Show for all sessions:
import matplotlib.gridspec as gridspec
nrankstoplot = 12
fig = plt.figure(figsize=(6*cm,5*cm))
gs = gridspec.GridSpec(3, 3)  # 3x3 grid

ax = fig.add_subplot(gs[1:3, 0:2])     

handles = []
labels = []
for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        ymeantoplot = np.nanmean(R2_ranks[isa][ita],axis=(0,1,3,4))
        yerrortoplot = np.nanstd(R2_ranks[isa][ita],axis=(0,1,3,4)) / np.sqrt(params['nSessions']*params['nStim'])
        handles.append(shaded_error(np.arange(params['nranks']),ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[isa,ita],alpha=0.3))
        labels.append(arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea))
    
        ax.plot(np.nanmean(optim_rank[isa][ita]),np.nanmean(R2_cv[isa][ita])+0.005,color=clrs_arealabelpairs[isa,ita],marker='v',markersize=5)

leg = ax.legend(handles,labels,frameon=False, reverse=True)
my_legend_strip(ax)
ax.set_xlabel('rank')
ax.set_ylabel('performance')
# ax.set_xticks(np.arange(params['nranks'])[::3]+1)
ax.set_xticks(np.arange(params['nranks'])[::3]+1)
ax.set_xlim([0,nrankstoplot])

#Show Performance R2 across stim and sessions:
ax = fig.add_subplot(gs[1:3, 2])     
handles = []
labels = []

data = np.full((nsourcearealabelpairs*ntargetarealabelpairs,params['nSessions']*params['nStim']),np.nan)
labels = np.full((nsourcearealabelpairs*ntargetarealabelpairs),'',dtype=object)
for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        data[isa*ntargetarealabelpairs+ita] = R2_cv[isa][ita].flatten()
        labels[isa*ntargetarealabelpairs+ita] = arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea)

# ax.plot(np.arange(4),data,color='black',marker='o',linestyle='-',alpha=0.1,markersize=2)
# ax.errorbar(np.arange(4),np.nanmean(data,axis=1),yerr=np.nanstd(data,axis=1)/np.sqrt(params['nSessions']*params['nStim']),
            # color='black',marker='',linestyle='-',alpha=1,markersize=8)
# for i in range(4):
#     ax.plot(i,np.nanmean(data,axis=1)[i],color=clrs_arealabelpairs.flatten()[i],marker='o',linestyle='',alpha=1,markersize=8)

ax.set_axis_off()

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

# Show optimal rank across stim and sessions:
ax = fig.add_subplot(gs[0,0:2])     
labels = []
data = np.full((nsourcearealabelpairs*ntargetarealabelpairs,params['nSessions']*params['nStim']),np.nan)
labels = np.full((nsourcearealabelpairs*ntargetarealabelpairs),'',dtype=object)
for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        data[isa*ntargetarealabelpairs+ita] = optim_rank[isa][ita].flatten()
        labels[isa*ntargetarealabelpairs+ita] = arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea)
# ax.set_ylabel('Optimal rank')
# ax.set_ylim([0,9])
# ax_nticks(ax,4)
# ax.set_yticks(np.arange(0,10,2))
ax.set_axis_off()

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

# plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True,offset=3)
ax.set_xticks(np.arange(4),labels,rotation=45,ha='right')
# my_savefig(fig,figdir,'RRR_joint_looped_%s_%dsessions' % (version,params['nSessions']))

#%% Show Performance R2 across stim and sessions:
fig, axes = plt.subplots(1,1,figsize=(4.5*cm,5.9*cm))
ax = axes
handles = []
labels = []
# normalized = True
normalized = False

data = np.full((nsourcearealabelpairs*ntargetarealabelpairs,params['nSessions']*params['nStim']),np.nan)
labels = np.full((nsourcearealabelpairs*ntargetarealabelpairs),'',dtype=object)
for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        # data[isa*ntargetarealabelpairs+ita] = R2_cv[isa][ita].flatten()
        # labels[isa*ntargetarealabelpairs+ita] = arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea)

        data[ita*ntargetarealabelpairs+isa] = R2_cv[isa][ita].flatten()
        labels[ita*ntargetarealabelpairs+isa] = arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea)

if normalized:
    data = data - data[0][np.newaxis,:] #normalize the R2 values of each group by the maximum R2 value across sessions and stims in that group to get a better sense of the relative performance across groups (since the absolute R2 values can differ a lot across groups)

# ax.plot(np.arange(4),data,color='black',marker='o',linestyle='-',alpha=0.1,markersize=2)
ax.errorbar(np.arange(4),np.nanmean(data,axis=1),yerr=np.nanstd(data,axis=1)/np.sqrt(params['nSessions']),
# ax.errorbar(np.arange(4),np.nanmean(data,axis=1),yerr=np.nanstd(data,axis=1),
            color='black',marker='',linestyle='-',alpha=1,markersize=6)

for i in range(4):
    ax.plot(i,np.nanmean(data,axis=1)[i],color=clrs_arealabelpairs.flatten()[i],marker='o',linestyle='',alpha=1,markersize=6)

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
        add_stat_annotation(ax, x1, x2, np.nanpercentile(data,70) + icomb*0.001, pvalues[icomb], h=0)


ax.set_ylabel('performance')
ax.set_ylim([my_floor(np.nanmin(np.nanmean(data,axis=1))*0.9,3),my_ceil(np.nanmax(np.nanmean(data,axis=1))*1.2,3)])
# ax.set_ylim([-.01,my_ceil(np.nanmax(data),2)])
# ax.set_ylim([-.002,0.015])
ax_nticks(ax,4)
ax.set_xticks(np.arange(4),labels,rotation=45,ha='right')

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
my_savefig(fig,figdir,'Looped_cvR2_bars_%s_%dsessions' % (version,params['nSessions']))


#%% Show the sum of each projection added R2 and the added R2 of V1Pm PMv1:
datatotest = R2_cv - R2_cv[0,0][np.newaxis,np.newaxis,:,:] #calculate the added R2 of each projection by subtracting the R2 of the model with only the first rank (which captures shared variance across all models) from the R2 of the full model. This gives the R2 uniquely explained by each projection.
xdata = datatotest[0,1] + datatotest[1,0] #sum of the added R2 of both projections
ydata = datatotest[1,1] #added R2 of V1Pm or PMv1 projection

fig, ax = plt.subplots(figsize=(4*cm,4*cm))
ax.plot(xdata.flatten(),ydata.flatten(),color='red',marker='o',linestyle='',alpha=0.8,markersize=2)
ax.plot([-1,1],[-1,1],color='black',linestyle='--')
ax.set_xlim([-0.01,my_ceil(np.nanmax([xdata,ydata]),2)])
ax.set_ylim([-0.01,my_ceil(np.nanmax([xdata,ydata]),2)])
add_paired_ttest_results(ax,xdata.flatten(),ydata.flatten(),pos=[0.8,0.2])
ax_nticks(ax,3)
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))
ax.set_xlabel(r'ΔR$^{2}$ V1$_{PM}$-PM$_{ND}$ + V1$_{ND}$-PM$_{V1}$')
ax.set_ylabel(r'ΔR$^{2}$ V1$_{PM}$-PM$_{V1}$')
sns.despine(fig=fig,trim=False,top=True,right=True,offset=2)
my_savefig(fig,figdir,'RRR_cvR2_linearsum_joint_looped_%s_%dsessions' % (version,params['nSessions']))
 


#%% 

####### ### #     # ####### 
   #     #  ##   ## #       
   #     #  # # # # #       
   #     #  #  #  # #####   
   #     #  #     # #       
   #     #  #     # #       
   #    ### #     # ####### 

#%% Load the data:
version = 'FF_original'
filename = 'RRR_Joint_looped_FF_original_2026-04-28_22-38-28'

# version = 'FF_behavout'
# filename = 'RRR_Joint_looped_FF_behavout_2026-02-24_07-30-21'

# version = 'FB_original'
# filename = 'RRR_Joint_looped_FB_original_2026-04-29_02-22-58'

# version = 'FB_behavout'
# filename = 'RRR_Joint_looped_FB_behavout_2026-02-24_10-57-25'

#%% Load the data:
data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)

for key in data.keys():
    exec(key+'=data[key]')

with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

#%% Plotting the mean across time across sessions: 

clrs_arealabelpairs = np.array([['#7D7D7D','#D100EB'],
                       ['#EB5200', '#EA0101']])
nsourcearealabelpairs = len(sourcearealabelpairs)
ntargetarealabelpairs = len(targetarealabelpairs)

R2_toplot = np.reshape(R2_cv,(nsourcearealabelpairs,ntargetarealabelpairs,params['nSessions']*params['nStim'],params['nT']))

t_ticks = np.array([-1,0,1,2])

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
handles = []
labels = []
for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        handles.append(shaded_error(params['t_axis'],R2_toplot[isa,ita,:,:],error='sem',
                                    color=clrs_arealabelpairs[isa,ita],alpha=0.3,ax=ax))#clrs[iapl],alpha=0.3,ax=ax))

        labels.append(arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea))

leg = ax.legend(handles,labels,frameon=False, reverse=True,bbox_to_anchor=(0.7,0.3))

ymin = 0
ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],3)])
ax.set_xlim([-1,2])
thickness = ax.get_ylim()[1]/15
ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('time (s)')
ax.set_ylabel('performance')

# plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,figdir,'RRR_looped_time_%s' % (version))

#%% Plot the ratio across time across sessions: 
R2_toplot = np.reshape(R2_cv,(nsourcearealabelpairs,ntargetarealabelpairs,params['nSessions']*params['nStim'],params['nT']))
# R2_toplot = np.clip(R2_toplot,np.nanpercentile(R2_toplot,1),np.nanpercentile(R2_toplot,99)) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
# plotcontrasts = np.array([[1,2],[1,3]])
noise_constant = 1e-4
clipval = 1e-4
R2_toplot = np.clip(R2_toplot,clipval,np.nanpercentile(R2_toplot,100)) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
# noise_constant = 0

# noise_constant = 0 #add a small constant to avoid division by zero and extreme ratios when R2 values are close to zero (e.g. for shuffled data)              
ymin = 0.9

if params['direction'] == 'FF': 
    figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
elif params['direction'] == 'FB': 
    figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']
clrs = ['grey','red']

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
handles = []
labels = []
for isa,sourcearea in enumerate(sourcearealabelpairs):
    for ita,targetarea in enumerate(targetarealabelpairs):
        R2_ratio = (R2_toplot[isa,ita,:,:]+noise_constant) / (R2_toplot[0,0,:,:]+noise_constant) #add a small constant to avoid division by zero
        # R2_ratio = R2_toplot[isa,ita,:,:] - R2_toplot[0,0,:,:] #add a small constant to avoid division by zero
        # print(np.nanmean(R2_ratio))
        handles.append(shaded_error(params['t_axis'],R2_ratio,error='sem',
                                    color=clrs_arealabelpairs[isa,ita],alpha=0.3,ax=ax))#clrs[iapl],alpha=0.3,ax=ax))

        labels.append(arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea))
leg = ax.legend(handles,labels,frameon=False, reverse=True,bbox_to_anchor=(0.7,0.3))

my_legend_strip(ax)
ax.set_xlim([-1,2])
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('time (s)')
ax.set_ylabel('performance ratio')
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,figdir,'RRR_looped_time_ratio_%s' % (version))



#%% Get the R2 ratio data for different dimensions:


#%% Plot the ratio across time across sessions: 
noise_constant = 1e-4
clipval = 1e-4

# clipval = -np.inf
# noise_constant = 0

clipval = 0
noise_constant = 1e-4

ranksplit       = [np.array([0,1]),np.array([2,3,4])]
ranksplit       = [np.array([0,1]),np.array([2,3,4])]
# ranksplit       = [np.array([0]),np.array([1,2,3,4])]
# ranksplit       = [np.array([0]),np.array([2,3,4])]
ranksplitlabels = ['ranks 1 and 2','ranks 3-5']

R2_ratiodata    = np.full((2,2,len(ranksplit),params['nSessions']*params['nStim'],params['nT']),np.nan)
from matplotlib.patches import Rectangle

ymin = 0.9
twin_iti = np.array([-1,0])
idx_iti = (params['t_axis']>=twin_iti[0]) & (params['t_axis']<=twin_iti[1])
twin_resp = np.array([0,1.25])
idx_resp = (params['t_axis']>=twin_resp[0]) & (params['t_axis']<=twin_resp[1])

if params['direction'] == 'FF': 
    figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
elif params['direction'] == 'FB': 
    figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']
clrs = ['grey','red']

for iranks,rankstoaverage in enumerate(ranksplit):
    data = R2_ranks
    R2_toplot = np.diff(R2_ranks,axis=5) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
    R2_toplot = np.nanmean(R2_toplot[:,:,:,:,:,rankstoaverage],axis=(5,6,7)) #average across ranks selected
    R2_toplot = np.reshape(R2_toplot,(2,2,params['nSessions']*params['nStim'],params['nT']))
    R2_toplot = np.clip(R2_toplot,clipval,np.inf) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
    # R2_toplot[R2_toplot < clipval] = np.nan #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)

    for isa,sourcearea in enumerate(sourcearealabelpairs):
        for ita,targetarea in enumerate(targetarealabelpairs):
            R2_ratiodata[isa,ita,iranks] = (R2_toplot[isa,ita,:,:]+noise_constant) / (R2_toplot[0,0,:,:]+noise_constant) #add a small constant to avoid division by zero
        
fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharex=True,sharey=False)
ax = axes
handles = []
labels = []

for iranks,rankstoaverage in enumerate(ranksplit):
    ax = axes[iranks]
    for isa,sourcearea in enumerate(sourcearealabelpairs):
        for ita,targetarea in enumerate(targetarealabelpairs):
            if isa == 1 and ita == 1: #only add legend for one of the subplots to avoid duplicates
                handles.append(shaded_error(params['t_axis'],R2_ratiodata[isa,ita,iranks],error='sem',
                                            color=clrs_arealabelpairs[isa,ita],alpha=0.3,ax=ax))#clrs[iapl],alpha=0.3,ax=ax))

                labels.append(arealabeled_to_figlabels(sourcearea) + ' - ' + arealabeled_to_figlabels(targetarea))

                # add_paired_ttest_results(ax,np.nanmean(R2_ratio[:,idx_iti],axis=1),np.nanmean(R2_ratio[:,idx_resp],axis=1),pos=[0.4,0.95])
                add_paired_ttest_results(ax,np.nanmean(R2_ratiodata[isa,ita,iranks,:,idx_iti],axis=0),
                                        np.nanmean(R2_ratiodata[isa,ita,iranks,:,idx_resp],axis=0),pos=[0.4,0.9],color=clrs[0])  
                # add_paired_ttest_results(ax,np.nanmean(R2_ratiodata[isa,ita,iranks,:,idx_iti],axis=0),
                                        # np.nanmean(R2_ratiodata[isa,ita,iranks,:,idx_resp],axis=0),pos=[0.4,0.8],color=clrs[1])

                thickness = ax.get_ylim()[1]/25
                ax.axhline(y=1,color='grey',linestyle='--')
                ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
                ax.legend(handles=handles,labels=figlabels,loc='best')
                my_legend_strip(ax)
                ax_nticks(ax,3)
                ax.set_xticks(t_ticks)
                ax.set_xticklabels(t_ticks)
                ax.set_xlabel('time (s)')
                ax.set_ylabel('performance ratio')
                # ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
                ax.set_ylim([ymin,my_ceil(np.max([1.5,ax.get_ylim()[1]]),2)])
                ax.set_xlim([-1,1.8])

                rect = Rectangle((twin_iti[0],ymin),twin_iti[1]-twin_iti[0],np.diff(ax.get_ylim())[0],fc='none',ec='black',alpha=0.5,lw=0.8,linestyle='--')
                ax.add_patch(rect)
                rect = Rectangle((twin_resp[0],ymin),twin_resp[1]-twin_resp[0],np.diff(ax.get_ylim())[0],fc='none',ec='blue',lw=0.8,linestyle='--')
                ax.add_patch(rect)

    if iranks == 0:  # Only add legend to the first subplot
        leg = ax.legend(handles,labels,frameon=False, reverse=True,bbox_to_anchor=(0.7,0.3))
        my_legend_strip(ax)
    ax_nticks(ax,3)
    ax.set_xticks(t_ticks)
    ax.set_xticklabels(t_ticks)
    ax.set_xlim([-1,1.8])
    ax.axhline(1, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('time (s)')
    # ax.set_ylabel('R$^{2}$ ratio')
    ax.set_ylabel('performance ratio')
    ax.set_title(ranksplitlabels[iranks])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,figdir,'RRR_looped_time_ratio_splitranks_%s' % (version))
