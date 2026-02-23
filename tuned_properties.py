# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy import stats
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.stats import ttest_ind

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.psth import compute_tensor
# from utils.explorefigs import plot_excerpt,plot_PCA_gratings,plot_tuned_response
from utils.tuning import *
from params import load_params

params = load_params()
figdir = os.path.join(params['figdir'],'ResponseProperties')
areas = ['V1','PM','AL']

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% 
session_list        = np.array([
                                # ['LPE09830_2023_04_10'], #
                                # ['LPE09665_2023_03_14'], #
                                # ['LPE11622_2024_03_26'], #GR with AL
                                ['LPE11998_2024_05_10'], #GR with AL
                                ['LPE11622_2024_03_25'], #GN with AL
                                ]) 

# sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_noiselevel=False,filter_areas=areas,
                                    #    only_session_id=session_list,
                                    #    )

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_noiselevel=False,filter_areas=areas,
                                       only_session_id=session_list,
                                       )


#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_noiselevel=True)
# sessions,nSessions   = filter_sessions(protocols = ['GR'],min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=areas,filter_areas=areas)

#%% 
report_sessions(sessions)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
for ises in range(nSessions):
    sessions[ises].load_data(load_calciumdata=True,calciumversion=params['calciumversion'])
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 t_pre=params['t_pre'], t_post=params['t_post'], method='nearby')
    sessions[ises].respmat = np.mean(sessions[ises].tensor[:,:,np.logical_and(t_axis>-0, t_axis<=1.5)] ,axis=2)

#%% ########################### Compute tuning metrics: ###################################
sessions = ori_remapping(sessions)
for ises in range(nSessions):
    if sessions[ises].sessiondata['protocol'][0] == 'GR':
        sessions[ises].trialdata['Direction'] = sessions[ises].trialdata['Orientation']
        sessions[ises].trialdata['Orientation'] = np.mod(sessions[ises].trialdata['Orientation'],180)
        junk,junk,oriconds  = np.unique(sessions[ises].trialdata['Orientation'],return_index=True,return_inverse=True)
        sessions[ises].trialdata['stimCond']    = oriconds

sessions = compute_tuning_wrapper(sessions)

#%% Show some tuned responses with calcium and deconvolved traces across orientations:
example_cells = [3,100,58,62,70]
fig = plot_tuned_response(sessions[0].tensor,sessions[0].trialdata,t_axis,example_cells)
fig.suptitle('%s - Deconvolved' % sessions[0].sessiondata['session_id'][0],fontsize=12)
# save the figure
# fig.savefig(os.path.join(savedir,'TunedResponse_deconv_%s.png' % sessions[0].sessiondata['session_id']))

#%% Construct matrix of trial-averaged responses for all cells and all orientations
celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
nCells      = len(celldata)
nStim       = len(np.unique(sessions[0].trialdata['stimCond']))
# stims       = np.unique(sessions[0].trialdata['Orientation'])
stims       = np.unique(sessions[0].trialdata['stimCond'])
nStim       = len(stims)
nStim = 8
stims = np.arange(nStim)
# nStim       = len(np.unique(sessions[0].trialdata['Orientation']))

tensor_avg  = np.full((nCells,nStim,len(t_axis)),np.nan)
tensor_std  = np.full((nCells,nStim,len(t_axis)),np.nan)
respmat_avg = np.full((nCells,nStim),np.nan)

for ises in range(nSessions):
    idx_ses = np.where(celldata['session_id']==sessions[ises].sessiondata['session_id'][0])[0]
    for istim in range(nStim):
        # idx_T               = sessions[ises].trialdata['Orientation']==stims[istim]
        idx_T               = sessions[ises].trialdata['stimCond']==stims[istim]
        tensor_avg[idx_ses,istim,:] = np.nanmean(sessions[ises].tensor[:,idx_T,:],axis=1)
        tensor_std[idx_ses,istim,:] = np.nanstd(sessions[ises].tensor[:,idx_T,:],axis=1)
        # respmat_avg[idx_ses,istim]  = np.nanmean(sessions[ises].respmat[:,idx_T],axis=1)

    # temp = np.mean(sessions[ises].tensor,axis=1)
    # tensor_avgall = np.concatenate((tensor_avgall,np.mean(sessions[ises].tensor,axis=1)))
    
    # temp = np.mean(sessions[ises].respmat,axis=1)
    # respmat_avgall  = np.concatenate((respmat_avgall,temp))

    # for iN in range(len(sessions[ises].celldata)):
    #     trialidx = sessions[ises].trialdata['Orientation'] == sessions[ises].celldata['pref_ori'][iN]
    #     temp = np.mean(sessions[ises].tensor[iN,trialidx,:],axis=0)
    #     tensor_avgpref = np.concatenate((tensor_avgpref,temp[np.newaxis,:]))

    #     temp = np.mean(sessions[ises].respmat[iN,trialidx],axis=0)
    #     respmat_avgpref  = np.concatenate((respmat_avgpref,[temp]))

#%% 
idx_sort = np.lexsort((-np.mod(celldata['pref_ori'],180),celldata['arealabel']))

arealabel_sorted = celldata['arealabel'][idx_sort]
prefori_sorted = celldata['pref_ori'][idx_sort]

arealabels  = np.unique(arealabel_sorted)
clrs_arealabels = get_clr_area_labeled(arealabels)

tensor_avg_sorted = tensor_avg[idx_sort,:,:]
tensor_std_sorted = tensor_std[idx_sort,:,:]
respmat_avg_sorted = respmat_avg[idx_sort,:]

t_len = t_axis[-1]-t_axis[0]
cmap = sns.color_palette('magma', as_cmap=True)
cmap = sns.color_palette('Reds', as_cmap=True)
cmap = sns.color_palette('Greens', as_cmap=True)

vmin = 0
vmax = 1
neuronoffset = 0.03*nCells

# tensor_sorted_norm = copy.copy(tensor_avg_sorted)
# tensor_sorted_norm -= np.nanmin(tensor_sorted_norm,axis=(1,2),keepdims=True)
# tensor_sorted_norm /= np.nanmax(tensor_sorted_norm,axis=(1,2),keepdims=True)

# tensor_sorted_norm = copy.copy(tensor_avg_sorted)
# tensor_sorted_norm -= np.nanpercentile(tensor_sorted_norm,5,axis=(1,2),keepdims=True)
# tensor_sorted_norm /= np.nanpercentile(tensor_sorted_norm,99,axis=(1,2),keepdims=True)

tensor_sorted_norm = copy.copy(tensor_avg_sorted)
tensor_sorted_norm -= np.nanmean(tensor_sorted_norm,axis=(1,2),keepdims=True)
tensor_sorted_norm /= np.nanstd(tensor_sorted_norm,axis=(1,2),keepdims=True)

tensor_std_sorted_norm = copy.copy(tensor_std_sorted)
tensor_std_sorted_norm -= np.nanmean(tensor_std_sorted_norm,axis=(1,2),keepdims=True)

vmin = -1
vmax = 1

fig,axes = plt.subplots(1,1,figsize=(6*cm,6*cm),sharey=True)
ax = axes
for ial,al in enumerate(arealabels):
    # ax = axes[iarea]
    idx_N = np.where(arealabel_sorted==al)[0]
    print(al,len(idx_N))
    for istim in range(nStim):
        # ax.pcolor(t_axis+istim*t_len,idx_N,np.nanmean(tensor_sorted_norm[idx_N,istim,:],axis=0),vmin=vmin,vmax=vmax,cmap=cmap)
        ax.pcolor(t_axis+istim*(t_len+1),idx_N+neuronoffset*ial,tensor_sorted_norm[idx_N,istim,:].squeeze(),vmin=vmin,vmax=vmax,cmap=cmap)

        ax.axvline(x=0 + istim*(t_len+1), color='black', linestyle='--', linewidth=0.5)
    ax.text(-0.1,np.mean(idx_N) / nCells + neuronoffset/nCells*ial,al,color=clrs_arealabels[ial],ha='center',va='center',transform=ax.transAxes,fontsize=8,rotation=90)

ax.set_ylabel('Neurons')
ax.set_xlabel('Time (s)')
ax.set_xticks(np.arange(nStim)*(t_len+1),stims)

plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=0)
# my_savefig(fig,figdir,'Tensor_av_heatmap_GR_%dneurons' % (nCells))

#%%

fig,axes = plt.subplots(1,1,figsize=(5.5*cm,5.5*cm),sharey=True)
ax = axes
handles = []

for ial,al in enumerate(arealabels):
    # ax = axes[iarea]
    idx_N = np.where(arealabel_sorted==al)[0]

    meantoplot = np.nanmean(tensor_sorted_norm[idx_N,:,:],axis=(0,1))
    # meantoplot = np.nanmean(tensor_avg[idx_N,:,:],axis=(0,1))
    errortoplot = np.nanmean(tensor_std_sorted_norm[idx_N,:,:],axis=(0,1))

    handles.append(shaded_error(x=t_axis,y=meantoplot,yerror=errortoplot,color=clrs_arealabels[ial],ax=ax))

ax.set_ylabel('Neurons')
ax.set_xlabel('Time (s)')

ax.legend(handles=handles,labels=list(arealabels),frameon=False)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
# my_savefig(fig,figdir,'Tensor_avg_timetrace_GR_%dneurons' % (nCells))
# zip(handles, arealabels)
# handles, labels = zip(*zip(handles, arealabels))

#%% ##################### Noise level for labeled vs unlabeled cells:

# ## plot precentage of labeled cells as a function of depth:
# # sns.barplot(x='depth', y='redcell', data=celldata[celldata['roi_name'].isin(['V1','PM'])], estimator=lambda y: sum(y==1)*100.0/len(y))
# sns.lineplot(data=celldata[celldata['roi_name'].isin(['V1','PM'])],x='depth', y='redcell', estimator=lambda y: sum(y==1)*100.0/len(y))
# # sns.lineplot(data=celldata,x='depth', y='redcell', hue='roi_name',estimator=lambda y: sum(y==1)*100.0/len(y),palette='Accent')
# plt.ylabel('% labeled cells')

# #Plot fraction of labeled cells across areas of recordings: 
# sns.barplot(x='roi_name', y='redcell', data=celldata, estimator=lambda x: sum(x==1)*100.0/len(x),palette='Accent')
# plt.ylabel('% labeled cells')

# ## plot number of cells per plane across depths:
# sns.histplot(data=celldata, x='depth',hue='roi_name',palette='Accent')

# ## plot quality of cells per plane across depths with skew:
# # sns.lineplot(data=celldata, x="depth",y=celldata['skew'],estimator='mean')
# sns.lineplot(x=np.round(celldata["depth"],-1),y=celldata['skew'],estimator='mean')

# ## plot quality of cells per plane across depths with noise level:
# # sns.lineplot(data=celldata, x="depth",y=celldata['noise_level'],estimator='mean')
# sns.lineplot(x=np.round(celldata["depth"],-1),y=celldata['noise_level'],estimator='mean')
# plt.ylim([0,0.3])
