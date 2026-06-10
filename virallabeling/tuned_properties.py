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

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.psth import compute_tensor
from utils.tuning import *
from utils.params import load_params

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
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_noiselevel=False,filter_areas=areas,
                                       only_session_id=session_list,
                                       )

# #%% Get all data 
# sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_noiselevel=True)
sessions,nSessions   = filter_sessions(protocols = ['GR'],min_lab_cells_V1=40,min_lab_cells_PM=40,filter_noiselevel=False)
# sessions,nSessions   = filter_sessions(protocols = ['GR'],only_all_areas=areas,filter_areas=areas)

#%% 
report_sessions(sessions)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
for ises in range(nSessions):
    sessions[ises].load_data(load_calciumdata=True,calciumversion=params['calciumversion'])
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 t_pre=params['t_pre'], t_post=params['t_post'], method='nearby')
    sessions[ises].respmat = np.mean(sessions[ises].tensor[:,:,np.logical_and(t_axis>0, t_axis<=1.5)] ,axis=2)

#%% ########################### Compute tuning metrics: ###################################
sessions = ori_remapping(sessions)
sessions = compute_tuning_wrapper(sessions)

#%% Function that takes in the tensor and computes the average response for some example neurons:
def plot_tuned_response(calciumdata, trialdata, t_axis, example_cells,plot_n_trials=0,celllabels=None):
    """
    The plot_tuned_response function is used to visualize the average response of specific neurons to different orientations. It takes in four inputs:
    calciumdata: a 3D tensor containing calcium imaging data for multiple cells, trials, and timepoints.
    trialdata: a pandas DataFrame containing trial information, including orientation.
    t_axis: a 1D array representing the time axis.
    example_cells: a list of cell indices to plot.
    plot_n_trials: the number of trials to plot as individual traces.
    celllabels: a list of labels for each cell.
    returns: a figure with subplots for each cell and each orientation.
    """
    if calciumdata.ndim != 3:
        raise ValueError("calciumdata must have shape (n_cells, n_trials, n_timepoints)")

    if t_axis.ndim != 1:
        raise ValueError("t_axis must be a 1D array")

    T = len(t_axis)
    oris = np.sort(pd.Series.unique(trialdata['Orientation']))
    colors = plt.cm.tab20(np.linspace(0, 1, len(oris)))  # assume this is the color palette used in plot_PCA_gratings
    pal = sns.color_palette('husl', len(oris))
    pal = np.tile(sns.color_palette('husl', int(len(oris)/2)),(2,1))

    fig, axs = plt.subplots(len(example_cells), len(oris), figsize=[8*cm, len(example_cells)*0.7*cm], sharex=True, sharey=False)
    axs = axs.flatten()
    for i, cell in enumerate(example_cells):
        row_max = 0
        for j, ori in enumerate(oris):
            resp_meanori = np.nanmean(calciumdata[np.ix_([cell], trialdata['Orientation'] == ori,range(T))], axis=1).squeeze()

            axs[i * len(oris) + j].plot(t_axis, resp_meanori,color=pal[j],linewidth=0.8)
            # axs[i * len(oris) + j].set_title(f'Cell {cell}, {ori} deg')
            axs[i * len(oris) + j].set_xticks([])
            axs[i * len(oris) + j].set_yticks([])
            # REMOVE axis borders
            axs[i * len(oris) + j].axis('off')
            row_max = max(row_max, np.max(resp_meanori))
        
            if plot_n_trials > 0:
                # for trial in range(plot_n_trials):
                trialsel = np.random.choice(np.where(trialdata['Orientation'] == ori)[0],plot_n_trials)
                tracedata = calciumdata[np.ix_([cell], trialsel,range(T))].squeeze().T
                axs[i * len(oris) + j].plot(t_axis, tracedata, color='k', linewidth=0.1)
                row_max = max(row_max, np.max(tracedata))
            if j == 0 and celllabels is not None:
                axs[i * len(oris) + j].text(0.5, 0.5, celllabels[i], transform=axs[i * len(oris) + j].transAxes, ha='center', va='center', fontsize=6,rotation=90)
        for j in range(len(oris)):
            axs[i * len(oris) + j].set_ylim(top=row_max * 1.1)  # add 10% padding
            # Add vertical dotted line at t=0
            axs[i * len(oris) + j].axvline(x=0, ymin=0,ymax=0.5,color='k', linestyle=':', linewidth=0.5)
    fig.subplots_adjust(hspace=0)

    return fig

#%% 
ises = 2
perc = 70
arealabels = np.array(['V1unl','V1lab','PMunl','PMlab'])
example_cells = []
for iarealabel,arealabel in enumerate(arealabels):
    idx_N = np.where(np.all((sessions[ises].celldata['arealabel']==arealabel,
                             sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                             sessions[ises].celldata['gOSI']>np.nanpercentile(sessions[ises].celldata['gOSI'][sessions[ises].celldata['arealabel']==arealabel], perc)
                             ),axis=0))[0]
    example_cells.append(np.random.choice(idx_N,size=1,replace=False)[0])

#%% 
ises = 0
example_cells = [553,77,1468,1531]

#%% Show some tuned responses with calcium and deconvolved traces across orientations:
np.random.seed(0)
fig = plot_tuned_response(sessions[ises].tensor,sessions[ises].trialdata,t_axis,example_cells,plot_n_trials=15,celllabels=arealabels)
# fig.suptitle('%s' % sessions[0].sessiondata['session_id'][0],fontsize=7)
my_savefig(fig,figdir,'TunedResponses_%s' % (sessions[ises].session_id))

#%% Construct matrix of trial-averaged responses for all cells and all orientations
celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
nCells      = len(celldata)
nStim       = len(np.unique(sessions[0].trialdata['stimCond']))
# stims       = np.unique(sessions[0].trialdata['Orientation'])
stims       = np.unique(sessions[0].trialdata['stimCond'])
nStim       = len(stims)
# nStim       = 8
stims       = np.arange(nStim)
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
