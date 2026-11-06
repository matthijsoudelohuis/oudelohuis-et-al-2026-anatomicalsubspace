# -*- coding: utf-8 -*-
"""
This script analyzes the quality of the recordings and their relation 
to various factors such as depth of recording, being labeled etc.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% Import packages
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import zscore,wilcoxon,ttest_rel
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from skimage.measure import block_reduce
from sklearn import preprocessing

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.psth import compute_tensor,compute_respmat
from utils.plot_lib import * #get all the fixed color schemes
from utils.regress_lib import *
from utils.tuning import compute_tuning_wrapper
from utils.params import load_params
from utils.RRRlib import *

params = load_params()
figdir = os.path.join(params['figdir'],'Labeling')

#%% 
def get_rand_trials(ses, ntrials=80):
    trialsel = [np.random.randint(low=5, high=len(ses.trialdata)-100)]
    trialsel.append(trialsel[0]+ntrials)
    return trialsel

def plot_excerpt(ses, trialsel=None, neuronsel=None, plot_neural=True, plot_behavioral=True, neural_version='traces'):
    if trialsel is None:
        trialsel = get_rand_trials(ses)
    if ses.sessiondata['protocol'][0] in ['GR','GN','IM']:
        tstart  = ses.trialdata['tOffset'][trialsel[0]-1]
        tstop   = ses.trialdata['tOnset'][trialsel[1]-1]
    elif ses.sessiondata['protocol'][0] == 'DN':
        tstart  = ses.trialdata['tStart'][trialsel[0]-1]
        tstop   = ses.trialdata['tEnd'][trialsel[1]-1]
    elif ses.sessiondata['protocol'][0] == 'SP':
        tstart  = ses.sessiondata['tStart'][0] + 30
        tstop   = ses.sessiondata['tEnd'][0]
        
    fig, ax = plt.subplots(figsize=[9*cm, 12*cm])
    counter = 0
    if plot_neural:
        if neural_version == 'traces':
            counter = plot_neural_traces(
                ses, ax, tstart,tstop, neuronsel=neuronsel, counter=counter)
        elif neural_version == 'raster':
            counter = plot_neural_raster(
                ses, ax, tstart,tstop, neuronsel=neuronsel, counter=counter)
        counter -= 1

    if plot_behavioral:
        counter = plot_behavioral_traces(
            ses, ax, tstart,tstop, counter=counter)

    plot_stimuli(ses, trialsel, ax)

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])

    ax.set_xlim([tstart-10, tstop])
    ax.set_ylim([counter-1, 1])

    ax.add_artist(AnchoredSizeBar(ax.transData, 10,
                  "10 Sec", loc=4, frameon=False))
    ax.axis('off')

    fig.tight_layout()

    return fig

def plot_norm_trace(x, y, offset=0, clr='k'):
    min_max_scaler = preprocessing.MinMaxScaler()
    y = np.array(y)
    y = min_max_scaler.fit_transform(y[:, np.newaxis])
    handle = plt.plot(x, y + offset, linewidth=0.5, color=clr)[0]
    return handle


def plot_stimuli(ses, trialsel, ax):

    # Add stimuli:
    if ses.protocol == 'GR':
        oris = np.unique(ses.trialdata['Orientation'])
        rgba_color = plt.get_cmap('hsv', lut=16)(np.linspace(0, 1, len(oris)))

        for i in np.arange(trialsel[0], trialsel[1]):
            ax.add_patch(plt.Rectangle([ses.trialdata['tOnset'][i], -1000], 1, 2000, alpha=0.1, linewidth=0,
                                       facecolor=rgba_color[np.where(oris == ses.trialdata['Orientation'][i])]))

        handles = []
        for i, ori in enumerate(oris):
            handles.append(ax.add_patch(plt.Rectangle(
                [0, 0], 1, 1000, alpha=0.3, linewidth=0, facecolor=rgba_color[i])))
        ax.legend(handles, oris, loc='center right',
                  bbox_to_anchor=(1.25, 0.5))
    elif ses.protocol == 'GN':
        oris = np.sort(np.unique(ses.trialdata['centerOrientation']))
        speeds = np.sort(np.unique(ses.trialdata['centerSpeed']))
        clrs, oo = get_clr_gratingnoise_stimuli(oris, speeds)

        for i in np.arange(trialsel[0], trialsel[1]):
            iO = np.where(oris == ses.trialdata['centerOrientation'][i])
            iS = np.where(speeds == ses.trialdata['centerSpeed'][i])
            ax.add_patch(plt.Rectangle([ses.trialdata['tOnset'][i], -1000], 1, 1000, alpha=0.3, linewidth=0,
                                       facecolor=clrs[iO, iS, :].flatten()))

        for iO, ori in enumerate(oris):
            for iS, speed in enumerate(speeds):
                ax.add_patch(plt.Rectangle(
                    [0, 0], 1, 3, alpha=0.3, linewidth=0, facecolor=clrs[iO, iS, :].flatten()))

    elif ses.protocol == 'IM':
        # rgba_color  = plt.get_cmap('prism',lut=np.diff(trialsel)[0])(np.linspace(0, 1, np.diff(trialsel)[0]))
        rgba_color = sns.color_palette("Set2", np.diff(trialsel)[0])

        for i, k in enumerate(np.arange(trialsel[0], trialsel[1])):
            ax.add_patch(plt.Rectangle([ses.trialdata['tOnset'][k], -1000],
                                       ses.trialdata['tOffset'][k] -
                                       ses.trialdata['tOnset'][k],
                                       2000, alpha=0.3, linewidth=0,
                                       # facecolor=rgba_color[i,:]))
                                       facecolor=rgba_color[i]))
            
    elif ses.protocol == 'DN':
        stimcats = np.unique(ses.trialdata['stimcat'])
        rgba_color = sns.color_palette("Set2", len(stimcats))

        for i, k in enumerate(np.arange(trialsel[0], trialsel[1])):
            ax.add_patch(plt.Rectangle([ses.trialdata['tStimStart'][k], -1000],
                                       ses.trialdata['tStimEnd'][k] -
                                       ses.trialdata['tStimStart'][k],
                                       2000, alpha=0.3, linewidth=0,
                                       # facecolor=rgba_color[i,:]))
                                       facecolor=rgba_color[np.where(stimcats == ses.trialdata['stimcat'][i])[0][0]]))

    return

def plot_behavioral_traces(ses, ax, tstart,tstop, nvideoPCs=8, counter=0):
    
    ts_V = ses.videodata['ts']
    idx_V = np.logical_and(ts_V > tstart, ts_V < tstop)
    handles = []
    labels = []

    clrs = sns.color_palette('husl', 4)
    clrs = sns.color_palette("crest", nvideoPCs)

    for iPC in range(nvideoPCs):
        motionenergy = ses.videodata['videoPC_%d' % iPC][idx_V]
        handles.append(plot_norm_trace(
            ts_V[idx_V], motionenergy, counter, clr=clrs[iPC]))
        # labels.append('videoPC%d' %iPC)
        counter -= 1

    ax.text(tstart, counter+nvideoPCs/2, 'video PCs',
            fontsize=9, color='black', horizontalalignment='right')

    pupil_area = ses.videodata['pupil_area'][idx_V]
    handles.append(plot_norm_trace(
        ts_V[idx_V], pupil_area, counter, clr='purple'))
    # labels.append('Pupil Size')
    ax.text(tstart, counter, 'Pupil Size', fontsize=9,
            color='black', horizontalalignment='right')
    counter -= 1

    pupil_area = ses.videodata['pupil_xpos'][idx_V]
    handles.append(plot_norm_trace(
        ts_V[idx_V], pupil_area, counter, clr='indigo'))
    # labels.append('Pupil X-pos')
    ax.text(tstart, counter, 'Pupil X-pos', fontsize=9,
            color='black', horizontalalignment='right')
    counter -= 1

    pupil_area = ses.videodata['pupil_ypos'][idx_V]
    handles.append(plot_norm_trace(
        ts_V[idx_V], pupil_area, counter, clr='plum'))
    # labels.append('Pupil Y-pos')
    ax.text(tstart, counter, 'Pupil Y-pos', fontsize=9,
            color='black', horizontalalignment='right')
    counter -= 1

    ts_B = ses.behaviordata['ts']
    idx_B = np.logical_and(ts_B > tstart, ts_B < tstop)

    runspeed = ses.behaviordata['runspeed'][idx_B]

    handles.append(plot_norm_trace(
        ts_B[idx_B], runspeed, counter, clr='saddlebrown'))
    # labels.append('Running Speed')
    ax.text(tstart, counter, 'Running Speed', fontsize=9,
            color='black', horizontalalignment='right')
    counter -= 1

    return counter

def plot_neural_traces(ses, ax, tstart, tstop , neuronsel=None, counter=0, nexcells=8):

    scaleddata      = np.array(ses.calciumdata)
    min_max_scaler  = preprocessing.MinMaxScaler()
    scaleddata      = min_max_scaler.fit_transform(scaleddata)
    scaleddata      = scaleddata[np.logical_and(
        ses.ts_F > tstart, ses.ts_F < tstop)]
    scaleddata      = min_max_scaler.fit_transform(scaleddata)

    areas           = np.unique(ses.celldata['roi_name'])
    labeled         = np.unique(ses.celldata['redcell'])
    labeltext       = ['unlabeled', 'labeled',]

    if neuronsel is None:
        example_cells   = np.array([])

        for iarea, area in enumerate(areas):
            for ilabel, label in enumerate(labeled):
                idx             = np.where(np.logical_and(ses.celldata['roi_name'] == area, 
                                            ses.celldata['redcell'] == label))[0]
                temp_excells    = np.min((len(idx), nexcells))
                excerpt_var     = np.var(scaleddata, axis=0)
                example_cells   = np.append(example_cells, idx[np.argsort(-excerpt_var[idx])[:temp_excells]])
    else:
        example_cells = np.array(neuronsel)

    clrs = get_clr_labeled()
    for iarea, area in enumerate(areas):
        for ilabel, label in enumerate(labeled):
            example_cells_area_label = example_cells[np.logical_and(ses.celldata['roi_name'][example_cells] == area,
                                                                    ses.celldata['redcell'][example_cells] == label)]

            excerpt = scaleddata[:, example_cells_area_label.astype(int)]

            ncells = np.shape(excerpt)[1]

            for i in range(ncells):
                counter -= 1
                ax.plot(ses.ts_F[np.logical_and(ses.ts_F > tstart, ses.ts_F < tstop)],
                        excerpt[:, i]+counter, linewidth=0.5, color=clrs[ilabel])

            ax.text(tstart, counter+ncells/2, area + ' - ' +
                    labeltext[ilabel], fontsize=9, color='black', horizontalalignment='right')

    return counter

#%% Load one session:
# session_list        = np.array([['LPE10919_2023_11_06']])
session_list        = np.array([['LPE09830_2023_04_10']])
# session_list        = np.array([['LPE11086_2024_01_10']])
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,filter_noiselevel=True)

for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=params['calciumversion'])

for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=params['calciumversion'],keepraw=True)

#%% 
example_cells   = [
                    #V1 - unlabeled
                    'LPE09830_2023_04_10_1_0120',
                   'LPE09830_2023_04_10_2_0121',
                    'LPE09830_2023_04_10_0_0080',
                    'LPE09830_2023_04_10_0_0081',
                    'LPE09830_2023_04_10_0_0194',
                    #V1 - labeled
                   'LPE09830_2023_04_10_2_0006',
                   'LPE09830_2023_04_10_3_0036',
                   'LPE09830_2023_04_10_2_0020',
                   'LPE09830_2023_04_10_2_0103',
                   'LPE09830_2023_04_10_0_0162',
                    #PM - unlabeled
                    'LPE09830_2023_04_10_5_0214',
                    'LPE09830_2023_04_10_5_0771',
                    'LPE09830_2023_04_10_6_0126',
                    'LPE09830_2023_04_10_7_0040',
                    'LPE09830_2023_04_10_4_0034',
                    #PM - labeled
                    'LPE09830_2023_04_10_6_0083',
                    'LPE09830_2023_04_10_5_0075',
                    'LPE09830_2023_04_10_5_0141',
                    'LPE09830_2023_04_10_5_0124',
                    'LPE09830_2023_04_10_5_0239'
                   ]
trialsel = [375,450]

example_cells = np.array([np.where(sessions[0].celldata['cell_id'] == cell)[0][0] for cell in example_cells])
fig = plot_excerpt(sessions[0],trialsel=trialsel,plot_neural=True,plot_behavioral=False,
                   neuronsel=example_cells,neural_version='traces')

my_savefig(fig,figdir,'ExTraces_%s' % sessions[0].session_id)



#%% code to find representative cells for each area and label combination:

# #%% ########################### Compute tuning metrics: ###################################
# sessions = compute_tuning_wrapper(sessions)

# #%% #####################################
# #Show some traces and some stimuli to see responses:
# trialsel = [375,450]

# #Get 10 neurons from V1 unlabeled with lowest noise level:
# idx = np.where(np.all((
#                         sessions[0].celldata['roi_name'] == 'V1',
#                         # sessions[0].celldata['roi_name'] == 'PM',
#                         sessions[0].celldata['redcell'] == 0,
#                         # sessions[0].celldata['redcell'] == 1,
#                         sessions[0].celldata['tuning_var'] > np.percentile(sessions[0].celldata['tuning_var'], 75)
#                         ),axis=0))[0]

# example_cells = np.random.choice(idx, size=10, replace=False)

# #Get 10 neurons from V1 unlabeled with lowest noise level:
# # idx = np.where(np.logical_and(sessions[0].celldata['roi_name'] == 'V1',
#                             #   sessions[0].celldata['redcell'] == 1))[0]

# # example_cells = np.random.choice(idx[np.argsort(sessions[0].celldata['noise_level'][idx])[:20]], size=10, replace=False)
# # example_cells = np.random.choice(idx[np.argsort(sessions[0].celldata['event_rate'][idx])[:20]], size=10, replace=False)
# # example_cells = np.random.choice(idx[np.argsort(sessions[0].celldata['tuning_var'][idx])[:20]], size=10, replace=False)

# # example_cells   = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110]
# print('Example cells: %s' % sessions[0].celldata['cell_id'][example_cells])
# fig = plot_excerpt(sessions[0],trialsel=[100,150],plot_neural=True,plot_behavioral=False,
#                    neuronsel=example_cells,neural_version='traces')