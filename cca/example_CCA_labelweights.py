# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from sklearn.cross_decomposition import CCA

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.rf_lib import filter_nearlabeled
from utils.regress_lib import *
from utils.CCAlib import *
from utils.RRRlib import *
from utils.pair_lib import value_matching
from utils.params import load_params

params = load_params()
figdir = os.path.join(params['figdir'],'CCA')

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% 
areas       = ['V1','PM']
nareas      = len(areas)

#%% Load example sessions:
session_list        = np.array([['LPE09665_2023_03_14'], #V1lab higher
                                ['LPE10885_2023_10_23'], #V1lab much higher
                                ])
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,only_all_areas=areas,filter_areas=areas)

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False)

#%%
n_components        = 20
idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
arealabels          = np.array(['V1unl', 'V1lab', 'PMunl', 'PMlab'])

#%% Example session and validation of CCA: 
model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)
ises        = 9
istim       = 2
ses         = sessions[ises]
np.random.seed(0) #for reproducibility of random sampling of neurons

if params['filter_nearby']:
    idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
else:
    idx_nearby = np.ones(len(ses.celldata),dtype=bool)


idx_N_all = np.empty(len(arealabels),dtype=object)
for ial, al in enumerate(arealabels):
    idx_N_all[ial]           = np.where(np.all((ses.celldata['arealabel']==al,
                            ses.celldata['noise_level']<params['maxnoiselevel'],
                            idx_nearby),axis=0))[0]

nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
                                        ses.celldata['noise_level']<params['maxnoiselevel'],
                                        idx_nearby),axis=0)) for i in arealabels])

idx_areax           = np.concatenate((np.random.choice(idx_N_all[0],nsampleneurons,replace=False),
                                    np.random.choice(idx_N_all[1],nsampleneurons,replace=False)))
idx_areay           = np.concatenate((np.random.choice(idx_N_all[2],nsampleneurons,replace=False),
                                    np.random.choice(idx_N_all[3],nsampleneurons,replace=False)))

idx_T               = ses.trialdata['stimCond']==istim

#on residual tensor during the response:
X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]

X                   = X.reshape(len(idx_areax),-1).T
Y                   = Y.reshape(len(idx_areay),-1).T

X                   = zscore(X,axis=0,nan_policy='omit')  #Z score activity for each neuron
Y                   = zscore(Y,axis=0,nan_policy='omit')

# Fit CCA MODEL:
model_CCA.fit(X,Y)

#%% Show the loadings for the first N dimensions:
ncompstoshow = 5
baralpha = 0.5
fig,axes = plt.subplots(ncompstoshow,2,figsize=(6*cm,ncompstoshow*2*cm),sharex=True,sharey=False)
for icomp in range(ncompstoshow):
    ax = axes[icomp,0]
    # sortidx = np.argsort(-model_CCA.x_loadings_[:nsampleneurons,0])
    # ax.stairs(model_CCA.x_loadings_[:nsampleneurons,icomp][sortidx], np.arange(nsampleneurons+1), 
    ax.stairs(np.sort(-model_CCA.x_loadings_[:nsampleneurons,icomp][sortidx]), np.arange(nsampleneurons+1), 
              fill=True, color='grey', edgecolor='black',alpha=baralpha)
    ax.stairs(np.sort(-model_CCA.x_loadings_[nsampleneurons:,icomp]), np.arange(nsampleneurons+1), 
              fill=True, color='red', edgecolor='red',alpha=baralpha)
    if icomp == 0: 
        ax.set_title(f'V1 loadings')
    ax.set_axis_off()
    ax = axes[icomp,1]
    ax.stairs(np.sort(-model_CCA.y_loadings_[:nsampleneurons,icomp]), np.arange(nsampleneurons+1), 
              fill=True, color='grey', edgecolor='black',alpha=baralpha)
    ax.stairs(np.sort(-model_CCA.y_loadings_[nsampleneurons:,icomp]), np.arange(nsampleneurons+1), 
              fill=True, color='red', edgecolor='red',alpha=baralpha)
    if icomp == 0: 
        ax.set_title(f'PM loadings')
    # ax.bar(np.arange(nsampleneurons),np.sort(-model_CCA.x_loadings_[:nsampleneurons,icomp]),color='grey',label='unlabeled')
    # ax.bar(np.arange(nsampleneurons),np.sort(-model_CCA.x_loadings_[nsampleneurons:,icomp]),color='red',label='labeled')
    # ax = axes[icomp,1]
    # ax.bar(np.arange(nsampleneurons),np.sort(-model_CCA.y_loadings_[:nsampleneurons,icomp]),color='grey',label='unlabeled')
    # ax.bar(np.arange(nsampleneurons),np.sort(-model_CCA.y_loadings_[nsampleneurons:,icomp]),color='red',label='labeled')
    if icomp==0:
        ax.legend(frameon=False,loc='upper right')
    if icomp==ncompstoshow-1:
        ax.set_xlabel('Neuron index (sorted)')
    ax.set_axis_off()
plt.tight_layout()
sns.despine(top=True,right=True,offset=2,trim=False)
# my_savefig(fig,figdir,'CCA_V1PM_labeled_exampleloadings_%s' % ses.session_id)
