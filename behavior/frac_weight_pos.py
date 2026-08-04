# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
import numpy as np
from scipy.stats import zscore
import pickle

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.params import load_params

#%% Load parameters and settings:
params = load_params()
figdir = os.path.join(params['figdir'],'RRR','Labeling','Behavior')

#%% 
session_list        = np.array([
                                # ['LPE12223_2024_06_10'], #V1lab actually lower
                                ['LPE09830_2023_04_10'], #V1 labeled higher predictive than V1unl
                                ['LPE10919_2023_11_06'],  #V1lab actually lower
                                # ['LPE12223_2024_06_08'], #V1lab actually lower
                                ['LPE11998_2024_05_02'], # V1lab lower?
                                # ['LPE11622_2024_03_25'], #same
                                ['LPE09665_2023_03_14'], #V1lab higher
                                # ['LPE10885_2023_10_23'], #V1lab much higher
                                ['LPE11086_2024_01_05'], #Really much higher, best session, first dimensions are more predictive.
                                ['LPE11086_2024_01_10'], #Few v1 labeled cells, very noisy
                                # ['LPE11998_2024_05_10'], #
                                # ['LPE12013_2024_05_07'], #
                                ['LPE11495_2024_02_28'], #
                                # ['LPE11086_2023_12_15'], #Same
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,
                                       min_lab_cells_V1=20,filter_noiselevel=False)

#%% Get all data 
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_noiselevel=False)
report_sessions(sessions)

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False)

#%% 
params['direction'] = 'FF'
params['direction'] = 'FB'

if params['direction'] =='FF': 
    sourcearealabelpair = 'V1'
    targetarealabelpair = 'PM'
elif params['direction'] =='FB': 
    sourcearealabelpairs = 'PM'
    targetarealabelpair = 'V1'

#%% 
Nsub                = 100
nmodelfits          = 20

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]

frac_pos_weight_out = np.full((nSessions,params['nStim'],params['nranks'],nmodelfits),np.nan)
frac_pos_weight_in  = np.full((nSessions,params['nStim'],params['nranks'],nmodelfits),np.nan)

for ises,ses in enumerate(sessions):
    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    idx_areax      = np.where(np.all((ses.celldata['roi_name']==sourcearealabelpair,
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
    idx_areay       = np.where(np.all((ses.celldata['roi_name']==targetarealabelpair,
                                            ses.celldata['noise_level']<params['maxnoiselevel'],
                                            idx_nearby
                                            ),axis=0))[0]
    
    for imf in tqdm(range(nmodelfits),total=nmodelfits,desc='Fitting RRR model for session %d/%d' % (ises+1,nSessions)):
        idx_areax_sub       = np.random.choice(idx_areax,Nsub,replace=False)
        idx_areay_sub        = np.random.choice(idx_areay,Nsub,replace=False)

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
       
            X                   = sessions[ises].tensor[np.ix_(idx_areax_sub,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

            # reshape to neurons x time points
            X                  = X.reshape(len(idx_areax_sub),-1).T
            Y                   = Y.reshape(len(idx_areay_sub),-1).T

            X                  = zscore(X,axis=0) #zscore the activity per neuron
            Y                   = zscore(Y,axis=0)

            #RRR X to Y
            B_hat        = LM(Y,X, lam=params['lam'])
            Y_hat         = X @ B_hat

            # decomposing and low rank approximation of Y_hat
            U, s, V = svds(Y_hat,k=params['nranks'],which='LM')
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

            #Fraction of weights that is projecting positively onto firing rate:
            for r in range(params['nranks']): #for each rank
                #find correct sign of weight by sign of inner product mean firing rate and left singular vector
                frac_pos_weight_out[ises,istim,r,imf] = np.sum(np.sign(V[r,:])==np.sign(U[:,r].T @ np.nanmean(Y, axis=1))) / np.shape(V)[1]
            
            # Predictive source directions
            W = B_hat @ V.T  # (N x k)
            # Mean source firing rate across timepoints
            mu_X = X.mean(axis=1)
            for r in range(params['nranks']): #for each rank compute weights
                # Align sign to mean source firing
                sign = np.sign(np.dot(X @ W[:, r], mu_X))
                frac_pos_weight_in[ises,istim,r,imf] = np.sum(np.sign(W[:, r])==sign) / np.shape(W)[0]

#%% Plot the fraction of output weights (onto target area) that have a positive projection onto firing rate for each rank:
nranktstoplot = 20
handles = []
fig,axes = plt.subplots(1,1,figsize=(4*cm,3.7*cm))
ax = axes
data = frac_pos_weight_out
ymeantoplot = np.nanmean(data,axis=(0,1,3)) #mean across sessions and stim and modelfits
yerrortoplot = np.nanstd(data,axis=(0,1,3)) / np.sqrt(nSessions*nmodelfits)
handles.append(shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,
                            linewidth=1,color='green',alpha=0.3))

data = frac_pos_weight_in 
ymeantoplot = np.nanmean(data,axis=(0,1,3)) #mean across sessions and stim and modelfits
yerrortoplot = np.nanstd(data,axis=(0,1,3)) / np.sqrt(nSessions*nmodelfits)
handles.append(shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,
                            linewidth=1,color='blue',alpha=0.3))
ax.legend(handles,['source area','target area'],fontsize=6,frameon=False)
ax_nticks(ax,4)
ax.axhline(y=0.5,color='grey',linestyle='--')
ax.set_xticks(np.arange(params['nranks'])[::3]+1)
ax.set_xlim([0,nranktstoplot])
ax.set_xlabel('dimension')
ax.set_ylabel('Frac. pos. projection')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=2)
my_savefig(fig,figdir,'Frac_pos_weights_%s_%dsessions' % (params['direction'],nSessions))
