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
from scipy.stats import zscore
from scipy import stats
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.RRRlib import *
from utils.regress_lib import *
from utils.psth import compute_tensor
from params import load_params

params = load_params()

params['regress_behavout'] = False
params['maxnoiselevel'] = 20
params['radius'] = 50

#%% 
figdir = os.path.join(params['figdir'],'RRR','Labeling')

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% 
session_list        = np.array([
                                # ['LPE12223_2024_06_10'], #V1lab actually lower
                                ['LPE09830_2023_04_10'], #V1 labeled higher predictive than V1unl
                                # ['LPE10919_2023_11_06'],  #V1lab actually lower
                                # ['LPE12223_2024_06_08'], #V1lab actually lower
                                # ['LPE11998_2024_05_02'], # V1lab lower?
                                # ['LPE11622_2024_03_25'], #same
                                # ['LPE09665_2023_03_14'], #V1lab higher
                                ['LPE10885_2023_10_23'], #V1lab much higher
                                # ['LPE11086_2024_01_05'], #Really much higher, best session, first dimensions are more predictive.
                                # ['LPE11086_2024_01_10'], #Few v1 labeled cells, very noisy
                                # ['LPE11998_2024_05_10'], #
                                # ['LPE12013_2024_05_07'], #
                                # ['LPE11495_2024_02_28'], #
                                # ['LPE11086_2023_12_15'], #Same
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,
                                       min_lab_cells_V1=20,filter_noiselevel=False)


#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=params['regress_behavout'])

#%% 
params['direction'] = 'FF'
sourcearealabelpairs = ['V1unl','V1lab']
targetarealabelpair = 'PMunl'
clrs_arealabelpairs = np.array(['grey','red'])

ises                = 0
stim                = 3
starttimepoint_idx  = 100
ntimebins           = 250
Nsub                = 150 #number of cells to sample

#%% 
params['direction'] = 'FB'
sourcearealabelpairs = ['PMunl','PMlab']
targetarealabelpair = 'V1unl'
clrs_arealabelpairs = np.array(['grey','red'])

ises                = 0
stim                = 5
starttimepoint_idx  = 1000
ntimebins           = 250
Nsub                = 50 #number of cells to sample

#%% Show example shared latents: 
idx_resp            = np.where((t_axis>=0) & (t_axis<=2))[0]

rank                = 3 #rank of RRR ranks to plot
nranks              = 20 
kernel_size         = 3 #size of smoothing kernel in frames

ses = sessions[ises]

idx_areax1          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[0],
                            ses.celldata['noise_level']<params['maxnoiselevel'],	
                            ),axis=0))[0]
idx_areax2          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[1],
                            ses.celldata['noise_level']<params['maxnoiselevel'],	
                            ),axis=0))[0]
idx_areay       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpair,
                                        ses.celldata['noise_level']<params['maxnoiselevel'],	
                                        ),axis=0))[0]
print(len(idx_areax2))

np.random.seed(0)

idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
idx_areax2_sub       = np.random.choice(idx_areax2,Nsub,replace=False)
idx_areay_sub        = np.random.choice(idx_areay,Nsub*2,replace=False)

idx_T               = ses.trialdata['stimCond']==stim

X1                  = ses.tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
X2                  = ses.tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
Y                   = ses.tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

# reshape to neurons x time points
X1                  = X1.reshape(len(idx_areax1_sub),-1).T
X2                  = X2.reshape(len(idx_areax2_sub),-1).T
Y                   = Y.reshape(len(idx_areay_sub),-1).T

X1  = zscore(X1,axis=0)
X2  = zscore(X2,axis=0)
Y   = zscore(Y,axis=0)

X = np.concatenate((X1,X2),axis=1)

#RRR X to Y
B_hat         = LM(Y,X, lam=params['lam'])
Y_hat         = X @ B_hat

# decomposing and low rank approximation of A
U, s, V = svds(Y_hat,k=nranks,which='LM')
U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

B_rrr           = B_hat @ V[:rank,:].T @ V[:rank,:] #project beta coeff into low rank subspace
Y_hat_test_rr   = X @ B_rrr

# How much of the variance in the source area is aligned with the predictive subspace:
Ub, sb, Vb = svds(B_rrr,k=rank,which='LM')
# Ub, sb, Vb = svds(B_hat,k=rank,which='LM')
# print(EV(X_test, X_test @ Ub @ Ub.T))

X_1 = copy.deepcopy(X)
X_1[:,Nsub:] = 0
# plt.imshow(X_1,vmin=-3,vmax=3,aspect='auto')
Z_1 = X_1 @ Ub 
# Y_hat_test_rr   = X_1 @ B_rrr

X_2 = copy.deepcopy(X)
X_2[:,:Nsub] = 0
# plt.imshow(X_2,vmin=-3,vmax=3,aspect='auto')
# Y_hat_test_rr   = X_2 @ B_rrr
Z_2 = X_2 @ Ub

#Little bit of smoothing:
kernel = np.ones(kernel_size) / kernel_size
for r in range(rank):
    Z_1[:,r] = np.convolve(Z_1[:,r], kernel, mode='same')
    Z_2[:,r] = np.convolve(Z_2[:,r], kernel, mode='same')

#Plot excerpt of latent dimensions over time
clrs = sns.color_palette('viridis',rank)
# fig,axes = plt.subplots(rank,1,figsize=(7*cm,rank*2*cm),sharex=True)

fig,axes = plt.subplots(rank,1,figsize=(7*cm,rank*2*cm),sharex=True,sharey=True)

idx_K = np.arange(starttimepoint_idx,starttimepoint_idx+ntimebins)
for r in range(rank):
    ax = axes[r]
    ax.plot(Z_1[idx_K,r],color=clrs_arealabelpairs[0],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[0]]))
    ax.plot(Z_2[idx_K,r],color=clrs_arealabelpairs[1],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[1]]))
    ax.set_ylabel('Latent dim %d' % (r+1))
    if r==0:
        ax.legend(frameon=False,loc='lower left',fontsize=8)
        my_legend_strip(ax)
    ax.set_title('Latent %d' % (r+1))
    ax.axis('off')
    if r==0:
        ax.add_artist(AnchoredSizeBar(ax.transData, 10*ses.sessiondata['fs'][0],
                        "10 Sec", loc=4, frameon=False))
plt.tight_layout()
# my_savefig(fig,figdir,'Example_Latents_Joint_%s_%dneurons_%s' % (params['direction'],Nsub,sessions[ises].session_id))

