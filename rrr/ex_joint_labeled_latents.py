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
# sourcearealabelpairs = ['V1unl','V1unl']
targetarealabelpair = 'PMunl'
clrs_arealabelpairs = np.array(['grey','red'])

ises                = 0
stim                = 4
starttimepoint_idx  = 300
ntimebins           = 250
Nsub                = 150 #number of cells to sample

Nsub  = np.sum(np.all((sessions[ises].celldata['arealabel']==sourcearealabelpairs[1],
                            sessions[ises].celldata['noise_level']<params['maxnoiselevel'],	
                            ),axis=0))

#%% 
params['direction'] = 'FB'
sourcearealabelpairs = ['PMunl','PMlab']
targetarealabelpair = 'V1unl'
clrs_arealabelpairs = np.array(['grey','red'])

ises                = 1
stim                = 3
starttimepoint_idx  = 1900
ntimebins           = 250
Nsub                = 50 #number of cells to sample

Nsub  = np.sum(np.all((sessions[ises].celldata['arealabel']==sourcearealabelpairs[1],
                            sessions[ises].celldata['noise_level']<params['maxnoiselevel'],	
                            ),axis=0))

#%% Show example shared latents: 
idx_resp            = np.where((t_axis>=0) & (t_axis<=2))[0]
scale_eigenvalues = True # scale eigenvalues
rank                = 10 #rank of RRR ranks to plot
kernel_size         = 3 #size of smoothing kernel in frames

idx_areax1          = np.where(np.all((sessions[ises].celldata['arealabel']==sourcearealabelpairs[0],
                            sessions[ises].celldata['noise_level']<params['maxnoiselevel'],	
                            ),axis=0))[0]
idx_areax2          = np.where(np.all((sessions[ises].celldata['arealabel']==sourcearealabelpairs[1],
                            sessions[ises].celldata['noise_level']<params['maxnoiselevel'],	
                            ),axis=0))[0]
idx_areay       = np.where(np.all((sessions[ises].celldata['arealabel']==targetarealabelpair,
                                        sessions[ises].celldata['noise_level']<params['maxnoiselevel'],	
                                        ),axis=0))[0]
print(len(idx_areax2))

np.random.seed(0)

idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
idx_areax2_sub       = np.random.choice(idx_areax2,Nsub,replace=False)
idx_areay_sub        = np.random.choice(idx_areay,Nsub*2,replace=False)

idx_T               = sessions[ises].trialdata['stimCond']==stim

X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
Y                   = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

# reshape to neurons x time points
X1                  = X1.reshape(len(idx_areax1_sub),-1).T
X2                  = X2.reshape(len(idx_areax2_sub),-1).T
Y                   = Y.reshape(len(idx_areay_sub),-1).T

X1                  = zscore(X1,axis=0)
X2                  = zscore(X2,axis=0)
Y                   = zscore(Y,axis=0)

X                   = np.concatenate((X1,X2),axis=1)
 
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
Ub, sb, Vb = Ub[:, ::-1], sb[::-1], Vb[::-1, :]

X_1 = copy.deepcopy(X)
X_1[:,Nsub:] = 0
Z_1 = X_1 @ Ub 

X_2 = copy.deepcopy(X)
X_2[:,:Nsub] = 0
Z_2 = X_2 @ Ub

if scale_eigenvalues:
    Z_1 = Z_1 * sb
    Z_2 = Z_2 * sb

#Little bit of smoothing:
Z_1_smooth = np.zeros_like(Z_1)
Z_2_smooth = np.zeros_like(Z_2)
kernel = np.ones(kernel_size) / kernel_size
for r in range(rank):
    Z_1_smooth[:,r] = np.convolve(Z_1[:,r], kernel, mode='same')
    Z_2_smooth[:,r] = np.convolve(Z_2[:,r], kernel, mode='same')

#Plot excerpt of latent dimensions over time
clrs = sns.color_palette('viridis',rank)
# fig,axes = plt.subplots(rank,1,figsize=(7*cm,rank*2*cm),sharex=True)

fig,axes = plt.subplots(rank,1,figsize=(7*cm,rank*2*cm),sharex=True,sharey=True)

idx_K = np.arange(starttimepoint_idx,starttimepoint_idx+ntimebins)
for r in range(rank):
    ax = axes[r]
    ax.plot(Z_1_smooth[idx_K,r],color=clrs_arealabelpairs[0],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[0]]))
    ax.plot(Z_2_smooth[idx_K,r],color=clrs_arealabelpairs[1],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[1]]))
    ax.set_ylabel('Latent dim %d' % (r+1))
    if r==0:
        ax.legend(frameon=False,loc='lower left',fontsize=8)
        my_legend_strip(ax)
    ax.set_title('Latent %d' % (r+1))
    ax.axis('off')
    if r==0:
        ax.add_artist(AnchoredSizeBar(ax.transData, 10*sessions[ises].sessiondata['fs'][0],
                        "10 Sec", loc=4, frameon=False))
plt.tight_layout()
my_savefig(fig,figdir,'Example_Latents_Joint_%s_%dneurons_%s' % (params['direction'],Nsub,sessions[ises].session_id))

#Apply DOC: 
doc_eigvecs, doc_eigvals = doc_rotation(Z_1,Z_2)

# Rotate into DOC space
Z_1_doc = Z_1 @ doc_eigvecs
Z_2_doc = Z_2 @ doc_eigvecs

#Little bit of smoothing:
Z_1_smooth = np.zeros_like(Z_1)
Z_2_smooth = np.zeros_like(Z_2)
kernel = np.ones(kernel_size) / kernel_size
for r in range(rank):
    Z_1_smooth[:,r] = np.convolve(Z_1_doc[:,r], kernel, mode='same')
    Z_2_smooth[:,r] = np.convolve(Z_2_doc[:,r], kernel, mode='same')

#Plot excerpt of latent dimensions over time
clrs = sns.color_palette('viridis',rank)
# fig,axes = plt.subplots(rank,1,figsize=(7*cm,rank*2*cm),sharex=True)

fig,axes = plt.subplots(rank,1,figsize=(7*cm,rank*2*cm),sharex=True,sharey=True)

idx_K = np.arange(starttimepoint_idx,starttimepoint_idx+ntimebins)
for r in range(rank):
    ax = axes[r]
    ax.plot(Z_1_smooth[idx_K,r],color=clrs_arealabelpairs[0],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[0]]))
    ax.plot(Z_2_smooth[idx_K,r],color=clrs_arealabelpairs[1],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[1]]))
    # ax.set_ylabel('Latent dim %d' % (r+1))
    if r==0:
        ax.legend(frameon=False,loc='lower left',fontsize=8)
        my_legend_strip(ax)
    # ax.set_title('%d' % (r+1),fontsize=7)
    ax.text(0.5,0.9,'%d' % (r+1),fontsize=7,transform=ax.transAxes)
    ax.axis('off')
    if r==0:
        ax.add_artist(AnchoredSizeBar(ax.transData, 10*sessions[ises].sessiondata['fs'][0],
                        "10 Sec", loc=4, frameon=False))
plt.suptitle('Maximally different latents')
plt.tight_layout()

my_savefig(fig,figdir,'Example_Latents_Joint_DOC_%s_%dneurons_%s' % (params['direction'],Nsub,sessions[ises].session_id))

#%% Now compute the R2 of the latents in the target area, to see if they are more predictive than the original RRR latents:

R2_latents_orig = np.zeros((2,rank))
R2_latents_doc = np.zeros((2,rank))

print('full R2: %f' % EV(Y,Y_hat))
Y_hat_rr = X @ B_rrr
print('RRR R2: %f' % EV(Y,Y_hat_rr))
Y_hat_doc = Z_1_doc @ np.linalg.pinv(Z_1_doc) @ Y

# latent -> Y mapping before rotation
if scale_eigenvalues:
    A = Vb[:rank, :]   # shape: (rank, n_Y)
else:
    A = np.diag(sb[:rank]) @ Vb[:rank, :]   # shape: (rank, n_Y)

# rotate into DOC space
A_doc = doc_eigvecs.T @ A               # shape: (rank, n_Y)

Z_doc = X @ Ub @ doc_eigvecs

Y_hat_doc = Z_doc @ A_doc

print('RRR R2 DOC: %f' % EV(Y,Y_hat_doc))

# R2 of original RRR latents:
for r in range(rank):
    Y_hat_latent_r = Z_1[:,r][:, np.newaxis] @ A[r,:][np.newaxis, :]
    R2_latents_orig[0,r] = EV(Y,Y_hat_latent_r)
    Y_hat_latent_r = Z_2[:,r][:, np.newaxis] @ A[r,:][np.newaxis, :]
    R2_latents_orig[1,r] = EV(Y,Y_hat_latent_r)

for r in range(rank):
    Y_hat_latent_r = Z_1_doc[:,r][:, np.newaxis] @ A_doc[r,:][np.newaxis, :]
    R2_latents_doc[0,r] = EV(Y,Y_hat_latent_r)
    Y_hat_latent_r = Z_2_doc[:,r][:, np.newaxis] @ A_doc[r,:][np.newaxis, :]
    R2_latents_doc[1,r] = EV(Y,Y_hat_latent_r)

#Plotting the R2 of the original and DOC latents:
fig,axes = plt.subplots(1,2,figsize=(7*cm,3*cm),sharey=True,sharex=True)
ax = axes[0]
x = np.arange(1,rank+1)
ax.plot(x,R2_latents_orig[0,:],color=clrs_arealabelpairs[0],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[0]]))
ax.plot(x,R2_latents_orig[1,:],color=clrs_arealabelpairs[1],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[1]]))
ax.set_xlabel('Latent dimension')
ax.set_ylabel('R$^{2}$')
ax.set_xticks(x)
ax.legend(frameon=False,loc='upper right',fontsize=8)
my_legend_strip(ax)

ax = axes[1]
ax.plot(x,R2_latents_doc[0,:],color=clrs_arealabelpairs[0],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[0]]))
ax.plot(x,R2_latents_doc[1,:],color=clrs_arealabelpairs[1],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[1]]))
ax.set_xlabel('Latent dimension')
ax.set_ylabel('R$^{2}$')
# ax.legend(frameon=False,loc='upper right',fontsize=8)
# my_legend_strip(ax)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,figdir,'R2_Latents_Joint_DOC_%s_%dneurons_%s' % (params['direction'],Nsub,sessions[ises].session_id))
