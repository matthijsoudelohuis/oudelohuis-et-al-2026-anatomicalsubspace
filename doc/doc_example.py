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
figdir = os.path.join(params['figdir'],'RRR','DOC')

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


#%% Show an example session, stimulus for feedforward direction
params['direction'] = 'FF'
sourcearealabelpairs = ['V1unl','V1lab']
targetarealabelpair = 'PMunl'
clrs_arealabelpairs = np.array(['grey','red'])

ises                = 0
stim                = 5
ntimebins           = 250

Nsub  = np.sum(np.all((sessions[ises].celldata['arealabel']==sourcearealabelpairs[1],
                            sessions[ises].celldata['noise_level']<params['maxnoiselevel'],	
                            ),axis=0))

#%% Show an example session, stimulus for feedback direction
params['direction'] = 'FB'
sourcearealabelpairs = ['PMunl','PMlab']
targetarealabelpair = 'V1unl'
clrs_arealabelpairs = np.array(['grey','red'])

ises                = 0
stim                = 5
ntimebins           = 250

Nsub  = np.sum(np.all((sessions[ises].celldata['arealabel']==sourcearealabelpairs[1],
                            sessions[ises].celldata['noise_level']<params['maxnoiselevel'],	
                            ),axis=0))

#%% Compute example shared latents: 
rank                = 5 #rank of RRR ranks to plot
idx_resp            = np.where((t_axis>=0) & (t_axis<=2))[0]

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

np.random.seed(99)

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
U, s, V = svds(Y_hat,k=rank,which='LM')
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

Z_1 = Z_1 @ np.diag(sb)
Z_2 = Z_2 @ np.diag(sb)

#Apply DOC: 
doc_eigvecs, doc_eigvals = doc_rotation(Z_1,Z_2)

# covariance matrices
Cx = np.cov(Z_1, rowvar=False)
Cy = np.cov(Z_2, rowvar=False)
# difference of covariances
S = Cy - Cx

# Rotate into DOC space
Z_1_doc = Z_1 @ doc_eigvecs
Z_2_doc = Z_2 @ doc_eigvecs

# new covariance matrices
Cx_doc = np.cov(Z_1_doc, rowvar=False)
Cy_doc = np.cov(Z_2_doc, rowvar=False)
# difference of covariances
S_doc = Cy_doc - Cx_doc

#%% Make figure of the covariance matrices before and after DOC:
nanperc = np.nanpercentile(np.abs([Cx,Cy,S,S_doc]), 95)
vmin,vmax = -nanperc,nanperc
lw = 0.4
fig,axes = plt.subplots(2,2,figsize=(7*cm,7*cm))
axes[0,0].pcolormesh(np.flipud(Cx),vmin=vmin,vmax=vmax,cmap='viridis',edgecolors='k',linewidth=lw)
axes[0,0].set_title(r'Cov %s latents' % arealabeled_to_figlabels(sourcearealabelpairs[0]),fontsize=7)
axes[0,1].pcolormesh(np.flipud(Cy),vmin=vmin,vmax=vmax,cmap='viridis',edgecolors='k',linewidth=lw)
axes[0,1].set_title(r'Cov %s latents' % arealabeled_to_figlabels(sourcearealabelpairs[1]),fontsize=7)
axes[1,0].pcolormesh(np.flipud(S),vmin=vmin,vmax=vmax,cmap='bwr',edgecolors='k',linewidth=lw)
axes[1,0].set_title(r'Diff Cov %s - %s' % (arealabeled_to_figlabels(sourcearealabelpairs[1]), arealabeled_to_figlabels(sourcearealabelpairs[0])),fontsize=7)
axes[1,1].pcolormesh(np.flipud(S_doc),vmin=vmin,vmax=vmax,cmap='bwr',edgecolors='k',linewidth=lw)
axes[1,1].set_title(r'Cov %s - %s after DOC' % (arealabeled_to_figlabels(sourcearealabelpairs[0]), arealabeled_to_figlabels(sourcearealabelpairs[1])),fontsize=7)

for ax in axes.flatten():
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_xlim(-0.5, rank+0.5)
    ax.set_ylim(-0.5, rank+0.5)
    ax.axis('off')
   # ax.set
    # ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    # ax.grid(True)

my_savefig(fig,figdir,'Example_Cov_%s_%dneurons_%s' % (params['direction'],Nsub,sessions[ises].session_id))

#%% Plot excerpt of latent dimensions over time
kernel_size         = 3 #size of smoothing kernel in frames

#Little bit of smoothing:
Z_1_smooth = np.zeros_like(Z_1)
Z_2_smooth = np.zeros_like(Z_2)
kernel = np.ones(kernel_size) / kernel_size
for r in range(rank):
    Z_1_smooth[:,r] = np.convolve(Z_1[:,r], kernel, mode='same')
    Z_2_smooth[:,r] = np.convolve(Z_2[:,r], kernel, mode='same')

#Little bit of smoothing:
Z_1_doc_smooth = np.zeros_like(Z_1_doc)
Z_2_doc_smooth = np.zeros_like(Z_2_doc)
kernel = np.ones(kernel_size) / kernel_size
for r in range(rank):
    Z_1_doc_smooth[:,r] = np.convolve(Z_1_doc[:,r], kernel, mode='same')
    Z_2_doc_smooth[:,r] = np.convolve(Z_2_doc[:,r], kernel, mode='same')

#Find a good chunk of time to plot where the DOC is highlighting the difference between the two populations:
offset = 5
lw = 0.7
diffs = []
for ichunk,idx_k0 in enumerate(np.arange(0,np.shape(X1)[0]-ntimebins,ntimebins)):
    idx_K = np.arange(idx_k0,idx_k0+ntimebins)
    diff = np.sum(Z_1_doc_smooth[idx_K,:] - Z_2_doc_smooth[idx_K,:],axis=0)
    diffs.append(diff[0] - diff[1])
ichunk = np.argmax(diffs)
idx_K = np.arange(ichunk*ntimebins,(ichunk+1)*ntimebins)

#make the figure:
fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharex=True,sharey=True)
ax = axes[0]
for r in range(rank):
    # ax = axes[r,0]
    ax.plot(Z_1_smooth[idx_K,r]-r*offset,color=clrs_arealabelpairs[0],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[0]]),lw=lw)
    ax.plot(Z_2_smooth[idx_K,r]-r*offset,color=clrs_arealabelpairs[1],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[1]]),lw=lw)
ax.set_ylabel('Latent dim %d' % (r+1))
# ax.text(0.5,0.9,'RRR Latent %d' % (r+1),fontsize=7,transform=ax.transAxes)
ax.axis('off')
ax.set_title('RRR Latents',fontsize=7)
ax = axes[1]
for r in range(rank):
    # ax = axes[r,1]
    ax.plot(Z_1_doc_smooth[idx_K,r]-r*offset,color=clrs_arealabelpairs[0],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[0]]),lw=lw)
    ax.plot(Z_2_doc_smooth[idx_K,r]-r*offset,color=clrs_arealabelpairs[1],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[1]]),lw=lw)
    # ax.set_ylabel('Latent dim %d' % (r+1))
handles = ax.get_legend_handles_labels()[0]
ax.legend(handles=handles[:2],frameon=False,bbox_to_anchor=(1.3,.8),loc='upper right',fontsize=8)
my_legend_strip(ax)
ax.axis('off')
ax.set_title('DOC Latents',fontsize=7)
ax.add_artist(AnchoredSizeBar(ax.transData, 10*sessions[ises].sessiondata['fs'][0],
                "10 Sec", loc=4, frameon=False))
plt.tight_layout()
# my_savefig(fig,figdir,'Example_Latents_Joint_DOC_%s_%dneurons_%s' % (params['direction'],Nsub,sessions[ises].session_id))

#%% Now compute the R2 of the latents in the target area, to see if they are more predictive than the original RRR latents:

R2_latents_orig = np.zeros((2,rank))
R2_latents_doc = np.zeros((2,rank))

print('full R2: %f' % EV(Y,Y_hat))
Y_hat_rr = X @ B_rrr
print('RRR R2: %f' % EV(Y,Y_hat_rr))
# Y_hat_doc = Z_1_doc @ np.linalg.pinv(Z_1_doc) @ Y

# latent -> Y mapping before rotation

Y_hat_rr_svd = X @ Ub @ np.diag(sb) @ Vb
print('RRR R2: %f' % EV(Y,Y_hat_rr_svd))

# print('RRR R2: %f' % EV(Y,Y_hat_rr_svd))

assert(np.all(np.max(np.abs(Y_hat_rr - Y_hat_rr_svd)) < 1e-10)), 'RRR reconstruction should be the same as the one from the latent space'

# rotate into DOC space
Vb_doc = doc_eigvecs.T @ Vb              # shape: (rank, n_Y)

Z_doc = X @ Ub @ np.diag(sb) @ doc_eigvecs

Y_hat_doc = Z_doc @ Vb_doc

print('RRR R2 DOC: %f' % EV(Y,Y_hat_doc))

assert(np.allclose(EV(Y,Y_hat_doc),EV(Y,Y_hat_rr),atol=1e-10)), 'DOC rotation should not change the overall R2'

# R2 of original RRR latents:
for r in range(rank):
    Y_hat_latent_r = Z_1[:,r][:, np.newaxis] @ Vb[r,:][np.newaxis, :]
    R2_latents_orig[0,r] = EV(Y,Y_hat_latent_r)
    Y_hat_latent_r = Z_2[:,r][:, np.newaxis] @ Vb[r,:][np.newaxis, :]
    R2_latents_orig[1,r] = EV(Y,Y_hat_latent_r)

for r in range(rank):
    Y_hat_latent_r = Z_1_doc[:,r][:, np.newaxis] @ Vb_doc[r,:][np.newaxis, :]
    R2_latents_doc[0,r] = EV(Y,Y_hat_latent_r)
    Y_hat_latent_r = Z_2_doc[:,r][:, np.newaxis] @ Vb_doc[r,:][np.newaxis, :]
    R2_latents_doc[1,r] = EV(Y,Y_hat_latent_r)

# # R2 of original RRR latents:
# for r in range(rank):
#     Y_hat_latent_r = Z_1[:,r][:, np.newaxis] @ A[r,:][np.newaxis, :]
#     # Y_hat_latent_r = Z_1[:,:r+1] @ A[:r+1,:]
#     R2_latents_orig[0,r] = EV(Y,Y_hat_latent_r)
#     Y_hat_latent_r = Z_2[:,r][:, np.newaxis] @ A[r,:][np.newaxis, :]
#     # Y_hat_latent_r = Z_2[:,:r+1] @ A[:r+1,:]
#     R2_latents_orig[1,r] = EV(Y,Y_hat_latent_r)
# R2_latents_orig = np.diff(np.concatenate((np.zeros((2,1)),R2_latents_orig),axis=1),axis=1)

# for r in range(rank):
#     Y_hat_latent_r = Z_1_doc[:,r][:, np.newaxis] @ Vb_doc[r,:][np.newaxis, :]
#     # Y_hat_latent_r = Z_1_doc[:,:r+1] @ A[:r+1,:]
#     R2_latents_doc[0,r] = EV(Y,Y_hat_latent_r)
#     Y_hat_latent_r = Z_2_doc[:,r][:, np.newaxis] @ Vb_doc[r,:][np.newaxis, :]
#     # Y_hat_latent_r = Z_2_doc[:,:r+1] @ A[:r+1,:]
#     R2_latents_doc[1,r] = EV(Y,Y_hat_latent_r)
# R2_latents_doc = np.diff(np.concatenate((np.zeros((2,1)),R2_latents_doc),axis=1),axis=1)

#Plotting the R2 of the original and DOC latents:
fig,axes = plt.subplots(1,2,figsize=(7*cm,4*cm),sharey=True,sharex=True)
ax = axes[0]
x = np.arange(1,rank+1)
ax.plot(x,R2_latents_orig[0,:],color=clrs_arealabelpairs[0],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[0]]))
ax.plot(x,R2_latents_orig[1,:],color=clrs_arealabelpairs[1],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[1]]))
ax.set_xlabel('RRR dimension')
ax.set_ylabel('R$^{2}$')
ax_nticks(ax,4)
ax.set_title('Original (RRR)')
ax.set_xticks(x)

ax.legend(frameon=False,loc='center right',fontsize=8)
my_legend_strip(ax)

ax = axes[1]
ax.plot(x,R2_latents_doc[0,:],color=clrs_arealabelpairs[0],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[0]]))
ax.plot(x,R2_latents_doc[1,:],color=clrs_arealabelpairs[1],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[1]]))
ax.set_xlabel('DOC dimension')
ax.set_title('After DOC rotation')
ax.set_ylim([0,ax.get_ylim()[1]])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 1,trim=True)
my_savefig(fig,figdir,'R2_DOC_Example_%s_%dneurons_%s' % (params['direction'],Nsub,sessions[ises].session_id))

#%%
#Plotting the R2 of the original and DOC latents:
fig,axes = plt.subplots(1,2,figsize=(7*cm,4*cm),sharey=True,sharex=True)
ax = axes[0]
x = np.arange(1,rank+1)
ax.plot(x,R2_latents_orig[1,:] - R2_latents_orig[0,:],color='grey',alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[0]]))
# ax.plot(x,R2_latents_orig[1,:],color=clrs_arealabelpairs[1],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[1]]))
ax.set_xlabel('Latent dimension')
ax.set_ylabel('$\Delta$R$^{2}$')
ax_nticks(ax,4)
ax.set_xticks(x)
ax.axhline(y=0,color='grey',linestyle='--')
# ax.legend(frameon=False,loc='upper right',fontsize=8)
# my_legend_strip(ax)

ax = axes[1]
ax.plot(x,R2_latents_doc[1,:]- R2_latents_doc[0,:],color='grey',alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[0]]))
# ax.plot(x,R2_latents_doc[1,:],color=clrs_arealabelpairs[1],alpha=1,label=arealabeled_to_figlabels([sourcearealabelpairs[1]]))
ax.set_xlabel('Latent dimension')
ax.set_ylabel('R$^{2}$')
ax.axhline(y=0,color='grey',linestyle='--')
sns.despine(fig=fig, top=True, right=True, offset = 3)
