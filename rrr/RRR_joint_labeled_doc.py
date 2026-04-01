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
from sklearn.decomposition import PCA
from scipy.stats import zscore
import pickle
from numpy.linalg import eigh

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.RRRlib import *
from utils.regress_lib import *
from params import load_params
from datetime import datetime

#%% Load parameters and settings:
params = load_params()

# params['regress_behavout'] = True
params['regress_behavout'] = False
params['direction'] = 'FF'
params['direction'] = 'FB'
# params['direction'] = 'FF_AL'
# params['direction'] = 'FB_AL'

version = 'Joint_labeled_%s_%s' % (params['direction'],'behavout' if params['regress_behavout'] else 'original')

resultdir = os.path.join(params['resultdir'])
if not os.path.exists(resultdir):
    os.makedirs(resultdir)
datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savefilename = os.path.join(resultdir,'RRR_%s_%s' % (version,datetime_str))

#%% Do RRR of V1 and PM labeled and unlabeled neurons simultaneously
if params['direction'] =='FF': 
    sourcearealabelpairs = ['V1unl','V1unl','V1lab']
    targetarealabelpair = 'PMunl'
    only_all_areas = np.array(['V1','PM'])
elif params['direction'] =='FB': 
    sourcearealabelpairs = ['PMunl','PMunl','PMlab']
    targetarealabelpair = 'V1unl'
    only_all_areas = np.array(['V1','PM'])
elif params['direction'] =='FF_AL': 
    sourcearealabelpairs = ['V1unl','V1unl','V1lab']
    targetarealabelpair = 'ALunl'
    only_all_areas = np.array(['V1','PM','AL'])
elif params['direction'] =='FB_AL': 
    sourcearealabelpairs = ['PMunl','PMunl','PMlab']
    targetarealabelpair = 'ALunl'
    only_all_areas = np.array(['V1','PM','AL'])

#%% 
session_list        = np.array([
                                # ['LPE12223_2024_06_10'], #V1lab actually lower
                                ['LPE09830_2023_04_10'], #V1 labeled higher predictive than V1unl
                                # ['LPE10919_2023_11_06'],  #V1lab actually lower
                                # ['LPE12223_2024_06_08'], #V1lab actually lower
                                # ['LPE11998_2024_05_02'], # V1lab lower?
                                ['LPE11622_2024_03_25'], #same
                                ['LPE09665_2023_03_14'], #V1lab higher
                                # ['LPE10885_2023_10_23'], #V1lab much higher
                                ['LPE11086_2024_01_05'], #Really much higher, best session, first dimensions are more predictive.
                                # ['LPE11086_2024_01_10'], #Few v1 labeled cells, very noisy
                                # ['LPE11998_2024_05_10'], #
                                # ['LPE12013_2024_05_07'], #
                                # ['LPE11495_2024_02_28'], #
                                ['LPE11086_2023_12_15'], #Same
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,
                                       min_lab_cells_V1=20,filter_noiselevel=False)

#%% Get all data 
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_noiselevel=False)
report_sessions(sessions)

#%% Wrapper function to load the tensor data
params['calciumversion'] = 'deconv'
# params['calciumversion'] = 'dF'

[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=params['regress_behavout'])

#%% 
narealabelpairs     = len(sourcearealabelpairs)

# Nsub                = 20
nmodelfits          = 6

params['nStim']     = 16

fixed_rank          = 5

idx_resp            = np.where((t_axis>=-99) & (t_axis<=99))[0]
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.95))[0]
idx_resp            = np.where((t_axis>=-99) & (t_axis<=1.95))[0]
nT                  = len(idx_resp)

contrasts            = np.array([[1,2],[1,3]]) #which contrasts to use for the DOC rotation, e.g. V1unl vs V1lab, or V1unl vs V1unl
contrasts            = np.array([[0,1],[0,2]]) #which contrasts to use for the DOC rotation, e.g. V1unl vs V1lab, or V1unl vs V1unl
ncontrasts            = len(contrasts)

R2_ranks_orig       = np.full((narealabelpairs+1,ncontrasts,nSessions,params['nStim'],fixed_rank,nmodelfits),np.nan)
R2_ranks_doc        = np.full((narealabelpairs+1,ncontrasts,nSessions,params['nStim'],fixed_rank,nmodelfits),np.nan)

R2_ranks_orig_t     = np.full((narealabelpairs+1,ncontrasts,nSessions,params['nStim'],nT,fixed_rank,nmodelfits),np.nan)
R2_ranks_doc_t      = np.full((narealabelpairs+1,ncontrasts,nSessions,params['nStim'],nT,fixed_rank,nmodelfits),np.nan)

# latents_ranks_orig  = np.full((narealabelpairs+1,2,nSessions,params['nStim'],nT,fixed_rank,nmodelfits),np.nan)
latents_ranks_doc   = np.full((narealabelpairs+1,ncontrasts,nSessions,params['nStim'],nT,fixed_rank,nmodelfits),np.nan)

kf          = KFold(n_splits=params['kfold'],shuffle=True)

params['minnneurons'] = 20

for ises,ses in enumerate(sessions):
    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    idx_areax1      = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[0],
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
    idx_areax2      = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[1],
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
    idx_areax3      = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[2],
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
    idx_areay       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpair,
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby
                                ),axis=0))[0]
    
    Nsub = min(len(idx_areax1)//2, len(idx_areax3), len(idx_areay)//narealabelpairs) #number of neurons to subselect from each population, based on the smallest population across all sessions
    if Nsub < params['minnneurons']: #skip exec if not enough neurons in one of the populations
        print('%d in %s, %d in %s' % (len(idx_areax3),sourcearealabelpairs[2],
                                                len(idx_areay),targetarealabelpair))
        continue

    for imf in tqdm(range(nmodelfits),total=nmodelfits,desc='Fitting RRR model for session %d/%d' % (ises+1,nSessions)):
        idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
        idx_areax2_sub       = np.random.choice(np.setdiff1d(idx_areax2,idx_areax1_sub),Nsub,replace=False)
        idx_areax3_sub       = np.random.choice(idx_areax3,Nsub,replace=False)
        idx_areay_sub        = np.random.choice(idx_areay,Nsub*narealabelpairs,replace=False)

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        # for istim,stim in enumerate([ses.trialdata['stimCond'][0]]): # loop over orientations 

            idx_T               = ses.trialdata['stimCond']==stim
            nK                  = np.sum(idx_T) #number of trials for this stimulus condition

            X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
            X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
            X3                  = sessions[ises].tensor[np.ix_(idx_areax3_sub,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

            #Zscore: 
            X1 -= np.nanmean(X1, axis=(1,2), keepdims=True)
            X2 -= np.nanmean(X2, axis=(1,2), keepdims=True)
            X3 -= np.nanmean(X3, axis=(1,2), keepdims=True)
            Y -= np.nanmean(Y, axis=(1,2), keepdims=True)

            X1 /= np.nanstd(X1, axis=(1,2), keepdims=True)
            X2 /= np.nanstd(X2, axis=(1,2), keepdims=True)
            X3 /= np.nanstd(X3, axis=(1,2), keepdims=True)
            Y /= np.nanstd(Y, axis=(1,2), keepdims=True)
            
            X                       = np.concatenate((X1,X2,X3),axis=0) #use this as source to predict the activity in Y with RRR

            # reshape to neurons x time points
            X_r               = X.reshape(np.shape(X)[0],-1).T
            Y_r               = Y.reshape(np.shape(Y)[0],-1).T
                        
            #RRR X to Y
            B_hat         = LM(Y_r,X_r, lam=params['lam'])
            Y_hat         = X_r @ B_hat

            # decomposing and low rank approximation of A
            U, s, V = svds(Y_hat,k=fixed_rank,which='LM')
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

            B_rrr           = B_hat @ V[:fixed_rank,:].T @ V[:fixed_rank,:] #project beta coeff into low rank subspace
            Y_hat_rr        = X_r @ B_rrr

            # How much of the variance in the source area is aligned with the predictive subspace:
            Ub, sb, Vb = svds(B_rrr,k=fixed_rank,which='LM')
            Ub, sb, Vb = Ub[:, ::-1], sb[::-1], Vb[::-1, :]

            #### NOTE: 
            # My latent space is not exactly the same as the one used in RRR reconstruction

            # X_split = np.full((*X_r.shape,narealabelpairs),np.nan)
            X_split = np.repeat(X_r[:,:,np.newaxis],narealabelpairs,axis=2)
            
            X_split[:,Nsub:,0] = 0
            X_split[:,:Nsub,1] = 0
            X_split[:,2*Nsub:,1] = 0
            X_split[:,:2*Nsub,2] = 0

            Z_orig = np.full((X_r.shape[0],fixed_rank,narealabelpairs),np.nan)
            for ial in range(narealabelpairs):
                Z_orig[:,:,ial] = X_split[:,:,ial] @ Ub @ np.diag(sb)

            for icontrast, contrast in enumerate(contrasts):
             
                #Apply DOC: 
                doc_eigvecs, doc_eigvals = doc_rotation(Z_orig[:,:,contrast[0]],Z_orig[:,:,contrast[1]])

                # latent -> Y mapping before rotation
                # A = np.diag(sb) @ Vb   # shape: (rank, n_Y)
                A = Vb   # shape: (rank, n_Y)

                # rotate into DOC space
                A_doc = doc_eigvecs.T @ A               # shape: (rank, n_Y)

                Z_full      = X_r @ Ub
                Y_hat_rr    = Z_full @ A
                Z_full_doc  = X_r @ Ub @ doc_eigvecs
                Y_hat_doc   = Z_full_doc @ A_doc

                Z_full_doc_r = np.reshape(Z_full_doc,(nK,nT,fixed_rank),order='F')
                # latents_ranks_doc[0,icontrast,ises,istim,:,:,imf] = np.nanmean(Z_full_doc_r,axis=0) #save the latents of the second contrast in the pair, which are the ones that should be more predictive of Y
                latents_ranks_doc[0,icontrast,ises,istim,:,:,imf] = np.nanvar(Z_full_doc_r,axis=0) #save the latents of the second contrast in the pair, which are the ones that should be more predictive of Y

                Y_hat_rr_svd = X_r @ Ub @ A
                assert(np.all(np.max(np.abs(Y_hat_rr - Y_hat_rr_svd)) < 1e-10)), 'RRR reconstruction should be the same as the one from the latent space'

                assert(np.allclose(EV(Y_r,Y_hat_doc),EV(Y_r,Y_hat_rr),atol=1e-10)), 'DOC rotation should not change the overall R2'

                # Rotate activity of each area into DOC space
                Z_doc = np.full_like(Z_orig,np.nan)
                for ial in range(narealabelpairs):
                    Z_doc[:,:,ial] = Z_orig[:,:,ial] @ doc_eigvecs
                    Z_doc_r = np.reshape(Z_doc[:,:,ial],(nK,nT,fixed_rank),order='F')

                    # latents_ranks_doc[ial+1,icontrast,ises,istim,:,:,imf] = np.nanmean(Z_doc_r,axis=0) #save the latents of the second contrast in the pair, which are the ones that should be more predictive of Y
                    latents_ranks_doc[ial+1,icontrast,ises,istim,:,:,imf] = np.nanvar(Z_doc_r,axis=0) #save the latents of the second contrast in the pair, which are the ones that should be more predictive of Y

                for r in range(fixed_rank):
                    # Y_hat
                    
                    Y_hat_doc_r = Z_full_doc[:,r][:, np.newaxis] @ A_doc[r,:][np.newaxis, :]
                    R2_ranks_doc[0,icontrast,ises,istim,r,imf] = EV(Y_r,Y_hat_doc_r)
                    # print('RRR R2 DOC Latent %d: %f' % (r+1, EV(Y_r,Y_hat_doc_r)))
                    
                    Y_hat_doc_perrank   = np.reshape(Y_hat_doc_r.T,(3*Nsub,nK,nT),order='C')
                    
                    for t in range(nT):
                        R2_ranks_doc_t[0,icontrast,ises,istim,t,r,imf] = EV(Y[:,:,t],Y_hat_doc_perrank[:,:,t])

                    for ial in range(narealabelpairs):
                        Z_area_doc = Z_doc[:,r,ial]
                        Y_hat_area_doc_r = Z_area_doc[:, np.newaxis] @ A_doc[r,:][np.newaxis, :]
                        R2_ranks_doc[ial+1,icontrast,ises,istim,r,imf] = EV(Y_r,Y_hat_area_doc_r)

                        Y_hat_area_doc = np.reshape(Y_hat_area_doc_r.T,(3*Nsub,nK,nT),order='C')
                        for t in range(nT):
                            R2_ranks_doc_t[ial+1,icontrast,ises,istim,t,r,imf] = EV(Y[:,:,t],Y_hat_area_doc[:,:,t])

#%% Plotting:
cm = 1/2.54
figdir = os.path.join(params['figdir'],'RRR','DOC')

t_ticks = np.array([-1,0,1,2])
# fig,axes = plt.subplots(3,4,figsize=(13,10),sharex=True,sharey=False)
fig,axes = plt.subplots(ncontrasts,fixed_rank,figsize=(12*cm,6*cm),sharex=True,sharey=True)
ax = axes
handles = []
# plotcontrast = [1,2,3]
clrs = ['grey','black','red']
for icontrast in range(ncontrasts):
    for r in range(fixed_rank):
        ax = axes[icontrast,r]
        handles = []
        for ial in range(narealabelpairs):
            datatoplot = np.nanmean(latents_ranks_doc[:,icontrast],axis=(1,2,5))
            # datatoplot = np.nanmean(latents_ranks_doc[:,icontrast,0],axis=(1,4))
            handles.append(ax.plot(t_axis[idx_resp],datatoplot[ial+1,:,r],color=clrs[ial])[0])
            thickness = ax.get_ylim()[1]/12
            ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
        if icontrast == 0 and r == fixed_rank-1: 
            ax.legend(handles=handles,labels=list(arealabeled_to_figlabels(sourcearealabelpairs)),loc='best',bbox_to_anchor=(0.65, 0.5), bbox_transform=ax.transAxes)
            my_legend_strip(ax)
        if r == 0:
            ax.set_ylabel('Variance')
# ax.legend(handles=handles,labels=sourcearealabelpairs,loc='best')
# my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,figdir,'DOC_latents_variance_acrosss_time')
# my_savefig(fig,figdir,'DOC_latents_variance_acrosss_time_10dim')
my_savefig(fig,figdir,'DOC_latents_variance_acrosss_time_%s' % params['direction'])

#%% Plotting the R2 of the latents across time:
# fig,axes = plt.subplots(3,4,figsize=(13,10),sharex=True,sharey=False)
fig,axes = plt.subplots(ncontrasts,fixed_rank,figsize=(12*cm,6*cm),sharex=True,sharey=True)
ax = axes
# plotcontrast = [1,2,3]
clrs = ['grey','black','red']
for icontrast in range(ncontrasts):
    for r in range(fixed_rank):
        ax = axes[icontrast,r]
        handles = []
        for ial in range(narealabelpairs):
            datatoplot = np.nanmean(R2_ranks_doc_t[:,icontrast],axis=(1,2,5))
            # datatoplot = np.nanmean(R2_ranks_doc_t[:,icontrast,0],axis=(1,4))
            handles.append(ax.plot(t_axis[idx_resp],datatoplot[ial+1,:,r],color=clrs[ial])[0])
        thickness = ax.get_ylim()[1]/20
        ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
            # ax.lege
        if icontrast == 0 and r == fixed_rank-1: 
            ax.legend(handles=handles,labels=list(arealabeled_to_figlabels(sourcearealabelpairs)),loc='best',bbox_to_anchor=(0.65, 0.5), bbox_transform=ax.transAxes)
            my_legend_strip(ax)
        if r == 0:
            ax.set_ylabel('R$^{2}$')
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')

# plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,figdir,'DOC_latents_r2_across_time_%s' % params['direction'])
# my_savefig(fig,figdir,'DOC_latents_r2_across_time_10dim')


#%% Plotting the R2 ratio of the latents across time:
fig,axes = plt.subplots(ncontrasts,fixed_rank,figsize=(12*cm,6*cm),sharex=True,sharey=True)
ax = axes
noise_constant = 1e-3
figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']

plotcontrasts = np.array([[1,2],[1,3]])
clrs = ['grey','red']
for icontrast in range(ncontrasts):
    for r in range(fixed_rank):
        ax = axes[icontrast,r]
        handles = []
        for iplotcontrast,plotcontrast in enumerate(plotcontrasts):

            data1 = np.nanmean(R2_ranks_doc_t[plotcontrast[0],icontrast],axis=(0,1,4))
            data2 = np.nanmean(R2_ranks_doc_t[plotcontrast[1],icontrast],axis=(0,1,4))
            datatoplot = (data2+noise_constant) / (data1+noise_constant)
            handles.append(ax.plot(t_axis[idx_resp],datatoplot[:,r],color=clrs[iplotcontrast])[0])

            # data1 = np.nanmean(R2_ranks_doc_t[plotcontrast[0],icontrast,:,:,:,r,:],axis=(3))
            # data2 = np.nanmean(R2_ranks_doc_t[plotcontrast[1],icontrast,:,:,:,r,:],axis=(3))
            # datatoplot = (data2+noise_constant) / (data1+noise_constant)
            # datatoplot = np.nanmean(datatoplot,axis=(0,1))
            # handles.append(ax.plot(t_axis[idx_resp],datatoplot,color=clrs[iplotcontrast])[0])
        thickness = ax.get_ylim()[1]/20
        ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
        ax.axhline(y=1,color='grey',linestyle='--')
        if icontrast == 0 and r == fixed_rank-1: 
            ax.legend(handles=handles,labels=figlabels,loc='best',bbox_to_anchor=(0.65, 0.5), bbox_transform=ax.transAxes)
            my_legend_strip(ax)
        if r == 0:
            ax.set_ylabel('R$^{2}$ ratio')
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,figdir,'DOC_latents_r2_ratio_across_time_%s' % params['direction'])
# my_savefig(fig,figdir,'DOC_latents_r2_across_time_10dim')

# #%% Plotting the R2 of the latents across time:
# # fig,axes = plt.subplots(3,4,figsize=(13,10),sharex=True,sharey=False)
# fig,axes = plt.subplots(ncontrasts,fixed_rank,figsize=(12*cm,6*cm),sharex=True,sharey=True)
# ax = axes
# # plotcontrast = [1,2,3]
# clrs = ['grey','black','red']
# for icontrast in range(ncontrasts):
#     for r in range(fixed_rank):
#         ax = axes[icontrast,r]
#         handles = []
#         for ial in range(narealabelpairs):
#             datatoplot = np.nanmean(R2_ranks_doc_t[:,icontrast],axis=(1,2,5))
#             datatoplot = np.nanmean(R2_ranks_doc_t[:,icontrast,0],axis=(1,4))
#             handles.append(ax.plot(t_axis[idx_resp],datatoplot[ial+1,:,r],color=clrs[ial])[0])
#         thickness = ax.get_ylim()[1]/20
#         ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
#             # ax.lege
#         if icontrast == 0 and r == fixed_rank-1: 
#             ax.legend(handles=handles,labels=list(arealabeled_to_figlabels(sourcearealabelpairs)),loc='best',bbox_to_anchor=(0.65, 0.5), bbox_transform=ax.transAxes)
#             my_legend_strip(ax)
#         if r == 0:
#             ax.set_ylabel('R$^{2}$')
# ax_nticks(ax,3)
# ax.set_xticks(t_ticks)
# ax.set_xticklabels(t_ticks)
# ax.set_xlabel('Time (sec)')

# # plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True, offset = 3)
# figdir = os.path.join(params['figdir'],'RRR','DOC')
# my_savefig(fig,figdir,'DOC_latents_r2_acrosss_time')

#%%
params['Nsub']     = Nsub
params['nranks']    = nranks
params['nmodelfits'] = nmodelfits
params['nSessions'] = nSessions

#%% Save the data:
np.savez(savefilename + '.npz',R2_cv=R2_cv,R2_ranks=R2_ranks,optim_rank=optim_rank,
         sourcearealabelpairs=sourcearealabelpairs,
         targetarealabelpair=targetarealabelpair,
         allow_pickle=True)

with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)

#%%