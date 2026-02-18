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

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
# from utils.corr_lib import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.pair_lib import value_matching
from utils.psth import compute_tensor
from params import load_params
from utils.corr_lib import filter_sharednan

params = load_params()
savedir = os.path.join(params['savedir'],'RRR','Labeling')

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
                                ['LPE09665_2023_03_14'], #V1lab higher
                                ['LPE10885_2023_10_23'], #V1lab much higher
                                # ['LPE11086_2024_01_05'], #Really much higher, best session, first dimensions are more predictive.
                                # ['LPE11086_2024_01_10'], #Few v1 labeled cells, very noisy
                                # ['LPE11998_2024_05_10'], #
                                # ['LPE12013_2024_05_07'], #
                                # ['LPE11495_2024_02_28'], #
                                ['LPE11086_2023_12_15'], #Same
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,
                                       min_lab_cells_V1=20,filter_noiselevel=False)
# np.sum(np.logical_and(sessions[0].celldata['roi_name']=='V1',sessions[0].celldata['redcell']==1))

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_noiselevel=True)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],min_lab_cells_V1=20,filter_noiselevel=True)
report_sessions(sessions)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False)
# [sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=True)
# sessions = load_resid_tensor(sessions,behavout=True)

#%% Do RRR of V1 and PM labeled and unlabeled neurons simultaneously
sourcearealabelpairs = ['V1unl','V1unl','V1lab']
targetarealabelpair = 'PMunl'

clrs_arealabelpairs = get_clr_area_labeled(sourcearealabelpairs)
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 25
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 10
# dim_method          = 'parallel_analysis'
# dim_method          = 'participation_ratio'
dim_method          = 'pca_shuffle'
nStim               = 16

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)
# minsampleneurons    = 10

R2_cv               = np.full((narealabelpairs+1,nSessions,nStim),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
optim_rank          = np.full((narealabelpairs+1,nSessions,nStim),np.nan)
R2_ranks            = np.full((narealabelpairs+1,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
R2_ranks_neurons    = np.full((narealabelpairs+1,Nsub*narealabelpairs,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
source_dim          = np.full((narealabelpairs+1,nSessions,nStim,nmodelfits),np.nan)
R2_sourcealigned    = np.full((narealabelpairs+1,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
R2_ranks_ctrl       = np.full((narealabelpairs+1,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
R2_ranks_neurons_ctrl = np.full((narealabelpairs+1,Nsub*narealabelpairs,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
frac_pos_weight_out = np.full((nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
frac_pos_weight_in  = np.full((narealabelpairs+1,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
# weights_in          = np.full((Nsub*narealabelpairs,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
weights_in          = np.full((narealabelpairs+1,Nsub,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)

params['maxnoiselevel'] = 20
params['radius'] = 50
# params['radius'] = 30
# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different populations'):
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

    if len(idx_areax1)<Nsub*2 or len(idx_areax2)<Nsub*2 or len(idx_areax3)<Nsub or len(idx_areay)<narealabelpairs*Nsub: #skip exec if not enough neurons in one of the populations
        continue

    for imf in tqdm(range(nmodelfits),total=nmodelfits,desc='Fitting RRR model for session %d/%d' % (ises+1,nSessions)):
        idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
        idx_areax2_sub       = np.random.choice(np.setdiff1d(idx_areax2,idx_areax1_sub),Nsub,replace=False)
        idx_areax3_sub       = np.random.choice(idx_areax3,Nsub,replace=False)
        idx_areay_sub        = np.random.choice(idx_areay,Nsub*narealabelpairs,replace=False)

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
       
            X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
            X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
            X3                  = sessions[ises].tensor[np.ix_(idx_areax3_sub,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

            # reshape to neurons x time points
            X1                  = X1.reshape(len(idx_areax1_sub),-1).T
            X2                  = X2.reshape(len(idx_areax2_sub),-1).T
            X3                  = X3.reshape(len(idx_areax3_sub),-1).T
            Y                   = Y.reshape(len(idx_areay_sub),-1).T

            X1                  = zscore(X1,axis=0) #zscore the activity per neuron
            X2                  = zscore(X2,axis=0)
            X3                  = zscore(X3,axis=0)
            Y                   = zscore(Y,axis=0)

            X                   = np.concatenate((X1,X2,X3),axis=1) #use this as source to predict the activity in Y with RRR

            # for i,data in enumerate([X,X1,X2,X3]):
                # source_dim[i,ises,istim,imf] = estimate_dimensionality(data,method=dim_method)

            # OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS    
            R2_kfold    = np.zeros((params['kfold']))
            kf          = KFold(n_splits=params['kfold'],shuffle=True)
            for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                X_train, X_test     = X[idx_train], X[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                #RRR X to Y
                B_hat_train         = LM(Y_train,X_train, lam=params['lam'])
                Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of Y_hat
                U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                for r in range(nranks):
                    B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                    Y_hat_test_rr   = X_test @ B_rrr

                    R2_ranks[0,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)
                    R2_ranks_neurons[0,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')
                    
                    X_test_1 = copy.deepcopy(X_test)
                    X_test_1[:,Nsub:] = 0
                    Y_hat_test_rr   = X_test_1 @ B_rrr

                    R2_ranks[1,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)
                    R2_ranks_neurons[1,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')

                    X_test_2 = copy.deepcopy(X_test)
                    X_test_2[:,:Nsub] = 0
                    X_test_2[:,2*Nsub:] = 0
                    Y_hat_test_rr   = X_test_2 @ B_rrr

                    R2_ranks[2,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)
                    R2_ranks_neurons[2,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')

                    X_test_3 = copy.deepcopy(X_test)
                    X_test_3[:,:2*Nsub] = 0
                    Y_hat_test_rr   = X_test_3 @ B_rrr

                    R2_ranks[3,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)
                    R2_ranks_neurons[3,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')

                for i,data in enumerate([X_test,X_test_1,X_test_2,X_test_3]):
                    # How much of the variance in the source area is aligned with the predictive subspace:
                    R2_sourcealigned[i,ises,istim,:,imf,ikf] = compute_rrr_sourcevariance(data, B_hat_train,nranks=20)

                #Fraction of weights that is projecting positively onto firing rate:
                for r in range(nranks): #for each rank
                    #find correct sign of weight by sign of inner product mean firing rate and left singular vector
                    frac_pos_weight_out[ises,istim,r,imf,ikf] = np.sum(np.sign(V[r,:])==np.sign(U[:,r].T @ np.nanmean(Y_train, axis=1))) / np.shape(V)[1]
                    
                # Predictive source directions
                W = B_hat_train @ V.T  # (N x k)
                # Mean source firing rate across timepoints
                mu_X = X_train.mean(axis=1)
                for r in range(nranks): #for each rank compute weights
                    # Align sign to mean source firing
                    sign = np.sign(np.dot(X_train @ W[:, r], mu_X))
                    # weights_in[:,ises,istim,r,imf,ikf] = sign * W[:, r]

                    idx_N = np.arange(Nsub)
                    weights_in[0,:,ises,istim,r,imf,ikf] = W[np.ix_(idx_N,[r])].flatten()*sign
                    idx_N = np.arange(Nsub,2*Nsub)
                    weights_in[1,:,ises,istim,r,imf,ikf] = W[np.ix_(idx_N,[r])].flatten()*sign
                    idx_N = np.arange(Nsub*2,Nsub*3)
                    weights_in[2,:,ises,istim,r,imf,ikf] = W[np.ix_(idx_N,[r])].flatten()*sign

                    frac_pos_weight_in[0,ises,istim,r,imf,ikf] = np.sum(np.sign(W[:, r])==sign) / np.shape(W)[0]
                    idx_N = np.arange(Nsub)
                    frac_pos_weight_in[1,ises,istim,r,imf,ikf] = np.sum(np.sign(W[np.ix_(idx_N,[r])])==sign) / Nsub
                    idx_N = np.arange(Nsub,2*Nsub)
                    frac_pos_weight_in[2,ises,istim,r,imf,ikf] = np.sum(np.sign(W[np.ix_(idx_N,[r])])==sign) / Nsub
                    idx_N = np.arange(Nsub*2,Nsub*3)
                    frac_pos_weight_in[3,ises,istim,r,imf,ikf] = np.sum(np.sign(W[np.ix_(idx_N,[r])])==sign) / Nsub
                    # print(frac_pos_weight_in[:,ises,istim,r,imf,ikf])

#%%
fixed_rank = 5
for ises in range(nSessions):
    if np.any(~np.isnan(R2_ranks[0,ises,:,:,:,:])):
        for istim in range(nStim):
            if fixed_rank is not None:
                rank = fixed_rank
                R2_cv[0,ises,istim] = np.nanmean(R2_ranks[0,ises,istim,rank,:,:])
                R2_cv[1,ises,istim] = np.nanmean(R2_ranks[1,ises,istim,rank,:,:])
                R2_cv[2,ises,istim] = np.nanmean(R2_ranks[2,ises,istim,rank,:,:])
                R2_cv[3,ises,istim] = np.nanmean(R2_ranks[3,ises,istim,rank,:,:])
            else:
                if not np.isnan(R2_ranks[0,ises,istim,:,:,:]).all():
                    R2_cv[0,ises,istim],optim_rank[0,ises,istim] = rank_from_R2(R2_ranks[0,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                    R2_cv[1,ises,istim],optim_rank[1,ises,istim] = rank_from_R2(R2_ranks[1,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                    R2_cv[2,ises,istim],optim_rank[2,ises,istim] = rank_from_R2(R2_ranks[2,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                    R2_cv[3,ises,istim],optim_rank[3,ises,istim] = rank_from_R2(R2_ranks[3,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])

#%% 
fig, axes = plt.subplots(1,1,figsize=(6*cm,5*cm))
# ax = axes[0]
ax = axes
ax.plot(range(nranks),np.nanmean(R2_ranks[0],axis=(0,1,3,4)),label='All neurons',color='grey')
ax.plot(np.nanmean(R2_ranks[1],axis=(0,1,3,4)),label=sourcearealabelpairs[0],color=clrs_arealabelpairs[0])
ax.plot(np.nanmean(R2_ranks[2],axis=(0,1,3,4)),label=sourcearealabelpairs[1],color=clrs_arealabelpairs[1])
ax.plot(np.nanmean(R2_ranks[3],axis=(0,1,3,4)),label=sourcearealabelpairs[2],color=clrs_arealabelpairs[2])
leg = ax.legend(frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Cross-validated R2')

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
my_savefig(fig,savedir,'RRR_joint_cvR2_labunl_FF_%dsessions' % (nSessions))
# my_savefig(fig,savedir,'RRR_joint_cvR2_labunl_FF_%dsessions_behavout' % (nSessions))


#%% Show figure for each of the arealabelpairs and each of the dataversions
#Reshape stim x sessions:
R2_data                 = np.reshape(R2_cv,(narealabelpairs+1,nSessions*nStim))
optim_rank_data         = np.reshape(optim_rank,(narealabelpairs+1,nSessions*nStim))
R2_ranks_data           = np.reshape(R2_ranks,(narealabelpairs+1,nSessions*nStim,nranks,nmodelfits,params['kfold']))
if np.any(~np.isnan(R2_data)):
    # for idx in np.array([[0,1],[2,3]]):
    for idx in np.array([[1,3]]):
    # for idx in np.array([[2,3]]):
    # for idx in np.array([[1,3]]):
        # clrs        = ['grey',get_clr_area_labeled([sourcearealabelpairs[idx[1]].split('-')[0]])]
        clrs        = ['grey','red']
        fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(sourcearealabelpairs)[idx-1],clrs)
        # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

#%% Identify which dimensions are particularly enhanced in labeled cells:
data = np.nanmean(R2_ranks,axis=(5)) #average across kfolds
data = np.diff(data,axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

diffmetric = 'ratio' #'difference'
# diffmetric = 'difference' #'difference'
noise_constant = 1e-3
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm),sharey=True,sharex=True)
ax = axes
handles = []
if diffmetric == 'ratio':
    # ymeantoplot = np.nanmean(data[2],axis=(0,1,3)) / (np.nanmean(data[1],axis=(0,1,3))+1e-3)
    # yerrortoplot = np.nanstd(data[2],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3)) / np.sqrt(nSessions*nmodelfits)
    ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)
    yerrortoplot = (np.nanstd(data[2],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1],axis=(0,1,3))+noise_constant) / np.sqrt(nSessions*nmodelfits)
elif diffmetric == 'difference':
    ymeantoplot = np.nanmean(data[2] - data[1],axis=(0,1,3))
    yerrortoplot = np.nanstd(data[2] - data[1],axis=(0,1,3)) / np.sqrt(nSessions*nmodelfits)
handles.append(shaded_error(np.arange(nranks-1)+1,ymeantoplot,yerrortoplot,ax=ax,color='black',alpha=0.3))

if diffmetric == 'ratio':
    ymeantoplot = (np.nanmean(data[3],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)
    yerrortoplot = (np.nanstd(data[3],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1],axis=(0,1,3))+noise_constant) / np.sqrt(nSessions*nmodelfits)
    # ymeantoplot = np.nanmean(data[3],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3))
    # yerrortoplot = np.nanstd(data[3],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3)) / np.sqrt(nSessions*nmodelfits)
elif diffmetric == 'difference':
    ymeantoplot = np.nanmean(data[3] - data[1],axis=(0,1,3))
    yerrortoplot = np.nanstd(data[3] - data[1],axis=(0,1,3)) / np.sqrt(nSessions*nmodelfits)

handles.append(shaded_error(np.arange(nranks-1)+1,ymeantoplot,yerrortoplot,ax=ax,color='red',alpha=0.3))
ax.legend(handles,['unl-unl','lab-unl'],frameon=False)
my_legend_strip(ax)
ax_nticks(ax,4)
ax.set_xticks(np.arange(nranks-1)[::3]+1)
ax.set_xlim([1,10])
ax.set_xlabel('dimension')
ax.set_ylabel('R2 %s' % diffmetric)
if diffmetric == 'ratio':
    ax.axhline(y=1,color='grey',linestyle='--')
elif diffmetric == 'difference':
    ax.axhline(y=0,color='grey',linestyle='--')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,savedir,'RRR_R2_%s_rank_noiseconstant_V1lab_V1unl_V1unl_%dneurons_%dsessions' % (diffmetric,Nsub,nSessions))
# my_savefig(fig,savedir,'RRR_unique_cvR2_V1lab_V1unl_V1unl_%dneurons' % Nsub)

#%% Are the dimensions which are enhanced in labeled cells unique or express in unlabeled cells as well?
nrankstoplot = 4
r2data = np.nanmean(R2_ranks,axis=(5)) #average across kfolds
r2data = np.diff(r2data[:,:,:,:nrankstoplot+1,:],axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
nbins = 20
r2lim = 0.05
bins = np.linspace(0,r2lim,nbins)
fig,axes = plt.subplots(1,3,figsize=(12*cm,4.5*cm),sharey=True,sharex=True)
ax = axes[0]
xdata = r2data[1].flatten()
ydata = r2data[3].flatten()
xdata,ydata = filter_sharednan(xdata,ydata)
histdata1 = np.histogram2d(xdata,ydata,bins=bins)[0].T
histdata1 /= np.sum(histdata1)
ax.pcolor(bins,bins,np.log(histdata1),cmap='viridis')
ax.set_ylabel('R2 (lab)')
ax.set_xlabel('R2 (unl)')
ax.set_title('lab-unl')

ax = axes[1]
ydata = r2data[2].flatten()
xdata = r2data[1].flatten()
xdata,ydata = filter_sharednan(xdata,ydata)
histdata2 = np.histogram2d(xdata,ydata,bins=bins)[0].T
histdata2 /= np.sum(histdata2)
ax.pcolor(bins,bins,np.log(histdata2),cmap='viridis')
ax.set_xlabel('R2 (unl)')
ax.set_ylabel('R2 (unl)')
ax.set_title('Unl-unl')

ax = axes[2]
diffdata = histdata1 - histdata2
# Apply gaussian filter
sigma = [1, 1]
diffdata = sp.ndimage.filters.gaussian_filter(diffdata, sigma, mode='constant')
vmin,vmax = -np.abs(np.nanpercentile(diffdata,99.5)),np.abs(np.nanpercentile(diffdata,99.5)) #diffdata.max()),diffdata.max()
ax.pcolor(bins,bins,diffdata,cmap='bwr',vmin=vmin,vmax=vmax)
ax.set_title('Difference')
ax.plot([0,r2lim],[0,r2lim],color='grey',linestyle='--')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,savedir,'R2_2dhist_V1lab_V1unl_V1unl_%dneurons_%dsessions' % (Nsub,nSessions))

#%% Plot the fraction of output weights (onto target area) that have a positive projection onto firing rate for each rank:
data = frac_pos_weight_out #take the maximum across all kfolds
ymeantoplot = np.nanmean(data,axis=(0,1,3,4)) #mean across sessions and stim and modelfits
# yerrortoplot = np.nanstd(data,axis=(0,1,3,4)) / np.sqrt(nSessions*nmodelfits)
yerrortoplot = np.nanstd(data,axis=(0,1,3,4))# / np.sqrt(nSessions*nmodelfits)

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm),sharey=True,sharex=True)
ax = axes
shaded_error(np.arange(nranks)+1,ymeantoplot,yerrortoplot,ax=ax,color='blue',alpha=0.3)
ax_nticks(ax,4)
ax.axhline(y=0.5,color='grey',linestyle='--')
ax.set_xticks(np.arange(nranks)[::3]+1)
# ax.set_xlim([1,10])
ax.set_xlabel('dimension')
ax.set_ylabel('Frac. pos. projection')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,savedir,'Frac_pos_weightsout_V1lab_V1unl_%dneurons_%dsessions' % (Nsub,nSessions))

#%% Are those dimensions that are enhanced in V1lab, dimensions that are leading to positive projections?
nrankstoplot = 5
r2data = np.nanmean(R2_ranks,axis=(5)) #average across kfolds
r2data = np.diff(r2data[:,:,:,:nrankstoplot+1,:],axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

weightdata = np.nanmean(frac_pos_weight_out[:,:,:nrankstoplot,:,:],axis=(4)) #average across kfolds

fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharey=True,sharex=True)
ax = axes[0]
xdata = (r2data[3] - r2data[1]).flatten()
ydata = weightdata.flatten()
xdata,ydata = filter_sharednan(xdata,ydata)
sns.regplot(x=xdata,y=ydata,marker="o",color='red',ax=ax,scatter_kws={'s':5, 'facecolors': 'black', 'edgecolors': 'None'})
ax.text(0.7,0.1,'r=%1.2f' % np.corrcoef(xdata.flatten(),ydata.flatten())[0,1],color='red',transform=ax.transAxes)
ax.set_ylabel('Frac. pos. projection')
ax.set_xlabel('Diff. R2')
ax.set_title('Lab-unl')
ax = axes[1]
xdata = (r2data[2] - r2data[1]).flatten()
ydata = weightdata.flatten()
xdata,ydata = filter_sharednan(xdata,ydata)
sns.regplot(x=xdata.flatten(),y=ydata.flatten(),marker="o",color='blue',ax=ax,scatter_kws={'s':5, 'facecolors': 'black', 'edgecolors': 'None'})
ax.text(0.7,0.1,'r=%1.2f' % np.corrcoef(xdata.flatten(),ydata.flatten())[0,1],color='blue',transform=ax.transAxes)
ax.set_title('Unl-unl')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,savedir,'Corr_pos_weight_V1lab_V1unl_%dneurons_%dsessions' % (Nsub,nSessions))

##

# WEIGHTS IN 


#%% Plot the fraction of output weights (onto target area) that have a positive projection onto firing rate for each rank:
data = frac_pos_weight_in #take the maximum across all kfolds
ymeantoplot = np.nanmean(data[0],axis=(0,1,3,4)) #mean across sessions and stim and modelfits
# yerrortoplot = np.nanstd(data,axis=(0,1,3,4)) / np.sqrt(nSessions*nmodelfits)
yerrortoplot = np.nanstd(data[0],axis=(0,1,3,4)) / np.sqrt(nSessions)

fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharey=True,sharex=True)
ax = axes[0]
shaded_error(np.arange(nranks)+1,ymeantoplot,yerrortoplot,ax=ax,color='blue',alpha=0.3)
ax_nticks(ax,4)
ax.axhline(y=0.5,color='grey',linestyle='--')
ax.set_xticks(np.arange(nranks)[::3]+1)
# ax.set_xlim([1,10])
ax.set_xlabel('dimension')
ax.set_ylabel('Frac. pos. projection')

ax = axes[1]
data = frac_pos_weight_in #take the maximum across all kfolds
handles = []
for i in range(narealabelpairs):
    ymeantoplot = np.nanmean(data[i+1],axis=(0,1,3,4)) #mean across sessions and stim and modelfits
    yerrortoplot = np.nanstd(data[i+1],axis=(0,1,3,4)) / np.sqrt(nSessions)

    handles.append(shaded_error(np.arange(nranks)+1,ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[i],alpha=0.3))
ax.axhline(y=0.5,color='grey',linestyle='--')
ax.legend(handles,sourcearealabelpairs,frameon=False)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,savedir,'Frac_pos_weightsin_V1lab_V1unl_%dneurons_%dsessions' % (Nsub,nSessions))


#%% Are those dimensions that are enhanced in V1lab, dimensions that are resulting from positive weights of labeled cells?
nrankstoplot = 4
r2data = np.nanmean(R2_ranks,axis=(5)) #average across kfolds
r2data = np.diff(r2data,axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

topfrac = 10 #show the top 10% latents with largest difference in R2 between labeled and unlabeled

weightdata = np.nanmean(weights_in,axis=(6)) #average across kfolds
bins = np.linspace(-1,1,50)
# bins = np.linspace(-1,1,50)
fig,axes    = plt.subplots(1,nrankstoplot,figsize=(nrankstoplot*4*cm,4*cm),sharey=True,sharex=True)
for r in range(nrankstoplot):
    ax          = axes[r]
    xdata       = r2data[3] - r2data[1]
    xdata       = xdata[:,:,r,:]
    xdata = xdata.flatten()

    ydata = weightdata[2]
    ydata = ydata[:,:,:,r,:]
    ydata = np.reshape(ydata,(Nsub,-1))

    assert xdata.shape[0] == ydata.shape[1]
    idx_top = np.where(xdata > np.nanpercentile(xdata,100-topfrac))[0]
    histdata = ydata[:,idx_top].flatten()
    histdata = histdata[~np.isnan(histdata)]
    ax.hist(histdata,bins=bins,color='red',alpha=0.25,density=True)

    idx_top = np.where(xdata < np.nanpercentile(xdata,100-topfrac))[0]
    histdata = ydata[:,idx_top].flatten()
    histdata = histdata[~np.isnan(histdata)]
    ax.hist(histdata,bins=bins,color='blue',alpha=0.25,density=True)

plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
# my_savefig(fig,savedir,'Corr_pos_weight_in_V1lab_V1unl_%dneurons_%dsessions' % (Nsub,nSessions))


#%% Which neurons are well predicted? Is this distribution Gaussian, or skewed?
data = np.nanmean(R2_ranks_neurons,axis=(6)) #average across kfolds
# data = np.nanmean(R2_ranks_neurons,axis=(3,6)) #average across stim and kfolds
bins = np.arange(-0.1,0.6,0.02)
clrs = sns.color_palette('viridis',nranks)
fig,axes = plt.subplots(1,1,figsize=(5*cm,5*cm),sharey=True,sharex=True)
ax = axes
for i in range(3):
    # sns.histplot(data=data[i].flatten(),bins=20,kde=True,ax=ax,color=clrs_arealabelpairs[i])
    sns.histplot(data=data[i+1].flatten(),bins=bins,ax=ax,color=clrs_arealabelpairs[i],fill=False,
                 stat='probability',cumulative=False,element='step')
ax.legend(sourcearealabelpairs,frameon=False)
ax.set_xlabel('Crossvalidated R2 per neuron')
ax.set_yscale('log')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=0)
my_savefig(fig,savedir,'cvR2_perneuron_PM_%dneurons_%dsessions' % (Nsub,nSessions))

#%% Are specific neurons much better predicted by labeled cells? Is this distribution Gaussian, or skewed?
data = np.nanmean(R2_ranks_neurons,axis=(6)) #average across kfolds
# data = np.nanmean(R2_ranks_neurons,axis=(3,6)) #average across kfolds
bins = np.arange(-0.1,0.7,0.02)
clrs = sns.color_palette('viridis',nranks)
fig,axes = plt.subplots(1,1,figsize=(5*cm,5*cm),sharey=True,sharex=True)
ax = axes
sns.histplot(data=(data[2] - data[1]).flatten(),bins=bins,ax=ax,color='grey',fill=False,
                stat='probability',cumulative=False,element='step')
sns.histplot(data=(data[3] - data[1]).flatten(),bins=bins,ax=ax,color='red',fill=False,
                stat='probability',cumulative=False,element='step')
ax.legend(['unl-unl','lab-unl'],frameon=False)
sns.despine(fig=fig,top=True,right=True,offset=0)
ax.set_yscale('log')
ax.set_xlabel('Crossvalidated R2 per neuron')
my_savefig(fig,savedir,'cvR2_perneuron_PM_Labunldiff_%dneurons' % Nsub)

#%% Are specific neurons much better predicted by labeled cells? Is this distribution Gaussian, or skewed?
data = np.nanmean(R2_ranks_neurons,axis=(3,6)) #average across kfolds
fig,axes = plt.subplots(1,1,figsize=(5*cm,5*cm),sharey=True,sharex=True)
ax = axes
xdata = data[2].flatten()
ydata = (data[3] - data[1]).flatten()
xdata,ydata = filter_sharednan(xdata,ydata)
# sns.regplot(x=data[1].flatten(),y=(data[2] - data[1]).flatten(),marker="o",
    # scatter_kws={'s': 1, 'facecolors': 'none', 'edgecolors': 'grey'},color='red',ax=ax)
sns.regplot(x=xdata,y=ydata,marker="o",
    scatter_kws={'s': 1, 'facecolors': 'none', 'edgecolors': 'grey'},color='red',ax=ax)
sns.despine(fig=fig,top=True,right=True,offset=0)
ax.text(0.7,0.9,'r=%1.2f' % np.corrcoef(data[2].flatten(),(data[3] - data[1]).flatten())[0,1],
        transform=ax.transAxes,color='red')
ax.set_ylabel('R2 lab - R2 unl')
ax.set_xlabel('Crossvalidated R2 per neuron')
my_savefig(fig,savedir,'Corr_cvR2_perneuron_PM_Labunldiff_%dneurons_%dsessions' % (Nsub,nSessions))

#%% Show the correlation between R2 predicted by labeled and unlabeled neurons:
data = np.nanmean(R2_ranks_neurons,axis=(3,6)) #average across stimuli/kfolds
data = np.diff(data,axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

# plotlims = [-0.1,my_ceil(np.nanmax([xdata,ydata]),1)]
plotlims = [0,my_ceil(np.nanmax(data[1:]),2)]
bintickres = 0.1
nrankstoshow = 5
clrs = sns.color_palette('Reds_r',nrankstoshow)

fig,axes = plt.subplots(1,nrankstoshow-1,figsize=(nrankstoshow*3.2*cm,4*cm),sharey=True,sharex=True)
for r in range(nrankstoshow-1):
    ax = axes[r]
    xdatatoplot = data[1,:,:,r,:].flatten()
    ydatatoplot = data[2,:,:,r,:].flatten()
    # ax.scatter(xdatatoplot,ydatatoplot,alpha=0.3,color=clrs[r],s=5)
    ax.scatter(xdatatoplot,ydatatoplot,alpha=0.7,color='grey',s=3)

    xdatatoplot = data[1,:,:,r,:].flatten()
    ydatatoplot = data[3,:,:,r,:].flatten()
    ax.scatter(xdatatoplot,ydatatoplot,alpha=0.7,color=clrs[r],s=3)

    ax.set_title('Rank %d' % (r+1))
    if r==0:
        ax.set_xlabel('R2 unl ')
        ax.set_ylabel('R2 unl/lab ')

    ax.legend(['unl-unl','lab-unl'],frameon=False)
    my_legend_strip(ax)
    ax.set_xlim(plotlims)
    ax.set_ylim(plotlims)
    ax.set_xticks(np.arange(plotlims[0],plotlims[1]+bintickres,bintickres))
    ax.set_yticks(np.arange(plotlims[0],plotlims[1]+bintickres,bintickres))
    # ax_nticks(ax[r],5)
    ax.plot(plotlims,plotlims,'--',linewidth=1,color='grey')
    xdatatoplot = xdatatoplot[~np.isnan(ydatatoplot)]
    ydatatoplot = ydatatoplot[~np.isnan(ydatatoplot)]
    print(np.corrcoef(xdatatoplot,ydatatoplot)[0,1])

plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'RRR_unique_cvR2_V1lab_V1unl_PMunl_%dneurons' % Nsub)

#%% 
 #####  ####### #     # ######   #####  #######    #     # #######  #####  #     #    #    #     # ###  #####  #     #  #####  
#     # #     # #     # #     # #     # #          ##   ## #       #     # #     #   # #   ##    #  #  #     # ##   ## #     # 
#       #     # #     # #     # #       #          # # # # #       #       #     #  #   #  # #   #  #  #       # # # # #       
 #####  #     # #     # ######  #       #####      #  #  # #####   #       ####### #     # #  #  #  #   #####  #  #  #  #####  
      # #     # #     # #   #   #       #          #     # #       #       #     # ####### #   # #  #        # #     #       # 
#     # #     # #     # #    #  #     # #          #     # #       #     # #     # #     # #    ##  #  #     # #     # #     # 
 #####  #######  #####  #     #  #####  #######    #     # #######  #####  #     # #     # #     # ###  #####  #     #  #####  

#%% How much of the variance in the source area is aligned with the predictive subspace:
fig, axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
# ax.plot(range(nranks),np.nanmean(R2_ranks[0],axis=(0,1,3,4)),label='All neurons',color='grey')
ax.plot(np.nanmean(R2_sourcealigned[1],axis=(0,1,3,4)),label=sourcearealabelpairs[0],color=clrs_arealabelpairs[0])
ax.plot(np.nanmean(R2_sourcealigned[2],axis=(0,1,3,4)),label=sourcearealabelpairs[1],color=clrs_arealabelpairs[1])
ax.plot(np.nanmean(R2_sourcealigned[3],axis=(0,1,3,4)),label=sourcearealabelpairs[2],color=clrs_arealabelpairs[2])
leg = ax.legend(frameon=False,fontsize=6)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_title('Source variance along\npredictive dimensions')
ax.set_xticks(np.arange(nranks)[::3]+1)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,savedir,'RRR_source_aligned_R2_V1lab_V1unl_PMunl_%dneurons_%dsessions' % (Nsub,nSessions))


#%% Are subpopulations that are more predictive lower dimensional?
fig, axes = plt.subplots(1,1,figsize=(5*cm,4*cm))
rank = 4
ax = axes
xdata = source_dim[0]
ydata = R2_ranks[0,:,:,rank,:,:].mean(axis=3)
xdata,ydata = filter_sharednan(xdata,ydata)
sns.regplot(x=xdata.flatten(),y=ydata.flatten(),marker="o",color='blue',ax=ax,scatter_kws={'s':5, 'facecolors': 'black', 'edgecolors': 'None'})
ax.set_xlabel(dim_method)
ax.set_ylabel('R2')
ax.text(0.1,0.05,'r=%1.2f' % np.corrcoef(xdata.flatten(),ydata.flatten())[0,1],color='blue',transform=ax.transAxes)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,savedir,'RRR_dimensionality_%s_PMprediction_%dsessions' % (dim_method,nSessions))

#%% Are subpopulations that are more predictive lower dimensional?
fig, axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
for i in range(3):
    meantoplot = np.nanmean(source_dim[i+1])
    errortoplot = np.nanstd(source_dim[i+1])/np.sqrt(nSessions)
    ax.errorbar(x=i,y=meantoplot,yerr=errortoplot,color=clrs_arealabelpairs[i],marker='o',linestyle='None')
ax.set_xticks([0,1,2],labels=sourcearealabelpairs)
ax.set_xlabel('Source area')
ax.set_ylabel(dim_method)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,savedir,'RRR_dimensionality_%s_subpopulations_%dsessions' % (dim_method,nSessions))

#%% Is the lower dimensionality of labeled population responsible for the increased predictive accuracy?
fig, axes = plt.subplots(1,1,figsize=(5*cm,4*cm))
rank = 4
ax = axes
for i in range(3):
    xdata = source_dim[i+1]
    ydata = R2_ranks[i+1,:,:,rank,:,:].mean(axis=3)
    sns.regplot(x=xdata.flatten(),y=ydata.flatten(),marker="o",color=clrs_arealabelpairs[i],
                ax=ax,scatter_kws={'s':5, 'facecolors': clrs_arealabelpairs[i], 'edgecolors': 'None'})
# ax.legend(sourcearealabelpairs,frameon=False)
ax.set_xlabel(dim_method)
ax.set_ylabel('R2')
ax_nticks(ax,3)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,savedir,'RRR_dimensionality_%s_subpopulations_PMprediction_%dsessions' % (dim_method,nSessions))

numobs = len(source_dim[1].flatten())
df = pd.DataFrame({'dim': source_dim[2].flatten(),
                   'R2': R2_ranks[np.ix_([2],range(nSessions),range(nStim),[rank],range(nmodelfits),range(params['kfold']))].mean(axis=5).flatten(),
                   'idx': np.arange(numobs),
                   'source': np.repeat(sourcearealabelpairs[1],numobs)})
df = pd.concat((df,pd.DataFrame({'dim': source_dim[3].flatten(),
                   'R2': R2_ranks[np.ix_([3],range(nSessions),range(nStim),[rank],range(nmodelfits),range(params['kfold']))].mean(axis=5).flatten(),
                   'idx': np.arange(numobs),
                   'source': np.repeat(sourcearealabelpairs[2],numobs)})))
df.dropna().reset_index()
# from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
import statsmodels.api as sm

model = ols('R2 ~ source', data=df).fit()
#summarise model
model.summary()
# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model = ols('R2 ~ source + dim', data=df).fit()
#summarise model
model.summary()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

#%% Do PCA of populations to see spectrum:
sourcearealabelpairs = ['V1unl','V1lab']

clrs_arealabelpairs = get_clr_area_labeled(sourcearealabelpairs)
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 20
nmodelfits          = 5
# dim_method          = 'parallel_analysis'
# dim_method          = 'participation_ratio'
# dim_method          = 'pca_shuffle'

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

source_spectrum    = np.full((2,2,Nsub,nSessions,nStim,nmodelfits),np.nan)
source_dim          = np.full((2,nSessions,nStim,nmodelfits),np.nan)

from sklearn.decomposition import FactorAnalysis as FA
from sklearn.decomposition import PCA as PCA
# PCA = PCA(n_components=Nsub)
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
    
    if len(idx_areax1)<Nsub or len(idx_areax2)<Nsub: #skip exec if not enough neurons in one of the populations
        continue

    for imf in tqdm(range(nmodelfits),total=nmodelfits,desc='Fitting RRR model for session %d/%d' % (ises+1,nSessions)):
        idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
        idx_areax2_sub       = np.random.choice(idx_areax2,Nsub,replace=False)

        # idx_areax1_sub       = idx_areax1
        # idx_areax2_sub       = idx_areax2

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
       
            X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
            X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]

            # reshape to neurons x time points
            X1                  = X1.reshape(len(idx_areax1_sub),-1).T
            X2                  = X2.reshape(len(idx_areax2_sub),-1).T

            X1                  = zscore(X1,axis=0)
            X2                  = zscore(X2,axis=0)

            # FA = FA(n_components=10)
            # where 𝐲 is a q-dimensional vector containing the observed residuals at a given time point, L is the 𝑞×𝑚 loading matrix that defines the relationship between the m-dimensional (𝑚<𝑞) latent variable 𝐳 and 𝐲, μ is a q-dimensional vector and Ψ is a 𝑞×𝑞 diagonal matrix. We estimated the dimensionality of the latent variable 𝐳 in two steps: (1) we found the number of dimensions 𝑚𝑝⁢𝑒⁢𝑎⁢𝑘 that maximized the cross-validated log-likelihood of the observed residuals; (2) we fitted a FA model with 𝑚𝑝⁢𝑒⁢𝑎⁢𝑘 dimensions and chose m, using the eigenvalue decomposition, as the smallest dimensionality that captured 95% of the variance in the shared covariance matrix 𝐿⁢𝐿𝑇. This procedure provides more robust estimates of the FA model dimensionality (Williamson et al., 2016).
            # FA.fit(X1)
            # FA.

            # for i,data in enumerate([X1,X2]):
            #     source_dim[i,ises,istim,imf] = estimate_dimensionality(data,method=dim_method)

            for i,data in enumerate([X1,X2]):
                X_shuffled = my_shuffle(data, method='random')
                pca_original = PCA(n_components=Nsub).fit(data)
                pca_shuffled = PCA(n_components=Nsub).fit(X_shuffled)
                source_spectrum[i,0,:,ises,istim,imf] = pca_original.explained_variance_ / np.sum(pca_original.explained_variance_)
                source_spectrum[i,1,:,ises,istim,imf] = pca_shuffled.explained_variance_ / np.sum(pca_shuffled.explained_variance_)

#%% Is the lower dimensionality of labeled population responsible for the increased predictive accuracy?
fig, axes = plt.subplots(1,1,figsize=(5*cm,4*cm))
ax = axes
handles = []
for i in range(2):
    ymeantoplot = np.nanmean(source_spectrum[i][0],axis=(1,2,3))
    yerrortoplot = np.nanstd(source_spectrum[i][0],axis=(1,2,3)) / np.sqrt(Nsub*nSessions*nmodelfits)
    handles.append(shaded_error(x=np.arange(Nsub),y=ymeantoplot,yerror=yerrortoplot,ax=ax,color=clrs_arealabelpairs[i]))

for i in range(2):
    ymeantoplot = np.nanmean(source_spectrum[i][1],axis=(1,2,3))
    yerrortoplot = np.nanstd(source_spectrum[i][1],axis=(1,2,3)) / np.sqrt(Nsub*nSessions*nmodelfits)
    handles.append(shaded_error(x=np.arange(Nsub),y=ymeantoplot,yerror=yerrortoplot,ax=ax,color='grey'))

ax.legend(handles,('Unl','Lab','Shuf'),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Component')
ax.set_ylabel('Fraction of variance')
ax_nticks(ax,4)
# ax.set_yscale('log')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,savedir,'PCA_Spectrum_V1unl_V1lab_%d_sessions' % (nSessions))

#%% Is the lower dimensionality of labeled population responsible for the increased predictive accuracy?
fig, axes = plt.subplots(1,1,figsize=(5*cm,4*cm))
ax = axes
handles = []
for i in range(2):
    ymeantoplot = np.nanmean(source_spectrum[i][0],axis=(1,2,3)) - np.nanmean(source_spectrum[i][1],axis=(1,2,3))
    yerrortoplot = np.nanstd(source_spectrum[i][0],axis=(1,2,3)) / np.sqrt(Nsub*nSessions*nmodelfits)
    handles.append(shaded_error(x=np.arange(Nsub),y=ymeantoplot,yerror=yerrortoplot,ax=ax,color=clrs_arealabelpairs[i]))

ax.axhline(y=0,color='grey',linestyle='--')
ax.legend(handles,('Unl','Lab','Shuf'),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Component')
ax.set_ylabel('Fraction of variance')
ax_nticks(ax,4)
# ax.set_yscale('log')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
# my_savefig(fig,savedir,'PCA_Spectrum_V1unl_V1lab_shuffle_%d_sessions' % (nSessions))


#%% 








#%% Do RRR of V1 and PM labeled and unlabeled neurons simultaneously
# sourcearealabelpairs = ['V1unl','V1lab']
# targetarealabelpair = 'PMunl'
# controlarealabelpairs = ['V1unl','V1unl']

# clrs_arealabelpairs = get_clr_area_labeled(sourcearealabelpairs)
# narealabelpairs     = len(sourcearealabelpairs)

# Nsub                = 25
# # nranks              = 10
# nranks              = 20 #number of ranks of RRR to be evaluated
# nmodelfits          = 10
# R2_cv               = np.full((3,nSessions,nStim),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
# optim_rank          = np.full((3,nSessions,nStim),np.nan)
# R2_ranks            = np.full((3,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
# R2_ranks_neurons    = np.full((3,Nsub*2,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
# source_dim          = np.full((3,nSessions,nStim,nmodelfits),np.nan)
# R2_sourcealigned    = np.full((3,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
# R2_ranks_ctrl       = np.full((3,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
# R2_ranks_neurons_ctrl = np.full((3,Nsub*2,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different populations'):
#     if params['filter_nearby']:
#         idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
#     else:
#         idx_nearby = np.ones(len(ses.celldata),dtype=bool)

#     idx_areax1      = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[0],
#                                 ses.celldata['noise_level']<params['maxnoiselevel'],
#                                 idx_nearby),axis=0))[0]
#     idx_areax2      = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[1],
#                                 ses.celldata['noise_level']<params['maxnoiselevel'],
#                                 idx_nearby),axis=0))[0]
#     idx_areay       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpair,
#                                             ses.celldata['noise_level']<params['maxnoiselevel'],
#                                             idx_nearby),axis=0))[0]

#     if len(idx_areax1)<Nsub or len(idx_areax2)<Nsub or len(idx_areay)<2*Nsub: #skip exec if not enough neurons in one of the populations
#         continue

#     for imf in range(nmodelfits):
#         idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
#         idx_areax2_sub       = np.random.choice(idx_areax2,Nsub,replace=False)
#         idx_areay_sub        = np.random.choice(idx_areay,Nsub*2,replace=False)

#         for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
#             idx_T               = ses.trialdata['stimCond']==stim
       
#             X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
#             X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
#             Y                   = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

#             # reshape to neurons x time points
#             X1                  = X1.reshape(len(idx_areax1_sub),-1).T
#             X2                  = X2.reshape(len(idx_areax2_sub),-1).T
#             Y                   = Y.reshape(len(idx_areay_sub),-1).T

#             X1                  = zscore(X1,axis=0)
#             X2                  = zscore(X2,axis=0)
#             Y                   = zscore(Y,axis=0)

#             X                   = np.concatenate((X1,X2),axis=1)

#             source_dim[0,ises,istim,imf] = estimate_dimensionality(X,method='participation_ratio')
#             source_dim[1,ises,istim,imf] = estimate_dimensionality(X1,method='participation_ratio')
#             source_dim[2,ises,istim,imf] = estimate_dimensionality(X2,method='participation_ratio')

#             # OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS    
#             R2_kfold    = np.zeros((params['kfold']))
#             kf          = KFold(n_splits=params['kfold'],shuffle=True)
#             for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
#                 X_train, X_test     = X[idx_train], X[idx_test]
#                 Y_train, Y_test     = Y[idx_train], Y[idx_test]

#                 #RRR X to Y
#                 B_hat_train         = LM(Y_train,X_train, lam=params['lam'])
#                 Y_hat_train         = X_train @ B_hat_train

#                 # decomposing and low rank approximation of Y_hat
#                 U, s, V = svds(Y_hat_train,k=nranks,which='LM')
#                 U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

#                 for r in range(nranks):
#                     B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
#                     Y_hat_test_rr   = X_test @ B_rrr

#                     R2_ranks[0,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)
#                     R2_ranks_neurons[0,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')
                    
#                     X_test_1 = copy.deepcopy(X_test)
#                     X_test_1[:,Nsub:] = 0
#                     Y_hat_test_rr   = X_test_1 @ B_rrr

#                     R2_ranks[1,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)
#                     R2_ranks_neurons[1,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')

#                     X_test_2 = copy.deepcopy(X_test)
#                     X_test_2[:,:Nsub] = 0
#                     Y_hat_test_rr   = X_test_2 @ B_rrr

#                     R2_ranks[2,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)
#                     R2_ranks_neurons[2,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')




#%% LAB and UNL in same population


#%% Parameters for RRR for size-matched populations of V1 and PM labeled and unlabeled neurons
lam                 = 0
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 10 #number of times new neurons are resampled - many for final run
params['kfold']               = 5
maxnoiselevel       = 20
nStim               = 16

# idx_resp            = np.where((t_axis>=0) & (t_axis<=1))[0]
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
ntimebins           = len(idx_resp)
minsampleneurons    = 10
params['filter_nearby']       = True

dataversions        = np.array(['original','behavout','neuralout'])
dataversions        = np.array(['original','behavout'])
# dataversions        = np.array(['original','','neuralout'])
# dataversions        = np.array(['original'])
# dataversions        = np.array(['neuralout'])
nversions           = len(dataversions) #0: original,1:behavior out, 2:neural out
rank_behavout       = 5
rank_neuralout      = 5
fixed_rank          = None

#%%

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Deleting data'):
    if hasattr(ses,'tensor_neuralout'):
        delattr(sessions[ises],'tensor_neuralout')
    if hasattr(ses,'tensor_behavout'):
        delattr(sessions[ises],'tensor_behavout')


#%% Do RRR of V1 and PM labeled and unlabeled neurons jointly

sourcearealabelpairs = ['V1unl','V1lab']
targetarealabelpair = 'PMunl'

clrs_arealabelpairs = get_clr_area_labeled(sourcearealabelpairs)
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 25
nranks              = 5
nmodelfits          = 20
R2_neuron           = np.full((narealabelpairs,Nsub,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)
# R2_cv               = np.full((narealabelpairs,3,nSessions,nStim),np.nan)
# optim_rank          = np.full((narealabelpairs,3,nSessions,nStim),np.nan)
# R2_ranks            = np.full((narealabelpairs,3,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different populations'):
    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(ses,radius=30)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    idx_areax1          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[0],
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    idx_areax2          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[1],
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
    idx_areay       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpair,
                                            ses.celldata['noise_level']<maxnoiselevel,	
                                            idx_nearby),axis=0))[0]

    if len(idx_areax1)<Nsub or len(idx_areax2)<Nsub or len(idx_areay)<Nsub: #skip exec if not enough neurons in one of the populations
        continue

    for imf in range(nmodelfits):
        idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
        idx_areax2_sub       = np.random.choice(idx_areax2,Nsub,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,Nsub,replace=False)

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim

            X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
            X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

            # reshape to neurons x time points
            X1                  = X1.reshape(len(idx_areax1_sub),-1).T
            X2                  = X2.reshape(len(idx_areax2_sub),-1).T
            Y                   = Y.reshape(len(idx_areay_sub),-1).T

            X1  = zscore(X1,axis=0)
            X2  = zscore(X2,axis=0)
            Y   = zscore(Y,axis=0)

            # OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS    
            R2_kfold    = np.zeros((params['kfold']))
            kf          = KFold(n_splits=params['kfold'],shuffle=True)
            for ikf, (idx_train, idx_test) in enumerate(kf.split(X1)):
                X1_train, X1_test     = X1[idx_train], X1[idx_test]
                X2_train, X2_test     = X2[idx_train], X2[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                #RRR X1 to Y
                B_hat_train         = LM(Y_train,X1_train, lam=params['lam'])
                Y_hat_train         = X1_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                for r in range(nranks):
                    B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                    Y_hat_test_rr   = X1_test @ B_rrr
                    R2_neuron[0,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')

                #RRR X2 to Y
                B_hat_train         = LM(Y_train,X2_train, lam=params['lam'])
                Y_hat_train         = X2_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                for r in range(nranks):
                    B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                    Y_hat_test_rr   = X2_test @ B_rrr
                    R2_neuron[1,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_test_rr, multioutput='raw_values')

# print('Population size: %d (%d-%d)' % (np.mean(sampleN[sampleN>minsampleneurons]),np.min(sampleN[sampleN>minsampleneurons]),np.max(sampleN[sampleN>minsampleneurons])))

#%% Show the correlation between R2 predicted by labeled and unlabeled neurons:
examplerank = 0
# xdata = R2_neuron[0,:,:,:,examplerank,:,:].flatten()
# ydata = R2_neuron[1,:,:,:,examplerank,:,:].flatten()
# xdata = np.nanmean(R2_neuron[0,:,:,:,:,:,:],axis=2) #average across stimuli
# ydata = np.nanmean(R2_neuron[1,:,:,:,:,:,:],axis=2)

xdata = np.nanmean(R2_neuron[0,:,:,:,:,:,:],axis=(2,4)) #average across stimuli
ydata = np.nanmean(R2_neuron[1,:,:,:,:,:,:],axis=(2,4))

xdata = np.diff(xdata,axis=2) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
ydata = np.diff(ydata,axis=2)

plotlims = [-0.1,my_ceil(np.nanmax([xdata,ydata]),1)]

clrs = sns.color_palette('viridis',nranks)
fig,ax = plt.subplots(1,nranks-1,figsize=(nranks*3.5*cm,4*cm),sharey=True,sharex=True)
for r in range(nranks-1):
    xdatatoplot = xdata[:,:,r,:].flatten()
    ydatatoplot = ydata[:,:,r,:].flatten()
    # xdatatoplot = xdata[:,:,r,:,:].flatten()
    # ydatatoplot = ydata[:,:,r,:,:].flatten()
    ax[r].set_title('Rank %d' % (r+1))
    if r==0:
        ax[r].set_xlabel('R2 unl ')
        ax[r].set_ylabel('R2 lab ')
    ax[r].scatter(xdatatoplot,ydatatoplot,alpha=0.3,color=clrs[r],s=5)
    ax[r].set_xlim(plotlims)
    ax[r].set_ylim(plotlims)
    ax[r].set_xticks(np.arange(plotlims[0],plotlims[1]+0.01,0.1))
    ax[r].set_yticks(np.arange(plotlims[0],plotlims[1]+0.01,0.1))
    # ax_nticks(ax[r],5)
    ax[r].plot(plotlims,plotlims,'k--',linewidth=1,color='grey')
    xdatatoplot = xdatatoplot[~np.isnan(ydatatoplot)]
    ydatatoplot = ydatatoplot[~np.isnan(ydatatoplot)]
    print(np.corrcoef(xdatatoplot,ydatatoplot)[0,1])
plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'RRR_unique_cvR2_V1lab_V1unl_PMunl_%dneurons' % Nsub)

#%%
# my_savefig(fig,savedir,'RRR_cvR2_V1unl_V

# #%% Show the correlation between R2 predicted by labeled and unlabeled neurons:

# # xdata = np.nanmean(R2_neuron[0,:,:,:,:,rank_behavout-1,:,:],axis=(0,4,5,6)).flatten()
# # ydata = np.nanmean(R2_neuron[1,:,:,:,:,rank_behavout-1,:,:],axis=(0,4,5,6)).flatten()
# examplerank = 0
# # xdata = R2_neuron[0,:,:,:,examplerank,:,:].flatten()
# # ydata = R2_neuron[1,:,:,:,examplerank,:,:].flatten()

# xdata = R2_neuron[0,:,:,:,:,:,:]
# ydata = R2_neuron[1,:,:,:,:,:,:]
# clrs = sns.color_palette('viridis',nranks)
# fig,ax = plt.subplots(1,nranks,figsize=(nranks*4,4),sharey=True)
# for r in range(nranks):
#     xdatatoplot = xdata[:,:,:,r,:,:].flatten()
#     ydatatoplot = ydata[:,:,:,r,:,:].flatten()
#     ax[r].set_title('Rank %d' % r)
#     ax[r].set_xlabel('R2 unlabeled neurons')
#     ax[r].set_ylabel('R2 labeled neurons')
#     ax[r].scatter(xdatatoplot,ydatatoplot,alpha=0.1,color=clrs[r],s=5)
#     ax[r].set_xlim([-0.1,0.6])
#     ax[r].set_ylim([-0.1,0.6])
#     ax[r].plot([-0.1,0.6],[-0.1,0.6],'k--')
#     print(np.corrcoef(xdatatoplot,ydatatoplot)[0,1])
# plt.tight_layout()
# sns.despine(fig=fig,trim=True,top=True,right=True)
# # my_savefig(fig,savedir,'RRR_cvR2_V1unl_V

#%% 
sourcearealabelpairs = ['V1unl','V1lab']
targetarealabelpair = 'PMunl'

clrs_arealabelpairs = get_clr_area_labeled(sourcearealabelpairs)
narealabelpairs     = len(sourcearealabelpairs)

#%% Show latents: 
ises = 1
stim = 5
Nsub = 50
rank = 3
idx_resp            = np.where((t_axis>=0) & (t_axis<=2))[0]

ses = sessions[ises]

idx_areax1          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[0],
                            ses.celldata['noise_level']<maxnoiselevel,	
                            ),axis=0))[0]
idx_areax2          = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[1],
                            ses.celldata['noise_level']<maxnoiselevel,	
                            ),axis=0))[0]
idx_areay       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpair,
                                        ses.celldata['noise_level']<maxnoiselevel,	
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

#%% Little bit of smoothing:
kernel_size = 3
kernel = np.ones(kernel_size) / kernel_size
for r in range(rank):
    Z_1[:,r] = np.convolve(Z_1[:,r], kernel, mode='same')
    Z_2[:,r] = np.convolve(Z_2[:,r], kernel, mode='same')

#%% Plot excerpt of latent dimensions over time
clrs = sns.color_palette('viridis',rank)
fig,axes = plt.subplots(rank,1,figsize=(9*cm,rank*3*cm),sharex=True)

starttimepoint_idx  = 100
ntimebins           = 250
idx_K = np.arange(starttimepoint_idx,starttimepoint_idx+ntimebins)
for r in range(rank):
    ax = axes[r]
    ax.plot(Z_1[idx_K,r],color=clrs_arealabelpairs[0],alpha=1,label='V1 unlabeled')
    ax.plot(Z_2[idx_K,r],color=clrs_arealabelpairs[1],alpha=1,label='V1 labeled')
    ax.set_ylabel('Latent dim %d' % (r+1))
    if r==0:
        ax.legend(frameon=False)
        my_legend_strip(ax)
    ax.set_title('Latent %d' % (r+1))
    ax.axis('off')
    if r==0:
        ax.add_artist(AnchoredSizeBar(ax.transData, 10*ses.sessiondata['fs'][0],
                        "10 Sec", loc=4, frameon=False))
plt.tight_layout()
my_savefig(fig,savedir,'Example_Latents_Joint_V1lab_V1unl_PMunl_%dneurons' % Nsub)


#%% 
######  ######  ######     #          #    ######     #     #  #####     #     # #     # #       
#     # #     # #     #    #         # #   #     #    #     # #     #    #     # ##    # #       
#     # #     # #     #    #        #   #  #     #    #     # #          #     # # #   # #       
######  ######  ######     #       #     # ######     #     #  #####     #     # #  #  # #       
#   #   #   #   #   #      #       ####### #     #     #   #        #    #     # #   # # #       
#    #  #    #  #    #     #       #     # #     #      # #   #     #    #     # #    ## #       
#     # #     # #     #    ####### #     # ######        #     #####      #####  #     # ####### 

#%%
# celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

# sns.histplot(data=celldata,x='noise_level',hue='arealabel',stat='probability',common_norm=False,cumulative=True,
#              bins=np.linspace(0,100,500),element='step',fill=False)

#%% Parameters for RRR for size-matched populations of V1 and PM labeled and unlabeled neurons
lam                 = 0
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 10 #number of times new neurons are resampled - many for final run
params['kfold']               = 5
maxnoiselevel       = 20
nStim               = 16

# idx_resp            = np.where((t_axis>=0) & (t_axis<=1))[0]
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
# idx_resp            = np.where((t_axis>=0.5) & (t_axis<=1.5))[0]
ntimebins           = len(idx_resp)
minsampleneurons    = 10
params['filter_nearby']       = True

dataversions        = np.array(['original','behavout','neuralout'])
dataversions        = np.array(['original','behavout'])
# dataversions        = np.array(['original','','neuralout'])
# dataversions        = np.array(['original'])
# dataversions        = np.array(['neuralout'])
nversions           = len(dataversions) #0: original,1:behavior out, 2:neural out
rank_behavout       = 5
rank_neuralout      = 5
fixed_rank          = None

#%%

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Deleting data'):
    if hasattr(ses,'tensor_neuralout'):
        delattr(sessions[ises],'tensor_neuralout')
    if hasattr(ses,'tensor_behavout'):
        delattr(sessions[ises],'tensor_behavout')

#%% Do RRR of V1 and PM labeled and unlabeled neurons

# Main analysis: identify pairs of populations and do crossvalidated RRR between these. 
# Compute crossvalidated EV for each rank, identify optimal rank, and store ev at this rank
# Do this for many pairs and subsequent analyses focus on different comparissons
# Additionally, do this for different versions of the data (original, behavior out, neural out)
# Behavior out: remove activity explained by behavioral variables
# Neural out: remove activity explained by simultaneous AL + RSP data 
# (comparisons with AL and RSP in arealabelpairs with neural_out thus don't make sense)

arealabelpairs  = [
                    'V1unl-PMunl', #Feedforward pairs
                    'V1lab-PMunl',

                    # 'PMunl-V1unl', #Feedback pairs
                    # 'PMlab-V1unl',

                    # # Comparing looped pairs:
                    # 'V1unl-PMlab',
                    # 'V1lab-PMlab',
                    # 'PMunl-V1lab',
                    # 'PMlab-V1lab',

                    # #Feedforward selectivity
                    # 'V1unl-ALunl', #to AL
                    # 'V1lab-ALunl',
                    # 'V1unl-RSPunl', #to RSP
                    # 'V1lab-RSPunl',

                    # #Feedback selectivity
                    # 'PMunl-ALunl', #to AL
                    # 'PMlab-ALunl',
                    # 'PMunl-RSPunl',#to RSP
                    # 'PMlab-RSPunl'
                    ]

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

R2_cv               = np.full((narealabelpairs,3,nSessions,nStim),np.nan)
optim_rank          = np.full((narealabelpairs,3,nSessions,nStim),np.nan)
R2_ranks            = np.full((narealabelpairs,3,nSessions,nStim,nranks,nmodelfits,params['kfold']),np.nan)

sampleN             = np.zeros((nSessions))

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different populations'):
for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting RRR model for different populations'):
    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(ses,radius=30)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    #take the smallest sample size
    allpops             = np.array([i.split('-') for i in arealabelpairs])
    sourcepops,targetpops          = allpops[:,0],allpops[:,1]
   
    nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
                                                 ses.celldata['noise_level']<maxnoiselevel,
                                                 idx_nearby),axis=0)) for i in sourcepops],
                                  )
    sampleN[ises] = nsampleneurons
    nsampleneurons = 50

    if not hasattr(ses,'tensor_behavout') and 'behavout' in dataversions: 
        sessions[ises].tensor_behavout = copy.copy(sessions[ises].tensor)
        #Get behavioral matrix:
        B                   = np.concatenate((sessions[ises].tensor_vid,
                                sessions[ises].tensor_run),axis=0)
        
        for area in ['V1','PM']:
            idx_N    = np.where(np.all((ses.celldata['roi_name']==area,
                                        idx_nearby,
                                        ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]

            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations
                idx_T               = sessions[ises].trialdata['stimCond']==stim

                Bstim                   = B[:,idx_T,:].reshape(np.shape(B)[0],-1).T
                Bstim                   = zscore(Bstim,axis=0,nan_policy='omit')
                Bstim                   = Bstim[:,~np.all(np.isnan(Bstim),axis=0)]

                tempdata            = sessions[ises].tensor[np.ix_(idx_N,idx_T,np.arange(len(t_axis)))]
                N,K,T = np.shape(tempdata)
                Y_r = np.reshape(tempdata,(N,K*T),order='C').T
                Y_orig,Y_hat,Y_out  = regress_out_cv(X=Bstim,Y=Y_r,rank=np.min([rank_behavout,len(idx_N)-1]),
                                                    lam=params['lam'],kfold=5)
                # print(area,EV(Y_orig,Y_hat))
                sessions[ises].tensor_behavout[np.ix_(idx_N,idx_T,np.arange(len(t_axis)))] = np.reshape(Y_out.T,(N,K,T),order='C')

    # plt.imshow(np.nanmean(sessions[ises].tensor,axis=2),aspect='auto',vmin=-0.1,vmax=0.1)
    # plt.imshow(np.nanmean(sessions[ises].tensor_behavout,axis=2),aspect='auto',vmin=-0.1,vmax=0.1)

    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                idx_nearby),axis=0))[0]

        if np.any(np.intersect1d(idx_areax,idx_areay)): #if interactions within one population:
            if not np.array_equal(idx_areax, idx_areay): 
                print('Arealabelpair %s has partly overlapping neurons'%arealabelpair)
            idx_areax, idx_areay = np.array_split(np.random.permutation(idx_areax), 2)

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons: #skip exec if not enough neurons in one of the populations
            continue
        
        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim

            # idx_T         = np.all((
            #                         ses.respmat_videome < params['maxvideome'],
            #                         ses.respmat_runspeed < params['maxrunspeed'],
            #                         ses.trialdata['stimCond']==stim),axis=0)
            
            if 'original' in dataversions:
                X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
                Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
                
                # reshape to neurons x time points
                X                   = X.reshape(len(idx_areax),-1).T
                Y                   = Y.reshape(len(idx_areay),-1).T

                #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
                R2_cv[iapl,0,ises,istim],optim_rank[iapl,0,ises,istim],R2_ranks[iapl,0,ises,istim,:,:,:]  = RRR_wrapper(Y, X, 
                                nN=nsampleneurons,nK=None,lam=params['lam'],nranks=nranks,kfold=params['kfold'],nmodelfits=nmodelfits,fixed_rank=fixed_rank)

            if 'behavout' in dataversions:
                X                   = sessions[ises].tensor_behavout[np.ix_(idx_areax,idx_T,idx_resp)]
                Y                   = sessions[ises].tensor_behavout[np.ix_(idx_areay,idx_T,idx_resp)]
                
                # reshape to neurons x time points
                X                   = X.reshape(len(idx_areax),-1).T
                Y                   = Y.reshape(len(idx_areay),-1).T

                #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
                R2_cv[iapl,1,ises,istim],optim_rank[iapl,1,ises,istim],R2_ranks[iapl,1,ises,istim,:,:,:]  = RRR_wrapper(Y, X, 
                                nN=nsampleneurons,nK=None,lam=params['lam'],nranks=nranks,kfold=params['kfold'],nmodelfits=nmodelfits,fixed_rank=fixed_rank)

            if 'neuralout' in dataversions and hasattr(ses,'tensor_neuralout') and iapl<8:
                X                   = sessions[ises].tensor_neuralout[np.ix_(idx_areax,idx_T,idx_resp)]
                Y                   = sessions[ises].tensor_neuralout[np.ix_(idx_areay,idx_T,idx_resp)]
                
                # reshape to neurons x time points
                X                   = X.reshape(len(idx_areax),-1).T
                Y                   = Y.reshape(len(idx_areay),-1).T

                #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
                R2_cv[iapl,2,ises,istim],optim_rank[iapl,2,ises,istim],R2_ranks[iapl,2,ises,istim,:,:,:]  = RRR_wrapper(Y, X, 
                                nN=nsampleneurons,nK=None,lam=params['lam'],nranks=nranks,kfold=params['kfold'],nmodelfits=nmodelfits,fixed_rank=fixed_rank)

print('Population size: %d (%d-%d)' % (np.mean(sampleN[sampleN>minsampleneurons]),np.min(sampleN[sampleN>minsampleneurons]),np.max(sampleN[sampleN>minsampleneurons])))

#%% 

#          #    ######     #     #  #####     #     # #     # #       
#         # #   #     #    #     # #     #    #     # ##    # #       
#        #   #  #     #    #     # #          #     # # #   # #       
#       #     # ######     #     #  #####     #     # #  #  # #       
#       ####### #     #     #   #        #    #     # #   # # #       
#       #     # #     #      # #   #     #    #     # #    ## #       
####### #     # ######        #     #####      #####  #     # ####### 


#%% Show figure for each of the arealabelpairs and each of the dataversions
for iversion, version in enumerate(dataversions):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    optim_rank_data         = np.reshape(optim_rank[:,iversion,:,:],(narealabelpairs,nSessions*nStim))
    R2_ranks_data           = np.reshape(R2_ranks[:,iversion,:,:,:,:],(narealabelpairs,nSessions*nStim,nranks,nmodelfits,params['kfold']))
    # R2_data                 = np.nanmean(R2_cv[:,iversion,:,:],axis=2)
    # optim_rank_data                 = np.nanmean(optim_rank[:,iversion,:,:],axis=2)
    if np.any(~np.isnan(R2_data)):
        for idx in np.array([[0,1]]):
        # for idx in np.array([[0,1],[2,3]]):
            clrs        = ['grey',get_clr_area_labeled([arealabelpairs[idx[1]].split('-')[0]])]
            fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(arealabelpairs)[idx],clrs)
            # my_savefig(fig,savedir,'RRR_cvR2_%s_%s_%dsessions' % (arealabelpairs[idx[1]],version,nSessions))

#%% Reshape data to stretch stim x sessions
nversions = 3
R2_cv_2         = np.reshape(R2_cv,(narealabelpairs,nversions,nSessions*nStim))
optim_rank_2    = np.reshape(optim_rank,(narealabelpairs,nversions,nSessions*nStim))
R2_ranks_2      = np.reshape(R2_ranks,(narealabelpairs,nversions,nSessions*nStim,nranks,nmodelfits,params['kfold']))

#%% Define the ratio of R2 between V1PM and V1ND
ratiodata_FF       = (R2_cv_2[1] / R2_cv_2[0])*100-100 #
ratiodata_FB       = (R2_cv_2[3] / R2_cv_2[2])*100-100

#%% Plot the R2 for each of the arealabelpairs and each of the dataversions
#Residual variance explained goes down with behavior or brainwide activity regressed out: 
nversions = 3
fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5*cm,4*cm))
ax = axes[0]
# ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[0],axis=1),yerr=np.nanstd(R2_cv_2[0],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[0])
# ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[0],axis=1),yerr=np.nanstd(R2_cv_2[0],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[0])
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[[0,1]],axis=(0,2)),yerr=np.nanstd(R2_cv_2[[0,1]],axis=(0,2))/np.sqrt(nSessions),color='k')
ax.set_ylabel("Performance $R^2$")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FF')

ax = axes[1]
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[[2,3]],axis=(0,2)),yerr=np.nanstd(R2_cv_2[[2,3]],axis=(0,2))/np.sqrt(nSessions),color='k')
# ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[2],axis=1),yerr=np.nanstd(R2_cv_2[2],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[2])
# ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[3],axis=1),yerr=np.nanstd(R2_cv_2[3],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[3])
ax.axhline(y=0,color='k',linestyle='--')
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_title('FB')

plt.tight_layout()
sns.despine(fig=fig,trim=True)
# my_savefig(fig,savedir,'RRR_cvR2_FF_FB_diffversions_%dsessions' % (nSessions))

#%% Plot the R2 for each of the arealabelpairs and each of the dataversions
# #Residual variance explained goes down with behavior or brainwide activity regressed out: 

fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5,4))
ax = axes[0]
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[0],axis=1),yerr=np.nanstd(R2_cv_2[0],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[0])
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[1],axis=1),yerr=np.nanstd(R2_cv_2[1],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[1])
ax.set_ylabel("Performance $R^2$")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FF')

ax = axes[1]
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[2],axis=1),yerr=np.nanstd(R2_cv_2[2],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[2])
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[3],axis=1),yerr=np.nanstd(R2_cv_2[3],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[3])
ax.axhline(y=0,color='k',linestyle='--')
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_title('FB')

plt.tight_layout()
sns.despine(fig=fig,trim=True)
# my_savefig(fig,savedir,'RRR_cvR2_FF_FB_labeled_diffversions_%dsessions' % (nSessions))

#%% Plot the R2 for each of the arealabelpairs and each of the dataversions
#Residual variance explained goes down with behavior or brainwide activity regressed out: 

fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5,4))
ax = axes[0]
ax.errorbar(x=range(nversions),y=np.nanmean(optim_rank_2[0],axis=1),yerr=np.nanstd(optim_rank_2[0],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[0])
ax.errorbar(x=range(nversions),y=np.nanmean(optim_rank_2[1],axis=1),yerr=np.nanstd(optim_rank_2[1],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[1])
ax.set_ylabel("Rank")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FF')

ax = axes[1]
ax.errorbar(x=range(nversions),y=np.nanmean(optim_rank_2[2],axis=1),yerr=np.nanstd(optim_rank_2[2],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[2])
ax.errorbar(x=range(nversions),y=np.nanmean(optim_rank_2[3],axis=1),yerr=np.nanstd(optim_rank_2[3],axis=1)/np.sqrt(nSessions*nStim),color=clrs_arealabelpairs[3])
ax.axhline(y=0,color='k',linestyle='--')
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_title('FB')

plt.tight_layout()
sns.despine(fig=fig,trim=True)
# my_savefig(fig,savedir,'RRR_rank_FF_FB_diffversions_%dsessions' % (nSessions))


#%% Make the figure: 
fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5,4))
ax = axes[0]
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FF,axis=1),yerr=np.nanstd(ratiodata_FF,axis=1)/np.sqrt(np.shape(ratiodata_FF)[1]))
ax.set_ylabel("Relative performance in %\n$V1_{PM}$ vs. $V1_{ND}$")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FF')

for it,(ix,iy) in enumerate(np.array([[0,1],[0,2],[1,2]])):
    h,p = stats.ttest_rel(ratiodata_FF[ix],ratiodata_FF[iy],nan_policy='omit')
    add_stat_annotation(ax, ix,iy, 30+it*2, p, h=1)

ax = axes[1]
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FB,axis=1),yerr=np.nanstd(ratiodata_FB,axis=1)/np.sqrt(np.shape(ratiodata_FF)[1]))
for it,(ix,iy) in enumerate(np.array([[0,1],[0,2],[1,2]])):
    h,p = stats.ttest_rel(ratiodata_FB[ix],ratiodata_FB[iy],nan_policy='omit')
    add_stat_annotation(ax, ix,iy, 30+it*2, p, h=1)

ax.axhline(y=0,color='k',linestyle='--')
ax.set_ylabel("Relative performance in %\n$PM_{V1}$ vs $PM_{ND}$")
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_title('FB')

plt.tight_layout()
sns.despine(fig=fig,trim=True)
# my_savefig(fig,savedir,'RRR_cvR2_ratio_FF_FB_diffversions_%dsessions' % (nSessions))


