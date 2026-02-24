# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('c:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy import stats
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pickle

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
# from utils.corr_lib import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.psth import compute_tensor
from params import load_params
from utils.corr_lib import filter_sharednan
from datetime import datetime

#%% Load parameters and settings:
params = load_params()
params['radius'] = 50

# params['regress_behavout'] = True
params['regress_behavout'] = False

figdir = os.path.join(params['figdir'],'RRR','RF')

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% 
session_list        = np.array([
                                ['LPE12223_2024_06_10'], 
                                # ['LPE09830_2023_04_10'], 
                                ['LPE12223_2024_06_08'], 
                                ['LPE11998_2024_05_02'],
                                # ['LPE11622_2024_03_25'], 
                                # ['LPE09665_2023_03_14'],
                                # ['LPE10885_2023_10_23'],
                                # ['LPE11086_2024_01_05'], #
                                # ['LPE11086_2024_01_10'], #
                                # ['LPE11998_2024_05_10'], #
                                ['LPE12013_2024_05_07'], #
                                ['LPE11495_2024_02_28'], #
                                # ['LPE11086_2023_12_15'], #
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,
                                       filter_noiselevel=False)

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],session_rf=True)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],min_lab_cells_V1=20,filter_noiselevel=True)
report_sessions(sessions)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=params['regress_behavout'])


#%%
for ses in sessions:
    # if 'rf_az_Fsmooth' in ses.celldata.columns:
    if 'rf_az_Fneu' in ses.celldata.columns:
        # print(np.sum(~np.isnan( ses.celldata['rf_az_Fsmooth'])) / len(ses.celldata))
        print(np.sum(~np.isnan( ses.celldata['rf_az_Fneu'])) / len(ses.celldata))
        # print(np.sum(ses.celldata['rf_r2_Fneu']>0.2) / len(ses.celldata))


#%% Matched and mismatched receptive fields across areas: 

binres              = 10 #deg steps in azimuth and elevation to select target neurons

vec_elevation       = [-16.7,50.2] #bottom and top of screen displays
vec_azimuth         = [-135,135] #left and right of screen displays

binedges_az         = np.arange(vec_azimuth[0],vec_azimuth[1]+binres,binres)
binedges_el         = np.arange(vec_elevation[0],vec_elevation[1]+binres,binres)
nbins_az            = len(binedges_az)
nbins_el            = len(binedges_el)

radius_match        = 15 #deg, radius of receptive field to match
radius_mismatch     = 20 #deg, radius of receptive field to mismatch, if within this radius then excluded
params['minrfR2']     = 0.2 #minimum R2 of receptive field fit to be included in analysis

# arealabelpairs      = ['PMunl-V1unl']
arealabelpairs      = ['V1unl-PMunl','PMunl-V1unl']

# arealabelpairs  = ['V1unl-PMunl',
#                     'V1lab-PMunl',
#                     'V1unl-PMlab',
#                     'V1lab-PMlab',
#                     'PMunl-V1unl',
#                     'PMunl-V1lab',
#                     'PMlab-V1unl',
#                     'PMlab-V1lab']

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

nsampleneurons       = 25

params['nranks']     = 20
params['nmodelfits'] = 15 #number of times new neurons are resampled 

R2_cv               = np.full((nbins_az,nbins_el,narealabelpairs,2,nSessions),np.nan)
optim_rank          = np.full((nbins_az,nbins_el,narealabelpairs,2,nSessions),np.nan)
R2_ranks            = np.full((nbins_az,nbins_el,narealabelpairs,2,nSessions,params['nranks'],params['nmodelfits'],params['kfold']),np.nan)

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for match/mismatch RF:'):
for ises,ses in enumerate(sessions):
    # if 'rf_az_Fsmooth' not in ses.celldata.columns:
    if 'rf_az_Fneu' not in ses.celldata.columns:
        continue
    # sesaz = ses.celldata['rf_az_Fsmooth'].to_numpy()
    # sesel = ses.celldata['rf_el_Fsmooth'].to_numpy()
    sesaz = ses.celldata['rf_az_Fneu'].to_numpy()
    sesel = ses.celldata['rf_el_Fneu'].to_numpy()

    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)

    # idx_T               = ses.trialdata['stimCond']==stim
    idx_T               = ses.trialdata['stimCond']==0

    for iaz,az in tqdm(enumerate(binedges_az),total=nbins_az,desc='Matching RFs across areas, session %d/%d:' % (ises+1,nSessions)):
        for iel,el in enumerate(binedges_el):
            for iapl, arealabelpair in enumerate(arealabelpairs):

                alx,aly = arealabelpair.split('-')
                idx_match = np.all((sesaz>=az-radius_match,
                                    sesaz<az+radius_match,
                                    sesel>=el-radius_match,
                                    sesel<el+radius_match),axis=0)
                
                idx_mismatch = ~np.all((sesaz>=az-radius_mismatch,
                                    sesaz<az+radius_mismatch,
                                    sesel>=el-radius_mismatch,
                                    sesel<el+radius_mismatch),axis=0)
                
                idx_x       = np.where(np.all((ses.celldata['arealabel']==alx,
                                idx_match,
                                ses.celldata['rf_r2_Fneu']>params['minrfR2'],
                                ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
            
                idx_y_match  = np.where(np.all((ses.celldata['arealabel']==aly,
                                idx_match,
                                ses.celldata['rf_r2_Fneu']>params['minrfR2'],
                                ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
            
                idx_y_mismatch  = np.where(np.all((ses.celldata['arealabel']==aly,
                                idx_mismatch,
                                ses.celldata['rf_r2_Fneu']>params['minrfR2'],
                                ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]

                if len(idx_x)<nsampleneurons or len(idx_y_match)<nsampleneurons or len(idx_y_mismatch)<nsampleneurons: #skip exec if not enough neurons in one of the populations
                    continue
                

                X                  = sessions[ises].tensor[np.ix_(idx_x,idx_T,idx_resp)]
                Y1                 = sessions[ises].tensor[np.ix_(idx_y_match,idx_T,idx_resp)]
                Y2                 = sessions[ises].tensor[np.ix_(idx_y_mismatch,idx_T,idx_resp)]

                # reshape to neurons x time points
                X                  = X.reshape(len(idx_x),-1).T
                Y1                 = Y1.reshape(len(idx_y_match),-1).T
                Y2                 = Y2.reshape(len(idx_y_mismatch),-1).T
                
                R2_cv[iaz,iel,iapl,0,ises],optim_rank[iaz,iel,iapl,0,ises],R2_ranks[iaz,iel,iapl,0,ises,:,:,:]  = \
                    RRR_wrapper(Y1, X, nN=nsampleneurons,nK=None,lam=params['lam'],nranks=params['nranks'],kfold=params['kfold'],nmodelfits=params['nmodelfits'])

                R2_cv[iaz,iel,iapl,1,ises],optim_rank[iaz,iel,iapl,1,ises],R2_ranks[iaz,iel,iapl,1,ises,:,:,:]  = \
                    RRR_wrapper(Y2, X, nN=nsampleneurons,nK=None,lam=params['lam'],nranks=params['nranks'],kfold=params['kfold'],nmodelfits=params['nmodelfits'])

#%%
print('Fraction of array filled with data: %.2f' % (np.sum(~np.isnan(R2_cv)) / R2_cv.size))

#%% Plot the results: 
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*2,3),sharey=True,sharex=True)
if narealabelpairs == 1:
    axes = np.array([axes])

for iapl, arealabelpair in enumerate(arealabelpairs):
    ax = axes[iapl]

    datatoplot = np.column_stack((R2_cv[:,:,iapl,0,:].flatten(),R2_cv[:,:,iapl,1,:].flatten())) 
    datatoplot = datatoplot[~np.isnan(datatoplot).any(axis=1)]

    ax.scatter(np.zeros(len(datatoplot))+np.random.randn(len(datatoplot))*0.05,datatoplot[:,0],color='k',marker='o',s=10)
    ax.errorbar(0.2,np.nanmean(datatoplot[:,0]),np.nanstd(datatoplot[:,0])/np.sqrt(nSessions),color='g',marker='o',zorder=10)

    ax.scatter(np.ones(len(datatoplot))+np.random.randn(len(datatoplot))*0.05,datatoplot[:,1],color='k',marker='o',s=10)
    ax.errorbar(1.2,np.nanmean(datatoplot[:,1]),np.nanstd(datatoplot[:,1])/np.sqrt(nSessions),color='r',marker='o',zorder=10)

    add_paired_ttest_results(ax,datatoplot[:,0],datatoplot[:,1],pos=[0.5,0.8],fontsize=6)
    ax.set_xticks([0,1],['Match','Mismatch'])
    ax.set_ylabel('R2')
    ax.set_title('%s' % arealabelpair,fontsize=8)
ax.set_ylim([0,my_ceil(np.nanmax(datatoplot),2)])
sns.despine(top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,figdir,'RRR_R2_MatchMismatch_RF_%dsessions' % (nSessions),formats = ['png'])

#%% Plot the results: 
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*1.3,3),sharey=True,sharex=True)
if narealabelpairs == 1:
    axes = np.array([axes])

for iapl, arealabelpair in enumerate(arealabelpairs):
    ax = axes[iapl]

    datatoplot = np.column_stack((optim_rank[:,:,iapl,0,:].flatten(),optim_rank[:,:,iapl,1,:].flatten())) 
    # datatoplot = np.column_stack((np.nanmean(optim_rank[:,:,iapl,0,:],axis=(0,1)).flatten(),np.nanmean(optim_rank[:,:,iapl,1,:],axis=(0,1)).flatten())) 
    datatoplot = datatoplot[~np.isnan(datatoplot).any(axis=1)]

    ax.scatter(np.zeros(len(datatoplot))+np.random.randn(len(datatoplot))*0.05,datatoplot[:,0],color='k',marker='o',s=10)
    ax.errorbar(0.2,np.nanmean(datatoplot[:,0]),np.nanstd(datatoplot[:,0])/np.sqrt(nSessions),color='g',marker='o',zorder=10)

    ax.scatter(np.ones(len(datatoplot))+np.random.randn(len(datatoplot))*0.05,datatoplot[:,1],color='k',marker='o',s=10)
    ax.errorbar(1.2,np.nanmean(datatoplot[:,1]),np.nanstd(datatoplot[:,1])/np.sqrt(nSessions),color='r',marker='o',zorder=10)

    add_paired_ttest_results(ax,datatoplot[:,0],datatoplot[:,1],pos=[0.5,0.8],fontsize=6)
    ax.set_xticks([0,1],['Match','Mismatch'])
    ax.set_title('%s' % arealabelpair,fontsize=8)
# ax.set_ylim([0,0.25])
axes[0].set_ylabel('Rank')
sns.despine(top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,figdir,'RRR_Rank_MatchMismatch_RF_%dsessions' % (nSessions),formats = ['png'])


#%%  Show percentage difference between match and mismatch:

fig,axes = plt.subplots(1,1,figsize=(2,3),sharey=True,sharex=True)

datatoplot = np.column_stack([R2_cv[:,:,iapl,0,:].flatten() / R2_cv[:,:,iapl,1,:].flatten() for iapl in range(narealabelpairs)]) 
ax = axes

for iapl, arealabelpair in enumerate(arealabelpairs):
    # ax.errorbar(iapl+0.5,np.nanmean(datatoplot[iapl,:]),np.nanstd(datatoplot[iapl,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)
    ax.errorbar(iapl,np.nanmean(datatoplot[:,iapl]),np.nanstd(datatoplot[:,iapl])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)
ax.axhline(1, color='k', linewidth=0.5, linestyle='--')
ax.set_ylabel('Ratio R2 (match/mismatch)')
ax_nticks(ax,4)
sns.despine(top=True,right=True,offset=3)	
ax.set_xticks(range(narealabelpairs))
ax.set_xticklabels(arealabelpairs,rotation=45,ha='right',fontsize=8)
my_savefig(fig,savedir,'R2_Ratio_MatchMismatch_RF_%dsessions' % (nSessions),formats = ['png'])




#%% 
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 25
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 100

params['dim_method'] = 'pca_shuffle'
params['nStim']     = 16

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

R2_cv               = np.full((narealabelpairs+1,nSessions,params['nStim']),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
optim_rank          = np.full((narealabelpairs+1,nSessions,params['nStim']),np.nan)
R2_ranks            = np.full((narealabelpairs+1,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
R2_ranks_neurons    = np.full((narealabelpairs+1,Nsub*narealabelpairs,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
source_dim          = np.full((narealabelpairs+1,nSessions,params['nStim'],nmodelfits),np.nan)
R2_sourcealigned    = np.full((narealabelpairs+1,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
frac_pos_weight_out = np.full((nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
frac_pos_weight_in  = np.full((narealabelpairs+1,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
weights_in          = np.full((narealabelpairs+1,Nsub,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)

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

            for i,data in enumerate([X,X1,X2,X3]):
                source_dim[i,ises,istim,imf] = estimate_dimensionality(data,method=params['dim_method'])

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

#%% Find best rank and cvR2 at this rank:
fixed_rank = None
for ises in range(nSessions):
    if np.any(~np.isnan(R2_ranks[0][ises])):
        for istim in range(params['nStim']):
            if fixed_rank is not None:
                rank = fixed_rank
                R2_cv[0,ises,istim] = np.nanmean(R2_ranks[0,ises,istim,rank,:,:])
                R2_cv[1,ises,istim] = np.nanmean(R2_ranks[1,ises,istim,rank,:,:])
                R2_cv[2,ises,istim] = np.nanmean(R2_ranks[2,ises,istim,rank,:,:])
                R2_cv[3,ises,istim] = np.nanmean(R2_ranks[3,ises,istim,rank,:,:])
            else:
                if not np.isnan(R2_ranks[0][ises][istim]).all():
                    R2_cv[0,ises,istim],optim_rank[0,ises,istim] = rank_from_R2(R2_ranks[0,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                    R2_cv[1,ises,istim],optim_rank[1,ises,istim] = rank_from_R2(R2_ranks[1,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                    R2_cv[2,ises,istim],optim_rank[2,ises,istim] = rank_from_R2(R2_ranks[2,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                    R2_cv[3,ises,istim],optim_rank[3,ises,istim] = rank_from_R2(R2_ranks[3,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])

#%%