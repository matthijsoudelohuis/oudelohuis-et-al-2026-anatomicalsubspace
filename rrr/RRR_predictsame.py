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
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_noiselevel=False)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],min_lab_cells_V1=20,filter_noiselevel=True)
report_sessions(sessions)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False)
# [sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=True)
# sessions = load_resid_tensor(sessions,behavout=True)

#%% Do RRR of V1 and PM labeled and unlabeled neurons
sourcearealabelpairs = ['V1unl-V1lab','V1lab-V1unl']
targetarealabelpair = 'PMunl'

# clrs_arealabelpairs = get_clr_area_labeled(sourcearealabelpairs)
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 25
Nsub_pops           = np.array([5,10,15,20,25,30,35,40,45,50])
# Nsub_pops           = np.array([10,20,30,40,50])
# Nsub_pops           = np.array([10,20,30,50,100])
# Nsub_pops           = np.array([10,50,100])
nsub_pops           = len(Nsub_pops)
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 4

nStim               = 16

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)
# minsampleneurons    = 10

fixed_rank          = 5

R2_cv               = np.full((narealabelpairs,nSessions,nStim),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
optim_rank          = np.full((narealabelpairs,nSessions,nStim),np.nan)
R2_Y1_X1            = np.full((narealabelpairs,nSessions,nStim,nmodelfits,params['kfold']),np.nan)
R2_Y2_X2            = np.full((narealabelpairs,nSessions,nStim,nsub_pops,nranks,nmodelfits,params['kfold']),np.nan)
R2_Y1_X2            = np.full((narealabelpairs,nSessions,nStim,nsub_pops,nranks,nmodelfits,params['kfold']),np.nan)

params['maxnoiselevel'] = 20
params['radius'] = 50
# params['radius'] = 30
# for ises,ses in enumerate(sessions):
# for ises,ses in enumerate(sessions[:8]):
for ises,ses in enumerate(sessions[7:10]):
    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    for iapl, sourcearealabelpair in enumerate(sourcearealabelpairs):
        
        alx1,alx2       = sourcearealabelpair.split('-')

        idx_areax1      = np.where(np.all((ses.celldata['arealabel']==alx1,
                                    ses.celldata['noise_level']<params['maxnoiselevel'],
                                    idx_nearby),axis=0))[0]
        idx_areax2      = np.where(np.all((ses.celldata['arealabel']==alx2,
                                    ses.celldata['noise_level']<params['maxnoiselevel'],
                                    idx_nearby),axis=0))[0]
        idx_areay       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpair,
                                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                                idx_nearby
                                                ),axis=0))[0]
        
        if len(idx_areax1)<max(Nsub_pops) or len(idx_areax2)<max(Nsub_pops) or len(idx_areay)<Nsub: #skip exec if not enough neurons in one of the populations
            continue

        for imf in tqdm(range(nmodelfits),total=nmodelfits,desc='Fitting RRR model for session %d/%d' % (ises+1,nSessions)):
            idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
            idx_areax2_sub       = np.random.choice(idx_areax2,max(Nsub_pops),replace=False)
            idx_areay_sub        = np.random.choice(idx_areay,Nsub,replace=False)

            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
                idx_T               = ses.trialdata['stimCond']==stim

                X1                  = ses.tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
                X2                  = ses.tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
                Y                   = ses.tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

                # reshape to neurons x time points
                X1                  = X1.reshape(len(idx_areax1_sub),-1).T
                X2                  = X2.reshape(len(idx_areax2_sub),-1).T
                Y                   = Y.reshape(len(idx_areay_sub),-1).T

                X1                  = zscore(X1,axis=0) #zscore the activity per neuron
                X2                  = zscore(X2,axis=0)
                Y                   = zscore(Y,axis=0)

                # OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS    
                R2_kfold    = np.zeros((params['kfold']))
                kf          = KFold(n_splits=params['kfold'],shuffle=True)
                for ikf, (idx_train, idx_test) in enumerate(kf.split(X1)):
                    X_train, X_test     = X1[idx_train], X1[idx_test]
                    Y_train, Y_test     = Y[idx_train], Y[idx_test]

                    #RRR X to Y
                    B_hat_train         = LM(Y_train,X_train, lam=params['lam'])
                    Y_hat_train         = X_train @ B_hat_train

                    # decomposing and low rank approximation of Y_hat
                    U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                    U, s, V = U[:, ::-1], s[::-1], V[::-1, :]
                    
                    B_rrr           = B_hat_train @ V[:fixed_rank,:].T @ V[:fixed_rank,:] #project beta coeff into low rank subspace
                    Y_hat_X1_test   = X_test @ B_rrr

                    R2_Y1_X1[iapl,ises,istim,imf,ikf] = EV(Y_test,Y_hat_X1_test)

                    for ipop,pop in enumerate(Nsub_pops):
                        X2_sub = X2[:,:pop]
                        #Now see if the same activity is predicted by X2 for different population sizes:
                        X2_train, X2_test     = X2_sub[idx_train], X2_sub[idx_test]

                        #RRR X to Y
                        B_hat_train         = LM(Y_train,X2_train, lam=params['lam'])
                        Y_hat_train         = X2_train @ B_hat_train

                        # decomposing and low rank approximation of Y_hat
                        U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                        U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                        for r in range(nranks):
                            B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                            Y_hat_X2_test   = X2_test @ B_rrr

                            R2_Y2_X2[iapl,ises,istim,ipop,r,imf,ikf] = EV(Y_test,Y_hat_X2_test)
                            # R2_Y1_X2[iapl,ises,istim,ipop,r,imf,ikf] = EV(Y_test-Y_hat_X1_test,Y_hat_X2_test)
                            # R2_Y1_X2[iapl,ises,istim,ipop,r,imf,ikf] = EV(Y_hat_X1_test,Y_hat_X2_test)
                            # R2_Y1_X2[iapl,ises,istim,ipop,r,imf,ikf] = np.corrcoef(Y_hat_X1_test.flatten(),Y_hat_X2_test.flatten())[0,1]

#%%
plt.imshow(Y_test,aspect='auto',vmin=-1,vmax=1)
plt.imshow(Y_hat_X1_test,aspect='auto',vmin=-1,vmax=1)
plt.imshow(Y_test-Y_hat_X1_test,aspect='auto',vmin=-1,vmax=1)
plt.imshow(Y_hat_X2_test,aspect='auto',vmin=-1,vmax=1)

# #%%
# fixed_rank = 5
# for ises in range(nSessions):
#     if np.any(~np.isnan(R2_ranks[0,ises,:,:,:,:])):
#         for istim in range(nStim):
#             if fixed_rank is not None:
#                 rank = fixed_rank
#                 R2_cv[0,ises,istim] = np.nanmean(R2_ranks[0,ises,istim,rank,:,:])
#                 R2_cv[1,ises,istim] = np.nanmean(R2_ranks[1,ises,istim,rank,:,:])
#                 R2_cv[2,ises,istim] = np.nanmean(R2_ranks[2,ises,istim,rank,:,:])
#                 R2_cv[3,ises,istim] = np.nanmean(R2_ranks[3,ises,istim,rank,:,:])
#             else:
#                 if not np.isnan(R2_ranks[0,ises,istim,:,:,:]).all():
#                     R2_cv[0,ises,istim],optim_rank[0,ises,istim] = rank_from_R2(R2_ranks[0,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
#                     R2_cv[1,ises,istim],optim_rank[1,ises,istim] = rank_from_R2(R2_ranks[1,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
#                     R2_cv[2,ises,istim],optim_rank[2,ises,istim] = rank_from_R2(R2_ranks[2,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
#                     R2_cv[3,ises,istim],optim_rank[3,ises,istim] = rank_from_R2(R2_ranks[3,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])

#%% 
from scipy.optimize import curve_fit

def power_law_func(x, a, b):
    return a * x**b

fig, axes = plt.subplots(1,2,figsize=(12*cm,5*cm),sharex=True,sharey=True)

fixed_rank = 5
clrs = ['grey','red']
for iapl in range(narealabelpairs):
    ax = axes[iapl]
    refperf = np.nanmean(R2_Y1_X1[iapl])
    ax.plot(Nsub,refperf,marker='o',color=clrs[iapl],label=sourcearealabelpairs[iapl])
    
    meantoplot = np.nanmean(R2_Y2_X2[iapl],axis=(0,1,4,5))
    meantoplot = meantoplot[:,fixed_rank]
    ax.plot(Nsub_pops,meantoplot,marker='o',color=clrs[1-iapl],label=sourcearealabelpairs[iapl])
   
    popt, pcov = curve_fit(power_law_func, Nsub_pops, meantoplot, p0=[1., 1.])
    x_domain = np.arange(0,50)
    y_domain = power_law_func(x_domain, *popt)

    ax.plot(x_domain, y_domain, color=clrs[1-iapl], linestyle='--', label='power law fit')
    # meantoplot = np.nanmean(R2_Y1_X2[iapl],axis=(0,1,3,4,5))
    meantoplot = np.nanmean(R2_Y1_X2[iapl],axis=(0,1,4,5))
    meantoplot = meantoplot[:,fixed_rank]

    ax.plot(Nsub_pops,meantoplot,marker='o',color=clrs[iapl],label=sourcearealabelpairs[iapl])
    nNeurons =  x_domain[np.where(y_domain>refperf)[0][0]]
    ax.text(0.7,0.1,'%d neurons\n%1.2f%%' % (nNeurons,100 - 100*Nsub/nNeurons), transform=ax.transAxes,color=clrs[1-iapl])
    ax.plot([nNeurons,nNeurons],[0,refperf],color=clrs[1-iapl],linestyle='--')
    # leg = ax.legend(frameon=False)
    # my_legend_strip(ax)
    ax.set_xlabel('#Neurons')
    if iapl==0: 
        ax.set_ylabel('Cross-validated R2')

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True,offset=0)
my_savefig(fig,figdir,'RRR_neurons_neededtomatch_cvR2_labunl_FF_%dsessions' % (nSessions))

#%% 

