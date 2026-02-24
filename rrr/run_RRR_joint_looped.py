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
import pickle

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
from datetime import datetime

#%% Load parameters and settings:
params = load_params()
params['radius'] = 50

params['regress_behavout'] = True
# params['regress_behavout'] = False
# params['direction'] = 'FF'
params['direction'] = 'FB'

version = 'Joint_looped_%s_%s' % (params['direction'],'behavout' if params['regress_behavout'] else 'original')

resultdir = os.path.join(params['resultdir'])
if not os.path.exists(resultdir):
    os.makedirs(resultdir)
datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savefilename = os.path.join(resultdir,'RRR_%s_%s' % (version,datetime_str))

#%% Do RRR of V1 and PM labeled and unlabeled neurons simultaneously
if params['direction'] =='FF': 
    sourcearealabelpairs = ['V1unl','V1lab']
    targetarealabelpairs = ['PMunl','PMlab']
    only_all_areas = np.array(['V1','PM'])
elif params['direction'] =='FB': 
    sourcearealabelpairs = ['PMunl','PMlab']
    targetarealabelpairs = ['V1unl','V1lab']
    only_all_areas = np.array(['V1','PM'])

#%% 
session_list        = np.array([
                                # ['LPE12223_2024_06_10'], #V1lab actually lower
                                ['LPE09830_2023_04_10'], #V1 labeled higher predictive than V1unl
                                # ['LPE10919_2023_11_06'],  #V1lab actually lower
                                # ['LPE12223_2024_06_08'], #V1lab actually lower
                                # ['LPE11998_2024_05_02'], # V1lab lower?
                                # ['LPE11622_2024_03_25'], #same
                                # ['LPE09665_2023_03_14'], #V1lab higher
                                # ['LPE10885_2023_10_23'], #V1lab much higher
                                # ['LPE11086_2024_01_05'], #Really much higher, best session, first dimensions are more predictive.
                                # ['LPE11086_2024_01_10'], #Few v1 labeled cells, very noisy
                                # ['LPE11998_2024_05_10'], #
                                # ['LPE12013_2024_05_07'], #
                                # ['LPE11495_2024_02_28'], #
                                # ['LPE11086_2023_12_15'], #Same
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,
                                       min_lab_cells_V1=20,filter_noiselevel=False)

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],min_lab_cells_V1=20,filter_noiselevel=True)
report_sessions(sessions)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=params['regress_behavout'])

#%% 
nsourcearealabelpairs     = len(sourcearealabelpairs)
ntargetarealabelpairs     = len(targetarealabelpairs)
# narealabelpairs         = nsourcearealabelpairs + ntargetarealabelpairs

Nsub                = 25
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 100

params['dim_method'] = 'pca_shuffle'
params['nStim']     = 16

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

R2_cv               = np.full((nsourcearealabelpairs,ntargetarealabelpairs,nSessions,params['nStim']),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
optim_rank          = np.full((nsourcearealabelpairs,ntargetarealabelpairs,nSessions,params['nStim']),np.nan)
R2_ranks            = np.full((nsourcearealabelpairs,ntargetarealabelpairs,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
R2_ranks_neurons    = np.full((nsourcearealabelpairs,ntargetarealabelpairs,Nsub,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
# source_dim          = np.full((nsourcearealabelpairs,ntargetarealabelpairs,nSessions,params['nStim'],nmodelfits),np.nan)
# R2_sourcealigned    = np.full((nsourcearealabelpairs,ntargetarealabelpairs,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
# frac_pos_weight_out = np.full((nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
# frac_pos_weight_in  = np.full((narealabelpairs+1,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
# weights_in          = np.full((narealabelpairs+1,Nsub,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)

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
    idx_areay1       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpairs[0],
                                            ses.celldata['noise_level']<params['maxnoiselevel'],
                                            idx_nearby),axis=0))[0]
    idx_areay2       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpairs[1],
                                            ses.celldata['noise_level']<params['maxnoiselevel'],
                                            idx_nearby),axis=0))[0]

    if len(idx_areax1)<Nsub*2 or len(idx_areax2)<Nsub or len(idx_areay1)<Nsub or len(idx_areay2)<Nsub: #skip exec if not enough neurons in one of the populations
        # print('Not enough neurons in one of the populations for session %s, skipping...' % ses.sessiondata['session_id'])
        continue

    for imf in tqdm(range(nmodelfits),total=nmodelfits,desc='Fitting RRR model for session %d/%d' % (ises+1,nSessions)):
        idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
        idx_areax2_sub       = np.random.choice(idx_areax2,Nsub,replace=False)
        idx_areay1_sub       = np.random.choice(idx_areay1,Nsub,replace=False)
        idx_areay2_sub       = np.random.choice(idx_areay2,Nsub,replace=False)

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
       
            X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
            X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
            Y1                  = sessions[ises].tensor[np.ix_(idx_areay1_sub,idx_T,idx_resp)]
            Y2                  = sessions[ises].tensor[np.ix_(idx_areay2_sub,idx_T,idx_resp)]



            # reshape to neurons x time points
            X1                  = X1.reshape(len(idx_areax1_sub),-1).T
            X2                  = X2.reshape(len(idx_areax2_sub),-1).T
            Y1                  = Y1.reshape(len(idx_areay1_sub),-1).T
            Y2                  = Y2.reshape(len(idx_areay2_sub),-1).T

            X1                  = zscore(X1,axis=0) #zscore the activity per neuron
            X2                  = zscore(X2,axis=0)
            Y1                  = zscore(Y1,axis=0)
            Y2                  = zscore(Y2,axis=0)

            X                   = np.concatenate((X1,X2),axis=1) #use this as source to predict the activity in Y with RRR
            Y                  = np.concatenate((Y1,Y2),axis=1) #use this as source to predict the activity in Y with RRR
            
            # for i,data in enumerate([X,X1,X2,X3]):
            #     source_dim[i,ises,istim,imf] = estimate_dimensionality(data,method=params['dim_method'])

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

                    for isource in range(nsourcearealabelpairs):
                        for itarget in range(ntargetarealabelpairs):
                            idx_X = np.repeat([True,False],Nsub) if isource==0 else np.repeat([False,True],Nsub)
                            idx_Y = np.repeat([True,False],Nsub) if itarget==0 else np.repeat([False,True],Nsub)
                            X_test_1 = copy.deepcopy(X_test)
                            X_test_1[:,~idx_X] = 0
                            Y_hat_test_rr   = X_test_1 @ B_rrr

                            R2_ranks[isource,itarget,ises,istim,r,imf,ikf] = EV(Y_test[:,idx_Y],Y_hat_test_rr[:,idx_Y])
                            R2_ranks_neurons[isource,itarget,:,ises,istim,r,imf,ikf] = r2_score(Y_test[:,idx_Y],Y_hat_test_rr[:,idx_Y], multioutput='raw_values')
                  
                # for i,data in enumerate([X_test,X_test_1,X_test_2,X_test_3]):
                #     # How much of the variance in the source area is aligned with the predictive subspace:
                #     R2_sourcealigned[i,ises,istim,:,imf,ikf] = compute_rrr_sourcevariance(data, B_hat_train,nranks=20)

                #Fraction of weights that is projecting positively onto firing rate:
                # for r in range(nranks): #for each rank
                #     #find correct sign of weight by sign of inner product mean firing rate and left singular vector
                #     frac_pos_weight_out[ises,istim,r,imf,ikf] = np.sum(np.sign(V[r,:])==np.sign(U[:,r].T @ np.nanmean(Y_train, axis=1))) / np.shape(V)[1]
                    
                # # Predictive source directions
                # W = B_hat_train @ V.T  # (N x k)
                # # Mean source firing rate across timepoints
                # mu_X = X_train.mean(axis=1)
                # for r in range(nranks): #for each rank compute weights
                #     # Align sign to mean source firing
                #     sign = np.sign(np.dot(X_train @ W[:, r], mu_X))
                #     # weights_in[:,ises,istim,r,imf,ikf] = sign * W[:, r]

                #     idx_N = np.arange(Nsub)
                #     weights_in[0,:,ises,istim,r,imf,ikf] = W[np.ix_(idx_N,[r])].flatten()*sign
                #     idx_N = np.arange(Nsub,2*Nsub)
                #     weights_in[1,:,ises,istim,r,imf,ikf] = W[np.ix_(idx_N,[r])].flatten()*sign
                #     idx_N = np.arange(Nsub*2,Nsub*3)
                #     weights_in[2,:,ises,istim,r,imf,ikf] = W[np.ix_(idx_N,[r])].flatten()*sign

                #     frac_pos_weight_in[0,ises,istim,r,imf,ikf] = np.sum(np.sign(W[:, r])==sign) / np.shape(W)[0]
                #     idx_N = np.arange(Nsub)
                #     frac_pos_weight_in[1,ises,istim,r,imf,ikf] = np.sum(np.sign(W[np.ix_(idx_N,[r])])==sign) / Nsub
                #     idx_N = np.arange(Nsub,2*Nsub)
                #     frac_pos_weight_in[2,ises,istim,r,imf,ikf] = np.sum(np.sign(W[np.ix_(idx_N,[r])])==sign) / Nsub
                #     idx_N = np.arange(Nsub*2,Nsub*3)
                #     frac_pos_weight_in[3,ises,istim,r,imf,ikf] = np.sum(np.sign(W[np.ix_(idx_N,[r])])==sign) / Nsub

#%% Find best rank and cvR2 at this rank:
fixed_rank = None
for ises in range(nSessions):
    if np.any(~np.isnan(R2_ranks[0][0][ises])):
        for istim in range(params['nStim']):
            for isource in range(nsourcearealabelpairs):
                for itarget in range(ntargetarealabelpairs):
                    if fixed_rank is not None:
                        rank = fixed_rank
                        R2_cv[isource,itarget,ises,istim] = np.nanmean(R2_ranks[isource,itarget,ises,istim,rank,:,:])
                        optim_rank[isource,itarget,ises,istim] = rank
                    else:
                        if not np.isnan(R2_ranks[isource,itarget,ises,istim]).all():
                            R2_cv[isource,itarget,ises,istim],optim_rank[isource,itarget,ises,istim] = rank_from_R2(R2_ranks[isource,itarget,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])

#%%
params['Nsub']     = Nsub
params['nranks']    = nranks
params['nmodelfits'] = nmodelfits
params['nSessions'] = nSessions

#%% Save the data:
np.savez(savefilename + '.npz',R2_cv=R2_cv,R2_ranks=R2_ranks,optim_rank=optim_rank,
         R2_ranks_neurons=R2_ranks_neurons,
        #  source_dim=source_dim,
        #  R2_sourcealigned=R2_sourcealigned,
        #  frac_pos_weight_out=frac_pos_weight_out,
        #  weights_in=weights_in,frac_pos_weight_in=frac_pos_weight_in,
         sourcearealabelpairs=sourcearealabelpairs,
         targetarealabelpairs=targetarealabelpairs,
        #  params=params,allow_pickle=True)
         allow_pickle=True)

with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)

#%%