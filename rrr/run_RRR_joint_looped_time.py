# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
import pickle

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
# params['direction'] = 'FF'
params['direction'] = 'FB'
params['calciumversion'] = 'deconv'

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
                                       filter_noiselevel=False)

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

Nsub                = 20 #number of neurons to subsample from each population (labeled and unlabeled in each area) for RRR model fitting, set to 0 to use all neurons that pass the noise level filter
nranks              = 15 #number of ranks of RRR to be evaluated
nmodelfits          = 100

params['nStim']     = 16

# idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
idx_resp            = np.where((t_axis>=-99) & (t_axis<=99))[0]
nT                  = len(idx_resp)

R2_cv               = np.full((nsourcearealabelpairs,ntargetarealabelpairs,nSessions,params['nStim'],nT),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
optim_rank          = np.full((nsourcearealabelpairs,ntargetarealabelpairs,nSessions,params['nStim'],nT),np.nan)
R2_ranks            = np.full((nsourcearealabelpairs,ntargetarealabelpairs,nSessions,params['nStim'],nT,nranks,nmodelfits,params['kfold']),np.nan)
# R2_ranks_neurons    = np.full((nsourcearealabelpairs,ntargetarealabelpairs,Nsub,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
kf          = KFold(n_splits=params['kfold'],shuffle=True)

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
            nK                  = np.sum(idx_T) #number of trials for this stimulus condition

            X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
            X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
            Y1                  = sessions[ises].tensor[np.ix_(idx_areay1_sub,idx_T,idx_resp)]
            Y2                  = sessions[ises].tensor[np.ix_(idx_areay2_sub,idx_T,idx_resp)]

            #Zscore: 
            X1 -= np.nanmean(X1, axis=(1,2), keepdims=True)
            X2 -= np.nanmean(X2, axis=(1,2), keepdims=True)
            Y1 -= np.nanmean(Y1, axis=(1,2), keepdims=True)
            Y2 -= np.nanmean(Y2, axis=(1,2), keepdims=True)

            X1 /= np.nanstd(X1, axis=(1,2), keepdims=True)
            X2 /= np.nanstd(X2, axis=(1,2), keepdims=True)
            Y1 /= np.nanstd(Y1, axis=(1,2), keepdims=True)
            Y2 /= np.nanstd(Y2, axis=(1,2), keepdims=True)
            
            X                   = np.concatenate((X1,X2),axis=0) #use this as source to predict the activity in Y with RRR
            Y                  = np.concatenate((Y1,Y2),axis=0) #use this as source to predict the activity in Y with RRR
            
            for ikf, (idx_train, idx_test) in enumerate(kf.split(np.arange(nK))): #Get indices of train and test trials
                
                X_train, X_test     = X[:,idx_train,:], X[:,idx_test,:]
                Y_train, Y_test     = Y[:,idx_train,:], Y[:,idx_test,:]

                # reshape to neurons x time points
                X_train_r               = X_train.reshape(np.shape(X)[0],-1).T
                Y_train_r               = Y_train.reshape(np.shape(Y)[0],-1).T
                X_test_r                = X_test.reshape(np.shape(X)[0],-1).T
                Y_test_r                = Y_test.reshape(np.shape(Y)[0],-1).T

                B_hat_train             = LM(Y_train_r,X_train_r, lam=params['lam']) #fit RRR model on training data

                Y_hat_train             = X_train_r @ B_hat_train


            # # OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS    
            # R2_kfold    = np.zeros((params['kfold']))
            # kf          = KFold(n_splits=params['kfold'],shuffle=True)
            # for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
            #     X_train, X_test     = X[idx_train], X[idx_test]
            #     Y_train, Y_test     = Y[idx_train], Y[idx_test]

            #     #RRR X to Y
            #     B_hat_train         = LM(Y_train,X_train, lam=params['lam'])
            #     Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of Y_hat
                U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                # for r in range(nranks):
                    # B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                    # Y_hat_test_rr   = X_test_r @ B_rrr
                    # Y_hat_test_rr   = np.reshape(Y_hat_test_rr.T,(nNtarget,len(idx_test),nT),order='C')
                    
                for r in range(nranks):
                    B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                    # Y_hat_test_rr   = X_test @ B_rrr

                    for isource in range(nsourcearealabelpairs):
                        for itarget in range(ntargetarealabelpairs):

                            idx_X = np.repeat([True,False],Nsub) if isource==0 else np.repeat([False,True],Nsub)
                            idx_Y = np.repeat([True,False],Nsub) if itarget==0 else np.repeat([False,True],Nsub)
                            X_test_1 = copy.deepcopy(X_test)
                            X_test_1[~idx_X,:,:] = 0
                            X_test_1r = X_test_1.reshape(np.shape(X_test_1)[0],-1).T
                            Y_hat_test_rr   = X_test_1r @ B_rrr

                            # Y_hat_test_rr   = X_test_r @ B_rrr
                            Y_hat_test_rr   = np.reshape(Y_hat_test_rr.T,(np.shape(Y_test)[0],len(idx_test),nT),order='C')
                    
                            for t in range(nT):
                                # R2_ranks[0,ises,istim,t,r,imf,ikf] = EV(Y_test[:,:,t],Y_hat_test_rr[:,:,t])
                    
                                R2_ranks[isource,itarget,ises,istim,t,r,imf,ikf] = EV(Y_test[idx_Y,:,t],Y_hat_test_rr[idx_Y,:,t])
                                # R2_ranks_neurons[isource,itarget,:,ises,istim,r,imf,ikf] = r2_score(Y_test[:,idx_Y],Y_hat_test_rr[:,idx_Y], multioutput='raw_values')

#%% Find best rank and cvR2 at this rank:
fixed_rank = None
for ises in range(nSessions):
    if np.any(~np.isnan(R2_ranks[0][0][ises])):
        for istim in range(params['nStim']):
            for t in range(nT):
                for isource in range(nsourcearealabelpairs):
                    for itarget in range(ntargetarealabelpairs):
                        if fixed_rank is not None:
                            rank = fixed_rank
                            R2_cv[isource,itarget,ises,istim,t] = np.nanmean(R2_ranks[isource,itarget,ises,istim,t,rank,:,:])
                            optim_rank[isource,itarget,ises,istim,t] = rank
                        else:
                            if not np.isnan(R2_ranks[isource,itarget,ises,istim,t]).all():
                                R2_cv[isource,itarget,ises,istim,t],optim_rank[isource,itarget,ises,istim,t] = rank_from_R2(R2_ranks[isource,itarget,ises,istim,t,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])

#%%
params['Nsub']     = Nsub
params['nranks']    = nranks
params['nmodelfits'] = nmodelfits
params['nSessions'] = nSessions
params['idx_resp'] = idx_resp
params['nT'] = nT
params['t_axis'] = t_axis

#%% Save the data:
np.savez(savefilename + '.npz',R2_cv=R2_cv,R2_ranks=R2_ranks,optim_rank=optim_rank,
        #  R2_ranks_neurons=R2_ranks_neurons,
         sourcearealabelpairs=sourcearealabelpairs,
         targetarealabelpairs=targetarealabelpairs,
         allow_pickle=True)

with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)

#%%