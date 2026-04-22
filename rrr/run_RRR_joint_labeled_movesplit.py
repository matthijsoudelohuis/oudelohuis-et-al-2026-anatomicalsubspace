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

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.RRRlib import *
from utils.regress_lib import *
from params import load_params
from datetime import datetime

#%% Load parameters and settings:
params = load_params()
params['regress_behavout'] = False
# params['direction'] = 'FF'
params['direction'] = 'FB'

version = 'Joint_labeled_%s_movesplit' % (params['direction'])

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
                                # ['LPE09830_2023_04_10'], #V1 labeled higher predictive than V1unl
                                ['LPE09665_2023_03_14'], #V1lab higher
                                # ['LPE10885_2023_10_23'], #V1lab much higher
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,
                                       min_lab_cells_V1=20,filter_noiselevel=False)

report_sessions(sessions)

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_noiselevel=False)
report_sessions(sessions)

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False,load_behav=True)

#%% 
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 20
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 100

params['nStim']     = 16
# params['radius']     = 30
params['maxvideome'] = 0.2
params['maxrunspeed'] = 0.5
params['mintrials'] = 30

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

R2_cv               = np.full((narealabelpairs+1,2,nSessions,params['nStim']),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
optim_rank          = np.full((narealabelpairs+1,2,nSessions,params['nStim']),np.nan)
R2_ranks            = np.full((narealabelpairs+1,2,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)

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
        print('%d in %s, %d in %s' % (len(idx_areax3),sourcearealabelpairs[2],
                                                len(idx_areay),targetarealabelpair))
        continue

    for imf in tqdm(range(nmodelfits),total=nmodelfits,desc='Fitting RRR model for session %d/%d' % (ises+1,nSessions)):
        idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
        idx_areax2_sub       = np.random.choice(np.setdiff1d(idx_areax2,idx_areax1_sub),Nsub,replace=False)
        idx_areax3_sub       = np.random.choice(idx_areax3,Nsub,replace=False)
        idx_areay_sub        = np.random.choice(idx_areay,Nsub*narealabelpairs,replace=False)

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            for istate in range(2): # loop over states
                idx_T_still  = np.logical_and( ses.respmat_videome < params['maxvideome'],
                                        ses.respmat_runspeed < params['maxrunspeed'])
                if istate==0: 
                    idx_T               = np.logical_and(ses.trialdata['stimCond']==stim,idx_T_still)
                elif istate==1:
                    idx_T               = np.logical_and(ses.trialdata['stimCond']==stim,~idx_T_still)

                if np.sum(idx_T)<params['mintrials']: #skip exec if not enough trials in session0:
                    continue
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

                        R2_ranks[0,istate,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)
                        
                        X_test_1 = copy.deepcopy(X_test)
                        X_test_1[:,Nsub:] = 0
                        Y_hat_test_rr   = X_test_1 @ B_rrr

                        R2_ranks[1,istate,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)

                        X_test_2 = copy.deepcopy(X_test)
                        X_test_2[:,:Nsub] = 0
                        X_test_2[:,2*Nsub:] = 0
                        Y_hat_test_rr   = X_test_2 @ B_rrr

                        R2_ranks[2,istate,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)

                        X_test_3 = copy.deepcopy(X_test)
                        X_test_3[:,:2*Nsub] = 0
                        Y_hat_test_rr   = X_test_3 @ B_rrr

                        R2_ranks[3,istate,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)

#%% Find best rank and cvR2 at this rank:
fixed_rank = None
for ises in range(nSessions):
    if np.any(~np.isnan(R2_ranks[0][istate][ises])):
        for istate in range(2):
            for istim in range(params['nStim']):
                if fixed_rank is not None:
                    rank = fixed_rank
                    R2_cv[0,istate,ises,istim] = np.nanmean(R2_ranks[0,istate,ises,istim,rank,:,:])
                    R2_cv[1,istate,ises,istim] = np.nanmean(R2_ranks[1,istate,ises,istim,rank,:,:])
                    R2_cv[2,istate,ises,istim] = np.nanmean(R2_ranks[2,istate,ises,istim,rank,:,:])
                    R2_cv[3,istate,ises,istim] = np.nanmean(R2_ranks[3,istate,ises,istim,rank,:,:])
                else:
                    if not np.isnan(R2_ranks[0][istate][ises][istim]).all():
                        R2_cv[0,istate,ises,istim],optim_rank[0,istate,ises,istim] = rank_from_R2(R2_ranks[0,istate,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                        R2_cv[1,istate,ises,istim],optim_rank[1,istate,ises,istim] = rank_from_R2(R2_ranks[1,istate,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                        R2_cv[2,istate,ises,istim],optim_rank[2,istate,ises,istim] = rank_from_R2(R2_ranks[2,istate,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                        R2_cv[3,istate,ises,istim],optim_rank[3,istate,ises,istim] = rank_from_R2(R2_ranks[3,istate,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])

#%%
params['Nsub']          = Nsub
params['nranks']        = nranks
params['nmodelfits']    = nmodelfits
params['nSessions']     = nSessions

#%% Save the data:
np.savez(savefilename + '.npz',R2_cv=R2_cv,R2_ranks=R2_ranks,optim_rank=optim_rank,
         sourcearealabelpairs=sourcearealabelpairs,
         targetarealabelpair=targetarealabelpair,
         allow_pickle=True)

with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)

#%%