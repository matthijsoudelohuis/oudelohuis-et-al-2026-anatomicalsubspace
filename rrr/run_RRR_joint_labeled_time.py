# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os

# from rrr.deprecated.RRR_V1PM import R2_cv_folds
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
import pickle
from sklearn.model_selection import KFold

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
# params['direction'] = 'FB'
# params['direction'] = 'FF_AL'
# params['direction'] = 'FB_AL'

version = 'Joint_labeled_%s_%s' % (params['direction'],'behavout' if params['regress_behavout'] else 'original')

resultdir = os.path.join(params['resultdir'])
if not os.path.exists(resultdir):
    os.makedirs(resultdir)
datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savefilename = os.path.join(resultdir,'RRR_time_%s_%s' % (version,datetime_str))

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

#%% Get all data 
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_noiselevel=False)
report_sessions(sessions)
# sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=params['regress_behavout'])

#%% 
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 20
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 15

params['nStim']     = 16

# idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
idx_resp            = np.where((t_axis>=-99) & (t_axis<=99))[0]
nT                  = len(idx_resp)

R2_cv               = np.full((narealabelpairs+1,nSessions,params['nStim'],nT),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
optim_rank          = np.full((narealabelpairs+1,nSessions,params['nStim'],nT),np.nan)
R2_ranks            = np.full((narealabelpairs+1,nSessions,params['nStim'],nT,nranks,nmodelfits,params['kfold']),np.nan)
# R2_ranks_neurons    = np.full((narealabelpairs+1,Nsub*narealabelpairs,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)

kf          = KFold(n_splits=params['kfold'],shuffle=True)
nNsource    = Nsub*3
nNtarget    = Nsub*3

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
                                idx_nearby),axis=0))[0]
    
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
        # for istim,stim in enumerate([ses.trialdata['stimCond'][0]]): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
       
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

            nK = np.sum(idx_T) #number of trials for this stimulus condition
            
            X                       = np.concatenate((X1,X2,X3),axis=0) #use this as source to predict the activity in Y with RRR
            R2_kfold                = np.zeros((params['kfold']))

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

                # decomposing and low rank approximation of A
                U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                U, s, V = U[:, ::-1], s[::-1], V[::-1, :] #sort by singular values

                X_test_1 = copy.deepcopy(X_test_r)
                X_test_1[:,Nsub:] = 0

                X_test_2 = copy.deepcopy(X_test_r)
                X_test_2[:,:Nsub] = 0
                X_test_2[:,2*Nsub:] = 0
                
                X_test_3 = copy.deepcopy(X_test_r)
                X_test_3[:,:2*Nsub] = 0

                for r in range(nranks):
                    B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                    Y_hat_test_rr   = X_test_r @ B_rrr
                    Y_hat_test_rr   = np.reshape(Y_hat_test_rr.T,(nNtarget,len(idx_test),nT),order='C')
                    
                    for t in range(nT):
                        R2_ranks[0,ises,istim,t,r,imf,ikf] = EV(Y_test[:,:,t],Y_hat_test_rr[:,:,t])
                    
                    Y_hat_test_rr_1     = X_test_1 @ B_rrr
                    Y_hat_test_rr       = np.reshape(Y_hat_test_rr_1.T,(nNtarget,len(idx_test),nT),order='C')

                    for t in range(nT):
                        R2_ranks[1,ises,istim,t,r,imf,ikf] = EV(Y_test[:,:,t],Y_hat_test_rr[:,:,t])
                    
                    Y_hat_test_rr_2     = X_test_2 @ B_rrr
                    Y_hat_test_rr       = np.reshape(Y_hat_test_rr_2.T,(nNtarget,len(idx_test),nT),order='C')

                    for t in range(nT):
                        R2_ranks[2,ises,istim,t,r,imf,ikf] = EV(Y_test[:,:,t],Y_hat_test_rr[:,:,t])

                    Y_hat_test_rr_3     = X_test_3 @ B_rrr
                    Y_hat_test_rr       = np.reshape(Y_hat_test_rr_3.T,(nNtarget,len(idx_test),nT),order='C')

                    for t in range(nT):
                        R2_ranks[3,ises,istim,t,r,imf,ikf] = EV(Y_test[:,:,t],Y_hat_test_rr[:,:,t])
               
#%% Find best rank and cvR2 at this rank:
fixed_rank = None
fixed_rank = 3
for ises in range(nSessions):
    if np.any(~np.isnan(R2_ranks[0][ises])):
        for istim in range(params['nStim']):
            for t in range(nT):
                if fixed_rank is not None:
                    rank = fixed_rank
                    R2_cv[0,ises,istim,t] = np.nanmean(R2_ranks[0,ises,istim,t,rank,:,:])
                    R2_cv[1,ises,istim,t] = np.nanmean(R2_ranks[1,ises,istim,t,rank,:,:])
                    R2_cv[2,ises,istim,t] = np.nanmean(R2_ranks[2,ises,istim,t,rank,:,:])
                    R2_cv[3,ises,istim,t] = np.nanmean(R2_ranks[3,ises,istim,t,rank,:,:])
                else:
                    if not np.isnan(R2_ranks[0][ises][istim]).all():
                        R2_cv[0,ises,istim,t],optim_rank[0,ises,istim] = rank_from_R2(R2_ranks[0,ises,istim,t,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                        R2_cv[1,ises,istim,t],optim_rank[1,ises,istim] = rank_from_R2(R2_ranks[1,ises,istim,t,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                        R2_cv[2,ises,istim,t],optim_rank[2,ises,istim] = rank_from_R2(R2_ranks[2,ises,istim,t,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                        R2_cv[3,ises,istim,t],optim_rank[3,ises,istim] = rank_from_R2(R2_ranks[3,ises,istim,t,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])


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
         sourcearealabelpairs=sourcearealabelpairs,
         targetarealabelpair=targetarealabelpair,
        #  params=params,allow_pickle=True)
         allow_pickle=True)

with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)

#%%