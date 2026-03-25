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
# params['direction'] = 'FB'
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
                                # ['LPE11622_2024_03_25'], #same
                                ['LPE09665_2023_03_14'], #V1lab higher
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
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_noiselevel=False)
report_sessions(sessions)

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=params['regress_behavout'])


#%%
def doc_rotation(Zx, Zy, center=True):
    """
    Zx: (n_samples, r) projections of X into predictive subspace
    Zy: (n_samples, r) projections of Y into predictive subspace
    """
    if center:
        Zx = Zx - Zx.mean(axis=0, keepdims=True)
        Zy = Zy - Zy.mean(axis=0, keepdims=True)

    # covariance matrices
    Cx = np.cov(Zx, rowvar=False)
    Cy = np.cov(Zy, rowvar=False)

    # difference of covariances
    S = Cy - Cx

    # eigen decomposition
    eigvals, eigvecs = eigh(S)

    # sort descending (Y-dominant first)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvecs, eigvals

#%% 
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 20
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 100

params['dim_method'] = 'pca_shuffle'
params['nStim']     = 16
# params['radius']     = 30

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)
fixed_rank  = 5

# R2_cv               = np.full((narealabelpairs+1,nSessions,params['nStim']),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
# optim_rank          = np.full((narealabelpairs+1,nSessions,params['nStim']),np.nan)
R2_cv            = np.full((narealabelpairs+1,nSessions,params['nStim'],nmodelfits,params['kfold']),np.nan)
R2_cv_rot        = np.full((narealabelpairs+1,nSessions,params['nStim'],fixed_rank,nmodelfits,params['kfold']),np.nan)
# R2_ranks_neurons    = np.full((narealabelpairs+1,Nsub*narealabelpairs,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)


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
                U, s, V = svds(Y_hat_train,k=fixed_rank,which='LM')
                U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                B_rrr           = B_hat_train @ V.T @ V #project beta coeff into low rank subspace
                
                W               = B_rrr @ V.T   # Get predictive X-directions
                
                ## Get partially filled X
                X_train_1 = copy.deepcopy(X_test)
                X_train_2 = copy.deepcopy(X_test)
                X_train_3 = copy.deepcopy(X_test)

                X_train_1[:,Nsub:] = 0

                X_train_2[:,:Nsub] = 0
                X_train_2[:,2*Nsub:] = 0
                
                X_train_3[:,:2*Nsub] = 0

                X_test_1 = copy.deepcopy(X_test)
                X_test_2 = copy.deepcopy(X_test)
                X_test_3 = copy.deepcopy(X_test)

                X_test_1[:,Nsub:] = 0

                X_test_2[:,:Nsub] = 0
                X_test_2[:,2*Nsub:] = 0
                
                X_test_3[:,:2*Nsub] = 0

                X_pred1 = X_train_1 @ W
                X_pred2 = X_train_2 @ W
                X_pred3 = X_train_3 @ W

                doc_eigvecs, doc_eigvals = doc_rotation(X_pred1,X_pred3)

                # Rotate into DOC space
                X_doc1 = X_pred1 @ doc_eigvecs
                X_doc2 = X_pred2 @ doc_eigvecs
                X_doc3 = X_pred3 @ doc_eigvecs

                R2_cv[0,ises,istim,imf,ikf] = EV(Y_test,Y_hat_test_rr)
                
                


                X_test_1 = copy.deepcopy(X_test)
                X_test_1[:,Nsub:] = 0
                Y_hat_test_rr   = X_test_1 @ B_rrr
                Xpred_test_1    = X_test_1 @ W   # Project X onto predictive dimensions
                
                for r in range(fixed_rank):
                    R2_cv_rot[1,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)
                R2_cv[1,ises,istim,imf,ikf] = EV(Y_test,Y_hat_test_rr)

                X_test_2 = copy.deepcopy(X_test)
                X_test_2[:,:Nsub] = 0
                X_test_2[:,2*Nsub:] = 0
                Y_hat_test_rr   = X_test_2 @ B_rrr

                R2_cv[2,ises,istim,imf,ikf] = EV(Y_test,Y_hat_test_rr)

                X_test_3 = copy.deepcopy(X_test)
                X_test_3[:,:2*Nsub] = 0
                Y_hat_test_rr   = X_test_3 @ B_rrr

                R2_cv[3,ises,istim,imf,ikf] = EV(Y_test,Y_hat_test_rr)

#    Y_hat_rr = np.full((Y.shape[0],Y.shape[1],narealabelpairs),np.nan)

#             # SVD of predicted activity
#             U, S, Vt = np.linalg.svd(Y_hat, full_matrices=False)
#             V = Vt.T
#             # choose rank-k approximation
#             V_k = V[:, :fixed_rank]

#             W = B_hat @ V_k   # Predictive X-directions

#             # Q, R = np.linalg.qr(W)   # Orthonormalize predictive X subspace

#             Xpred = X @ W   # Project X onto predictive dimensions

#             R2[0,ises,istim,imf] = EV(Y,X @ B_rrr)
#             # print(EV(Y,Y_hat_rr))
#             # R2_ranks_neurons[0,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_rr, multioutput='raw_values')
            
#             X_sub_0 = copy.deepcopy(X)
#             X_sub_0[:,Nsub:] = 0
#             Y_hat_rr[:,:,0]   = X_sub_0 @ B_rrr

#             R2[1,ises,istim,imf] = EV(Y,Y_hat_rr[:,:,0])
#             # R2_ranks_neurons[1,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_rr, multioutput='raw_values')

#             X_sub_1 = copy.deepcopy(X)
#             X_sub_1[:,:Nsub] = 0
#             X_sub_1[:,2*Nsub:] = 0
#             Y_hat_rr[:,:,1]   = X_sub_1 @ B_rrr

#             R2[2,ises,istim,imf] = EV(Y,Y_hat_rr[:,:,1])
#             # R2_ranks_neurons[2,:,ises,istim,r,imf,ikf] = r2_score(Y_test,Y_hat_rr, multioutput='raw_values')

#             X_sub_2 = copy.deepcopy(X)
#             X_sub_2[:,:2*Nsub] = 0
#             Y_hat_rr[:,:,2]   = X_sub_2 @ B_rrr

#             R2[3,ises,istim,imf] = EV(Y,Y_hat_rr[:,:,2])

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
params['Nsub']     = Nsub
params['nranks']    = nranks
params['nmodelfits'] = nmodelfits
params['nSessions'] = nSessions

#%% Save the data:
np.savez(savefilename + '.npz',R2_cv=R2_cv,R2_ranks=R2_ranks,optim_rank=optim_rank,
         R2_ranks_neurons=R2_ranks_neurons,source_dim=source_dim,
         R2_sourcealigned=R2_sourcealigned,
         frac_pos_weight_out=frac_pos_weight_out,
         weights_in=weights_in,frac_pos_weight_in=frac_pos_weight_in,
         sourcearealabelpairs=sourcearealabelpairs,
         targetarealabelpair=targetarealabelpair,
        #  params=params,allow_pickle=True)
         allow_pickle=True)

with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)

#%%