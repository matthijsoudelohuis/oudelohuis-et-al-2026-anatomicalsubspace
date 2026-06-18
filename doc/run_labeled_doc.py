# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
import numpy as np
from scipy.stats import zscore
import pickle

from loaddata.session_info import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.params import load_params
from datetime import datetime
from utils.tuning import compute_tuning_wrapper

#%% Load parameters and settings:
params = load_params()

# params['direction'] = 'FF'
params['direction'] = 'FB'

version = 'doc_labeled_%s' % (params['direction'])

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
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_noiselevel=False)
report_sessions(sessions)

#%% Wrapper function to load the tensor data
# params['calciumversion'] = 'deconv'
[sessions,t_axis] = load_resid_tensor(sessions,params,compute_respmat=True)

#%%
sessions = compute_tuning_wrapper(sessions)

#%% 
narealabelpairs     = len(sourcearealabelpairs)

np.random.seed(0)

params['nsubprojection'] = 25
params['nmodelfits'] = 100

fixed_rank          = 4

idx_resp            = (t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end'])
idx_resp            = t_axis>-9

# nT                  = len(t_axis)
nT                  = np.sum(idx_resp)

contrasts           = np.array([[0,1],[0,2]]) #which contrasts to use for the DOC rotation, e.g. V1unl vs V1lab, or V1unl vs V1unl
ncontrasts          = len(contrasts)

R2_ranks_orig       = np.full((narealabelpairs+1,ncontrasts,nSessions,params['nStim'],fixed_rank,params['nmodelfits']),np.nan)
R2_ranks_doc        = np.full((narealabelpairs+1,ncontrasts,nSessions,params['nStim'],fixed_rank,params['nmodelfits']),np.nan)

# R2_ranks_orig_t     = np.full((narealabelpairs+1,ncontrasts,nSessions,params['nStim'],nT,fixed_rank,params['nmodelfits']),np.nan)
# R2_ranks_doc_t      = np.full((narealabelpairs+1,ncontrasts,nSessions,params['nStim'],nT,fixed_rank,params['nmodelfits']),np.nan)

kf                  = KFold(n_splits=params['kfold'],shuffle=True)

for ises,ses in enumerate(sessions):
# for ises,ses in enumerate(sessions[:2]):
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
    
    if len(idx_areax1)<params['nsubprojection']*2 or len(idx_areax2)<params['nsubprojection']*2 or len(idx_areax3)<params['nsubprojection'] or len(idx_areay)<narealabelpairs*params['nsubprojection']: #skip exec if not enough neurons in one of the populations
        print('%d in %s, %d in %s' % (len(idx_areax3),sourcearealabelpairs[2],
                                                len(idx_areay),targetarealabelpair))
        continue

    for imf in tqdm(range(params['nmodelfits']),total=params['nmodelfits'],desc='Fitting RRR model for session %d/%d' % (ises+1,nSessions)):
        idx_areax1_sub       = np.random.choice(idx_areax1,params['nsubprojection'],replace=False)
        idx_areax2_sub       = np.random.choice(np.setdiff1d(idx_areax2,idx_areax1_sub),params['nsubprojection'],replace=False)
        idx_areax3_sub       = np.random.choice(idx_areax3,params['nsubprojection'],replace=False)
        idx_areay_sub        = np.random.choice(idx_areay,params['nsubprojection']*narealabelpairs,replace=False)

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        # for istim,stim in enumerate([0,4,7]): # loop over orientations 

            idx_T               = ses.trialdata['stimCond']==stim
            nK                  = np.sum(idx_T) #number of trials for this stimulus condition

            X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
            X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
            X3                  = sessions[ises].tensor[np.ix_(idx_areax3_sub,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

            # X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,np.arange(nT))]
            # X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,np.arange(nT))]
            # X3                  = sessions[ises].tensor[np.ix_(idx_areax3_sub,idx_T,np.arange(nT))]
            # Y                   = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,np.arange(nT))]

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

            B_rrr           = B_hat @ V.T @ V #project beta coeff into low rank subspace
            Y_hat_rr        = X_r @ B_rrr

            # OLD DOC with svd on the weight matrix
            # note that the latent space is not exactly the same as the one used in RRR reconstruction
            # Ub, sb, Vb = svds(B_rrr,k=fixed_rank,which='LM')
            # Ub, sb, Vb = Ub[:, ::-1], sb[::-1], Vb[::-1, :]
            
            # # Project latents into predictive subspace: Predictive X-directions scaled by their predictive power (eigenvalues)
            # Z = X @ B_hat @ V.T @ np.diag(s)

            # X_split = np.full((*X_r.shape,narealabelpairs),np.nan)
            X_split = np.repeat(X_r[:,:,np.newaxis],narealabelpairs,axis=2)
            
            X_split[:,params['nsubprojection']:,0] = 0
            X_split[:,:params['nsubprojection'],1] = 0
            X_split[:,2*params['nsubprojection']:,1] = 0
            X_split[:,:2*params['nsubprojection'],2] = 0

            Z_orig = np.full((X_r.shape[0],fixed_rank,narealabelpairs),np.nan)
            for ial in range(narealabelpairs):
                # Z_orig[:,:,ial] = X_split[:,:,ial] @ Ub @ np.diag(sb)
                Z_orig[:,:,ial] = X_split[:,:,ial] @ B_hat @ V.T  #project into the same latent space as the RRR reconstruction

            for icontrast, contrast in enumerate(contrasts):
                idx_resp_r = np.tile(idx_resp,nK)
                #Apply DOC: 
                # doc_eigvecs, doc_eigvals = doc_rotation(Z_orig[:,:,contrast[0]],Z_orig[:,:,contrast[1]])
                # doc_eigvecs, doc_eigvals = doc_rotation(Z_orig[idx_resp_r,:,contrast[0]],Z_orig[idx_resp_r,:,contrast[1]])
                doc_eigvecs, doc_eigvals = doc_rotation(Z_orig[:,:,contrast[0]],Z_orig[:,:,contrast[1]])

                # rotate into DOC space (OLD)
                # Vb_doc      = doc_eigvecs.T @ Vb              # shape: (rank, n_Y)
                # Z_full      = X_r @ Ub @ np.diag(sb)
                # Y_hat_rr    = Z_full @ Vb
                # Z_full_doc  = X_r @ Ub @ np.diag(sb) @ doc_eigvecs
                # Y_hat_doc   = Z_full_doc @ Vb_doc

                Z_full      = X_r @ B_hat @ V.T
                Y_hat_rr    = Z_full @ V
                Z_full_doc  = X_r @ B_hat @ V.T @ doc_eigvecs
                Y_hat_doc   = Z_full_doc @ doc_eigvecs.T @ V

                Y_hat_rr_svd = X_r @ B_rrr
                assert(np.all(np.max(np.abs(Y_hat_rr - Y_hat_rr_svd)) < 1e-10)), 'RRR reconstruction should be the same as the one from the latent space'

                assert(np.allclose(EV(Y_r,Y_hat_doc),EV(Y_r,Y_hat_rr),atol=1e-10)), 'DOC rotation should not change the overall R2'

                for r in range(fixed_rank):                    
                    # Y_hat_r = Z_full[:,r][:, np.newaxis] @ Vb[r,:][np.newaxis, :]
                    Y_hat_r = Z_full[:,r][:, np.newaxis] @ V[r,:][np.newaxis, :]
                    R2_ranks_orig[0,icontrast,ises,istim,r,imf] = EV(Y_r,Y_hat_r)

                    for ial in range(narealabelpairs):
                        Z_area_orig = Z_orig[:,r,ial]
                        # Y_hat_area_r = Z_area_orig[:, np.newaxis] @ Vb[r,:][np.newaxis, :]
                        Y_hat_area_r = Z_area_orig[:, np.newaxis] @ V[r,:][np.newaxis, :]
                        R2_ranks_orig[ial+1,icontrast,ises,istim,r,imf] = EV(Y_r,Y_hat_area_r)
                        # print(EV(Y_r,Y_hat_area_r))

                # Rotate activity of each area into DOC space
                Z_doc = np.full_like(Z_orig,np.nan)
                for ial in range(narealabelpairs):
                    Z_doc[:,:,ial] = Z_orig[:,:,ial] @ doc_eigvecs

                for r in range(fixed_rank):                    
                    # Y_hat_doc_r = Z_full_doc[:,r][:, np.newaxis] @ Vb_doc[r,:][np.newaxis, :]
                    Y_hat_doc_r = Z_full_doc[:,r][:, np.newaxis] @ doc_eigvecs[:,r].T[np.newaxis, :] @ V
                    R2_ranks_doc[0,icontrast,ises,istim,r,imf] = EV(Y_r,Y_hat_doc_r)
                    
                    # Y_hat_doc_perrank   = np.reshape(Y_hat_doc_r.T,(3*params['nsubprojection'],nK,nT),order='C')
                    
                    # for t in range(nT):
                    #     R2_ranks_doc_t[0,icontrast,ises,istim,t,r,imf] = EV(Y[:,:,t],Y_hat_doc_perrank[:,:,t])

                    for ial in range(narealabelpairs):
                        Z_area_doc = Z_doc[:,r,ial]
                        # Y_hat_area_doc_r = Z_area_doc[:, np.newaxis] @ Vb_doc[r,:][np.newaxis, :]
                        Y_hat_area_doc_r = Z_area_doc[:, np.newaxis] @ doc_eigvecs[:,r].T[np.newaxis, :] @ V
                        R2_ranks_doc[ial+1,icontrast,ises,istim,r,imf] = EV(Y_r,Y_hat_area_doc_r)

                        # Y_hat_area_doc = np.reshape(Y_hat_area_doc_r.T,(3*params['nsubprojection'],nK,nT),order='C')
                        # for t in range(nT):
                        #     R2_ranks_doc_t[ial+1,icontrast,ises,istim,t,r,imf] = EV(Y[:,:,t],Y_hat_area_doc[:,:,t])

#%%
params['fixed_rank']    = fixed_rank
params['nSessions'] = nSessions

#%% Save the data:
np.savez(savefilename + '.npz',
         contrasts=contrasts,
         R2_ranks_doc=R2_ranks_doc,
         R2_ranks_orig=R2_ranks_orig,
         sourcearealabelpairs=sourcearealabelpairs,
         targetarealabelpair=targetarealabelpair,
         narealabelpairs=narealabelpairs,
         allow_pickle=True)

with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)
