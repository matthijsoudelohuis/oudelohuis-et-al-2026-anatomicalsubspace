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

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.params import load_params
from datetime import datetime

#%% Load parameters and settings:
params = load_params()

# params['direction'] = 'FF'
params['direction'] = 'FB'

version = 'Joint_labeled_behavsubspace_%s' % (params['direction'])

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
                                # ['LPE11622_2024_03_25'], #same
                                ['LPE09665_2023_03_14'], #V1lab higher
                                # ['LPE10885_2023_10_23'], #V1lab much higher
                                ['LPE11086_2024_01_05'], #Really much higher, best session, first dimensions are more predictive.
                                ['LPE11086_2024_01_10'], #Few v1 labeled cells, very noisy
                                # ['LPE11998_2024_05_10'], #
                                # ['LPE12013_2024_05_07'], #
                                ['LPE11495_2024_02_28'], #
                                # ['LPE11086_2023_12_15'], #Same
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,
                                       min_lab_cells_V1=20,filter_noiselevel=False)

#%% Get all data 
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_noiselevel=False)
report_sessions(sessions)

#%% Wrapper function to load the tensor data
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False,load_behav=True)

#%% 
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 20
nranks              = 20 #number of ranks of RRR to be evaluated
# nmodelfits          = 100
nmodelfits          = 5
rankbehavout        = 3

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

nsubspaces          = 3 #full, behav, non-behav
R2_cv               = np.full((narealabelpairs+1,nsubspaces,nSessions,params['nStim']),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
optim_rank          = np.full((narealabelpairs+1,nsubspaces,nSessions,params['nStim']),np.nan)
R2_ranks            = np.full((narealabelpairs+1,nsubspaces,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)

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

    S                 = np.concatenate((sessions[ises].tensor_vid,
                        sessions[ises].tensor_run),axis=0)
    # print(np.shape(S))

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

            #Get the behavioraldata in the same shape as the neural data:
            Sstim                   = S[np.ix_(range(np.shape(S)[0]),idx_T,idx_resp)].reshape(np.shape(S)[0],-1).T
            Sstim                   = zscore(Sstim,axis=0,nan_policy='omit')
            Sstim                   = Sstim[:,~np.all(np.isnan(Sstim),axis=0)]
        
            X_orig,X_hat,X_out      = regress_out_cv(X=Sstim,Y=X,rank=rankbehavout,lam=0,kfold=5)

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

                for isubspace,X_test in enumerate([X[idx_test],X_hat[idx_test],X_out[idx_test]]):
                    for r in range(nranks):
                        B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                        Y_hat_test_rr   = X_test @ B_rrr

                        R2_ranks[0,isubspace,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)
                        
                        X_test_1 = copy.deepcopy(X_test)
                        X_test_1[:,Nsub:] = 0
                        Y_hat_test_rr   = X_test_1 @ B_rrr

                        R2_ranks[1,isubspace,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)

                        X_test_2 = copy.deepcopy(X_test)
                        X_test_2[:,:Nsub] = 0
                        X_test_2[:,2*Nsub:] = 0
                        Y_hat_test_rr   = X_test_2 @ B_rrr

                        R2_ranks[2,isubspace,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)

                        X_test_3 = copy.deepcopy(X_test)
                        X_test_3[:,:2*Nsub] = 0
                        Y_hat_test_rr   = X_test_3 @ B_rrr

                        R2_ranks[3,isubspace,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)

#%% Find best rank and cvR2 at this rank:
fixed_rank = None
for isubspace in range(nsubspaces):
    for ises in range(nSessions):
        if np.any(~np.isnan(R2_ranks[0,isubspace,ises])):
            for istim in range(params['nStim']):
                if fixed_rank is not None:
                    rank = fixed_rank
                    R2_cv[0,isubspace,ises,istim] = np.nanmean(R2_ranks[0,isubspace,ises,istim,rank,:,:])
                    R2_cv[1,isubspace,ises,istim] = np.nanmean(R2_ranks[1,isubspace,ises,istim,rank,:,:])
                    R2_cv[2,isubspace,ises,istim] = np.nanmean(R2_ranks[2,isubspace,ises,istim,rank,:,:])
                    R2_cv[3,isubspace,ises,istim] = np.nanmean(R2_ranks[3,isubspace,ises,istim,rank,:,:])
                else:
                    if not np.isnan(R2_ranks[0,isubspace,ises,istim]).all():
                        R2_cv[0,isubspace,ises,istim],optim_rank[0,isubspace,ises,istim] = rank_from_R2(R2_ranks[0,isubspace,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                        R2_cv[1,isubspace,ises,istim],optim_rank[1,isubspace,ises,istim] = rank_from_R2(R2_ranks[1,isubspace,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                        R2_cv[2,isubspace,ises,istim],optim_rank[2,isubspace,ises,istim] = rank_from_R2(R2_ranks[2,isubspace,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                        R2_cv[3,isubspace,ises,istim],optim_rank[3,isubspace,ises,istim] = rank_from_R2(R2_ranks[3,isubspace,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])

#%%
params['Nsub']          = Nsub
params['nranks']        = nranks
params['nmodelfits']    = nmodelfits
params['nSessions']     = nSessions
params['nsubspaces']    = nsubspaces

#%% Save the data:
np.savez(savefilename + '.npz',R2_cv=R2_cv,R2_ranks=R2_ranks,optim_rank=optim_rank,
         sourcearealabelpairs=sourcearealabelpairs,
         targetarealabelpair=targetarealabelpair,
        #  params=params,allow_pickle=True)
         allow_pickle=True)

with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)

#%%


# #%% Show an example session:
# clrs_arealabelpairs = ['grey','grey','red']
# narealabelpairs = 3
# statelabels = np.array(['Full','Behav','Non-behav'])
# ises = 0
# fig, axes = plt.subplots(1,3,figsize=(12*cm,4.6*cm),sharex=True,sharey=True)
# for isubspace in range(3):
#     handles = []
#     ax = axes[isubspace]
#     for iapl,apl in enumerate(sourcearealabelpairs):
#         ymeantoplot = np.nanmean(R2_ranks[iapl+1][isubspace][ises],axis=(0,2,3))
#         yerrortoplot = np.nanstd(R2_ranks[iapl+1][isubspace][ises],axis=(0,2,3)) / np.sqrt(nmodelfits)
#         handles.append(shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[iapl],alpha=0.3))

#     leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs),frameon=False)
#     my_legend_strip(ax)
#     ax.set_xlabel('Rank')
#     if isubspace == 0: 
#         ax.set_ylabel(r'R$^{2}$')
#     ax.set_title(statelabels[isubspace])

# plt.tight_layout()
# sns.despine(fig=fig,trim=False,top=True,right=True)
# # my_savefig(fig,figdir,'RRR_joint_cvR2_labunl_%s_ExampleSesion' % (version))
