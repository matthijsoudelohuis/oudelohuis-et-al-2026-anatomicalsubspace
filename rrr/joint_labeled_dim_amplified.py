# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
import numpy as np
from sklearn.decomposition import PCA
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
figdir = os.path.join(params['figdir'],'RRR','Labeling')

# params['regress_behavout'] = True
params['direction'] = 'FF'
params['direction'] = 'FB'

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
                                ['LPE10885_2023_10_23'], #V1lab much higher
                                ['LPE11086_2024_01_05'], #Really much higher, best session, first dimensions are more predictive.
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
# report_sessions(sessions)

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params)


#%% 
Nsub                = 25
fixed_rank          = 5

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

# R2_cv               = np.full((narealabelpairs+1,nSessions,params['nStim']),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
# optim_rank          = np.full((narealabelpairs+1,nSessions,params['nStim']),np.nan)
# R2_ranks            = np.full((narealabelpairs+1,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)
ises = 0
ses = sessions[ises]
narealabelpairs = 3

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

idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
idx_areax2_sub       = np.random.choice(np.setdiff1d(idx_areax2,idx_areax1_sub),Nsub,replace=False)
idx_areax3_sub       = np.random.choice(idx_areax3,Nsub,replace=False)
idx_areay_sub        = np.random.choice(idx_areay,Nsub*3,replace=False)

istim = 0
stim = 0
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

X_split = np.repeat(X_r[:,:,np.newaxis],narealabelpairs,axis=2)

X_split[:,params['nsubprojection']:,0] = 0
X_split[:,:params['nsubprojection'],1] = 0
X_split[:,2*params['nsubprojection']:,1] = 0
X_split[:,:2*params['nsubprojection'],2] = 0

Z = np.full((X_r.shape[0],fixed_rank,narealabelpairs),np.nan)
for ial in range(narealabelpairs):
    Z[:,:,ial] = X_split[:,:,ial] @ B_hat @ V.T  #project into the same latent space as the RRR reconstruction



A = LM(Z[:,:,2],Z[:,:,0])
A = LM(Z[:,:,1],Z[:,:,0])

U,s,V = svd(A)

plt.plot(s)




#%% 
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 25
params['nmodelfits'] = 50
fixed_rank          = 5

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

sing_vals           = np.full((2,fixed_rank,nSessions,params['nStim'],params['nmodelfits']),np.nan) 

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

            # X_split = np.full((*X_r.shape,narealabelpairs),np.nan)
            X_split = np.repeat(X_r[:,:,np.newaxis],narealabelpairs,axis=2)
            
            X_split[:,params['nsubprojection']:,0] = 0
            X_split[:,:params['nsubprojection'],1] = 0
            X_split[:,2*params['nsubprojection']:,1] = 0
            X_split[:,:2*params['nsubprojection'],2] = 0

            Z = np.full((X_r.shape[0],fixed_rank,narealabelpairs),np.nan)
            for ial in range(narealabelpairs):
                Z[:,:,ial] = X_split[:,:,ial] @ B_hat @ V.T  #project into the same latent space as the RRR reconstruction

            A = LM(Z[:,:,2],Z[:,:,0])
            U,s,V = svd(A)
            sing_vals[0,:,ises,istim,imf] = s

            A = LM(Z[:,:,1],Z[:,:,0])
            U,s,V = svd(A)
            sing_vals[1,:,ises,istim,imf] = s

#%%
fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm))
ax = axes[0]
ax.plot(np.arange(fixed_rank),np.nanmean(sing_vals[0],axis=(1,2,3)),color='red')
ax.plot(np.arange(fixed_rank),np.nanmean(sing_vals[1],axis=(1,2,3)),color='grey')
ax.set_ylabel('singular value')
ax.set_xlabel('singular index')

ax = axes[1]
ax.plot(np.arange(fixed_rank),np.nanmean(sing_vals[0]-sing_vals[1],axis=(1,2,3)),color='red')
ax.axhline(0,linestyle=':',color='grey')
ax.set_ylabel('difference in singular value')
ax.set_xlabel('singular index')
sns.despine(fig,trim=True,offset=2)
my_savefig(fig,figdir,'Number_Amplified_Latents_Joint_%s_%dneurons' % (params['direction'],Nsub))




