# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cross_decomposition import CCA
from datetime import datetime
import pickle

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.rf_lib import filter_nearlabeled
from utils.regress_lib import *
from utils.CCAlib import *
from utils.RRRlib import *
from utils.params import load_params

#%% Load parameters and settings:
params = load_params()

resultdir = os.path.join(params['resultdir'])
if not os.path.exists(resultdir):
    os.makedirs(resultdir)
datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savefilename = os.path.join(resultdir,'CCA_labeled_%s' % (datetime_str))

#%% Load example sessions:
session_list        = np.array([['LPE09665_2023_03_14'], #V1lab higher
                                ['LPE10885_2023_10_23'], #V1lab much higher
                                ])
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list)

#%% 
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],min_lab_cells_V1=20,min_lab_cells_PM=20)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False)

#%%

#%% 
 #####   #####     #       #     #    #   ######  #     #    #     # ####### ###  #####  #     # #######  #####  
#     # #     #   # #      #     #   ##   #     # ##   ##    #  #  # #        #  #     # #     #    #    #     # 
#       #        #   #     #     #  # #   #     # # # # #    #  #  # #        #  #       #     #    #    #       
#       #       #     #    #     #    #   ######  #  #  #    #  #  # #####    #  #  #### #######    #     #####  
#       #       #######     #   #     #   #       #     #    #  #  # #        #  #     # #     #    #          # 
#     # #     # #     #      # #      #   #       #     #    #  #  # #        #  #     # #     #    #    #     # 
 #####   #####  #     #       #     ##### #       #     #     ## ##  ####### ###  #####  #     #    #     #####  

#%% Are the weights higher for V1lab or PMlab than unlabeled neurons?
params['n_components']        = 20

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
arealabels          = np.array(['V1unl', 'V1lab', 'PMunl', 'PMlab'])
nStim               = 16
nmodelfits          = 10
minsampleneurons    = 10
weights_CCA         = np.full((params['n_components'],len(arealabels),nSessions,nStim,nmodelfits),np.nan)
cancorr_CCA         = np.full((params['n_components'],nSessions,nStim,nmodelfits),np.nan)
do_cv_cca           = True
fit_fast            = False

#%% Fit:
model_CCA           = CCA(n_components=params['n_components'],scale = False, max_iter = 1000)
# model_CCA           = PLSCanonical(n_components=params['n_components'],scale = False, max_iter = 1000)
for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
                                          ses.celldata['noise_level']<params['maxnoiselevel'],
                                          idx_nearby),axis=0)) for i in arealabels])
    # nsampleneurons = 10

    if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
        continue

    idx_N_all = np.empty(len(arealabels),dtype=object)
    for ial, al in enumerate(arealabels):
        idx_N_all[ial]           = np.where(np.all((ses.celldata['arealabel']==al,
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
    
    for imf in range(nmodelfits):
        idx_areax           = np.concatenate((np.random.choice(idx_N_all[0],nsampleneurons,replace=False),
                                            np.random.choice(idx_N_all[1],nsampleneurons,replace=False)))
        idx_areay           = np.concatenate((np.random.choice(idx_N_all[2],nsampleneurons,replace=False),
                                            np.random.choice(idx_N_all[3],nsampleneurons,replace=False)))
        assert len(idx_areax)==2*nsampleneurons and len(idx_areay)==2*nsampleneurons

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
        
            #on residual tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            X                   = zscore(X,axis=0,nan_policy='omit')  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0,nan_policy='omit')

            if fit_fast: 
                x_load, y_load, cancorr = cca_svd(X, Y, params['n_components'])
                weights_CCA[:,0,ises,istim,imf] = np.mean(np.abs(x_load[:nsampleneurons,:]),axis=0)
                weights_CCA[:,1,ises,istim,imf] = np.mean(np.abs(x_load[nsampleneurons:,:]),axis=0)
                weights_CCA[:,2,ises,istim,imf] = np.mean(np.abs(y_load[:nsampleneurons,:]),axis=0)
                weights_CCA[:,3,ises,istim,imf] = np.mean(np.abs(y_load[nsampleneurons:,:]),axis=0)
            else: 
                # Fit CCA MODEL:
                model_CCA.fit(X,Y)
                
                weights_CCA[:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_loadings_[:nsampleneurons,:]),axis=0)
                weights_CCA[:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_loadings_[nsampleneurons:,:]),axis=0)

                weights_CCA[:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_loadings_[:nsampleneurons,:]),axis=0)
                weights_CCA[:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_loadings_[nsampleneurons:,:]),axis=0)

            if do_cv_cca: #Implementing cross validation
                kf  = KFold(n_splits=params['kfold'], random_state=None,shuffle=True)
                corr_test = np.full((params['n_components'],params['kfold']),np.nan)
                # for train_index, test_index in kf.split(X):
                for ikf,(train_index, test_index) in enumerate(kf.split(X)):
                    X_train , X_test = X[train_index,:],X[test_index,:]
                    Y_train , Y_test = Y[train_index,:],Y[test_index,:]
                    
                    model_CCA.fit(X_train,Y_train)

                    X_c, Y_c = model_CCA.transform(X_test,Y_test)
                    for icomp in range(params['n_components']):
                        corr_test[icomp,ikf] = np.corrcoef(X_c[:,icomp],Y_c[:,icomp], rowvar = False)[0,1]
                cancorr_CCA[:,ises,istim,imf] = np.nanmean(corr_test,axis=1)

#%% 
 #####   #####     #       ######  ####### ######     ######  ####### ######     ######     #    ### ######  
#     # #     #   # #      #     # #       #     #    #     # #     # #     #    #     #   # #    #  #     # 
#       #        #   #     #     # #       #     #    #     # #     # #     #    #     #  #   #   #  #     # 
#       #       #     #    ######  #####   ######     ######  #     # ######     ######  #     #  #  ######  
#       #       #######    #       #       #   #      #       #     # #          #       #######  #  #   #   
#     # #     # #     #    #       #       #    #     #       #     # #          #       #     #  #  #    #  
 #####   #####  #     #    #       ####### #     #    #       ####### #          #       #     # ### #     # 

#%% Are the weights higher for V1lab or PMlab than unlabeled neurons?
Nsub                = 25
arealabelpairs      = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab']

narealabelpairs     = len(arealabelpairs)

CCA_corrtest        = np.full((narealabelpairs,params['n_components'],nSessions,nStim),np.nan)

#%% Fit:
model_CCA           = CCA(n_components=params['n_components'],scale = False, max_iter = 1000)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    #take the smallest sample size
    allpops             = np.array([i.split('-') for i in arealabelpairs]).flatten()
    nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
                                          ses.celldata['noise_level']<params['maxnoiselevel'],
                                          idx_nearby),axis=0)) for i in allpops])
    
    if nsampleneurons<Nsub: #skip session if less than minsampleneurons in either population
        continue
    
    for iapl, arealabelpair in enumerate(arealabelpairs):
        alx,aly = arealabelpair.split('-')

        if params['filter_nearby']:
            idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
        else:
            idx_nearby = np.ones(len(ses.celldata),dtype=bool)

        idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                ses.celldata['noise_level']<params['maxnoiselevel'],	
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                ses.celldata['noise_level']<params['maxnoiselevel'],	
                                idx_nearby),axis=0))[0]
    
        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
        
            #on residual tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            X                   = zscore(X,axis=0)  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0)

            [g,_] = CCA_subsample(X,Y,nN=Nsub,resamples=nmodelfits,kFold=params['kfold'],prePCA=None,n_components=np.min([params['n_components'],nsampleneurons]))
            CCA_corrtest[iapl,:len(g),ises,istim] = g

#%% Save parameters
params['Nsub']     = Nsub
params['nmodelfits'] = nmodelfits
params['nSessions'] = nSessions
params['idx_resp'] = idx_resp
params['t_axis'] = t_axis

#%% Save the data:
np.savez(savefilename + '.npz',
        arealabels=arealabels,
        weights_CCA=weights_CCA,
        cancorr_CCA=cancorr_CCA,
        CCA_corrtest=CCA_corrtest,
        arealabelpairs=arealabelpairs,
        allow_pickle=True)

with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)
