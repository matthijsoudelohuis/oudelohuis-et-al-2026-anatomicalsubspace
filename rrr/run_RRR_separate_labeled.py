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
params['regress_behavout'] = False
params['direction'] = 'FF'
# params['direction'] = 'FB'
# params['direction'] = 'FF_AL'
# params['direction'] = 'FB_AL'

version = 'Separate_labeled_%s_%s' % (params['direction'],'behavout' if params['regress_behavout'] else 'original')

resultdir = os.path.join(params['resultdir'])
if not os.path.exists(resultdir):
    os.makedirs(resultdir)
datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savefilename = os.path.join(resultdir,'RRR_%s_%s' % (version,datetime_str))

#%% Do RRR of V1 and PM labeled and unlabeled neurons simultaneously
if params['direction'] =='FF': 
    sourcearealabelpairs = ['V1unl','V1lab']
    targetarealabelpair = 'PMunl'
    only_all_areas = np.array(['V1','PM'])
elif params['direction'] =='FB': 
    sourcearealabelpairs = ['PMunl','PMlab']
    targetarealabelpair = 'V1unl'
    only_all_areas = np.array(['V1','PM'])
elif params['direction'] =='FF_AL': 
    sourcearealabelpairs = ['V1unl','V1lab']
    targetarealabelpair = 'ALunl'
    only_all_areas = np.array(['V1','PM','AL'])
elif params['direction'] =='FB_AL': 
    sourcearealabelpairs = ['PMunl','PMlab']
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
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_noiselevel=False)
report_sessions(sessions)

#%% Wrapper function to load the tensor data, 
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=params['regress_behavout'])

#%% 
narealabelpairs     = len(sourcearealabelpairs)

params['nsampleneurons']    = 20
params['nranks']     = 20 #number of ranks of RRR to be evaluated
params['nmodelfits']  = 100
params['nmodelfits']  = 10

params['nStim']     = 16
# params['radius']     = 30
fixed_rank          = None

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

R2_cv               = np.full((narealabelpairs,nSessions,params['nStim']),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
optim_rank          = np.full((narealabelpairs,nSessions,params['nStim']),np.nan)
R2_ranks            = np.full((narealabelpairs,nSessions,params['nStim'],
                               params['nranks'],params['nmodelfits'],params['kfold']),np.nan)

for ises,ses in enumerate(sessions):
    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    for isa, sourcearea in enumerate(sourcearealabelpairs):
        
        # alx,aly = arealabelpair.split('-')

    
        idx_areax           = np.where(np.all((ses.celldata['arealabel']==sourcearea,
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['arealabel']==targetarealabelpair,
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
        
        if len(idx_areax)<params['nsampleneurons']  or len(idx_areay)<params['nsampleneurons'] : #skip exec if not enough neurons in one of the populations
            print('%d in %s, %d in %s' % (len(idx_areax),sourcearea,
                                                len(idx_areay),targetarealabelpair))
            continue

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'][:2])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
       
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
                    
            # reshape to neurons x time points
            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
            R2_cv[isa,ises,istim],optim_rank[isa,ises,istim],R2_ranks[isa,ises,istim,:,:,:]  = RRR_wrapper(Y, X, 
                            nN=params['nsampleneurons'] ,nK=None,lam=params['lam'],nranks=params['nranks'],kfold=params['kfold'],
                            nmodelfits=params['nmodelfits'],fixed_rank=fixed_rank)
    
   

# #%% Find best rank and cvR2 at this rank:
# fixed_rank = None
# for ises in range(nSessions):
#     if np.any(~np.isnan(R2_ranks[0][ises])):
#         for istim in range(params['nStim']):
#             if fixed_rank is not None:
#                 rank = fixed_rank
#                 R2_cv[0,ises,istim] = np.nanmean(R2_ranks[0,ises,istim,rank,:,:])
#                 R2_cv[1,ises,istim] = np.nanmean(R2_ranks[1,ises,istim,rank,:,:])
#                 R2_cv[2,ises,istim] = np.nanmean(R2_ranks[2,ises,istim,rank,:,:])
#                 R2_cv[3,ises,istim] = np.nanmean(R2_ranks[3,ises,istim,rank,:,:])
#             else:
#                 if not np.isnan(R2_ranks[0][ises][istim]).all():
#                     R2_cv[0,ises,istim],optim_rank[0,ises,istim] = rank_from_R2(R2_ranks[0,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
#                     R2_cv[1,ises,istim],optim_rank[1,ises,istim] = rank_from_R2(R2_ranks[1,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
#                     R2_cv[2,ises,istim],optim_rank[2,ises,istim] = rank_from_R2(R2_ranks[2,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
#                     R2_cv[3,ises,istim],optim_rank[3,ises,istim] = rank_from_R2(R2_ranks[3,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])

#%%
params['nSessions'] = nSessions

#%% Save the data:
np.savez(savefilename + '.npz',R2_cv=R2_cv,R2_ranks=R2_ranks,optim_rank=optim_rank,
         sourcearealabelpairs=sourcearealabelpairs,
         targetarealabelpair=targetarealabelpair,
        #  params=params,allow_pickle=True)
         allow_pickle=True)

with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)

#%%