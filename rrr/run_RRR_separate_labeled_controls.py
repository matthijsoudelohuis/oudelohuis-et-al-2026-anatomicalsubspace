# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
import numpy as np
import pickle
from datetime import datetime

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.params import load_params
from utils.pair_lib import value_matching
from utils.tuning import compute_tuning_wrapper

#%% Load parameters and settings:
params = load_params()
params['regress_behavout'] = False
# params['direction'] = 'FF'
params['direction'] = 'FB'

version = 'Separate_labeled_controls_%s_%s' % (params['direction'],'behavout' if params['regress_behavout'] else 'original')

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

if 'sessions' in locals():
    del sessions # type: ignore

#%% 
session_list        = np.array([
                                # ['LPE12223_2024_06_10'], #V1lab actually lower
                                ['LPE09830_2023_04_10'], #V1 labeled higher predictive than V1unl
                                ['LPE09665_2023_03_14'], #V1lab higher
                                # ['LPE10885_2023_10_23'], #V1lab much higher
                                # ['LPE11086_2024_01_05'], #Really much higher, best session, first dimensions are more predictive.
                                # ['LPE11086_2024_01_10'], #Few v1 labeled cells, very noisy
                                # ['LPE11086_2023_12_15'], #Same
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,filter_noiselevel=False)

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_noiselevel=False)
report_sessions(sessions)

#%% Wrapper function to load the tensor data
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=params['regress_behavout'],compute_respmat=True)

#%% Compute tuning metrics
sessions = compute_tuning_wrapper(sessions)

#%% 
narealabelpairs     = len(sourcearealabelpairs)

params['nsampleneurons']    = 20
# params['nmodelfits']  = 100
# params['nmodelfits']  = 5

fixed_rank          = None

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

# valuematch_fields   = np.array(['depth','radius','noise_level','event_rate','skew','meanF'])
# valuematch_fields   = np.array(['radius','noise_level','event_rate','tuning_var'])
valuematch_fields   = np.array(['radius','noise_level','event_rate','gOSI','tuning_var'])

nvaluefields        = len(valuematch_fields)
nmatchbins          = 10
                               
R2_cv               = np.full((nvaluefields,narealabelpairs,nSessions,params['nStim']),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
optim_rank          = np.full((nvaluefields,narealabelpairs,nSessions,params['nStim']),np.nan)
R2_ranks            = np.full((nvaluefields,narealabelpairs,nSessions,params['nStim'],
                               params['nranks'],params['nmodelfits'],params['kfold']),np.nan)

for ises,ses in enumerate(sessions):
    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    for ivaluematch,valuematching in enumerate(valuematch_fields):
        idx_areax1           = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[0],
                                    ses.celldata['noise_level']<params['maxnoiselevel'],
                                    idx_nearby),axis=0))[0]
        idx_areax2           = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[1],
                                    ses.celldata['noise_level']<params['maxnoiselevel'],
                                    idx_nearby),axis=0))[0]
    
        if not valuematching in ses.celldata.keys():
                continue
        
        #Get value to match from celldata:
        values      = sessions[ises].celldata[valuematching].to_numpy()
        idx_joint   = np.concatenate((idx_areax1,idx_areax2))
        group       = np.concatenate((np.zeros(len(idx_areax1)),np.ones(len(idx_areax2))))
        idx_sub     = value_matching(idx_joint,group,values[idx_joint],bins=nmatchbins,showFig=False)
        idx_areax1   = np.intersect1d(idx_areax1,idx_sub) #recover subset from idx_joint
        idx_areax2   = np.intersect1d(idx_areax2,idx_sub)
    
        for isa, sourcearea in enumerate(sourcearealabelpairs):
            
            if isa==0:
                idx_areax = idx_areax1
            else:
                idx_areax = idx_areax2
                # idx_areax           = np.where(np.all((ses.celldata['arealabel']==sourcearea,
                #                     ses.celldata['noise_level']<params['maxnoiselevel'],
                #                     idx_nearby),axis=0))[0]
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
                R2_cv[ivaluematch,isa,ises,istim],optim_rank[ivaluematch,isa,ises,istim],R2_ranks[ivaluematch,isa,ises,istim,:,:,:]  = RRR_wrapper(Y, X, 
                                nN=params['nsampleneurons'] ,nK=None,lam=params['lam'],nranks=params['nranks'],kfold=params['kfold'],
                                nmodelfits=params['nmodelfits'],fixed_rank=fixed_rank)
    
#%% Save the data:
np.savez(savefilename + '.npz',R2_cv=R2_cv,R2_ranks=R2_ranks,optim_rank=optim_rank,
         sourcearealabelpairs=sourcearealabelpairs,
         targetarealabelpair=targetarealabelpair,
         valuematch_fields=valuematch_fields,
         allow_pickle=True)

#%% Save the parameters:
params['nSessions'] = nSessions
params['nvaluefields'] = nvaluefields
with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)
