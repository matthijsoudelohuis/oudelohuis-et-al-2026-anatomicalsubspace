"""
This script has data loading functions used to get dirs, filter and select sessions
Actual loading happens as method of instances of sessions (session.py)
Matthijs Oude Lohuis, 2023, Champalimaud Foundation
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import zscore

from loaddata.get_data_folder import get_data_folder
from loaddata.session import Session
from utils.psth import * # get support functions for psth generation
from utils.RRRlib import regress_out_cv

def load_sessions(protocol, session_list, load_behaviordata=False, load_calciumdata=False, load_videodata=False, 
                calciumversion='dF',filter_areas=None):
    """
    This function loads and outputs the session objects that have to be loaded.
    session_list is a 2D np array with animal_id and session_id pairs (each row one session)
    example:
    session_list = np.array([['LPE11086', '2024_01_05']])
    sessions = load_sessions(protocol='GR',session_list)
    """
    sessions = []

    assert np.shape(session_list)[1] == 2, 'session list does not seem to have two columns for animal and dates'

    # iterate over sessions in requested array:
    for i, ses in enumerate(session_list):
        ses = Session(
            protocol=protocol, animal_id=session_list[i, 0], session_id=session_list[i, 1])

        if filter_areas is not None:
            ses.load_data(load_behaviordata=False,
                              load_calciumdata=False, load_videodata=False)
            ses.cellfilter = np.isin(ses.celldata['roi_name'],filter_areas)

        ses.load_data(load_behaviordata, load_calciumdata,
                      load_videodata, calciumversion)

        sessions.append(ses)

    report_sessions(sessions)

    return sessions, len(sessions)

def filter_sessions(protocols,load_behaviordata=False, load_calciumdata=False,
                    load_videodata=False, calciumversion='dF',only_animal_id=None,
                    only_session_id=None,min_noise_trials=None,
                    min_cells=None, min_lab_cells_V1=None, min_lab_cells_PM=None, 
                    filter_areas=None,min_trials=None, session_rf=None,
                    any_of_areas=None,only_all_areas=None,has_pupil=False,
                    filter_noiselevel=False,im_ses_with_repeats=False):
    """
    This function filters and returns a list of session objects that meet specific 
    criteria based on the input arguments. It allows the user to specify conditions 
    related to the behavioral data, calcium imaging data, video data, and session-specific 
    parameters like the number of trials and cell counts.
    Usage is as follows:

    sessions = filter_sessions(protocols,min_trials=100)
    
    'protocols' Specifies the experimental protocols to filter sessions by. 
    If a single string is provided, it's treated as one protocol. If a list is provided, 
    it filters by any of the listed protocols. If None, it defaults to a list of all protocols 
    (['VR','IM','GR','GN','RF','SP','DM','DN','DP']).

    min_lab_cells_V1:
    Description: Filters sessions to include only those with at least this many labeled 
    cells in the V1 region. If None, no filtering by labeled cell count in V1 is applied.
    Example: 10

    min_lab_cells_PM:
    Description: Filters sessions to include only those with at least this many labeled 
    cells in the PM region. If None, no filtering by labeled cell count in PM is applied.
    Example: 10

    any_of_areas (list of str, default: None):
    Description: Filters sessions to include only those with cells in any of the specified brain areas. 
    Example: ['V1', 'PM']

    only_all_areas (list of str, default: None):
    Description: Filters sessions to include only those that have cells in all the specified brain areas.
    Additional areas are allowed 
    Example: ['V1', 'PM']

    filter_areas (list of str, default: None):
    Description: Filters data to include only those that have cells in the specified brain areas. 
    Example: ['V1', 'PM']
    
    has_pupil (bool, default: False): Filters sessions to include only those that have pupil data available.
    Example: True    
    """
    sessions = []
    if isinstance(protocols, str):
        protocols = [protocols]

    if protocols is None:
        protocols = ['VR', 'IM', 'GR', 'GN', 'RF', 'SP', 'DM', 'DN', 'DP']

    # iterate over files in that directory
    for protocol in protocols:
        for animal_id in os.listdir(os.path.join(get_data_folder(), protocol)):
            for sessiondate in os.listdir(os.path.join(get_data_folder(), protocol, animal_id)):
                # ses = Session(
                # protocol=protocol, animal_id=session_list[i, 0], session_id=session_list[i, 1])


                ses = Session(protocol=protocol,animal_id=animal_id, sessiondate=sessiondate)
                ses.load_data(load_behaviordata=False,load_calciumdata=False, load_videodata=False)
                assert(ses.session_id == animal_id + '_' + sessiondate), "session_id != animal_id + '_' + sessiondate"
                
                # go through specified conditions that have to be met for the session to be included:
                sesflag = True

                # SELECT BASED ON ANIMAL ID
                if only_animal_id is not None:
                    sesflag = sesflag and animal_id in only_animal_id

                # SELECT BASED ON SESSION ID
                if only_session_id is not None:
                    sesflag = sesflag and ses.session_id in only_session_id

                #Remove sessions with too much drift in them:
                driftses = ['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']
                if ses.session_id in driftses and protocol in ['GR','GN','IM']:
                    sesflag = False

                # SELECT BASED ON # TRIALS
                if min_trials is not None:
                    sesflag = sesflag and len(ses.trialdata) >= min_trials

                # SELECT BASED ON # TRIALS
                if min_noise_trials is not None:
                    sesflag = sesflag and np.sum(ses.trialdata['stimcat']=='N') >= min_noise_trials

                # SELECT BASED ON # CELLS
                if sesflag and min_cells is not None:
                    sesflag = sesflag and hasattr(ses, 'celldata') and len(ses.celldata) >= min_cells
                
                # SELECT BASED ON # LABELED CELLS
                if sesflag and min_lab_cells_V1 is not None:
                    sesflag = sesflag and hasattr(ses, 'celldata') and np.sum(np.logical_and(ses.celldata['roi_name']=='V1',ses.celldata['redcell']==1)) >= min_lab_cells_V1
                if sesflag and min_lab_cells_PM is not None:
                    sesflag = sesflag and hasattr(ses, 'celldata') and np.sum(np.logical_and(ses.celldata['roi_name']=='PM',ses.celldata['redcell']==1)) >= min_lab_cells_V1

                # SELECT BASED ON RF
                if sesflag and session_rf is not None:
                    sesflag = sesflag and hasattr(
                        ses, 'celldata') and 'rf_r2_F' in ses.celldata

                # SELECT BASED ON WHETHER SESSION HAS DATA IN A PARTICULAR AREA:
                if sesflag and any_of_areas is not None:
                    sesflag = sesflag and hasattr(ses, 'celldata') and np.any(np.isin(any_of_areas,np.unique(ses.celldata['roi_name'])))

                # SELECT BASED ON WHETHER SESSION HAS DATA IN ALL SPECIFIED AREAS:
                if sesflag and only_all_areas is not None:
                    sesflag = sesflag and hasattr(ses, 'celldata') and np.all(np.isin(only_all_areas,np.unique(ses.celldata['roi_name'])))
                    # sesflag = sesflag and hasattr(ses, 'celldata') and np.all(np.isin(np.unique(ses.celldata['roi_name']),only_all_areas))
                
                # FILTER DATA TO ONLY LOAD DATA FROM SPECIFIED AREAS
                if sesflag and filter_areas is not None and hasattr(ses, 'celldata'):
                    cellfilter = np.isin(ses.celldata['roi_name'],filter_areas)
                    ses.cellfilter = np.logical_and(ses.cellfilter, cellfilter) if getattr(ses, 'cellfilter', None) is not None else cellfilter

                # FILTER DATA TO ONLY LOAD DATA BELOW NOISE LEVEL 20 (rupprecht et al. 2021)
                if sesflag and filter_noiselevel and hasattr(ses, 'celldata'):
                    cellfilter = np.array(ses.celldata['noise_level']<20)
                    ses.cellfilter = np.logical_and(ses.cellfilter, cellfilter) if getattr(ses, 'cellfilter', None) is not None else cellfilter
                
                # Select based on whether session has subset of natural images with more than 2 repeats:
                if sesflag and im_ses_with_repeats and protocol == 'IM':
                    sesflag = sesflag and np.any(ses.trialdata['ImageNumber'].value_counts()>2)

                # SELECT BASED ON WHETHER SESSION HAS PUPIL DATA MEASUREMENTS
                if sesflag and has_pupil:
                    ses.load_data(load_videodata=True)
                    sesflag = sesflag and hasattr(ses, 'videodata') and 'pupil_area' in ses.videodata and np.any(ses.videodata['pupil_area'])
                
                if sesflag: #if session meets all criteria, load data and append to list of sessions:
                    ses.load_data(load_behaviordata, load_calciumdata, load_videodata, calciumversion)
                    sessions.append(ses)

    # report_sessions(sessions)

    return sessions, len(sessions)

def report_sessions(sessions):
    """
    This function reports show stats about the loaded sessions 
    """

    sessiondata = pd.DataFrame()
    trialdata = pd.DataFrame()
    celldata = pd.DataFrame()

    for ses in sessions:
        sessiondata = pd.concat([sessiondata, ses.sessiondata])
        trialdata = pd.concat([trialdata, ses.trialdata])
        if hasattr(ses, 'celldata'):
            celldata = pd.concat([celldata, ses.celldata])

    print(
        f'{pd.unique(sessiondata["protocol"])} dataset: {len(pd.unique(sessiondata["animal_id"]))} mice, {len(sessiondata)} sessions, {len(trialdata)} trials')

    if np.any(celldata):
        for area in np.unique(celldata['roi_name']):
            print(
                f"Number of neurons in {area}: {len(celldata[celldata['roi_name'] == area])}")
        print(f"Total number of neurons: {len(celldata)}")

def load_resid_tensor(sessions,params,compute_respmat=True,subtract_mean_evoked=True,regressbehavout=False):
    #  Load data properly:        
    ## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
    ## Parameters for temporal binning

    nSessions = len(sessions)
    vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
                                ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

    behavfields = np.array(['runspeed','diffrunspeed'])

    # t_pre       = -1         #pre s
    # t_post      = 2.17        #post s
    # binsize     = 1/5.35

    for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
        sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                    calciumversion=params['calciumversion'])
        [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                    method='nearby')
        delattr(sessions[ises],'calciumdata')
        if regressbehavout:
            [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
                                        params['t_pre'], params['t_post'], method='binmean',binsize=params['binsize'])
            #Subsample behavioral data 10 times before binning:
            sessions[ises].behaviordata.drop('session_id',axis=1,inplace=True)
            sessions[ises].behaviordata = sessions[ises].behaviordata.groupby(sessions[ises].behaviordata.index // 10).mean()
            sessions[ises].behaviordata['diffrunspeed'] = np.diff(sessions[ises].behaviordata['runspeed'],prepend=0)
            [sessions[ises].tensor_run,t_axis] = compute_tensor(sessions[ises].behaviordata[behavfields], sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'], 
                                        params['t_pre'], params['t_post'], method='binmean',binsize=params['binsize'])
            
            delattr(sessions[ises],'behaviordata')
            delattr(sessions[ises],'videodata')

    if compute_respmat:
        for ises in range(nSessions):
            sessions[ises].respmat = np.nanmean(sessions[ises].tensor[:,:,t_axis>0],axis=(2))
            
            if regressbehavout: 
                sessions[ises].respmat_videome = np.nanmean(sessions[ises].tensor_vid[np.ix_([0],range(sessions[ises].tensor_vid.shape[1]),t_axis>0)],axis=(2)).squeeze()
                sessions[ises].respmat_runspeed = np.nanmean(sessions[ises].tensor_run[np.ix_([0],range(sessions[ises].tensor_run.shape[1]),t_axis>0)],axis=(2)).squeeze()

                sessions[ises].respmat_videome -= np.nanmin(sessions[ises].respmat_videome,keepdims=True)
                sessions[ises].respmat_videome /= np.nanmax(sessions[ises].respmat_videome,keepdims=True)

    if subtract_mean_evoked:
        #Subtracting mean response across trials for each stimulus condition
        for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Subtracting mean response across trials'):
            N = len(sessions[ises].celldata)
            idx_resp = t_axis>0
            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
                idx_T               = sessions[ises].trialdata['stimCond']==stim

                #on tensor during the response:
                sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)] -= np.nanmean(sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)],axis=1,keepdims=True)
            
            idx_resp = t_axis<0
            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
                idx_T               = np.concatenate([[0],sessions[ises].trialdata['stimCond'][:-1]])==stim

                #on tensor during the response:
                sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)] -= np.nanmean(sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)],axis=1,keepdims=True)

    if regressbehavout:
        rank_behavout = 5

        for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Regressing out behavior-related variability'):

            # if filter_nearby:
            #     idx_nearby  = filter_nearlabeled(ses,radius=30)
            # else:
            #     idx_nearby = np.ones(len(ses.celldata),dtype=bool)

            # sessions[ises].tensor_behavout = copy.copy(sessions[ises].tensor)
            #Get behavioral matrix:
            B                   = np.concatenate((sessions[ises].tensor_vid,
                                    sessions[ises].tensor_run),axis=0)
            areas = np.unique(ses.celldata['roi_name'])
            for area in areas:
                idx_N    = np.where(np.all((ses.celldata['roi_name']==area,
                                            # idx_nearby,
                                            ses.celldata['noise_level']<20),axis=0))[0]

                for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations
                    idx_T               = sessions[ises].trialdata['stimCond']==stim

                    Bstim                   = B[:,idx_T,:].reshape(np.shape(B)[0],-1).T
                    Bstim                   = zscore(Bstim,axis=0,nan_policy='omit')
                    Bstim                   = Bstim[:,~np.all(np.isnan(Bstim),axis=0)]

                    tempdata            = sessions[ises].tensor[np.ix_(idx_N,idx_T,np.arange(len(t_axis)))]
                    N,K,T = np.shape(tempdata)
                    Y_r = np.reshape(tempdata,(N,K*T),order='C').T
                    Y_orig,Y_hat,Y_out  = regress_out_cv(X=Bstim,Y=Y_r,rank=np.min([rank_behavout,len(idx_N)-1]),
                                                        lam=0,kfold=5)
                    # print(area,EV(Y_orig,Y_hat))
                    sessions[ises].tensor[np.ix_(idx_N,idx_T,np.arange(len(t_axis)))] = np.reshape(Y_out.T,(N,K,T),order='C')
                    # sessions[ises].tensor_behavout[np.ix_(idx_N,idx_T,np.arange(len(t_axis)))] = np.reshape(Y_out.T,(N,K,T),order='C')

    return sessions,t_axis


# def assign_layer(celldata):
#     celldata['layer'] = ''

#     layers = {
#         'V1': {
#             'L2/3': (0, 200),
#             'L4': (200, 275),
#             'L5': (275, np.inf)
#         },
#         'PM': {
#             'L2/3': (0, 200),
#             'L4': (200, 275),
#             'L5': (275, np.inf)
#         },
#         'AL': {
#             'L2/3': (0, 200),
#             'L4': (200, 275),
#             'L5': (275, np.inf)
#         },
#         'RSP': {
#             'L2/3': (0, 300),
#             'L5': (300, np.inf)
#         }
#     }

#     for roi, layerdict in layers.items():
#         for layer, (mindepth, maxdepth) in layerdict.items():
#             idx = celldata[(celldata['roi_name'] == roi) & (mindepth <= celldata['depth']) & (celldata['depth'] < maxdepth)].index
#             celldata.loc[idx, 'layer'] = layer
    
#     assert(celldata['layer'].notnull().all()), 'problematic assignment of layer based on ROI and depth'
    
#     #References: 
#     # V1: 
#     # Niell & Stryker, 2008 Journal of Neuroscience
#     # Gilman, et al. 2017 eNeuro
#     # RSC/PM:
#     # Zilles 1995 Rat cortex areal and laminar structure

#     return celldata


# def assign_layer2(celldata,splitdepth=300):
#     celldata['layer'] = ''

#     layers = {
#         'V1': {
#             'L2/3': (0, splitdepth),
#             'L5': (splitdepth, np.inf)
#         },
#         'PM': {
#             'L2/3': (0, splitdepth),
#             'L5': (splitdepth, np.inf)
#         },
#         'AL': {
#             'L2/3': (0, splitdepth),
#             'L5': (splitdepth, np.inf)
#         },
#         'RSP': {
#             'L2/3': (0, splitdepth),
#             'L5': (splitdepth, np.inf)
#         }
#     }

#     for roi, layerdict in layers.items():
#         for layer, (mindepth, maxdepth) in layerdict.items():
#             idx = celldata[(celldata['roi_name'] == roi) & (mindepth <= celldata['depth']) & (celldata['depth'] < maxdepth)].index
#             celldata.loc[idx, 'layer'] = layer
    
#     assert(celldata['layer'].notnull().all()), 'problematic assignment of layer based on ROI and depth'

#     return celldata