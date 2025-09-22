"""
This script has data loading functions used to get dirs, filter and select sessions
Actual loading happens as method of instances of sessions (session.py)
Matthijs Oude Lohuis, 2023, Champalimaud Foundation
"""

import os
import numpy as np
import pandas as pd
import logging
from scipy.stats import zscore

from loaddata.get_data_folder import get_data_folder
from loaddata.session import Session
from utils.behaviorlib import * # get support functions for beh analysis 
from utils.psth import * # get support functions for psth generation
logger = logging.getLogger(__name__)


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

def load_neural_performing_sessions(calciumversion='deconv'):
    #Get signal as relative to psychometric curve for all sessions:
    sessions,nSessions = filter_sessions(protocols='DN',min_cells=1) #Load specified list of sessions
    np.random.seed(0)
    sessions = noise_to_psy(sessions,filter_engaged=True,bootstrap=True)
    # plot_psycurve(sessions,filter_engaged=True)

    idx_ses = get_idx_performing_sessions(sessions)

    # Filter sessions:
    sessions = [sessions[i] for i in np.where(idx_ses)[0]]
    nSessions = len(sessions)

    # Load the data:           
    for ises in range(nSessions):    # iterate over sessions
        sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                    calciumversion=calciumversion)
    # Z-score the calciumdata: 
    for i in range(nSessions):
        sessions[i].calciumdata = sessions[i].calciumdata.apply(zscore,axis=0)

    ## Construct spatial tensor: 3D 'matrix' of K trials by N neurons by S spatial bins
    s_pre       = -60  #pre cm
    s_post      = 80   #post cm
    binsize     = 10     #spatial binning in cm

    for i in range(nSessions):
        sessions[i].stensor,sbins    = compute_tensor_space(sessions[i].calciumdata,sessions[i].ts_F,sessions[i].trialdata['stimStart'],
                                        sessions[i].zpos_F,sessions[i].trialnum_F,s_pre=s_pre,s_post=s_post,binsize=binsize,method='binmean')
        # Compute average response in stimulus response zone:
        sessions[i].respmat             = compute_respmat_space(sessions[i].calciumdata, sessions[i].ts_F, sessions[i].trialdata['stimStart'],
                                        sessions[i].zpos_F,sessions[i].trialnum_F,s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)

        temp = pd.DataFrame(np.reshape(np.array(sessions[i].behaviordata['runspeed']),(len(sessions[i].behaviordata['runspeed']),1)))
        sessions[i].respmat_runspeed    = compute_respmat_space(temp, sessions[i].behaviordata['ts'], sessions[i].trialdata['stimStart'],
                                        sessions[i].behaviordata['zpos'],sessions[i].behaviordata['trialNumber'],s_resp_start=0,s_resp_stop=20,method='mean',subtr_baseline=False)


    return sessions,nSessions,sbins

    # print("{nneurons} dataset: {nsessions} sessions, {ntrials} trials".format(
        # protocol = sessions[0].sessiondata.protocol,nsessions = len(sessiondata),ntrials = len(trialdata))

    # # SELECT BASED ON NUMBER OF UNITS PER AREA
    # if min_units is not None:
    #     units_df = make_units_overview()
    #     units_df = units_df[units_df['n_units'] >= min_units]
    #     sel_df = pd.merge(units_df, trial_df, on=['animal_id', 'session_id'])

    # # SELECT BASED ON NUMBER OF UNITS/CHANNELS PER AREA/LAYER
    # if min_units_per_layer is not None or min_channels_per_layer is not None:
    #     layers_df = make_units_and_channels_overview()
    #     #layers_df_copy = layers_df.copy()

    #     if min_units_per_layer is not None:
    #         layers_df = layers_df[layers_df['n_units'] >= min_units_per_layer]

    #     if min_channels_per_layer is not None:
    #         layers_df = layers_df[
    #             layers_df['n_channels'] >= min_channels_per_layer]

    #     if exclude_NA_layer:
    #         layers_df = layers_df[layers_df['layer'] != 'NA']

    #     sel_df = pd.merge(layers_df, trial_df, on=['animal_id', 'session_id'])

    # try:
    #     return sel_df
    # except NameError:
    #     return trial_df


# if __name__ == '__main__':
#     # performance

#     only_correct = True
#     stimulus_type = None
#     min_trials_per_stim = 10
#     min_units = None
#     min_units_per_layer = 12
#     min_channels_per_layer = 10
#     min_perc_correct = 40
#     exclude_NA_layer = True

#     if min_units is not None and min_units_per_layer is not None:
#         raise ValueError('You can select either the minimum number of (total) units'
#                          'per area, or the minimum number of units per area/layer, '
#                          'not both!')

#     trial_df = make_trial_overview()

#     if min_trials_per_stim is not None:
#         trial_df = trial_df.loc[(trial_df.only_correct == only_correct) &
#                                     (trial_df.nt_0 >= min_trials_per_stim) &
#                                     (trial_df.nt_1 >= min_trials_per_stim)]

#     if stimulus_type is not None:
#         trial_df = trial_df[trial_df['stimulus_type'] == stimulus_type]

#     if min_perc_correct is not None:
#         trial_df = trial_df[trial_df['perc_corr'] >= min_perc_correct]


#     if min_units is not None:
#         units_df = make_units_overview()

#         units_df = units_df[units_df['n_units'] >= min_units]
#         sel_df = pd.merge(units_df, trial_df, on=['animal_id', 'session_id'])


#     if min_units_per_layer is not None or min_channels_per_layer is not None:
#         layers_df = make_units_and_channels_overview()
#         layers_df_copy = layers_df.copy()

#         if min_units_per_layer is not None:
#             layers_df = layers_df[layers_df['n_units'] >= min_units_per_layer]

#         if min_channels_per_layer is not None:
#             layers_df = layers_df[layers_df['n_channels'] >= min_channels_per_layer]

#         if exclude_NA_layer:
#             layers_df = layers_df[layers_df['layer'] != 'NA']

#         sel_df = pd.merge(layers_df, trial_df, on=['animal_id', 'session_id'])


# def make_trial_overview():

#     dfs = []

#     for animal_id in decoding_sessions.keys():
#         for session_id in decoding_sessions[animal_id]:
#             session = Session(animal_id=animal_id, session_id=session_id)
#             session.load_data(load_spikes=False, load_lfp=False)
#             df = session.get_session_overview_trials()
#             print(df)
#             dfs.append(df)

#     df = pd.concat(dfs).reset_index().drop('index', axis=1)

#     ind = df.loc[(df['animal_id'] == '2009') &
#            (df['session_id'] == '2018-08-24_11-56-35') &
#            (df['target_name'] == 'visualOriPostNorm')].index

#     df = df.drop(ind)

#     # just to make sure
#     sel = df.loc[(df['animal_id'] == '2009') &
#            (df['session_id'] == '2018-08-24_11-56-35') &
#            (df['target_name'] == 'visualOriPostNorm')]
#     assert sel.shape[0] == 0

#     return df


# def make_units_overview():

#     dfs = []

#     for animal_id in decoding_sessions.keys():
#         for session_id in decoding_sessions[animal_id]:
#             session = Session(animal_id=animal_id, session_id=session_id)
#             # TODO we should be able to provide the unit layer without loading
#             # the LFP
#             session.load_data(load_spikes=True, load_lfp=False)
#             df = session.get_session_overview_n_units()
#             dfs.append(df)

#     return pd.concat(dfs)
