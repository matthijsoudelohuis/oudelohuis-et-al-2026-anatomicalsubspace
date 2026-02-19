"""
This script contains functions to compute noise correlations
on simultaneously acquired calcium imaging data with mesoscope
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

## Import libs:
import os
import copy
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic,binned_statistic_2d
from skimage.measure import block_reduce
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

#Repeated measures ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
import itertools
import scipy.stats as ss
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from scipy.signal import detrend
from scipy.optimize import curve_fit
from scipy.stats import linregress

from utils.plot_lib import * #get all the fixed color schemes
from utils.tuning import mean_resp_gn,mean_resp_gr,mean_resp_image 
from utils.rf_lib import filter_nearlabeled
from utils.pair_lib import *
from statannotations.Annotator import Annotator

from utils.shuffle_lib import * 

 #####  ####### #     # ######  #     # ####### #######     #####  ####### ######  ######  
#     # #     # ##   ## #     # #     #    #    #          #     # #     # #     # #     # 
#       #     # # # # # #     # #     #    #    #          #       #     # #     # #     # 
#       #     # #  #  # ######  #     #    #    #####      #       #     # ######  ######  
#       #     # #     # #       #     #    #    #          #       #     # #   #   #   #   
#     # #     # #     # #       #     #    #    #          #     # #     # #    #  #    #  
 #####  ####### #     # #        #####     #    #######     #####  ####### #     # #     # 

def compute_trace_correlation(sessions,uppertriangular=True,binwidth=1):
    """
    Compute the trace correlation between the calcium traces of all neurons in a session
    Trace correlation is computed by taking the mean of the fluorescence traces over a specified time window (binwidth)
    Parameters
    sessions : Session
        list of Session objects
    uppertriangular : bool
        if set to True, only upper triangular part of the correlation matrix is computed
    binwidth : float
        time window over which to compute the mean of the fluorescence trace
    Returns sessions
    """

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing trace correlations: '):
    
        avg_nframes     = int(np.round(sessions[ises].sessiondata['fs'][0] * binwidth))

        if avg_nframes > 1:
            arr_reduced     = block_reduce(sessions[ises].calciumdata.T, block_size=(1,avg_nframes), func=np.mean, cval=np.mean(sessions[ises].calciumdata.T))
        else:
            arr_reduced     = sessions[ises].calciumdata.T.to_numpy()

        sessions[ises].trace_corr                   = np.corrcoef(arr_reduced)

        N           = np.shape(sessions[ises].calciumdata)[1] #get dimensions of response matrix

        idx_triu    = np.tri(N,N,k=0)==1 #index only upper triangular part
        
        if uppertriangular:
            sessions[ises].trace_corr[idx_triu] = np.nan
        else:
            np.fill_diagonal(sessions[ises].trace_corr,np.nan)

        assert np.all(sessions[ises].trace_corr[~idx_triu] > -1)
        assert np.all(sessions[ises].trace_corr[~idx_triu] < 1)
    return sessions    

def compute_signal_noise_correlation(sessions,uppertriangular=True,filter_stationary=False,remove_method=None,remove_rank=0):
    # computing the pairwise correlation of activity that is shared due to mean response (signal correlation)
    # or residual to any stimuli in GR and GN protocols (noise correlation).

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing signal and noise correlations: '):
        if sessions[ises].sessiondata['protocol'][0]=='IM':
            [respmean,imageids]         = mean_resp_image(sessions[ises])
            [N,K]                       = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            sessions[ises].sig_corr     = np.corrcoef(respmean)

            if np.any(sessions[ises].trialdata['ImageNumber'].value_counts()>2):
                stims = sessions[ises].trialdata['ImageNumber'].to_numpy()
                idx = sessions[ises].trialdata['ImageNumber'].value_counts().index
                ustim = idx[np.where(sessions[ises].trialdata['ImageNumber'].value_counts()>2)[0]]
                
                # noise_corr = np.empty((N,N,len(ustim)))
                # for istim,stim in enumerate(ustim):
                #     respmat_res             = sessions[ises].respmat[:,stims==stim]
                #     respmat_res             -= np.nanmean(respmat_res,axis=1,keepdims=True)
                #     noise_corr[:,:,istim]   = np.corrcoef(respmat_res)

                respmat_res = np.full((N,K),np.nan)
                for istim,stim in enumerate(ustim):
                    temp                    = sessions[ises].respmat[:,stims==stim]
                    respmat_res[:,stims==stim]   = temp - np.nanmean(temp,axis=1,keepdims=True)
                respmat_res = respmat_res[:,~np.isnan(respmat_res).all(axis=0)]
                sessions[ises].noise_corr       = np.corrcoef(respmat_res)
            else:
                sessions[ises].noise_corr = np.full((np.shape(sessions[ises].sig_corr)),np.nan)
            
            if uppertriangular:
                idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
                sessions[ises].sig_corr[idx_triu] = np.nan
                sessions[ises].noise_corr[idx_triu] = np.nan
            else: #set only autocorrelation to nan
                np.fill_diagonal(sessions[ises].sig_corr,np.nan)
                np.fill_diagonal(sessions[ises].noise_corr,np.nan)

        elif sessions[ises].sessiondata['protocol'][0]=='GR':
            [N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            oris                            = np.sort(sessions[ises].trialdata['Orientation'].unique())
            trialfilter                     = sessions[ises].respmat_runspeed<2 if filter_stationary else np.ones(K,bool)
            resp_meanori,respmat_res        = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
            prefori                         = oris[np.argmax(resp_meanori,axis=1)]

            sessions[ises].delta_pref       = np.abs(np.mod(np.subtract.outer(prefori, prefori),180))
            
            # Compute signal correlations on all trials: 
            # sessions[ises].sig_corr         = np.corrcoef(resp_meanori)

            #Compute signal correlation on separate halfs of trials:
            trialfilter                     = np.random.choice([True,False],size=(K),p=[0.5,0.5])
            resp_meanori1,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
            resp_meanori2,_                 = mean_resp_gr(sessions[ises],trialfilter=~trialfilter)
            sessions[ises].sig_corr         = 0.5 * (np.corrcoef(resp_meanori1, resp_meanori2)[:N, N:] +
                                                np.corrcoef(resp_meanori2, resp_meanori1)[:N, N:])

            # plt.imshow(sessions[ises].sig_corr,vmin=-0.4,vmax=0.4)

            if remove_method is not None:
                if remove_method in ['PCA','FA','RRR']:

                    assert remove_rank > 0, 'remove_rank must be > 0'	
                    
                    trial_ori   = sessions[ises].trialdata['Orientation']
                    respmat_res = copy.deepcopy(sessions[ises].respmat)
                    respmat_res = zscore(respmat_res,axis=1)
                    
                    # for iarea,area in enumerate(sessions[ises].celldata['roi_name'].unique()):
                    #     idx = sessions[ises].celldata['roi_name'] == area
                    #     data = respmat_res[idx,:]

                        # data_hat = remove_dim(data,remove_method,remove_rank)

                    #     #Remove low rank prediction from data:
                    #     respmat_res[idx,:] = data - data_hat
                    
                    for i,ori in enumerate(oris):
                        data = respmat_res[:,trial_ori==ori]
                        
                        data_hat = remove_dim(data,remove_method,remove_rank)
                        
                        #Remove low rank prediction from data:
                        respmat_res[:,trial_ori==ori] = data - data_hat
                elif remove_method == 'GM':
                    stimuli         = np.array(sessions[ises].trialdata['stimCond'])
                    data_hat        = pop_rate_gain_model(sessions[ises].respmat, stimuli)
                    respmat_res     = sessions[ises].respmat - data_hat

            # Compute noise correlations from residuals:
            # sessions[ises].noise_corr       = np.corrcoef(respmat_res)
            # Compute per stimulus, then average:
            trial_ori   = sessions[ises].trialdata['Orientation']
            noise_corr = np.empty((N,N,len(oris)))  
            for i,ori in enumerate(oris):
                noise_corr[:,:,i] = np.corrcoef(respmat_res[:,trial_ori==ori])
            sessions[ises].noise_corr       = np.mean(noise_corr,axis=2)

            idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
            if uppertriangular:
                sessions[ises].noise_corr[idx_triu] = np.nan
                sessions[ises].sig_corr[idx_triu] = np.nan
                sessions[ises].delta_pref[idx_triu] = np.nan
            else: #set only autocorrelation to nan
                np.fill_diagonal(sessions[ises].sig_corr,np.nan)
                np.fill_diagonal(sessions[ises].delta_pref,np.nan)
                np.fill_diagonal(sessions[ises].noise_corr,np.nan)

            assert np.all(sessions[ises].sig_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].sig_corr[~idx_triu] < 1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] < 1)
        
        elif sessions[ises].sessiondata['protocol'][0]=='GN':
            [N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            oris                            = np.sort(pd.Series.unique(sessions[ises].trialdata['centerOrientation']))
            speeds                          = np.sort(pd.Series.unique(sessions[ises].trialdata['centerSpeed']))
            trialfilter                     = sessions[ises].respmat_runspeed<2 if filter_stationary else np.ones(K,bool)
            resp_mean,respmat_res           = mean_resp_gn(sessions[ises],trialfilter)
            prefori, prefspeed              = np.unravel_index(resp_mean.reshape(N,-1).argmax(axis=1), (len(oris), len(speeds)))
            sessions[ises].prefori          = oris[prefori]
            sessions[ises].prefspeed        = speeds[prefspeed]

            # Compute signal correlations on all trials: 
            # sessions[ises].sig_corr         = np.corrcoef(resp_mean.reshape(N,len(oris)*len(speeds)))
            
            #Compute signal correlation on separate halfs of trials:
            trialfilter                     = np.random.choice([True,False],size=(K),p=[0.5,0.5])
            resp_mean1,_                    = mean_resp_gn(sessions[ises],trialfilter = trialfilter)
            resp_mean2,_                    = mean_resp_gn(sessions[ises],trialfilter = ~trialfilter)
            # sessions[ises].sig_corr         = 0.5 * (np.corrcoef(resp_mean1, resp_mean2)[:N, N:] +
                                                # np.corrcoef(resp_mean2, resp_mean1)[:N, N:])
            sessions[ises].sig_corr         = 0.5 * (np.corrcoef(resp_mean1.reshape(N,-1), resp_mean2.reshape(N,-1))[:N, N:] +
                                                np.corrcoef(resp_mean2.reshape(N,-1), resp_mean1.reshape(N,-1))[:N, N:])
            if remove_method is not None:
                if remove_method in ['PCA','FA','RRR']:
                    assert remove_rank > 0, 'remove_rank must be > 0'	
                    respmat_res = copy.deepcopy(sessions[ises].respmat)
                    respmat_res = zscore(respmat_res,axis=1)

                    trial_ori   = sessions[ises].trialdata['centerOrientation']
                    trial_spd   = sessions[ises].trialdata['centerSpeed']
                    for iO,ori in enumerate(oris):
                        for iS,speed in enumerate(speeds):
                            idx_trial = np.logical_and(trial_ori==ori,trial_spd==speed)
                            data = respmat_res[:,idx_trial]
                            data_hat = remove_dim(data,remove_method,remove_rank)
                            #Remove low rank prediction from data:
                            respmat_res[:,idx_trial] = data - data_hat
                elif remove_method == 'GM':
                    stimuli         = np.array(sessions[ises].trialdata['stimCond'])
                    data_hat        = pop_rate_gain_model(sessions[ises].respmat, stimuli)
                    respmat_res     = sessions[ises].respmat - data_hat

            # Detrend the data:
            # respmat_res = detrend(respmat_res,axis=1)

            #Compute noise correlations from residuals:
            sessions[ises].noise_corr       = np.corrcoef(respmat_res)

            idx_triu = np.tri(N,N,k=0)==1   #index upper triangular part
            if uppertriangular:
                sessions[ises].sig_corr[idx_triu] = np.nan
                sessions[ises].noise_corr[idx_triu] = np.nan
            else: #set autocorrelation to nan
                np.fill_diagonal(sessions[ises].sig_corr,np.nan)
                np.fill_diagonal(sessions[ises].noise_corr,np.nan)

            assert np.all(sessions[ises].sig_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].sig_corr[~idx_triu] < 1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] < 1)
        # else, do nothing, skipping protocol other than GR, GN, and IM'

    return sessions

#     # ###  #####  #######     #####  ####### ######  ######  
#     #  #  #     #    #       #     # #     # #     # #     # 
#     #  #  #          #       #       #     # #     # #     # 
#######  #   #####     #       #       #     # ######  ######  
#     #  #        #    #       #       #     # #   #   #   #   
#     #  #  #     #    #       #     # #     # #    #  #    #  
#     # ###  #####     #        #####  ####### #     # #     # 

def hist_corr_areas_labeling(sessions,corr_type='trace_corr',filternear=True,minNcells=10, 
                        areapairs=' ',layerpairs=' ',projpairs=' ',noise_thr=100,valuematching=None,
                        zscore=False,binres=0.01):
    # areas               = ['V1','PM']
    # redcells            = [0,1]
    # redcelllabels       = ['unl','lab']
    # legendlabels        = np.empty((4,4),dtype='object')

    binedges            = np.arange(-1,1,binres)
    bincenters          = binedges[:-1] + binres/2
    nbins               = len(bincenters)

    if zscore:
        binedges            = np.arange(-5,5,binres)
        bincenters          = binedges[:-1] + binres/2
        nbins               = len(bincenters)

    histcorr           = np.full((nbins,len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    meancorr           = np.full((len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    varcorr            = np.full((len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    fraccorr           = np.full((2,len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)

    for ises in tqdm(range(len(sessions)),desc='Averaging %s across sessions' % corr_type):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            if valuematching is not None:
                #Get value to match from celldata:
                values  = sessions[ises].celldata[valuematching].to_numpy()

                #For both areas match the values between labeled and unlabeled cells
                idx_V1      = sessions[ises].celldata['roi_name']=='V1'
                idx_PM      = sessions[ises].celldata['roi_name']=='PM'
                group       = sessions[ises].celldata['redcell'].to_numpy()
                idx_sub_V1  = value_matching(np.where(idx_V1)[0],group[idx_V1],values[idx_V1],bins=20,showFig=False)
                idx_sub_PM  = value_matching(np.where(idx_PM)[0],group[idx_PM],values[idx_PM],bins=20,showFig=False)
                
                # matchfilter2d  = np.isin(sessions[ises].celldata.index[:,None], np.concatenate([idx_sub_V1,idx_sub_PM])[None,:])
                # matchfilter    = np.logical_and(matchfilter2d,matchfilter2d.T)

                matchfilter1d = np.zeros(len(sessions[ises].celldata)).astype(bool)
                matchfilter1d[idx_sub_V1] = True
                matchfilter1d[idx_sub_PM] = True

                matchfilter    = np.meshgrid(matchfilter1d,matchfilter1d)
                matchfilter    = np.logical_and(matchfilter[0],matchfilter[1])

            else: 
                matchfilter = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

            if filternear:
                nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
                nearfilter      = np.meshgrid(nearfilter,nearfilter)
                nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
            else: 
                nearfilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

            if zscore:
                corrdata = corrdata/np.nanstd(corrdata,axis=None) - np.nanmean(corrdata,axis=None)
            
            rf_type = 'Fsmooth'
            if 'rf_r2_' + rf_type in sessions[ises].celldata:
                el              = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                az              = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                
                delta_el        = el[:,None] - el[None,:]
                delta_az        = az[:,None] - az[None,:]

                delta_rf        = np.sqrt(delta_az**2 + delta_el**2)
                rffilter        = delta_rf<50
            else: 
                rffilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

            for iap,areapair in enumerate(areapairs):
                for ilp,layerpair in enumerate(layerpairs):
                    for ipp,projpair in enumerate(projpairs):
                        signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<noise_thr,sessions[ises].celldata['noise_level']<noise_thr)
                        signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                        areafilter      = filter_2d_areapair(sessions[ises],areapair)

                        layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                        projfilter      = filter_2d_projpair(sessions[ises],projpair)

                        nanfilter       = ~np.isnan(corrdata)

                        proxfilter      = ~(sessions[ises].distmat_xy<10)

                        cellfilter      = np.all((signalfilter,areafilter,layerfilter,matchfilter,
                                                projfilter,proxfilter,nanfilter,nearfilter,rffilter),axis=0)

                        if np.sum(np.any(cellfilter,axis=0))>minNcells and np.sum(np.any(cellfilter,axis=1))>minNcells:
                            # if ipp==3:
                                # print(np.sum(cellfilter))
                            data      = corrdata[cellfilter].flatten()

                            histcorr[:,ises,iap,ilp,ipp]    = np.histogram(data,bins=binedges,density=True)[0]
                            meancorr[ises,iap,ilp,ipp]      = np.nanmean(data)
                            varcorr[ises,iap,ilp,ipp]       = np.nanstd(data)

                            if corr_type == 'trace_corr':
                                n = len(sessions[ises].ts_F)
                            elif corr_type in ['noise_corr','sig_corr','resp_corr','corr_shuffle']:
                                n = np.shape(sessions[ises].respmat)[1]

                            sigcorrdata = corrdata.copy()
                            sigcorrdata = filter_corr_p(sigcorrdata,n,p_thr=0.01)
                            fraccorr[0,ises,iap,ilp,ipp]       = np.sum(np.logical_and(cellfilter,sigcorrdata>0)) / np.sum(cellfilter)
                            fraccorr[1,ises,iap,ilp,ipp]       = np.sum(np.logical_and(cellfilter,sigcorrdata<0)) / np.sum(cellfilter)

    return bincenters,histcorr,meancorr,varcorr,fraccorr


#     # #######    #    #     #     #####  ####### ######  ######  
##   ## #         # #   ##    #    #     # #     # #     # #     # 
# # # # #        #   #  # #   #    #       #     # #     # #     # 
#  #  # #####   #     # #  #  #    #       #     # ######  ######  
#     # #       ####### #   # #    #       #     # #   #   #   #   
#     # #       #     # #    ##    #     # #     # #    #  #    #  
#     # ####### #     # #     #     #####  ####### #     # #     # 

def mean_corr_areas_labeling(sessions,corr_type='trace_corr',absolute=False,
                             filternear=True,filtersign=None,minNcells=10):
    areas               = ['V1','PM']
    redcells            = [0,1]
    redcelllabels       = ['unl','lab']
    legendlabels        = np.empty((4,4),dtype='object')

    meancorr            = np.full((4,4,len(sessions)),np.nan)
    fraccorr            = np.full((4,4,len(sessions)),np.nan)

    for ises in tqdm(range(len(sessions)),desc='Averaging %s across sessions' % corr_type):
        idx_nearfilter = filter_nearlabeled(sessions[ises],radius=50)
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            
            if filtersign == 'neg':
                corrdata[corrdata>0] = np.nan
            
            if filtersign =='pos':
                corrdata[corrdata<0] = np.nan

            if absolute:
                corrdata = np.abs(corrdata)

            for ixArea,xArea in enumerate(areas):
                for iyArea,yArea in enumerate(areas):
                    for ixRed,xRed in enumerate(redcells):
                        for iyRed,yRed in enumerate(redcells):

                                idx_source = sessions[ises].celldata['roi_name']==xArea
                                idx_target = sessions[ises].celldata['roi_name']==yArea

                                idx_source = np.logical_and(idx_source,sessions[ises].celldata['redcell']==xRed)
                                idx_target = np.logical_and(idx_target,sessions[ises].celldata['redcell']==yRed)

                                idx_source = np.logical_and(idx_source,sessions[ises].celldata['noise_level']<20)
                                idx_target = np.logical_and(idx_target,sessions[ises].celldata['noise_level']<20)

                                # if 'rf_p_F' in sessions[ises].celldata:
                                #     idx_source = np.logical_and(idx_source,sessions[ises].celldata['rf_p_F']<0.001)
                                    # idx_target = np.logical_and(idx_target,sessions[ises].celldata['rf_p_F']<0.001)

                                # if 'tuning_var' in sessions[ises].celldata:
                                #     idx_source = np.logical_and(idx_source,sessions[ises].celldata['tuning_var']>0.05)
                                #     idx_target = np.logical_and(idx_target,sessions[ises].celldata['tuning_var']>0.05)

                                if filternear:
                                    idx_source = np.logical_and(idx_source,idx_nearfilter)
                                    idx_target = np.logical_and(idx_target,idx_nearfilter)

                                if np.sum(idx_source)>minNcells and np.sum(idx_target)>minNcells:	
                                    meancorr[ixArea*2 + ixRed,iyArea*2 + iyRed,ises]  = np.nanmean(corrdata[np.ix_(idx_source, idx_target)])
                                    fraccorr[ixArea*2 + ixRed,iyArea*2 + iyRed,ises] = (
                                        np.sum(~np.isnan(corrdata[np.ix_(idx_source, idx_target)])) /
                                        corrdata[np.ix_(idx_source, idx_target)].size
                                    )

                                legendlabels[ixArea*2 + ixRed,iyArea*2 + iyRed]  = areas[ixArea] + redcelllabels[ixRed] + '-' + areas[iyArea] + redcelllabels[iyRed]

    # assuming meancorr and legeldlabels are 4x4xnSessions array
    upper_tri_indices           = np.triu_indices(4, k=0)
    meancorr_upper_tri          = meancorr[upper_tri_indices[0], upper_tri_indices[1], :]
    fraccorr_upper_tri          = fraccorr[upper_tri_indices[0], upper_tri_indices[1], :]
    
    # assuming legendlabels is a 4x4 array
    # legendlabels_upper_tri      = legendlabels[np.triu_indices(4, k=0)]
    legendlabels_upper_tri      = legendlabels[upper_tri_indices[0], upper_tri_indices[1]]

    df_mean                     = pd.DataFrame(data=meancorr_upper_tri.T,columns=legendlabels_upper_tri)
    df_frac                     = pd.DataFrame(data=fraccorr_upper_tri.T,columns=legendlabels_upper_tri)

    colorder                    = [0,1,4,7,8,9,2,3,5,6]
    legendlabels_upper_tri      = legendlabels_upper_tri[colorder]
    df_mean                     = df_mean[legendlabels_upper_tri]
    df_frac                     = df_frac[legendlabels_upper_tri]

    return df_mean,df_frac


######  ### #     #    #     # #######    #    #     #           #     # #     # 
#     #  #  ##    #    ##   ## #         # #   ##    #    #####   #   #   #   #  
#     #  #  # #   #    # # # # #        #   #  # #   #    #    #   # #     # #   
######   #  #  #  #    #  #  # #####   #     # #  #  #    #    #    #       #    
#     #  #  #   # #    #     # #       ####### #   # #    #    #   # #      #    
#     #  #  #    ##    #     # #       #     # #    ##    #    #  #   #     #    
######  ### #     #    #     # ####### #     # #     #    #####  #     #    #    

def bin_corr_deltaxy(sessions,method='mean',areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='noise_corr',rf_type='F',
                    rotate_prefori=False,deltaori=None,noise_thr=100,onlysameplane=False,
                    binresolution=5,tuned_thr=0,absolute=False,normalize=False,dsi_thr=0,
                    filtersign=None,corr_thr=0.05,shufflefield=None):
    """
    Binning pairwise correlations as a function of pairwise delta x and y position.
    - Sessions are binned by areapairs, layerpairs, and projpairs.
    - Returns binmean,bincount,binedges

    Parameters
    ----------
    sessions : list
        list of sessions
    areapairs : list (if ' ' then all areapairs are used)
        list of areapairs
    layerpairs : list  (if ' ' then all layerpairs are used)
        list of layerpairs
    projpairs : list  (if ' ' then all projpairs are used)
        list of projpairs
    corr_type : str, optional
        type of correlation to use, by default 'trace_corr'
    normalize : bool, optional
        whether to normalize correlations to the mean correlation at distances < 60 um, by default False
    sig_thr : float, optional
        significance threshold for including cells in the analysis, by default 0.001
    """

    #Binning parameters 2D:
    binlim          = 600
    binedges_2d     = np.arange(-binlim,binlim,binresolution)+binresolution/2 
    bincenters_2d   = binedges_2d[:-1]+binresolution/2 
    nBins           = len(bincenters_2d)

    bin_2d          = np.zeros((nBins,nBins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_2d_count    = np.zeros((nBins,nBins,len(areapairs),len(layerpairs),len(projpairs)))

    #Binning parameters 1D distance
    binlim          = 600
    binedges_dist   = np.arange(0,binlim,binresolution)+binresolution/2 
    binsdRF = binedges_dist[:-1]+binresolution/2 
    nBins           = len(binsdRF)

    bin_dist        = np.zeros((nBins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_dist_count  = np.zeros((nBins,len(areapairs),len(layerpairs),len(projpairs)))

    #Binning parameters 1D angle
    polarbinres         = 45
    centerthr           = [15,15,15]
    binedges_angle      = np.deg2rad(np.arange(0-polarbinres/2,360,step=polarbinres))
    bincenters_angle    = binedges_angle[:-1]+np.deg2rad(polarbinres/2)
    npolarbins          = len(bincenters_angle)

    bin_angle_cent      = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_angle_cent_count = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))

    bin_angle_surr      = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_angle_surr_count = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D corr histograms maps: '):

        celldata        = copy.deepcopy(sessions[ises].celldata)
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()

            if shufflefield == 'RF':
                celldata['rf_el_' + rf_type],celldata['rf_az_' + rf_type] = my_shuffle_celldata_joint(celldata['rf_el_' + rf_type],celldata['rf_az_' + rf_type],
                                                                celldata['roi_name'])
            elif shufflefield == 'XY':
                celldata['xloc'],celldata['yloc'] = my_shuffle_celldata_joint(celldata['xloc'],celldata['yloc'],
                                                                celldata['roi_name'])
            elif shufflefield == 'corrdata':
                corrdata = my_shuffle(corrdata,method='random',axis=None)
            elif shufflefield is not None:
                celldata = my_shuffle_celldata(celldata,shufflefield,keep_roi_name=True)

            delta_x        = celldata['xloc'].to_numpy()[:,None] - celldata['xloc'].to_numpy()[None,:]
            delta_y        = celldata['yloc'].to_numpy()[:,None] - celldata['yloc'].to_numpy()[None,:]
            delta_xy       = np.sqrt(delta_x**2 + delta_y**2)
            angle_xy       = np.mod(np.arctan2(delta_x,delta_y)-np.pi,np.pi*2)
            angle_xy       = np.mod(angle_xy+np.deg2rad(polarbinres/2),np.pi*2) - np.deg2rad(polarbinres/2)
            
            if absolute == True:
                corrdata = np.abs(corrdata)

            if normalize == True:
                corrdata = corrdata/np.nanstd(corrdata,axis=None) - np.nanmean(corrdata,axis=None)

            if method=='mean':
                if filtersign == 'neg':
                    corrsignfilter              = corrdata < 0
                elif filtersign =='pos':
                    corrsignfilter              = corrdata > 0
                else:
                    corrsignfilter = np.ones((len(celldata),len(celldata))).astype(bool)
            elif method=='frac':
                corrsignfilter = np.ones((len(celldata),len(celldata))).astype(bool)
                if filtersign == 'neg':
                    fracsignfilter              = corrdata < np.nanpercentile(corrdata,(corr_thr*100))
                elif filtersign =='pos':
                    fracsignfilter              = corrdata > np.nanpercentile(corrdata,(100-corr_thr*100))
                else:
                    raise ValueError('filtersign must be either pos or neg if metohd==frac is chosen')
            else: 
                raise ValueError('invalid method to apply to bins')

            if onlysameplane:
                planefilter    = np.meshgrid(celldata['plane_idx'],celldata['plane_idx'])
                planefilter    = planefilter[0] == planefilter[1]
            else:
                planefilter    = np.ones((len(celldata),len(celldata))).astype(bool)

            for iap,areapair in enumerate(areapairs):
                for ilp,layerpair in enumerate(layerpairs):
                    for ipp,projpair in enumerate(projpairs):
                        signalfilter    = np.meshgrid(celldata['noise_level']<noise_thr,celldata['noise_level']<noise_thr)
                        signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                        if tuned_thr:
                            tuningfilter    = np.meshgrid(celldata['tuning_var']>tuned_thr,celldata['tuning_var']>tuned_thr)
                            tuningfilter    = np.logical_and(tuningfilter[0],tuningfilter[1])
                        else: 
                            tuningfilter    = np.ones(np.shape(signalfilter))

                        areafilter      = filter_2d_areapair(sessions[ises],areapair)

                        layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                        projfilter      = filter_2d_projpair(sessions[ises],projpair)

                        nanfilter       = np.all((~np.isnan(corrdata),~np.isnan(delta_xy)),axis=0)

                        if deltaori is not None:
                            if isinstance(deltaori,(float,int)):
                                deltaori = np.array([deltaori,deltaori])
                            if np.shape(deltaori) == (1,):
                                deltaori = np.tile(deltaori,2)
                            assert np.shape(deltaori) == (2,),'deltaori must be a 2x1 array'
                            delta_pref = np.mod(sessions[ises].delta_pref,90) #convert to 0-90, direction tuning is ignored
                            delta_pref[sessions[ises].delta_pref == 90] = 90 #after modulo operation, restore 90 as 90
                            deltaorifilter = np.all((delta_pref >= deltaori[0], #find all entries with delta_pref between deltaori[0] and deltaori[1]
                                                    delta_pref <= deltaori[1]),axis=0)
                        else:
                            deltaorifilter = np.ones(np.shape(signalfilter)).astype(bool)

                        #Combine all filters into a single filter:
                        cellfilter      = np.all((signalfilter,tuningfilter,areafilter,corrsignfilter,
                                            layerfilter,projfilter,nanfilter,deltaorifilter),axis=0)

                        if np.any(cellfilter):
                            # valuedata are the correlation values, these are going to be binned
                            vdata           = corrdata[cellfilter].flatten()

                            #First 2D binning: x is elevation, y is azimuth, 
                            xdata               = delta_x[cellfilter].flatten()
                            ydata               = delta_y[cellfilter].flatten()
                            
                            #Take the sum of the correlations in each bin:
                            if method == 'mean': 
                                bin_2d[:,:,iap,ilp,ipp]   += binned_statistic_2d(x=xdata, y=ydata, values=vdata,bins=binedges_2d, statistic='sum')[0]
                            elif method == 'frac':
                                bin_2d[:,:,iap,ilp,ipp]   += np.histogram2d(x=delta_x[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), 
                                        y=delta_y[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), bins=binedges_2d)[0]                                       

                            # Count how many correlation observations are in each bin:
                            bin_2d_count[:,:,iap,ilp,ipp]  += np.histogram2d(x=xdata,y=ydata,bins=binedges_2d)[0]

                            #Now 1D, so only by deltarf:
                            xdata           = delta_xy[cellfilter].flatten()
                            if method == 'mean': 
                                bin_dist[:,iap,ilp,ipp] += binned_statistic(x=xdata,values=vdata,statistic='sum', bins=binedges_dist)[0]
                            elif method == 'frac':
                                bin_dist[:,iap,ilp,ipp] += np.histogram(delta_xy[np.all((cellfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_dist)[0]
                            bin_dist_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_dist)[0]

                            #Now polar binning:
                            tempfilter      = np.all((cellfilter,delta_xy<centerthr[iap]),axis=0)
                            vdata           = corrdata[tempfilter].flatten()
                            xdata           = angle_xy[tempfilter].flatten() #x is angle of rf difference

                            if method == 'mean': 
                                if np.any(tempfilter):
                                    bin_angle_cent[:,iap,ilp,ipp]  += binned_statistic(x=xdata,values=vdata,
                                                                statistic='sum',bins=binedges_angle)[0]
                            elif method == 'frac':
                                bin_angle_cent[:,iap,ilp,ipp] += np.histogram(angle_xy[np.all((tempfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_angle)[0]
                            bin_angle_cent_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_angle)[0]
                            
                            tempfilter      = np.all((cellfilter,delta_xy>centerthr[iap]),axis=0)
                            vdata           = corrdata[tempfilter].flatten()
                            xdata           = angle_xy[tempfilter].flatten() #x is angle of rf difference
                            
                            if method == 'mean': 
                                if np.any(tempfilter):
                                    bin_angle_surr[:,iap,ilp,ipp]  += binned_statistic(x=xdata,values=vdata,
                                                                statistic='sum',bins=binedges_angle)[0]
                            elif method == 'frac':
                                bin_angle_surr[:,iap,ilp,ipp] += np.histogram(angle_xy[np.all((tempfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_angle)[0]
                            bin_angle_surr_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_angle)[0]
        
    # divide the total summed correlations by the number of counts in that bin to get the mean:
    bin_2d = bin_2d / bin_2d_count
    bin_dist = bin_dist / bin_dist_count
    bin_angle_cent = bin_angle_cent / bin_angle_cent_count
    bin_angle_surr = bin_angle_surr / bin_angle_surr_count

    return bincenters_2d,bin_2d,bin_2d_count,bin_dist,bin_dist_count,binsdRF,bin_angle_cent,bin_angle_cent_count,bin_angle_surr,bin_angle_surr_count,bincenters_angle


def bin_corr_distance(sessions,areapairs,corr_type='trace_corr',normalize=False,absolute=False):
    binedges = np.arange(0,1000,20) 
    nbins= len(binedges)-1
    binmean = np.full((len(sessions),len(areapairs),nbins),np.nan)
    for ises in tqdm(range(len(sessions)),desc= 'Computing pairwise correlations across antom. distance: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            
            if absolute:
                corrdata = np.abs(corrdata)
            # corrdata[corrdata<0] = np.nan
            for iap,areapair in enumerate(areapairs):
                areafilter      = filter_2d_areapair(sessions[ises],areapair)
                nanfilter       = ~np.isnan(corrdata)
                cellfilter      = np.all((areafilter,nanfilter),axis=0)
                # binmean[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_xy[cellfilter].flatten(),
                binmean[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_xyz[cellfilter].flatten(),
                                                    values=corrdata[cellfilter].flatten(),
                                                    statistic='mean', bins=binedges)[0]
            
    if normalize: # subtract mean NC from every session:
        binmean = binmean - np.nanmean(binmean[:,:,binedges[:-1]<600],axis=2,keepdims=True)

    return binmean,binedges


def plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type):
    clrs_areapairs = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs = [clrs_areapairs]
    fig,axes = plt.subplots(1,1,figsize=(3.5,3))
    handles = []
    ax = axes
    for iap,areapair in enumerate(areapairs):
        for ises in range(len(sessions)):
            ax.plot(binedges[:-1],binmean[ises,iap,:].squeeze(),linewidth=0.15,color=clrs_areapairs[iap])
        handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,:].squeeze(),
                                    error='sem',color=clrs_areapairs[iap],linewidth=3))
        # plt.savefig(os.path.join(figdir,'NoiseCorr_distRF_RegressOut_' + areapair + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

    ax.legend(handles,areapairs,loc='upper right',frameon=False,fontsize=9)	
    ax.set_xlabel('Anatomical distance ($\mu$m)')
    ax.set_ylabel('Correlation')
    ax.set_xlim([20,600])
    ax_nticks(ax,3)
    # ax.set_title('%s (%s)' % (corr_type,protocol))
    # ax.set_ylim([-0.01,0.04])
    # ax.set_ylim([0,ax.get_ylim()[1]])
    ax.set_ylim([0,0.04])
    ax.set_aspect('auto')
    ax.tick_params(axis='both', which='major', labelsize=8)
    sns.despine(top=True,right=True,offset=3)
    plt.tight_layout()
    return fig


def plot_bin_corr_distance_projs(binsdRF,bin_dist,areapairs,layerpairs,projpairs):
    clrs_projpairs = get_clr_labelpairs(projpairs)
    clrs_areapairs = get_clr_area_pairs(areapairs)
    # nSessions = binsdRF.shape[0]
    nprojpairs = len(projpairs)
    nareapairs = len(areapairs)

    ilp = 0
    fig,axes = plt.subplots(1,nareapairs,figsize=(6.5,3),sharey=True,sharex=True)
    handles = []
    for iap,areapair in enumerate(areapairs):
        ax = axes[iap]
        for ipp,projpair in enumerate(projpairs):
            ax.plot(binsdRF,bin_dist[:,iap,ilp,ipp].squeeze(),
                                        color=clrs_projpairs[ipp],linewidth=3)
            # handles.append(shaded_error(x=binsdRF,y=bin_dist[:,iap,ilp,ipp].squeeze(),ax=ax,
                                        # error='sem',color=clrs_projpairs[ipp],linewidth=3))
        # data = 
        # for ises in range(nSessions):
            # ax.plot(binsdRF,binmean[ises,iap,:].squeeze(),linewidth=0.15,color=clrs_areapairs[iap])
        # handles.append(shaded_error(ax=ax,x=binsdRF,y=bin_dist[:,iap,ilp,ipp].squeeze(),
                                    # error='sem',color=clrs_areapairs[iap],linewidth=3))

        ax.legend(projpairs,loc='upper right',frameon=False,fontsize=9)	
        ax.set_xlabel('Anatomical distance ($\mu$m)')
        ax.set_ylabel('Correlation')
        ax.set_xlim([20,600])
        ax_nticks(ax,3)
    # ax.set_title('%s (%s)' % (corr_type,protocol))
    # ax.set_ylim([-0.01,0.04])
    # ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_ylim([0,0.04])
        ax.tick_params(axis='both', which='major', labelsize=8)
    sns.despine(top=True,right=True,offset=3)
    plt.tight_layout()
    return fig


def filter_corr_p(r,n,p_thr=0.01):
    """Filter out non-significant correlations in a correlation matrix.
    Parameters
    r : array
        Correlation matrix.
    n : int
        Number of datapoints.
    p_thr : float, optional
        Threshold for significant correlations. Default is 0.01.
    Returns
    r : array
        Correlation matrix with non-significant correlations set to nan.
    """
    t           = np.clip(r * np.sqrt((n-2)/(1-r*r)),a_min=-30,a_max=30)#convert correlation to t-statistic
    p           = ss.t.pdf(t, n-2) #convert to p-value using pdf of t-distribution and deg of freedom
    r[p>p_thr]  = np.nan #set all nonsignificant to nan
    # plt.scatter(r.flatten(),p.flatten())
    return r


def filter_sharednan(x,y):
    """
    Filter out shared nans in x and y.
    """
    isnan = np.logical_or(np.isnan(x),np.isnan(y))
    x = x[~isnan]
    y = y[~isnan]
    return x,y

