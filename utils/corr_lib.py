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
