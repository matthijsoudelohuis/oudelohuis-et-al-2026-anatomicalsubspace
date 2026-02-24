# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('c:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from skimage.measure import block_reduce
from tqdm import tqdm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plot_lib import * #get all the fixed color schemes
from utils.tuning import compute_tuning_wrapper
from utils.regress_lib import *
from utils.RRRlib import *
from params import load_params

params = load_params()
figdir = os.path.join(params['figdir'],'Behavior')

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches


#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR
sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%% 
areas = ['V1','PM','AL','RSP']
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],only_all_areas=areas,min_lab_cells_V1=50,min_lab_cells_PM=50)


#%%


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

ises                = 0 #which session to plot


#%% Load data :        
sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                            calciumversion=params['calciumversion'])


#%% Show mean population rate and behavioral variables:
excerptlength = 100 #seconds to show in the plot, starting from the first trial onset
t_start = sessions[ises].ts_F[10200]
t_start = sessions[ises].ts_F[4600]
t_stop = t_start+excerptlength
linewidth = 0.6
t_smooth = 1
clrs = sns.color_palette('dark',n_colors=5)
meanV1 = np.nanmean(sessions[ises].calciumdata.iloc[:,(sessions[ises].celldata['roi_name']=='V1').values],axis=1)
meanPM = np.nanmean(sessions[ises].calciumdata.iloc[:,(sessions[ises].celldata['roi_name']=='PM').values],axis=1)

from scipy.signal import savgol_filter

fig,ax = plt.subplots(1,1,figsize=(10*cm,3*cm))

#Mean V1:
idx_T = np.where((sessions[ises].ts_F>=t_start) & (sessions[ises].ts_F<=t_stop))[0]

plotdata = np.convolve(meanV1,np.ones(int(t_smooth * sessions[ises].sessiondata['fs'][0])),mode='same')
plotdata = plotdata[idx_T]
plotdata = scaler.fit_transform(plotdata.reshape(-1,1)).flatten()
ax.plot(sessions[ises].ts_F[idx_T],plotdata,color=clrs[0],label='Mean V1 Activity',linewidth=linewidth)

plotdata = np.convolve(meanPM,np.ones(int(t_smooth * sessions[ises].sessiondata['fs'][0])),mode='same')
plotdata = plotdata[idx_T]
plotdata = scaler.fit_transform(plotdata.reshape(-1,1)).flatten()
ax.plot(sessions[ises].ts_F[idx_T],plotdata,color=clrs[1],label='Mean PM Activity',linewidth=linewidth)

idx_T = np.where((sessions[ises].behaviordata['ts']>=t_start) & (sessions[ises].behaviordata['ts']<=t_stop))[0]
plotdata = np.convolve(sessions[ises].behaviordata['runspeed'],np.ones(int(t_smooth * 100)),mode='same')
plotdata = plotdata[idx_T]
plotdata = scaler.fit_transform(plotdata.reshape(-1,1)).flatten()
ax.plot(sessions[ises].behaviordata['ts'].iloc[idx_T],plotdata,color=clrs[2],label='Run speed',linewidth=linewidth)

idx_T = np.where((sessions[ises].videodata['ts']>=t_start) & (sessions[ises].videodata['ts']<=t_stop))[0]
plotdata = np.convolve(sessions[ises].videodata['pupil_area'],np.ones(int(t_smooth * sessions[ises].sessiondata['video_fs'][0])),mode='same')
plotdata = plotdata[idx_T]
plotdata = scaler.fit_transform(plotdata.reshape(-1,1)).flatten()
ax.plot(sessions[ises].videodata['ts'].iloc[idx_T],plotdata,color=clrs[3],label='Pupil area',linewidth=linewidth)

plotdata = np.convolve(sessions[ises].videodata['motionenergy'],np.ones(int(t_smooth * sessions[ises].sessiondata['video_fs'][0])),mode='same')
plotdata = plotdata[idx_T]
plotdata = scaler.fit_transform(plotdata.reshape(-1,1)).flatten()
ax.plot(sessions[ises].videodata['ts'].iloc[idx_T],plotdata,color=clrs[4],label='Motion energy',linewidth=linewidth)

ax.set_xlim(t_start,t_stop)
ax.legend(loc='upper right',fontsize=6,frameon=False,bbox_to_anchor=(1.5,1))
my_legend_strip(ax)
# ax.set_ylabel('Normalized signal')
ax.axis('off')
sns.despine(fig,top=True,right=True,offset=3)
plt.tight_layout()

ax.add_artist(AnchoredSizeBar(ax.transData, 10,
                "10 Sec", loc='upper right', frameon=False))

my_savefig(fig,figdir,'Example_Behavior_V1PM_%s' % sessions[ises].session_id)




#%%  Load data properly:        
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
# t_pre       = -1    #pre s
# t_post      = 1.9     #post s
# binsize     = 0.2
calciumversion = 'dF'
vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
                            ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

behavfields = np.array(['runspeed','diffrunspeed'])

t_pre       = -1         #pre s
t_post      = 2.17        #post s
binsize     = 1/5.35

for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion)
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 method='nearby')
    
    [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    #Subsample behavioral data 10 times before binning:
    sessions[ises].behaviordata.drop('session_id',axis=1,inplace=True)
    sessions[ises].behaviordata = sessions[ises].behaviordata.groupby(sessions[ises].behaviordata.index // 10).mean()
    sessions[ises].behaviordata['diffrunspeed'] = np.diff(sessions[ises].behaviordata['runspeed'],prepend=0)
    [sessions[ises].tensor_run,t_axis] = compute_tensor(sessions[ises].behaviordata[behavfields], sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    
    delattr(sessions[ises],'calciumdata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'videodata')



####### #     #    #    #     # ######  #       #######     #####  ####### #     # 
#        #   #    # #   ##   ## #     # #       #          #     # #     # #     # 
#         # #    #   #  # # # # #     # #       #          #       #     # #     # 
#####      #    #     # #  #  # ######  #       #####      #       #     # #     # 
#         # #   ####### #     # #       #       #          #       #     #  #   #  
#        #   #  #     # #     # #       #       #          #     # #     #   # #   
####### #     # #     # #     # #       ####### #######     #####  #######    #    



#%% Show example covariance matrix predicted by behavior:  
ises                = 1 #which session to compute covariance matrix for
stim                = 4 #which stimulus to compute covariance matrix for
rank                = 5

idx_T               = sessions[ises].trialdata['stimCond']==stim
idx_N               = np.ones(len(sessions[ises].celldata),dtype=bool)
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]

#on residual tensor during the response:
Y                   = sessions[ises].tensor[np.ix_(idx_N,idx_T,idx_resp)]
Y                   -= np.mean(Y,axis=1,keepdims=True)
Y                   = Y.reshape(len(idx_N),-1).T
Y                   = zscore(Y,axis=0,nan_policy='omit')  #Z score activity for each neuron

#Get behavioral matrix: 
B                   = np.concatenate((sessions[ises].tensor_vid[np.ix_(range(np.shape(sessions[ises].tensor_vid)[0]),idx_T,idx_resp)],
                        sessions[ises].tensor_run[np.ix_(range(np.shape(sessions[ises].tensor_run)[0]),idx_T,idx_resp)]),axis=0)
B                   = B.reshape(np.shape(B)[0],-1).T
B                   = zscore(B,axis=0,nan_policy='omit')

si                  = SimpleImputer()
Y                   = si.fit_transform(Y)
B                   = si.fit_transform(B)

#Reduced rank regression: 
B_hat               = LM(Y,B,lam=0)
Y_hat               = B @ B_hat

# decomposing and low rank approximation of Y_hat
U, s, V             = svds(Y_hat,k=rank)
U, s, V             = U[:, ::-1], s[::-1], V[::-1, :]

S                   = linalg.diagsvd(s,U.shape[0],s.shape[0])

Y_cov               = np.cov(Y.T)
np.fill_diagonal(Y_cov,np.nan)

Y_hat_rr            = U[:,:rank] @ S[:rank,:rank] @ V[:rank,:]
Y_cov_rrr           = np.cov(Y_hat_rr.T)
np.fill_diagonal(Y_cov_rrr,np.nan)

#%% Plot: 
vmin,vmax       = np.nanpercentile(Y_cov,5),np.nanpercentile(Y_cov,95)
# arealabeled     = np.array(['V1unl','V1lab','PMunl','PMlab'])
arealabeled     = np.array(['V1unl','V1lab','PMunl','PMlab','ALunl','RSPunl'])

al_fig          = arealabeled_to_figlabels(arealabeled)

idx_sort       = np.argsort(sessions[ises].celldata['arealabel'])[::-1]
# idx_sort       = sorted(sessions[ises].celldata['arealabel'],key=arealabeled)[::-1]

# al_sorted      = np.sort(np.array(sessions[ises].celldata['arealabel']),order=arealabeled)[::-1]
al_sorted      = sessions[ises].celldata['arealabel'][idx_sort]

Y_cov_sort      = copy.deepcopy(Y_cov)
Y_cov_sort      = Y_cov_sort[idx_sort,:]
Y_cov_sort      = Y_cov_sort[:,idx_sort]

Y_cov_rrr_sort  = copy.deepcopy(Y_cov_rrr)
Y_cov_rrr_sort  = Y_cov_rrr_sort[idx_sort,:]
Y_cov_rrr_sort  = Y_cov_rrr_sort[:,idx_sort]

#%% Join the two matrices: 
N = np.shape(Y_cov)[0]

Y_cov_joint = np.full_like(Y_cov_sort,np.nan)
idx_tri_upper = np.triu_indices(N, k=1)
Y_cov_joint[idx_tri_upper] = Y_cov_sort[idx_tri_upper]
idx_tri_lower = np.tril_indices(N, k=1)
Y_cov_joint[idx_tri_lower] = Y_cov_rrr_sort[idx_tri_lower]
# Y_cov_joint[np.diag_indices(N)] = 

vmin,vmax       = np.nanpercentile(Y_cov_joint,15),np.nanpercentile(Y_cov_joint,90)

fig,ax = plt.subplots(1,1,figsize=(3,3))
ax.imshow(Y_cov_joint,vmin=vmin,vmax=vmax,cmap='magma')
# ax.set_title('Covariance\n(original)')
ax.set_yticks([])
for ial,arealabel in enumerate(arealabeled):
    start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
    ax.plot([-5,-5],[start,stop],color=get_clr_area_labeled([arealabel]),linestyle='-',linewidth=5)
    labeltext = '%s\nn=%d' % (al_fig[ial],stop-start)
    ax.text(-85,(start+stop)/2,labeltext,fontsize=9,color=get_clr_area_labeled([arealabel]),
               rotation=0,ha='right',va='center')
for ial,arealabel in enumerate(arealabeled):
    start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
    ax.plot([start,stop],[-5,-5],color=get_clr_area_labeled([arealabel]),linestyle='-',linewidth=5)
    ax.text((start+stop)/2,-85,al_fig[ial],fontsize=9,color=get_clr_area_labeled([arealabel]),
               rotation=90,ha='center',va='bottom')
ax.set_xticks([0,np.shape(Y_cov)[0]-1])
ax.set_xlabel('Behavior-predicted',)
ax.set_ylabel('Original')
ax.set_xticks([])
ax.set_yticks([])
ax.yaxis.set_label_position("right")
plt.tight_layout()
my_savefig(fig,figdir,'CovarianceMatrix_V1PM_Behavior_%s' % sessions[ises].session_id)


#%% Make as separate figures: 
# fig,ax = plt.subplots(1,2,figsize=(6,3))
# ax[0].imshow(Y_cov_sort,vmin=vmin,vmax=vmax,cmap='magma')
# ax[0].set_title('Covariance\n(original)')
# ax[0].set_yticks([])
# for ial,arealabel in enumerate(arealabeled):
#     start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
#     ax[0].plot([-5,-5],[start,stop],color=get_clr_area_labeled([arealabel]),linestyle='-',linewidth=5)
#     ax[0].text(-85,(start+stop)/2,arealabel,fontsize=9,color=get_clr_area_labeled([arealabel]),
#                rotation=45,ha='center',va='center')
# for ial,arealabel in enumerate(arealabeled):
#     start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
#     ax[0].plot([start,stop],[-5,-5],color=get_clr_area_labeled([arealabel]),linestyle='-',linewidth=5)
#     ax[0].text((start+stop)/2,-85,arealabel,fontsize=9,color=get_clr_area_labeled([arealabel]),
#                rotation=45,ha='center',va='center')
# ax[0].set_xticks([0,np.shape(Y_cov)[0]-1])

# ax[1].imshow(Y_cov_rrr_sort,vmin=vmin,vmax=vmax,cmap='magma')
# ax[1].set_title('Covariance\n(predicted from behavior)')
# ax[1].set_xticks([0,np.shape(Y_cov)[0]-1])
# for ial,arealabel in enumerate(arealabeled):
#     start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
#     ax[1].plot([-5,-5],[start,stop],color=get_clr_area_labeled([arealabel]),linestyle='-',linewidth=5)
#     ax[1].text(-85,(start+stop)/2,arealabel,fontsize=9,color=get_clr_area_labeled([arealabel]),
#                rotation=45,ha='center',va='center')
# for ial,arealabel in enumerate(arealabeled):
#     start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
#     ax[1].plot([start,stop],[-5,-5],color=get_clr_area_labeled([arealabel]),linestyle='-',linewidth=5)
#     ax[1].text((start+stop)/2,-85,arealabel,fontsize=9,color=get_clr_area_labeled([arealabel]),
#                rotation=45,ha='center',va='center')
# for axi in ax:
#     # w = ax.get_xaxis()
#     # w.set_visible(False)
#     # axi.axis["left"].set_visible(False)
#     # axi.axis["top"].set_visible(False)
#     # axi.axis["right"].set_visible(False)
#     axi.set_axis_off()
# plt.tight_layout()
# my_savefig(fig,figdir,'CovarianceMatrix_V1PM_%s' % sessions[ises].session_id,formats=['png'])






#%% 

   #    #       #           #####  #######  #####   #####  ### ####### #     #  #####  
  # #   #       #          #     # #       #     # #     #  #  #     # ##    # #     # 
 #   #  #       #          #       #       #       #        #  #     # # #   # #       
#     # #       #           #####  #####    #####   #####   #  #     # #  #  #  #####  
####### #       #                # #             #       #  #  #     # #   # #       # 
#     # #       #          #     # #       #     # #     #  #  #     # #    ## #     # 
#     # ####### #######     #####  #######  #####   #####  ### ####### #     #  #####  

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%% 
areas = ['V1','PM']
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
calciumversion = 'dF'
vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
                            ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

behavfields = np.array(['runspeed','diffrunspeed'])

t_pre       = -1         #pre s
t_post      = 2.17        #post s
binsize     = 1/5.35

for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion)
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 method='nearby')
    
    [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    #Subsample behavioral data 10 times before binning:
    sessions[ises].behaviordata.drop('session_id',axis=1,inplace=True)
    sessions[ises].behaviordata = sessions[ises].behaviordata.groupby(sessions[ises].behaviordata.index // 10).mean()
    sessions[ises].behaviordata['diffrunspeed'] = np.diff(sessions[ises].behaviordata['runspeed'],prepend=0)
    [sessions[ises].tensor_run,t_axis] = compute_tensor(sessions[ises].behaviordata[behavfields], sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'], 
                                 t_pre, t_post, method='binmean',binsize=binsize)
    
    delattr(sessions[ises],'calciumdata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'videodata')



#%% Compute the variance and covariance explained by the behavior: 

# Variance: 
arealabeled         = np.array(['V1unl','V1lab','PMunl','PMlab'])
clrs_arealabels     = get_clr_area_labeled(arealabeled)
narealabels         = len(arealabeled)

# Covariance:
arealabelpairs  = np.array(['V1unl-V1unl',
                    'V1unl-V1lab',
                    'V1lab-V1lab',
                    'PMunl-PMunl',
                    'PMunl-PMlab',
                    'PMlab-PMlab',
                    'V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab'])

clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

#Parameters:
lam                 = 0
nranks              = 20
kfold               = 5
maxnoiselevel       = 20
nStim               = 16
filter_nearby       = True

idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
ntimebins           = len(idx_resp)

#Explained (co)variance
EV_pops             = np.full((narealabels,nranks,nStim,nSessions,kfold),np.nan)
EC_poppairs         = np.full((narealabelpairs,nranks,nStim,nSessions,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Covariance by behavior, fitting models'):
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Covariance by behavior, fitting models'):

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=30)
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        idx_T               = ses.trialdata['stimCond']==stim

        idx_N               = np.ones(len(ses.celldata),dtype=bool)

        #on residual tensor during the response:
        Y                   = sessions[ises].tensor[np.ix_(idx_N,idx_T,idx_resp)]
        Y                   -= np.mean(Y,axis=1,keepdims=True)
        Y                   = Y.reshape(len(idx_N),-1).T
        Y                   = zscore(Y,axis=0,nan_policy='omit')  #Z score activity for each neuron

        #Get behavioral matrix: 
        B                   = np.concatenate((sessions[ises].tensor_vid[np.ix_(range(np.shape(sessions[ises].tensor_vid)[0]),idx_T,idx_resp)],
                                sessions[ises].tensor_run[np.ix_(range(np.shape(sessions[ises].tensor_run)[0]),idx_T,idx_resp)]),axis=0)
        B                   = B.reshape(np.shape(B)[0],-1).T
        B                   = zscore(B,axis=0,nan_policy='omit')

        si      = SimpleImputer()
        Y       = si.fit_transform(Y)
        B       = si.fit_transform(B)

        #Reduced rank regression: 
        B_hat           = LM(Y,B,lam=lam)

        Y_hat           = B @ B_hat

        # decomposing and low rank approximation of Y_hat
        U, s, V = svds(Y_hat,k=nranks)
        U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

        S = linalg.diagsvd(s,U.shape[0],s.shape[0])

        Y_cov = np.cov(Y.T)

        for irank,rank in enumerate(range(1,nranks+1)):
            #construct low rank subspace prediction
            Y_hat_rr       = U[:,:rank] @ S[:rank,:rank] @ V[:rank,:]

            Y_out           = Y - Y_hat_rr #subtract prediction

            Y_cov_rrr = np.cov(Y_hat_rr.T)

            for ial,al in enumerate(arealabeled):
                idx_N           = np.where(np.all((ses.celldata['arealabel']==al,
                                        ses.celldata['noise_level']<maxnoiselevel,	
                                        idx_nearby),axis=0))[0]
                
                EV_pops[ial,irank,istim,ises,0] = EV(Y[:,idx_N],Y_hat_rr[:,idx_N])
            
            for ialp,arealabelpair in enumerate(arealabelpairs):

                alx,aly             = arealabelpair.split('-')

                idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                        ses.celldata['noise_level']<maxnoiselevel,	
                                        idx_nearby),axis=0))[0]
                idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                        ses.celldata['noise_level']<maxnoiselevel,	
                                        idx_nearby
                                        ),axis=0))[0]
                
                EC_poppairs[ialp,irank,istim,ises,0] = EV(Y_cov[np.ix_(idx_areax,idx_areay)],Y_cov_rrr[np.ix_(idx_areax,idx_areay)])


#%% Plotting:
fig,axes = plt.subplots(1,3,figsize=(7.5,2.5))
ax = axes[0]
handles = []

for ial, arealabel in enumerate(arealabeled):
    ialdata = np.reshape(EV_pops[ial,:,:,:],(nranks,-1))
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(range(nranks),ialdata.T,error='sem',color=clrs_arealabels[ial],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.legend(handles=handles,labels=list(arealabeled),loc='lower right',fontsize=8)
my_legend_strip(ax)
ax.set_xticks(np.arange(0,nranks+5,5))
ax.set_xticklabels(np.arange(0,nranks+5,5)+1)
ax.set_xlabel('Rank')
ax.set_ylabel('Variance explained')
ax.set_title('Variance explained',fontsize=10)

idx = [0,3,6]
ax = axes[1]
handles = []
for ialp in idx:
    ialpdata = np.reshape(EC_poppairs[ialp,:,:,:],(nranks,-1))
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(range(nranks),ialpdata.T,error='sem',color=clrs_arealabelpairs[ialp],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.legend(handles=handles,labels=list(arealabelpairs[idx]),loc='lower right',fontsize=8)
my_legend_strip(ax)
ax.set_xticks(np.arange(0,nranks+5,5))
ax.set_xticklabels(np.arange(0,nranks+5,5)+1)
ax.set_xlabel('Rank')
ax.set_ylabel('Covariance explained')
ax.set_title('Covariance explained',fontsize=10)

idx = [6,7,8,9]
ax = axes[2]
handles = []
for ialp in idx:
    ialpdata = np.reshape(EC_poppairs[ialp,:,:,:],(nranks,-1))
    # ax.plot(binframes,np.nanmean(R2_cv[:,iapl,:],axis=0),color=clrs_arealabelpairs[iapl],label=arealabelpair)
    handles.append(shaded_error(range(nranks),ialpdata.T,error='sem',color=clrs_arealabelpairs[ialp],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.legend(handles=handles,labels=list(arealabelpairs[idx]),loc='lower right',fontsize=8)
my_legend_strip(ax)
ax.set_xticks(np.arange(0,nranks+5,5))
ax.set_xticklabels(np.arange(0,nranks+5,5)+1)
ax.set_xlabel('Rank')
ax.set_ylabel('Covariance explained')
ax.set_title('Labeled covariance explained',fontsize=10)
sns.despine(fig,top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,figdir,'CoVarianceExplained_V1PM_%dsessions' % nSessions)

#%% Plotting:
fig,axes = plt.subplots(1,1,figsize=(1,1.5))
ax = axes
# ax = axes[0]
# handles = []
ax.scatter(np.random.randn(nSessions*nStim)*0.1,np.nanmean(EC_poppairs[:,5,:,:,0],axis=(0)).flatten(),s=15,color='k',marker='.')
ax.plot(0,np.nanmean(EC_poppairs[:,5,:,:,0]),color='purple',marker='o',markersize=8)
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],1)])
ax.set_ylabel('Cov. explained')
ax.set_xlim([-0.25,0.25])
ax.set_xticks([])
# ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
sns.despine(fig,top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,figdir,'Mean_CoVarianceExplained_V1PM_%dsessions' % nSessions)
