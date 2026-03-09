# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')
from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from tqdm import tqdm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn.preprocessing import MinMaxScaler

from loaddata.session_info import *
from utils.psth import compute_tensor
from utils.plot_lib import * #get all the fixed color schemes
from utils.regress_lib import *
from utils.RRRlib import *
from params import load_params

params = load_params()
figdir = os.path.join(params['figdir'],'Behavior')

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches


#%% 
session_list        = np.array(['LPE12223_2024_06_10']) #GR
sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%% Load data :        
ises                = 0 #which session to plot
sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                            calciumversion=params['calciumversion'])

#%% Show mean population rate and behavioral variables:
scaler = MinMaxScaler()
excerptlength = 100 #seconds to show in the plot, starting from the first trial onset
# t_start = sessions[ises].ts_F[10200]
# t_start = sessions[ises].ts_F[4600]
# t_start = sessions[ises].ts_F[5100]
t_start = sessions[ises].ts_F[2600]
t_stop = t_start+excerptlength
linewidth = 0.6
t_smooth = 1
scalefactor = 0.6
clrs = sns.color_palette('dark',n_colors=10)


idx_areax           = np.where(sessions[ises].celldata['roi_name']=='V1')[0]
idx_areay           = np.where(sessions[ises].celldata['roi_name']=='PM')[0]

meanV1 = np.nanmean(sessions[ises].calciumdata.iloc[:,idx_areax],axis=1)
meanPM = np.nanmean(sessions[ises].calciumdata.iloc[:,idx_areay],axis=1)

idx_areax_sub = np.random.choice(idx_areax,100,replace=False)
idx_areay_sub = np.random.choice(idx_areay,100,replace=False)

#RRR latent 1:
X               = zscore(sessions[ises].calciumdata.iloc[:,idx_areax_sub],axis=0).to_numpy()
Y               = zscore(sessions[ises].calciumdata.iloc[:,idx_areay_sub],axis=0).to_numpy()
B_hat           = LM(Y,X, lam=0) #linear regression
Y_hat           = X @ B_hat #project X onto B_hat to obtain Y_hat
U, s, V         = svds(Y_hat,k=1,which='LM') #decompose Y_hat, get maximally predictive dimensions
W               = B_hat @ V.T   # Predictive X-directions
Z               = X @ W   # Project X onto predictive dimensions
Z               = Z.flatten() #remove redundant dim 
Z               = Z*np.sign(np.corrcoef(Z,meanV1)[0,1]) #align sign with mean V1 rate

fig,ax = plt.subplots(1,1,figsize=(10*cm,5*cm)) #make the figure
data        = [sessions[ises].behaviordata['runspeed'],
               sessions[ises].videodata['pupil_area'],
               sessions[ises].videodata['motionenergy'],
               meanV1,
               meanPM,
               Z]
ts          = [
                sessions[ises].behaviordata['ts'],
                sessions[ises].videodata['ts'],
                sessions[ises].videodata['ts'],
                sessions[ises].ts_F,
               sessions[ises].ts_F,
               sessions[ises].ts_F,
               ]
labels      = ['Run speed','Pupil area','Motion energy','Mean V1 Activity','Mean PM Activity','Subspace Latent 1']
for i,(idata,its,ilabel) in enumerate(zip(data,ts,labels)):
    idx_T = np.where((its>=t_start) & (its<=t_stop))[0]
    plotdata = np.convolve(idata,np.ones(int(t_smooth * 1 / np.mean(np.diff(its)))),mode='same')
    plotdata = plotdata[idx_T]
    plotdata = scaler.fit_transform(plotdata.reshape(-1,1)).flatten()
    ax.plot(its[idx_T],-i*scalefactor+plotdata,color=clrs[i],label=ilabel,linewidth=linewidth)

ax.set_xlim(t_start,t_stop)
# ax.legend(loc='upper right',fontsize=7,frameon=False,bbox_to_anchor=(1.4,0.9),labelspacing=1.5)
ax.legend(loc='upper left',fontsize=7,frameon=False,bbox_to_anchor=(-0.45,0.92),labelspacing=1.2)
my_legend_strip(ax)
ax.axis('off')
sns.despine(fig,top=True,right=True,offset=3)
# plt.tight_layout()
ax.add_artist(AnchoredSizeBar(ax.transData, 10,
                "10 Sec", loc='lower right', frameon=False))

# my_savefig(fig,figdir,'Example_Behavior_V1PM_%s' % sessions[ises].session_id)


#%% 





####### #     #    #    #     # ######  #       #######     #####  ####### #     # 
#        #   #    # #   ##   ## #     # #       #          #     # #     # #     # 
#         # #    #   #  # # # # #     # #       #          #       #     # #     # 
#####      #    #     # #  #  # ######  #       #####      #       #     # #     # 
#         # #   ####### #     # #       #       #          #       #     #  #   #  
#        #   #  #     # #     # #       #       #          #     # #     #   # #   
####### #     # #     # #     # #       ####### #######     #####  #######    #    


#%% 
areas = ['V1','PM','AL']
session_list        = np.array(['LPE12223_2024_06_10']) #GR
sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list,filter_areas=areas)


#%% 
# areas = ['V1','PM','AL','RSP']
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],only_all_areas=areas,filter_areas=areas,min_lab_cells_V1=50,min_lab_cells_PM=50)

#%%  Load data properly:        
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
                            ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

behavfields = np.array(['runspeed','diffrunspeed'])

for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=params['calciumversion'])
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 method='nearby')
    
    [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
                                 params['t_pre'], params['t_post'], method='binmean',binsize=params['binsize'])
    #Subsample behavioral data 10 times before binning:
    sessions[ises].behaviordata.drop('session_id',axis=1,inplace=True)
    sessions[ises].behaviordata = sessions[ises].behaviordata.groupby(sessions[ises].behaviordata.index // 10).mean()
    sessions[ises].behaviordata['diffrunspeed'] = np.diff(sessions[ises].behaviordata['runspeed'],prepend=0)
    [sessions[ises].tensor_run,t_axis] = compute_tensor(sessions[ises].behaviordata[behavfields], sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'], 
                                 params['t_pre'], params['t_post'], method='binmean',binsize=params['binsize'])
    
    delattr(sessions[ises],'calciumdata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'videodata')

#%% Show example covariance matrix predicted by behavior:  
ises                = 0 #which session to compute covariance matrix for
stim                = 7 #which stimulus to compute covariance matrix for
rank                = 5

idx_T               = sessions[ises].trialdata['stimCond']==stim
idx_N               = np.ones(len(sessions[ises].celldata),dtype=bool)
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]

#on residual tensor during the response:
Y                   = sessions[ises].tensor[np.ix_(idx_N,idx_T,idx_resp)]
# Y                   -= np.mean(Y,axis=1,keepdims=True)
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

EV_cells = r2_score(Y,Y_hat,multioutput='raw_values')
# decomposing and low rank approximation of Y_hat
U, s, V             = svds(Y_hat,k=rank)
U, s, V             = U[:, ::-1], s[::-1], V[::-1, :]

S                   = linalg.diagsvd(s,U.shape[0],s.shape[0])

Y_cov               = np.cov(Y.T)
np.fill_diagonal(Y_cov,np.nan)

Y_hat_rr            = U[:,:rank] @ S[:rank,:rank] @ V[:rank,:]
Y_cov_rrr           = np.cov(Y_hat_rr.T)
np.fill_diagonal(Y_cov_rrr,np.nan)

#Plot: 
vmin,vmax       = np.nanpercentile(Y_cov,5),np.nanpercentile(Y_cov,95)
# arealabeled     = np.array(['V1unl','V1lab','PMunl','PMlab'])
# arealabeled     = np.array(['V1unl','V1lab','PMunl','PMlab','ALunl','RSPunl'])
arealabeled     = np.array(['V1unl','V1lab','PMunl','PMlab','ALunl'])

al_fig          = arealabeled_to_figlabels(arealabeled)

idx_sort       = np.argsort(sessions[ises].celldata['arealabel'])[::-1]
idx_sort = np.lexsort((-EV_cells,sessions[ises].celldata['arealabel']))[::-1]
al_sorted      = sessions[ises].celldata['arealabel'][idx_sort]

Y_cov_sort      = copy.deepcopy(Y_cov)
Y_cov_sort      = Y_cov_sort[idx_sort,:]
Y_cov_sort      = Y_cov_sort[:,idx_sort]

Y_cov_rrr_sort  = copy.deepcopy(Y_cov_rrr)
Y_cov_rrr_sort  = Y_cov_rrr_sort[idx_sort,:]
Y_cov_rrr_sort  = Y_cov_rrr_sort[:,idx_sort]

# Join the two matrices: 
N = np.shape(Y_cov)[0]

Y_cov_joint = np.full_like(Y_cov_sort,np.nan)
idx_tri_upper = np.triu_indices(N, k=1)
Y_cov_joint[idx_tri_upper] = Y_cov_sort[idx_tri_upper]
idx_tri_lower = np.tril_indices(N, k=1)
Y_cov_joint[idx_tri_lower] = Y_cov_rrr_sort[idx_tri_lower]
# Y_cov_joint[np.diag_indices(N)] = 

vmin,vmax       = np.nanpercentile(Y_cov_joint,15),np.nanpercentile(Y_cov_joint,90)

fig,ax = plt.subplots(1,1,figsize=(8*cm,8*cm))
im = ax.imshow(Y_cov_joint,vmin=vmin,vmax=vmax,cmap='magma')
# ax.pcolor(np.arange(N),np.arange(N),Y_cov_joint,vmin=vmin,vmax=vmax,cmap='magma')
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
ax.set_xlabel('Behavior-predicted',labelpad=3)
ax.set_ylabel('Original',labelpad=3)
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(im,ax=ax,shrink=0.3,location='right',label='Covariance')
# fig.colorbar(cm.ScalarMappable(norm=norm, cmap='magma'),
            #  ax=ax,shrink=0.6,location='right',label='R$^2$')
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
areas = ['V1','PM','AL']
session_list        = np.array(['LPE12223_2024_06_10']) #GR
sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list,filter_areas=areas)

#%% 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%% Get all data 
only_all_areas = np.array(['V1','PM'])
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_areas=only_all_areas,
                                       min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],min_lab_cells_V1=20,filter_noiselevel=True)
report_sessions(sessions)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load data:        
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
calciumversion = 'dF'
vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
                            ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

behavfields = np.array(['runspeed','diffrunspeed'])

for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                    calciumversion=params['calciumversion'])
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                method='nearby')
    delattr(sessions[ises],'calciumdata')
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

#%% Subtracting mean response across trials for each stimulus condition
for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Subtracting mean response across trials'):
    N = len(sessions[ises].celldata)
    idx_resp = t_axis>0
    for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over stim 
        idx_T               = sessions[ises].trialdata['stimCond']==stim

        #on tensor during the response:
        sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)] -= np.nanmean(sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)],axis=1,keepdims=True)
    
    idx_resp = t_axis<=0 #subtract mean response of previous trial
    for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over stim 
        idx_T               = np.concatenate([[0],sessions[ises].trialdata['stimCond'][:-1]])==stim

        #on tensor during the response:
        sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)] -= np.nanmean(sessions[ises].tensor[np.ix_(range(N),idx_T,idx_resp)],axis=1,keepdims=True)

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
nranks              = 20
nStim               = 16
filter_nearby       = True

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

#Explained (co)variance
EV_pops             = np.full((narealabels,nranks,nStim,nSessions,params['kfold']),np.nan)
EC_poppairs         = np.full((narealabelpairs,nranks,nStim,nSessions,params['kfold']),np.nan)

# EV_pops             = np.full((narealabels,nranks,nStim,nSessions),np.nan)
# EC_poppairs         = np.full((narealabelpairs,nranks,nStim,nSessions),np.nan)

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Covariance by behavior, fitting models'):
# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Covariance by behavior, fitting models'):
for ises,ses in enumerate(sessions):

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
    for istim,stim in tqdm(enumerate(np.unique(ses.trialdata['stimCond'])),
    # for istim,stim in tqdm(enumerate(np.unique(ses.trialdata['stimCond'])[:2]),
                           total=nStim,desc='Fitting RRR model for session %d/%d' %(ises+1,nSessions)): # loop over orientations 
        idx_T               = ses.trialdata['stimCond']==stim

        idx_N               = np.ones(len(ses.celldata),dtype=bool)

        #on residual tensor during the response:
        Y                   = sessions[ises].tensor[np.ix_(idx_N,idx_T,idx_resp)]
        # Y                   -= np.mean(Y,axis=1,keepdims=True)
        Y                   = Y.reshape(len(idx_N),-1).T
        Y                   = zscore(Y,axis=0,nan_policy='omit')  #Z score activity for each neuron

        #Get behavioral matrix: 
        X                   = np.concatenate((sessions[ises].tensor_vid[np.ix_(range(np.shape(sessions[ises].tensor_vid)[0]),idx_T,idx_resp)],
                                sessions[ises].tensor_run[np.ix_(range(np.shape(sessions[ises].tensor_run)[0]),idx_T,idx_resp)]),axis=0)
        X                   = X.reshape(np.shape(X)[0],-1).T
        X                   = zscore(X,axis=0,nan_policy='omit')

        si      = SimpleImputer()
        Y       = si.fit_transform(Y)
        X       = si.fit_transform(X)

        kf = KFold(n_splits=params['kfold'],shuffle=True)
        for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
            X_train, X_test     = X[idx_train], X[idx_test]
            Y_train, Y_test     = Y[idx_train], Y[idx_test]

            B_hat_train         = LM(Y_train,X_train, lam=params['lam'])

            Y_hat_train         = X_train @ B_hat_train

            # decomposing and low rank approximation of A
            U, s, V = svds(Y_hat_train,k=nranks,which='LM')
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

            Y_cov = np.cov(Y_train.T)

            for r in range(nranks):
                B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                #construct low rank subspace prediction
                Y_hat_test_rr   = X_test @ B_rrr
                #How much variance is explained for each of the populations?
                for ial,al in enumerate(arealabeled):
                    idx_N           = np.where(np.all((ses.celldata['arealabel']==al,
                                            ses.celldata['noise_level']<params['maxnoiselevel'],	
                                            idx_nearby),axis=0))[0]
                    
                    EV_pops[ial,r,istim,ises,ikf] = EV(Y_test[:,idx_N],Y_hat_test_rr[:,idx_N])
                
                #How much covariance is explained for each of the population pairs?
                Y_cov_rrr       = np.cov(Y_hat_test_rr.T)
                for ialp,arealabelpair in enumerate(arealabelpairs):

                    alx,aly             = arealabelpair.split('-')

                    idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                                            ses.celldata['noise_level']<params['maxnoiselevel'],
                                            idx_nearby),axis=0))[0]
                    idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                                            ses.celldata['noise_level']<params['maxnoiselevel'],
                                            idx_nearby
                                            ),axis=0))[0]
                    
                    EC_poppairs[ialp,r,istim,ises,ikf] = EV(Y_cov[np.ix_(idx_areax,idx_areay)],Y_cov_rrr[np.ix_(idx_areax,idx_areay)])

#%% Plotting:
fig,axes = plt.subplots(1,1,figsize=(4.3*cm,3.7*cm))
ax = axes
handles = []
# R2_max,optim_rank = rank_from_R2(np.reshape(np.nanmean(EV_pops,axis=0),(nranks,-1)),nranks,nSessions*nStim*params['kfold'])
R2_max,optim_rank = rank_from_R2(np.reshape(np.nanmean(EV_pops,axis=(0,2)),(nranks,-1)),nranks,nSessions*params['kfold'])
print('Optimal rank: %d' % optim_rank)
data = np.reshape(np.nanmean(EV_pops,axis=0),(nranks,-1))
shaded_error(range(nranks),data.T,error='sem',color='k',alpha=0.3,ax=ax)
ax.plot(optim_rank,R2_max+0.007,color='k',marker='v',markersize=5)
# R2_max,optim_rank = rank_from_R2(data,nranks,nSessions*nStim*params['kfold'])
# R2_max,optim_rank = rank_from_R2(data,nranks,nSessions*params['kfold'])

ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
# ax.legend(handles=handles,labels=list(arealabeled),loc='lower right',fontsize=8)
# my_legend_strip(ax)
ax_nticks(ax,4)
ax.set_xticks(np.arange(0,nranks+5,5))
ax.set_xticklabels(np.arange(0,nranks+5,5))
ax.set_xlabel('Rank')
ax.set_ylabel(r'Variance explained (R$^2$)')
sns.despine(fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'RRR_Behavior_R2_Ranks_V1PM_%dsessions' % nSessions)

#%% Plotting:
fig,axes = plt.subplots(1,3,figsize=(12*cm,4*cm))
ax = axes[0]
handles = []

for ial, arealabel in enumerate(arealabeled):
    ialdata = np.reshape(EV_pops[ial],(nranks,-1))
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
    ialpdata = np.reshape(EC_poppairs[ialp],(nranks,-1))
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
    ialpdata = np.reshape(EC_poppairs[ialp],(nranks,-1))
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
# plt.tight_layout()
# my_savefig(fig,figdir,'CoVarianceExplained_V1PM_%dsessions' % nSessions)

#%% Plotting:
clrs_arealabels = ['grey','red','grey','red']
fig,axes = plt.subplots(1,1,figsize=(4*cm,3.9*cm))
ax = axes
data = np.nanmean(EV_pops,axis=-1)
data = np.reshape(data[:,optim_rank,:,:],(narealabels,-1))
for ial, arealabel in enumerate(arealabeled):
    ax.scatter(np.random.randn(nSessions*nStim)*0.1+ial,data[ial,:].flatten(),s=5,color='k',marker='.')
    ax.plot(ial,np.nanmean(data[ial,:]),color=clrs_arealabels[ial],marker='o',markersize=5)

ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax_nticks(ax,4)
ax.set_xticks(range(narealabels),arealabeled_to_figlabels(arealabeled),rotation=45,fontsize=6)
ax.set_ylabel('Variance explained (R$^2$)')

#Statistical testing:
df = pd.DataFrame({'EV': data.flatten(),
                  'arealabel': np.repeat(arealabeled,nSessions*nStim),
                  'session': np.tile(np.arange(nSessions*nStim),narealabels)})
order = arealabeled #for statistical testing purposes
pairs = [('V1unl','V1lab'),('PMunl','PMlab'),('V1unl','PMunl'),('V1lab','PMlab')]

annotator = Annotator(ax, pairs, data=df, x="arealabel", y='EV', order=order)
annotator.configure(test='Wilcoxon', text_format='star', loc='inside',verbose=False,
                    line_offset_to_group=0.2, line_width=1,
                    comparisons_correction="Bonferroni",line_height=0, text_offset=-3,fontsize=9)
annotator.apply_and_annotate()

sns.despine(fig,top=True,right=True,offset=3)
# plt.tight_layout()
my_savefig(fig,figdir,'VarianceExplained_V1PM_labeled_%dsessions' % nSessions)

#%% Plotting:
fig,axes = plt.subplots(1,1,figsize=(4*cm,3.5*cm))
ax = axes
# data = np.nanmean(EC_poppairs,axis=-1)
# data = np.reshape(data[:,optim_rank,:,:],(narealabels,-1))
ax.scatter(np.random.randn(nSessions*nStim)*0.1,np.nanmean(EC_poppairs[:,optim_rank,:,:],axis=(0)).flatten(),s=15,color='k',marker='.')
ax.plot(0,np.nanmean(EC_poppairs[:,optim_rank,:,:]),color='purple',marker='o',markersize=8)
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],1)])
ax.set_ylabel('Cov. explained')
ax.set_xlim([-0.25,0.25])
ax.set_xticks([])
# ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
sns.despine(fig,top=True,right=True,offset=3)
plt.tight_layout()
# my_savefig(fig,figdir,'Mean_CoVarianceExplained_V1PM_%dsessions' % nSessions)
