# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from tqdm import tqdm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn.preprocessing import MinMaxScaler

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.psth import compute_tensor
from utils.plot_lib import * 
from utils.regress_lib import *
from utils.RRRlib import *
from utils.params import load_params

params = load_params()
figdir = os.path.join(params['figdir'],'Behavior')

#%% 
session_list        = np.array(['LPE12223_2024_06_10']) #GR
sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)


#%% Load data :        
ises                = 0 #which session to plot
sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                            calciumversion=params['calciumversion'])


#%% Identify sessions to load
areas = ['V1','PM','AL']
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],filter_areas=areas)
# sessions = sessions[:5]
report_sessions(sessions)

for ses in sessions:
    ses.load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                            calciumversion=params['calciumversion'])
    
#%%
si                  = SimpleImputer(keep_empty_features=True)

nranks = 4
labels      = ['run speed','pupil area','video ME','mean V1 rate',
                'mean PM rate', 'mean AL rate']
nvars = len(labels)
for irank in range(nranks):
    labels.append('RRR latent %d' % (irank+1))

corrmat = np.full((len(labels),len(labels),nSessions),np.nan)

for ises,ses in enumerate(sessions):
    t_smooth = 1

    idx_areax           = np.where(sessions[ises].celldata['roi_name']=='V1')[0]
    idx_areay           = np.where(sessions[ises].celldata['roi_name']=='PM')[0]
    idx_areaz           = np.where(sessions[ises].celldata['roi_name']=='AL')[0]

    meanV1 = np.nanmean(sessions[ises].calciumdata.iloc[:,idx_areax],axis=1)
    meanPM = np.nanmean(sessions[ises].calciumdata.iloc[:,idx_areay],axis=1)
    meanAL = np.nanmean(sessions[ises].calciumdata.iloc[:,idx_areaz],axis=1)

    datamat = np.full((len(sessions[ises].ts_F),nvars+nranks),np.nan)

    datamat[:,0] = np.interp(sessions[ises].ts_F,sessions[ises].behaviordata['ts'],
                               sessions[ises].behaviordata['runspeed'])
    datamat[:,1] = np.interp(sessions[ises].ts_F,sessions[ises].videodata['ts'],
                                   sessions[ises].videodata['pupil_area'])
    datamat[:,2] = np.interp(sessions[ises].ts_F,sessions[ises].videodata['ts'],
                                   sessions[ises].videodata['motionenergy'])
    datamat[:,3] = meanV1
    datamat[:,4] = meanPM
    datamat[:,5] = meanAL

    np.random.seed(42)
    idx_areax_sub = np.random.choice(idx_areax,100,replace=False)
    idx_areay_sub = np.random.choice(idx_areay,100,replace=False)

    #RRR latents:
    X               = zscore(sessions[ises].calciumdata.iloc[:,idx_areax_sub],axis=0).to_numpy()
    Y               = zscore(sessions[ises].calciumdata.iloc[:,idx_areay_sub],axis=0).to_numpy()
    B_hat           = LM(Y,X, lam=0) #linear regression
    Y_hat           = X @ B_hat #project X onto B_hat to obtain Y_hat
    U, s, V         = svds(Y_hat,k=nranks,which='LM') #decompose Y_hat, get maximally predictive dimensions
    U, s, V         = U[:, ::-1], s[::-1], V[::-1, :]
    W               = B_hat @ V.T   # Predictive X-directions
    Z               = X @ W   # Project X onto predictive dimensions
    # Z               = Z.flatten() #remove redundant dim 
    
    for irank in range(nranks):
        datamat[:,irank+nvars]    = Z[:,irank]*np.sign(np.corrcoef(Z[:,irank],meanV1)[0,1]) #align sign with mean V1 rate

    datamat                   = si.fit_transform(datamat)

    for i in range(datamat.shape[1]):
        datamat[:,i] = np.convolve(datamat[:,i],np.ones(int(t_smooth * 1 / np.mean(np.diff(sessions[ises].ts_F)))),mode='same')
    
    #     plotdata = np.convolve(idata,np.ones(int(t_smooth * 1 / np.mean(np.diff(its)))),mode='same')
    corrmat[:,:,ises] = np.corrcoef(datamat.T)

#%% 
meancorrmat = np.nanmean(corrmat,axis=2)
fig,ax = plt.subplots(1,1,figsize=(6*cm,5*cm))
sns.heatmap(meancorrmat,xticklabels=labels,yticklabels=labels,cmap='RdBu_r',center=0,ax=ax)
my_savefig(fig,figdir,'Behavior_corrmat_%dsessions' % (nSessions))

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
report_sessions(sessions)
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False,compute_respmat=True,
                                      load_behav=True)

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
# my_savefig(fig,figdir,'CovarianceMatrix_V1PM_Behavior_%s' % sessions[ises].session_id)
