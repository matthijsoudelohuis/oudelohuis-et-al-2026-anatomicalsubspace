# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
os.chdir('c:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

from loaddata.get_data_folder import get_local_drive
# os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import zscore,wilcoxon,ttest_rel
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from skimage.measure import block_reduce

from loaddata.session_info import *
from utils.psth import compute_tensor,compute_respmat
from utils.plot_lib import * #get all the fixed color schemes
from utils.regress_lib import *
from utils.tuning import compute_tuning_wrapper
from params import load_params
from utils.RRRlib import *

params = load_params()
figdir = os.path.join(params['figdir'],'RRR','Validation')

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches


#%% 
session_list        = np.array([['LPE12223_2024_06_10'], #GR
                                ['LPE10919_2023_11_06']]) #GR

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,filter_noiselevel=True)

# #%% Get all data 
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])
# report_sessions(sessions)

#%%  Load data properly:        
for ises in range(nSessions):
    sessions[ises].load_tensor(load_calciumdata=True,calciumversion=params['calciumversion'],keepraw=True)

t_axis = sessions[0].t_axis

#%% 
for ises in range(nSessions):
    sessions[ises].respmat = np.nanmean(sessions[ises].tensor[:,:,(t_axis>0) & (t_axis<1)],axis=2) #average across time axis to get mean response to each stimulus condition

#%%
sessions = compute_tuning_wrapper(sessions)

#%% Example reconstruction of activity in the target area:
#Perform RRR on raw calcium data: 
ises    = 0
ses     = sessions[ises]

nN                  = 500
nM                  = 500
nranks              = 5
lam                 = 0

# idx_T               = ses.trialdata['Orientation']==0
excerpt_length      = 1000
t_start             = np.random.choice(ses.ts_F,1,replace=False)[0]
idx_T               = (ses.ts_F>t_start) & (ses.ts_F<t_start+excerpt_length)
ts                  = ses.ts_F[idx_T]
nTs                 = np.sum(idx_T)

idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                        ses.celldata['noise_level']<10),axis=0))[0]
idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                        ses.celldata['noise_level']<10),axis=0))[0]

np.random.seed(9)
idx_areax_sub       = np.random.choice(idx_areax,nN,replace=False)
idx_areay_sub       = np.random.choice(idx_areay,nM,replace=False)

X                   = sessions[ises].calciumdata.to_numpy()[np.ix_(idx_T,idx_areax_sub)]
Y                   = sessions[ises].calciumdata.to_numpy()[np.ix_(idx_T,idx_areay_sub)]

X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
Y                   = zscore(Y,axis=0)

Y_hat_rr               = np.full((nTs,nM,nranks),np.nan)  

B_hat         = LM(Y,X, lam=lam)

Y_hat         = X @ B_hat

# decomposing and low rank approximation of A
U, s, V = linalg.svd(Y_hat)
S = linalg.diagsvd(s,U.shape[0],s.shape[0])

for r in range(nranks):
    Y_hat_rr[:,:,r]       = X @ B_hat @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace

sourceweights = np.mean(np.abs(B_hat @ V[:r,:].T @ V[:r,:]),axis=1)
targetweights = np.mean(np.abs(B_hat @ V[:r,:].T @ V[:r,:]),axis=0)

targetr2 =np.full((nM,nranks),np.nan)
for r in range(nranks):
    targetr2[:,r]       = r2_score(Y,Y_hat_rr[:,:,r],multioutput='raw_values') 
    
r = nranks-1
# targetweights = r2_score(Y,Y_hat_rr[:,:,r],multioutput='raw_values') + (r2_score(Y,Y_hat_rr[:,:,r],multioutput='raw_values') - r2_score(Y,Y_hat_rr[:,:,1],multioutput='raw_values'))
targetweights = np.median(np.diff(targetr2,axis=1),axis=1)

#%% Identify a chunk in time where there is a high prediction for high rank across areas:
window_size = 10
window_step = 1
nTs = len(ts)
nWindows = int(np.ceil((nTs-window_size)/window_step))
variance_explained = np.full((nWindows,nranks),np.nan)  

t_starts = np.empty(nWindows)
for iw in range(nWindows):
    t_starts[iw] = ts[0]+iw*window_step
    t_end = t_starts[iw]+window_size
    idx_window = (ts>=t_starts[iw]) & (ts<t_end)
    Y_window = Y[idx_window]
    for r in range(nranks):
        variance_explained[iw,r] = EV(Y_window,Y_hat_rr[idx_window,:,r])

plt.plot(t_starts,variance_explained)

example_tstart = t_starts[np.nanargmax(variance_explained[:,1])]

#%% Make figure
scale = 0.2
# lw_orig = 0.7
lw = 0.4
nshowneurons = 10
nrankstoplot = 5
clrs_ranks = sns.color_palette('magma_r',n_colors=nrankstoplot+1)

fig,axes = plt.subplots(1,2,figsize=(7*cm,4*cm),sharex=True,sharey=True)
ax = axes[0]

source_example_neurons = np.argsort(-sourceweights)[:nshowneurons]
for i,iN in enumerate(source_example_neurons):
    ax.plot(ts,X[:,iN]*scale+i,color='k',lw=lw)
ax.set_title('source activity (V1)')
ax.axis('off')
ax.add_artist(AnchoredSizeBar(ax.transData, 1,
            "1 Sec", loc=4, frameon=False))

target_example_neurons = np.argsort(-targetweights)[:nshowneurons]
ax = axes[1]
plothandles = []
for r in range(0,nrankstoplot):
    plothandles.append(plt.Line2D([0], [0], color=clrs_ranks[r], lw=lw))
ax.legend(plothandles,['1','2','3','4','5'],title='Rank',bbox_to_anchor=(1.05, 1),
          frameon=False,fontsize=6,ncol=1,loc='upper left')
# handles2 = []
for i,iN in enumerate(target_example_neurons):
    ax.plot(ts,Y[:,iN]*scale+i,color='k',lw=lw)
#     handles2.append(ax.plot(ts,Y[:,iN]*scale+i,color='k',lw=lw)[0])
# ax.legend(handles2,['Target Neurons'],bbox_to_anchor=(1.05, 0.5),frameon=False,fontsize=6,ncol=1,loc='center left') 
ax.set_title('target activity (PM)')
ax.set_xlim([example_tstart-10,example_tstart+20])
ax.axis('off')

# my_savefig(fig,figdir,'V1PM_LowRank_Excerpt_%s_Rank%d.png' % (ses.sessiondata['session_id'][0],0),formats=['png']) 
for r in range(1,nrankstoplot):
    for i,iN in enumerate(target_example_neurons):
        ax.plot(ts,Y_hat_rr[:,iN,r]*scale+i,color=clrs_ranks[r],lw=lw,linestyle='--')
    # my_savefig(fig,figdir,'V1PM_LowRank_Excerpt_%s_Rank%d.png' % (ses.sessiondata['session_id'][0],r+1),formats=['png']) 
# my_savefig(fig,figdir,'V1PM_LowRank_Excerpt_%s_Rank%d' % (ses.session_id,r+1)) 




#%% 
####### #     #    #    #     # ######  #       #######    ######     #    #     # #    # 
#        #   #    # #   ##   ## #     # #       #          #     #   # #   ##    # #   #  
#         # #    #   #  # # # # #     # #       #          #     #  #   #  # #   # #  #   
#####      #    #     # #  #  # ######  #       #####      ######  #     # #  #  # ###    
#         # #   ####### #     # #       #       #          #   #   ####### #   # # #  #   
#        #   #  #     # #     # #       #       #          #    #  #     # #    ## #   #  
####### #     # #     # #     # #       ####### #######    #     # #     # #     # #    # 

#%% Show RRR example: 
nranks              = 40
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
nsampleneurons      = 25
nsampleneurons      = 500
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
R2_ranks            = np.full((nranks,nmodelfits,kfold),np.nan)

ses                 = sessions[0]
idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                        ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                        ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]

idx_T               = ses.trialdata['stimCond']==0

#on tensor during the response:
X                   = ses.tensor[np.ix_(idx_areax,idx_T,idx_resp)]
Y                   = ses.tensor[np.ix_(idx_areay,idx_T,idx_resp)]

#subtract mean response across trials:
X                   -= np.mean(X,axis=1,keepdims=True)
Y                   -= np.mean(Y,axis=1,keepdims=True)

# reshape to time points x neurons
X                   = X.reshape(len(idx_areax),-1).T
Y                   = Y.reshape(len(idx_areay),-1).T

ev,rank,R2_ranks = RRR_wrapper(Y, X, nN=nsampleneurons,nM=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% 
R2_ranks_r = np.reshape(R2_ranks,(nranks,nmodelfits*kfold))

#%% Plotting, show performance across ranks:
xticks = [0,10,20,30,40]
fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
handles = []
handles.append(shaded_error(range(nranks),R2_ranks_r.T,error='sem',color='k',alpha=0.3,ax=ax))
# ax.text(rank,ev,'|',color='b',fontsize=15,horizontalalignment='center',verticalalignment='bottom')
ax.axhline(y=ev,color='k',linestyle='--',alpha=0.5) #ax.text(rank,ev,'|',color='b',fontsize=15,horizontalalignment='center',verticalalignment='bottom')
ax.axvline(x=rank,color='k',linestyle='--',alpha=0.5) #ax.text(rank,ev,'|',color='b',fontsize=15,horizontalalignment='center',verticalalignment='bottom')
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.set_xticks(xticks)
ax.set_xlabel('Rank')
ax.set_ylabel('Performance ($R^2$)')
sns.despine(fig=fig,top=True,right=True,trim=True)
my_savefig(fig,figdir,'RRR_rank_ev_procedure_%d' % nsampleneurons)


#%% Show the effect of averaging over time, of reducing the signal and only having noise, 
# and of selecting responsive neurons

nranks              = 40
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5
nsampleneurons      = 25
# nsampleneurons      = 500
idx_resp            = np.where((t_axis>=0) & (t_axis<=2))[0]
R2_ranks            = np.full((nranks,nmodelfits,kfold),np.nan)

nT = len(idx_resp)
R2_cv = np.full((nT,2),np.nan) #time x raw/signal x all/onlynoise

ses                 = sessions[0]
idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                        ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                        ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]

idx_T               = ses.trialdata['stimCond']==0

for iT in range(nT):
    for ifilter in range(2):
        if ifilter:
            idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                    ses.celldata['noise_level']<params['maxnoiselevel'],
                    ses.celldata['gOSI']>0.5
                    # ses.celldata['tuning_var']>0.025
                    ),axis=0))[0]
            idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<params['maxnoiselevel'],
                            ses.celldata['gOSI']>0.5
                            # ses.celldata['tuning_var']>0.025
                            ),axis=0))[0]

        else:
            idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                    ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
            idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]

        #on tensor during the response:
        X                   = ses.tensor[np.ix_(idx_areax,idx_T,idx_resp)]
        Y                   = ses.tensor[np.ix_(idx_areay,idx_T,idx_resp)]

        # reshape to time points x neurons
        X                   = X.reshape(len(idx_areax),-1).T
        Y                   = Y.reshape(len(idx_areay),-1).T

        X           = block_reduce(X, block_size=(1,iT+1), func=np.mean, cval=np.mean(X))
        Y           = block_reduce(Y, block_size=(1,iT+1), func=np.mean, cval=np.mean(Y))
        # if
        R2_cv[iT,ifilter],_,_ = RRR_wrapper(Y, X, nN=nsampleneurons,nM=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%%
fig,axes = plt.subplots(1,1,figsize=(3.2*cm,3.2*cm))
ax = axes
handles = []
for ifilter in range(2):
    handles.append(plt.plot(np.arange(1,nT+1),R2_cv[:,ifilter],color='k',label='Raw, All Neurons',linestyle=['-','--'][ifilter])[0])
# handles.append(plt.plot(R2_cv[:,0,0],color='k',label='Raw, All Neurons')[0])
# handles.append(plt.plot(R2_cv[:,0,1],color='k',label='Raw, All Neurons')[0])
ax.legend(handles=handles,labels=['All neurons','Tuned neurons'],loc='center',frameon=False,fontsize=6,reverse=True)
my_legend_strip(ax)
ax.set_xlim([1,nT])
ax.set_xticks(np.arange(1,nT,3))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax.set_xlabel('# Frames Averaged')
ax.set_ylabel('Performance ($R^2$)')
sns.despine(fig=fig,top=True,right=True,trim=True,offset=3)
my_savefig(fig,figdir,'RRR_effect_of_averaging_and_selecting_responsive_neurons')

#%%
from scipy.stats import pearsonr

# Y: shape (n_timepoints, n_neurons, n_trials)
def split_half_reliability(Y):
    n_trials = Y.shape[2]
    idx = np.random.permutation(n_trials)
    half = n_trials // 2
    Y1 = np.mean(Y[:, :, idx[:half]], axis=2)
    Y2 = np.mean(Y[:, :, idx[half:]], axis=2)
    
    # Flatten over time and neurons for population-level reliability
    r, _ = pearsonr(Y1.flatten(), Y2.flatten())
    
    # Spearman–Brown correction
    r_sb = (2 * r) / (1 + r)
    R2_ceiling = r_sb ** 2
    return R2_ceiling






#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])
report_sessions(sessions)

#%%  Load data properly:    
# params['calciumversion'] = 'dF'
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False,compute_respmat=True)

#%%
# sessions = compute_tuning_wrapper(sessions)

#%% 
   #     #####  ######  #######  #####   #####     ####### ### #     # ####### 
  # #   #     # #     # #     # #     # #     #       #     #  ##   ## #       
 #   #  #       #     # #     # #       #             #     #  # # # # #       
#     # #       ######  #     #  #####   #####        #     #  #  #  # #####   
####### #       #   #   #     #       #       #       #     #  #     # #       
#     # #     # #    #  #     # #     # #     #       #     #  #     # #       
#     #  #####  #     # #######  #####   #####        #    ### #     # ####### 

#%% RRR across time relative to stimulus onset:
lam                 = 0
nranks              = 10 #number of ranks of RRR to be evaluated
nmodelfits          = 10 #number of times new neurons are resampled - many for final run
kfold               = 5
maxnoiselevel       = 20
nStim               = 8

ntimebins           = len(t_axis)
nsampleneurons      = 100

arealabelpairs      = ['V1-PM','PM-V1']
narealabelpairs     = len(arealabelpairs)
clrs_arealabelpairs = get_clr_area_pairs(arealabelpairs)

R2_cv               = np.full((narealabelpairs,ntimebins,nSessions,nStim),np.nan)
optim_rank          = np.full((narealabelpairs,ntimebins,nSessions,nStim),np.nan)
R2_ranks            = np.full((narealabelpairs,ntimebins,nSessions,nStim,nranks,nmodelfits,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model across time relative to stimulus onset'):
    for iapl, arealabelpair in enumerate(arealabelpairs):
        ses.trialdata['stimCond'] = np.mod(ses.trialdata['stimCond'],8)
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['roi_name']==alx,
                                ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==aly,
                                ses.celldata['noise_level']<maxnoiselevel,	
                                ),axis=0))[0]
        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
            continue

        # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        for istim,stim in enumerate([1,2]): # loop over orientations 
            idx_T   = ses.trialdata['stimCond']==stim

            X       = ses.tensor[np.ix_(idx_areax,idx_T,np.arange(ntimebins))]#.squeeze().T
            Y       = ses.tensor[np.ix_(idx_areay,idx_T,np.arange(ntimebins))]#.squeeze().T

            R2_cv[iapl,:,ises,istim],optim_rank[iapl,:,ises,istim],R2_ranks[iapl,:,ises,istim,:,:,:]      = RRR_wrapper_tensor(
                                                                        Y,X, nN=nsampleneurons,lam=lam,
                                                                       nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

            # for ibin in range(ntimebins):
                # X       = ses.tensor[np.ix_(idx_areax,idx_T,[ibin])].squeeze().T
                # Y       = ses.tensor[np.ix_(idx_areay,idx_T,[ibin])].squeeze().T

                # R2_cv[iapl,ibin,ises,istim],optim_rank[iapl,ibin,ises,istim],R2_ranks[iapl,ibin,ises,istim,:,:,:]      = RRR_wrapper(
                #                                                         Y[:,:,ibin].T, X[:,:,ibin].T, nN=nsampleneurons,lam=lam,
                #                                                        nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)
       
            # for ibin in range(ntimebins):
            #     idx_bins = np.arange(ibin-1,ibin+2)
            #     idx_bins = idx_bins[(idx_bins>=0) & (idx_bins<ntimebins)]
            #     # print(idx_bins)
            #     X       = ses.tensor[np.ix_(idx_areax,idx_T,idx_bins)].reshape(len(idx_areax),-1).T
            #     Y       = ses.tensor[np.ix_(idx_areay,idx_T,idx_bins)].reshape(len(idx_areay),-1).T

            #     R2_cv[iapl,ibin,ises,istim],optim_rank[iapl,ibin,ises,istim],R2_ranks[iapl,ibin,ises,istim,:,:,:]      = RRR_wrapper(
            #                                                             Y, X, nN=nsampleneurons,lam=lam,
            #                                                            nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

print(np.nanmean(R2_cv))



#%% Average across FF and FB and across folds: 
# R2_toplot            = np.nanmean(R2_cv,axis=(0,3))
# rank_toplot          = np.nanmean(optim_rank,axis=(0,3))

#%% Plotting, show performance across time:
t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(1,1,figsize=(3.5*cm,3.5*cm))
ax = axes
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs):
    R2_toplot            = R2_cv[iapl].reshape(ntimebins,nSessions*nStim)
    handles.append(shaded_error(t_axis,R2_toplot.T,error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))

# handles.append(shaded_error(t_axis,R2_toplot[:,:].T,error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=arealabelpairs,loc='best',fontsize=8)
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
# ax.set_xlim([0,2])
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('$R^2$')
sns.despine(fig=fig,top=True,right=True,trim=False,offset=3)
my_savefig(fig,figdir,'RRR_perf_across_time')

#%% Plotting, showing rank across time:
t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(1,1,figsize=(3.5*cm,3.5*cm))
ax = axes
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs):
    rank_toplot            = optim_rank[iapl].reshape(ntimebins,nSessions*nStim)
    handles.append(shaded_error(t_axis,rank_toplot.T,error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))

# for iapl, arealabelpair in enumerate(arealabelpairs):
# handles.append(shaded_error(t_axis,rank_toplot[:,:].T,error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/20
ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
# ax.legend(handles=handles,labels=arealabelpairs,loc='best',fontsize=8)
# my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('RRR R2')
sns.despine(fig=fig,top=True,right=True,trim=True)
# my_savefig(fig,figdir,'RRR_rank_across_time')



#%% 
######  ####### ######      #####  ### ####### ####### 
#     # #     # #     #    #     #  #       #  #       
#     # #     # #     #    #        #      #   #       
######  #     # ######      #####   #     #    #####   
#       #     # #                #  #    #     #       
#       #     # #          #     #  #   #      #       
#       ####### #           #####  ### ####### ####### 

#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons
popsizes            = np.array([5,10,20,50,100,200,500])
# popsizes            = np.array([5,10,20])
npopsizes           = len(popsizes)
nranks              = 40
nStim               = 16
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]
R2_cv               = np.full((nSessions,nStim,npopsizes,npopsizes),np.nan)
optim_rank          = np.full((nSessions,nStim,npopsizes,npopsizes),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
    
    # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
    for istim,stim in enumerate([0]): # loop over orientations 
        idx_T               = ses.trialdata['stimCond']==stim

        #on tensor during the response:
        X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
        Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
        
        #subtract mean response across trials:
        X                   -= np.mean(X,axis=1,keepdims=True)
        Y                   -= np.mean(Y,axis=1,keepdims=True)

        # reshape to neurons x time points
        X                   = X.reshape(len(idx_areax),-1).T
        Y                   = Y.reshape(len(idx_areay),-1).T

        for ixpop,xpop in enumerate(popsizes):
            for iypop,ypop in enumerate(popsizes):
                if len(idx_areax)>=xpop and len(idx_areay)>=ypop:
                    R2_cv[ises,istim,ixpop,iypop],optim_rank[ises,istim,ixpop,iypop],_ = RRR_wrapper(Y, X, nN=xpop,nM=ypop,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plot R2 for different number of V1 and PM neurons
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["r","g"], N=256) 
# cmap=LinearSegmentedColormap.from_list('rp',["green","white","purple"], N=256) 
cmap = sns.color_palette('magma', as_cmap=True)
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
ax = axes[0]
# s = ax.imshow(np.nanmean(R2_cv,axis=(0,1)).T,cmap=cmap,vmin=0,vmax=np.nanmax(np.nanmean(R2_cv,axis=(0,1))),origin='lower')
sns.heatmap(np.nanmean(R2_cv,axis=(0,1)).T,cmap=cmap,vmin=0,vmax=np.nanmax(np.nanmean(R2_cv,axis=(0,1))),annot=True,ax=ax,
            xticklabels=popsizes,yticklabels=popsizes,linewidths=0.1,cbar=False,fmt="1.2f",
            annot_kws={'size': 8},square=True)
ax.set_title('Performance ($R^2$)')
ax.set_xlabel('# source neurons')
ax.set_ylabel('# target neurons')

# Does the dimensionality increase with increasing number of neurons?
ax = axes[1]
sns.heatmap(np.nanmean(optim_rank,axis=(0,1)).T,cmap=cmap,vmin=0,vmax=np.nanmax(np.nanmean(optim_rank,axis=(0,1))),annot=True,ax=ax,
            xticklabels=popsizes,yticklabels=popsizes,linewidths=0.1,cbar=False,fmt="2.0f",
            annot_kws={'size': 8},square=True)
ax.invert_yaxis()
ax.set_xlabel('# source neurons')
ax.set_title('Rank')
sns.despine(top=True,right=True,offset=3)
plt.tight_layout()
my_savefig(fig,figdir,'R2_RRR_Tensor_Rank_PopSize_V1PM_%dsessions' % nSessions,formats=['png'])



#%% 
#     # ### ####### #     # ### #     #    #     #  #####        #     #####  ######  #######  #####   #####  
#  #  #  #     #    #     #  #  ##    #    #     # #     #      # #   #     # #     # #     # #     # #     # 
#  #  #  #     #    #     #  #  # #   #    #     # #           #   #  #       #     # #     # #       #       
#  #  #  #     #    #######  #  #  #  #    #     #  #####     #     # #       ######  #     #  #####   #####  
#  #  #  #     #    #     #  #  #   # #     #   #        #    ####### #       #   #   #     #       #       # 
#  #  #  #     #    #     #  #  #    ##      # #   #     #    #     # #     # #    #  #     # #     # #     # 
 ## ##  ###    #    #     # ### #     #       #     #####     #     #  #####  #     # #######  #####   #####  

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealabelpairs  = ['V1-V1',
                    'PM-PM',
                    'V1-PM',
                    'PM-V1'
                    ]

clrs_arealabelpairs = get_clr_area_pairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

nsampleneurons      = 50
nsampleneurons      = 100
nranks              = 25
nmodelfits          = 15 #number of times new neurons are resampled 
nStim               = 16
idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]

R2_cv               = np.full((narealabelpairs,nSessions,nStim),np.nan)
optim_rank          = np.full((narealabelpairs,nSessions,nStim),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for within vs across populations'):
# for ises,ses in tqdm(enumerate(sessions[:2]),total=nSessions,desc='Fitting RRR model for within vs across populations'):
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    if np.sum((ses.celldata['roi_name']=='V1') & (ses.celldata['noise_level']<params['maxnoiselevel']))<(nsampleneurons*2):
        continue
    
    if np.sum((ses.celldata['roi_name']=='PM') & (ses.celldata['noise_level']<params['maxnoiselevel']))<(nsampleneurons*2):
        continue

    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['roi_name']==alx,
                                ses.celldata['noise_level']<params['maxnoiselevel']
                                ),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==aly,
                                ses.celldata['noise_level']<params['maxnoiselevel']
                                ),axis=0))[0]
    
        if np.any(np.intersect1d(idx_areax,idx_areay)): #if interactions within one population:
            if not np.array_equal(idx_areax, idx_areay): 
                print('Arealabelpair %s has partly overlapping neurons'%arealabelpair)
            idx_areax, idx_areay = np.array_split(np.random.permutation(idx_areax), 2)

        # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        for istim,stim in enumerate([0,4,7]): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim

            #on tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            # subtract mean response across trials:
            # X                   -= np.mean(X,axis=1,keepdims=True)
            # Y                   -= np.mean(Y,axis=1,keepdims=True)

            # reshape to time points x neurons
            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
            R2_cv[iapl,ises,istim],optim_rank[iapl,ises,istim],_  = RRR_wrapper(Y, X, nN=nsampleneurons,nranks=nranks,nmodelfits=nmodelfits)

#%% Plotting:
clr = clrs_arealabelpairs[0]
R2_toplot = np.reshape(R2_cv,(narealabelpairs,nSessions*nStim))
rank_toplot = np.reshape(optim_rank,(narealabelpairs,nSessions*nStim))
# R2_toplot = np.nanmean(R2_cv,axis=2)
# rank_toplot = np.nanmean(optim_rank,axis=2)

fig,axes = plt.subplots(1,2,figsize=(6.5*cm,3.5*cm))
markersize=30
clrs = get_clr_areas(['V1','PM'])
ax = axes[0]
comps = [[0,3],[1,2]]
for icomp,comp in enumerate(comps):
    ax.scatter(R2_toplot[comp[0],:],R2_toplot[comp[1],:],s=markersize,edgecolor='w',marker='.',color=clrs[icomp],
               linewidth=0.5)
    # ax.scatter(R2_cv[comp[0],:],R2_cv[comp[1],:],s=80,alpha=0.8,marker='.',color=clrs[icomp])
    ax_nticks(ax,3)
    _,pval = ttest_rel(R2_toplot[comp[0],:],R2_toplot[comp[1],:],nan_policy='omit')
    ax.text(0.6,0.1*(1+icomp),'%sp=%1.2e' % (get_sig_asterisks(pval),pval),transform=ax.transAxes,fontsize=5,color=clrs[icomp])
ax.set_xlabel('Within')
ax.set_ylabel('Across')
ax.set_title(r'$R^2$')
    # ax.legend(['V1 (V1->V1 vs PM->V1)','PM (V1->PM vs PM->PM)'],frameon=False,loc='upper left')
ax.legend(['V1','PM'],frameon=False,fontsize=7,loc='upper left')
my_legend_strip(ax)
ax.set_xlim([0,0.22])
ax.set_ylim([0,0.22])
ax.plot([0,0.2],[0,0.2],':',color='grey',linewidth=1)

ax = axes[1]
for icomp,comp in enumerate(comps):
    ax.scatter(rank_toplot[comp[0],:],rank_toplot[comp[1],:],s=markersize,edgecolor='w',marker='.',
               linewidth=0.5,color=clrs[icomp])
    _,pval = ttest_rel(rank_toplot[comp[0],:],rank_toplot[comp[1],:],nan_policy='omit')
    ax.text(0.6,0.1*(1+icomp),'%sp=%1.2e' % (get_sig_asterisks(pval),pval),transform=ax.transAxes,fontsize=5,color=clrs[icomp])
# ax.legend(['V1','PM'],frameon=False,fontsize=10)ax.set_xlabel('Within')
ax.set_ylabel('Across')
ax.set_title('Rank')

# ax.legend(['V1 (V1->V1 vs PM->V1)','PM (V1->PM vs PM->PM)'],frameon=False,loc='upper left')
# my_legend_strip(ax)
ax.set_xlim([0,20])
ax.set_ylim([0,20])
ax_nticks(ax,4)
ax.plot([0,25],[0,25],':',color='grey',linewidth=1)
# ax.legend(['V1 (V1->V1 vs PM->V1','PM (PM->PM vs V1->PM'],frameon=False,fontsize=8)
plt.tight_layout()

sns.despine(offset=3,top=True,right=True)
my_savefig(fig,figdir,'RRR_R2Rank_WithinVSAcross_%dneurons' % nsampleneurons)


#%% Get the ratio of within to across:
R2_toplot = np.reshape(R2_cv,(narealabelpairs,nSessions*nStim))
rank_toplot = np.reshape(optim_rank,(narealabelpairs,nSessions*nStim))

R2_ratio_within_across = R2_toplot[[0,1],:] /R2_toplot[[3,2],:]
rank_ratio_within_across = rank_toplot[[0,1],:] /rank_toplot[[3,2],:]

fig,axes = plt.subplots(1,2,figsize=(6.5*cm,3.5*cm))
ax = axes[0]
sns.stripplot(R2_ratio_within_across.T,ax=ax,color='k')
# sns.barplot(R2_ratio_within_across.T,ax=ax,color='k')
ax.set_ylim([0.8,2.5])
ax.axhline(y=1,color='k',linestyle='--')
ax.set_ylabel('Ratio R2\n(Within/Across)')
ax.set_xticks(range(2),['V1','PM' ],rotation=0,fontsize=6)
ax = axes[1]
sns.stripplot(rank_ratio_within_across.T,ax=ax,color='k')
ax.set_ylim([0.8,2.5])
ax.axhline(y=1,color='k',linestyle='--')
ax.set_ylabel('Ratio rank\n(Within/Across)')
ax.set_xticks(range(2),['V1','PM' ],rotation=0,fontsize=6)
plt.tight_layout()
sns.despine(offset=3,top=True,right=True)
my_savefig(fig,figdir,'RRR_R2Rank_ratio_WithinVSAcross_%dneurons' % nsampleneurons)



#%% 
######  ####### ######  ######  #######  #####     #    ####### ####### ######  
#     # #       #     # #     # #       #     #   # #      #    #       #     # 
#     # #       #     # #     # #       #        #   #     #    #       #     # 
#     # #####   ######  ######  #####   #       #     #    #    #####   #     # 
#     # #       #       #   #   #       #       #######    #    #       #     # 
#     # #       #       #    #  #       #     # #     #    #    #       #     # 
######  ####### #       #     # #######  #####  #     #    #    ####### ######  



#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons
popsizes            = np.array([5,10,20,50,100,200,500])
# popsizes            = np.array([5,10,20])
npopsizes           = len(popsizes)
nranks              = 40
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
R2_cv               = np.full((nSessions,npopsizes,npopsizes),np.nan)
optim_rank          = np.full((nSessions,npopsizes,npopsizes),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
    
    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    for ixpop,xpop in enumerate(popsizes):
        for iypop,ypop in enumerate(popsizes):
            if len(idx_areax)>=xpop and len(idx_areay)>=ypop:
                R2_cv[ises,ixpop,iypop],optim_rank[ises,ixpop,iypop] = RRR_wrapper(Y, X, nN=xpop,nM=ypop,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)


#%% Plot R2 for different number of V1 and PM neurons
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["r","g"], N=256) 
cmap = sns.color_palette('magma', as_cmap=True)
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
ax = axes[0]
s = ax.imshow(np.nanmean(R2_cv,axis=0).T,cmap=cmap,vmin=0,vmax=np.nanmax(np.nanmean(R2_cv,axis=0)),origin='lower')
ax.set_xticks(range(npopsizes),labels=popsizes)
ax.set_yticks(range(npopsizes),labels=popsizes)
fig.colorbar(s, ax=ax,shrink=0.3,orientation='vertical')
ax.set_title('R2')
ax.set_xlabel('# source neurons')
ax.set_ylabel('# target neurons')

# Does the dimensionality increase with increasing number of neurons?
ax = axes[1]
s = ax.imshow(np.nanmean(optim_rank,axis=0).T,cmap=cmap,vmin=0,vmax=20,origin='lower')
ax.set_xticks(range(npopsizes),labels=popsizes)
ax.set_yticks(range(npopsizes),labels=popsizes)
fig.colorbar(s, ax=ax,shrink=0.3,orientation='vertical')
ax.set_xlabel('# source neurons')
# ax.set_ylabel('# target neurons')
ax.set_title('Rank')

sns.despine(top=True,right=True,offset=3)

plt.tight_layout()
my_savefig(fig,figdir,'R2_RRR_Rank_PopSize_Both_V1PM_%dsessions.png' % nSessions,formats=['png'])


#%% Does performance increase with increasing number of neurons? Predicting PM from V1 with different number of V1 and PM neurons
popsizes            = np.array([5,10,20,50,100,200,500])
# popsizes            = np.array([5,10,20,50,100])
npopsizes           = len(popsizes)
nranks              = 50
nmodelfits          = 5 #number of times new neurons are resampled 
kfold               = 5
R2_cv               = np.full((nSessions,npopsizes),np.nan)
optim_rank          = np.full((nSessions,npopsizes),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
    
    X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
    Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
    for ipop,pop in enumerate(popsizes):
        if len(idx_areax)>=pop and len(idx_areay)>=pop:
            R2_cv[ises,ipop],optim_rank[ises,ipop]             = RRR_wrapper(Y, X, nN=pop,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)


#%% Plot R2 for different number of V1 and PM neurons
clrs_popsizes = sns.color_palette("rocket",len(popsizes))

fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=True)
ax = axes[0]
ax.scatter(popsizes, np.nanmean(R2_cv,axis=0), marker='o', color=clrs_popsizes)
# ax.plot(popsizes, np.nanmean(R2_cv,axis=0),color='k', linewidth=2)
shaded_error(popsizes,R2_cv,center='mean',error='sem',color='k',ax=ax)
ax.set_ylim([0,0.3])
# ax.set_xticks(popsizes)
ax.axhline(y=0,color='k',linestyle='--')

ax.set_xlabel('Population size')
ax.set_ylabel('RRR R2')
ax_nticks(ax,4)
# ax.set_xscale('log')

# Does the dimensionality increase with increasing number of neurons?
ax = axes[1]
ax.scatter(popsizes, np.nanmean(optim_rank,axis=0), marker='o', color=clrs_popsizes)
shaded_error(popsizes,optim_rank,center='mean',error='sem',color='k',ax=ax)

ax.plot(popsizes,popsizes**0.5,color='r',linestyle='--',linewidth=1)
ax.text(100,13,'$n^{1/2}$',color='r',fontsize=12)
ax.plot(popsizes,popsizes**0.3333,color='g',linestyle='--',linewidth=1)
ax.text(100,2,'$n^{1/3}$',color='g',fontsize=12)

ax.set_ylim([0,20])
ax.set_ylabel('Dimensionality')
ax.set_xlabel('Population size')
ax.set_xticks(popsizes[::2])
ax.set_xticks([10,100,200,500])

sns.despine(top=True,right=True,offset=3)

plt.tight_layout()
my_savefig(fig,figdir,'R2_RRR_Rank_PopSize_V1PM_%dsessions.png' % nSessions,formats=['png'])



#%% Show RRR performance as a function of the number of trials:
nsampleneurons  = 500
ntrials         = np.array([10,20,50,100,200,400,600,1000,2000,3200])
# ntrials         = np.array([50,100,200,400,1000])
ntrialsubsets   = len(ntrials)
kfold           = 5
nranks          = 50
lam             = 1
R2_cv           = np.full((nSessions,ntrialsubsets,nranks,kfold),np.nan)

kf              = KFold(n_splits=kfold,shuffle=True)

for ises,ses in enumerate(sessions):
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]

    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:

        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)

        for intrials,nt in enumerate(ntrials):
            idx_T           = np.random.choice(ses.respmat.shape[1],nt,replace=False)

            X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
            Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
            
            X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
            Y                   = zscore(Y,axis=0)

            for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                
                X_train, X_test     = X[idx_train], X[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                B_hat_train         = LM(Y_train,X_train, lam=lam)

                Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = linalg.svd(Y_hat_train)
                S = linalg.diagsvd(s,U.shape[0],s.shape[0])

                for r in range(nranks):
                    Y_hat_rr_test       = X_test @ B_hat_train @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace

                    R2_cv[ises,intrials,r,ikf] = EV(Y_test,Y_hat_rr_test)


#%% 
R2data = np.full((nSessions,ntrialsubsets),np.nan)
rankdata = np.full((nSessions,ntrialsubsets),np.nan)

for ises,ses in enumerate(sessions):
    for intrials,nt in enumerate(ntrials):
        R2data[ises,intrials],rankdata[ises,intrials] = rank_from_R2(R2_cv[ises,intrials,:,:],nranks,kfold)

#%% plot the results:
lambdacolors = sns.color_palette('magma',ntrialsubsets)

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax  = axes[0]
for intrials,nt in enumerate(ntrials):
    tempdata = np.nanmean(R2_cv[:,intrials,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[intrials],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('R2')
ax.set_ylim([-0.05,0.5])
ax.legend(ntrials,frameon=False,ncol=2,fontsize=8,title='number of trials')

ax = axes[1]
for ises,ses in enumerate(sessions):
    ax.plot(ntrials,rankdata[ises,:],color='grey',linewidth=1)
ax.plot(ntrials,np.nanmean(rankdata,axis=0),color='k',linewidth=1.5)
ax.scatter(ntrials,np.nanmean(rankdata,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('number of trials')
# ax.set_xscale('log')
ax.set_ylabel('estimated rank')
ax.set_ylim([0,np.max(rankdata)+1])

ax = axes[2]
for ises,ses in enumerate(sessions):
    ax.plot(ntrials,R2data[ises,:],color='grey',linewidth=1)
ax.plot(ntrials,np.nanmean(R2data,axis=0),color='k',linewidth=1.5)
ax.scatter(ntrials,np.nanmean(R2data,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('number of trials')
# ax.set_xscale('log')
ax.set_ylabel('R2 at optimal rank')
ax.set_ylim([0,np.max(R2data)+.05])
plt.tight_layout()
plt.savefig(os.path.join(figdir,'RRR_nTrials_lam%d_Rank_%dneurons.png' % (lam,nsampleneurons)), format = 'png')


#%%  Within to across area dimensionality comparison: 

figdir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\WithinAcross\\')


#%% 
min_cells = 1500
sessions,nSessions   = filter_sessions(protocols = 'GR',min_cells=min_cells,filter_areas=['V1','PM'])

#%% 
print('Number of cells in V1 and in PM:')
for ises,ses in enumerate(sessions):
    print('Session %d: %d cells in V1, %d cells in PM' % (ises,np.sum(ses.celldata['roi_name']=='V1'),np.sum(ses.celldata['roi_name']=='PM')))

#%%  Load data properly:        
calciumversion = 'dF'
# calciumversion = 'deconv'

for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=False, load_calciumdata=True,load_videodata=False,
                                calciumversion=calciumversion,keepraw=False)

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealabelpairs  = ['V1-V1',
                    'PM-PM',
                    'V1-PM',
                    'PM-V1'
                    ]
alp_withinacross = ['Within',
                   'Within',
                   'Across',
                   'Across'] 

clrs_arealabelpairs = get_clr_area_pairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

lam                 = 0
nsampleneurons      = 100
nranks              = 50
nmodelfits          = 10 #number of times new neurons are resampled 
kfold               = 5

R2_cv               = np.full((narealabelpairs,nSessions),np.nan)
optim_rank          = np.full((narealabelpairs,nSessions),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    if np.sum((ses.celldata['roi_name']=='V1') & (ses.celldata['noise_level']<params['maxnoiselevel']))<(nsampleneurons*2):
        continue
    
    if np.sum((ses.celldata['roi_name']=='PM') & (ses.celldata['noise_level']<params['maxnoiselevel']))<(nsampleneurons*2):
        continue

    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['roi_name']==alx,
                                ses.celldata['noise_level']<params['maxnoiselevel']
                                ),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==aly,
                                ses.celldata['noise_level']<params['maxnoiselevel']
                                ),axis=0))[0]
    
        if np.any(np.intersect1d(idx_areax,idx_areay)): #if interactions within one population:
            if not np.array_equal(idx_areax, idx_areay): 
                print('Arealabelpair %s has partly overlapping neurons'%arealabelpair)
            idx_areax, idx_areay = np.array_split(np.random.permutation(idx_areax), 2)

        X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T

        R2_cv[iapl,ises],optim_rank[iapl,ises],_  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

#%% Plotting:
clr = clrs_arealabelpairs[0]

fig,axes = plt.subplots(1,2,figsize=(4.5,2.5))

clrs = get_clr_areas(['V1','PM'])
ax = axes[0]
comps = [[0,3],[1,2]]
for icomp,comp in enumerate(comps):
    ax.scatter(R2_cv[comp[0],:],R2_cv[comp[1],:],s=120,edgecolor='w',marker='.',color=clrs[icomp])
    # ax.scatter(R2_cv[comp[0],:],R2_cv[comp[1],:],s=80,alpha=0.8,marker='.',color=clrs[icomp])
    ax_nticks(ax,3)
    ax.set_xlabel('Within')
    ax.set_ylabel('Across')
    ax.set_title('R2')

    _,pval = ttest_rel(R2_cv[comp[0],:],R2_cv[comp[1],:],nan_policy='omit')
    ax.text(0.6,0.1*(1+icomp),'%sp=%.3f' % (get_sig_asterisks(pval),pval),transform=ax.transAxes,fontsize=10,color=clrs[icomp])
ax.legend(['V1','PM'],frameon=False,fontsize=10)
my_legend_strip(ax)
ax.set_xlim([0,0.4])
ax.set_ylim([0,0.4])
ax.plot([0,0.4],[0,0.4],':',color='grey',linewidth=1)

ax = axes[1]
for icomp,comp in enumerate(comps):
    ax.scatter(optim_rank[comp[0],:],optim_rank[comp[1],:],s=120,edgecolor='w',marker='.',color=clrs[icomp])
    ax_nticks(ax,3)
    ax.set_xlabel('Within')
    ax.set_title('Rank')
    _,pval = ttest_rel(optim_rank[comp[0],:],optim_rank[comp[1],:],nan_policy='omit')
    ax.text(0.6,0.1*(1+icomp),'%sp=%.3f' % (get_sig_asterisks(pval),pval),transform=ax.transAxes,fontsize=10,color=clrs[icomp])
ax.legend(['V1','PM'],frameon=False,fontsize=10)
my_legend_strip(ax)
ax.set_xlim([0,25])
ax.set_ylim([0,25])
ax.plot([0,25],[0,25],':',color='grey',linewidth=1)
# ax.legend(['V1 (V1->V1 vs PM->V1','PM (PM->PM vs V1->PM'],frameon=False,fontsize=8)

sns.despine(offset=3,top=True,right=True)
plt.tight_layout()

my_savefig(fig,figdir,'RRR_R2Rank_WithinVSAcross_%dneurons' % nsampleneurons,formats=['png'])






#%% 

# DEPRECATED:





#%% get optimal lambda
nsampleneurons  = 500
lambdas         = np.logspace(-6, 5, 10)
# lambdas         = np.array([0,0.01,0.1,1])
nlambdas        = len(lambdas)
kfold           = 5
nranks          = 50

R2_cv           = np.full((nSessions,nlambdas,nranks,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different labmdas'):
    idx_T               = ses.trialdata['Orientation']==0
    # idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]

    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:

        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)

        X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
        
        X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
        Y                   = zscore(Y,axis=0)

        # Explanation of steps
        # X is of shape K x N (samples by features), Y is of shape K x M
        # K is the number of samples, N is the number of neurons in area 1,
        # M is the number of neurons in area 2

        # multiple linear regression, B_hat is of shape N x M:
        # B_hat               = LM(Y,X, lam=lam) 
        #RRR: do SVD decomp of Y_hat, 
        # U is of shape K x r, S is of shape r x r, V is of shape r x M
        # Y_hat_rr,U,S,V     = RRR(Y, X, B_hat, r) 

        for ilam,lam in enumerate(lambdas):
            # cross-validation version
            # R2_cv   = np.zeros(kfold)
            kf          = KFold(n_splits=kfold,shuffle=True)
            for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                
                X_train, X_test     = X[idx_train], X[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                B_hat_train         = LM(Y_train,X_train, lam=lam)

                Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = linalg.svd(Y_hat_train)
                S = linalg.diagsvd(s,U.shape[0],s.shape[0])

                for r in range(nranks):
                    Y_hat_rr_test       = X_test @ B_hat_train @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace

                    R2_cv[ises,ilam,r,ikf] = EV(Y_test,Y_hat_rr_test)


#%% plot the results for lam = 0
lambdacolors = sns.color_palette('magma',nlambdas)

fig,ax = plt.subplots(1,1,figsize=(3,3))
for ilam,lam in enumerate([lambdas[0]]):
    tempdata = np.nanmean(R2_cv[:,ilam,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[ilam],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('Test R2')
ax.set_xticks(range(nranks+5)[::5])
sns.despine(ax=ax,top=True,right=True,trim=True)
# ax.legend(my_ceil(np.log10(lambdas),1),frameon=False,ncol=2,title='lambda (10^x)')

#%% Compute optimal rank:
R2data = np.full((nSessions,nlambdas),np.nan)
rankdata = np.full((nSessions,nlambdas),np.nan)
for ises,ses in enumerate(sessions):
    if np.sum(np.isnan(R2_cv[ises,:,:,:]))>0:
        continue
    for ilam,lam in enumerate(lambdas):
        R2data[ises,ilam],rankdata[ises,ilam] = rank_from_R2(R2_cv[ises,ilam,:,:],nranks,kfold)

#%% plot the results:
lambdacolors = sns.color_palette('magma',nlambdas)

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax  = axes[0]
for ilam,lam in enumerate(lambdas):
    tempdata = np.nanmean(R2_cv[:,ilam,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[ilam],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('R2')
ax.legend(my_ceil(np.log10(lambdas),1),frameon=False,ncol=2,title='lambda (10^x)')

ax = axes[1]
for ises,ses in enumerate(sessions):
    ax.plot(lambdas,rankdata[ises,:],color='grey',linewidth=1)
ax.plot(lambdas,np.nanmean(rankdata,axis=0),color='k',linewidth=1.5)
ax.scatter(lambdas,np.nanmean(rankdata,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('lambda')
ax.set_xscale('log')
ax.set_ylabel('estimated rank')
ax.set_ylim([0,np.nanmax(rankdata)+1])

ax = axes[2]
for ises,ses in enumerate(sessions):
    ax.plot(lambdas,R2data[ises,:],color='grey',linewidth=1)
ax.plot(lambdas,np.nanmean(R2data,axis=0),color='k',linewidth=1.5)
ax.scatter(lambdas,np.nanmean(R2data,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('lambda')
ax.set_xscale('log')
ax.set_ylabel('R2 at optimal rank')
ax.set_ylim([0,np.nanmax(R2data)+.05])
plt.tight_layout()
# plt.savefig(os.path.join(figdir,'RRR_perOri_Lam_Rank_%dneurons.png' % nsampleneurons), format = 'png')
# plt.savefig(os.path.join(figdir,'RRR_Lam_Rank_%dneurons.png' % nsampleneurons), format = 'png')
my_savefig(fig,figdir,'RRR_Lam_Rank_%dneurons.png' % nsampleneurons,formats=['png'])
# my_savefig(fig,figdir,'RRR_perOri_Lam_Rank_%dneurons.png' % nsampleneurons,formats=['png'])



#%% get optimal pre pCA
nsampleneurons  = 500
PCAdims         = np.array([1,2,5,10,20,50,100])
# lambdas         = np.array([0,0.01,0.1,1])
nPCAdims        = len(PCAdims)
kfold           = 5
nranks          = 50
lam             = 5000
R2_cv           = np.full((nSessions,nPCAdims,nranks,kfold),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different pre PCA dims'):
    # idx_T               = ses.trialdata['Orientation']==0
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                            ses.celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]

    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:

        idx_areax_sub       = np.random.choice(idx_areax,nsampleneurons,replace=False)
        idx_areay_sub       = np.random.choice(idx_areay,nsampleneurons,replace=False)

        X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
        Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
        
        X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
        Y                   = zscore(Y,axis=0)

        # Explanation of steps
        # X is of shape K x N (samples by features), Y is of shape K x M
        # K is the number of samples, N is the number of neurons in area 1,
        # M is the number of neurons in area 2

        # multiple linear regression, B_hat is of shape N x M:
        # B_hat               = LM(Y,X, lam=lam) 
        #RRR: do SVD decomp of Y_hat, 
        # U is of shape K x r, S is of shape r x r, V is of shape r x M
        # Y_hat_rr,U,S,V     = RRR(Y, X, B_hat, r) 

        for ipc,PCAdim in enumerate(PCAdims):
            # cross-validation version
            # R2_cv   = np.zeros(kfold)

            Xmodel      = PCA(n_components=PCAdim)
            Xpca        = Xmodel.fit_transform(X)
            Ymodel      = PCA(n_components=PCAdim)
            Ypca        = Ymodel.fit_transform(Y)

            kf          = KFold(n_splits=kfold,shuffle=True)

            for ikf, (idx_train, idx_test) in enumerate(kf.split(Xpca)):
                
                X_train, X_test     = Xpca[idx_train], Xpca[idx_test]
                Y_train, Y_test     = Ypca[idx_train], Ypca[idx_test]

                B_hat_train         = LM(Y_train,X_train, lam=lam)

                Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of A
                U, s, V = linalg.svd(Y_hat_train)
                S = linalg.diagsvd(s,U.shape[0],s.shape[0])

                for r in range(nranks):
                    Y_hat_rr_test       = X_test @ B_hat_train @ V[:r,:].T @ V[:r,:] #project test data onto low rank subspace

                    R2_cv[ises,ipc,r,ikf] = EV(Y_test,Y_hat_rr_test) * Ymodel.explained_variance_ratio_.sum()


#%% 
R2data = np.full((nSessions,nPCAdims),np.nan)
rankdata = np.full((nSessions,nPCAdims),np.nan)

for ises,ses in enumerate(sessions):
    for ipc,PCAdim in enumerate(PCAdims):
        R2data[ises,ipc],rankdata[ises,ipc] = rank_from_R2(R2_cv[ises,ipc,:,:],nranks,kfold)

#%% plot the results:
lambdacolors = sns.color_palette('magma',nPCAdims)

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax  = axes[0]
for ilam,lam in enumerate(PCAdims):
    tempdata = np.nanmean(R2_cv[:,ilam,:,:],axis=(0,2))
    ax.plot(range(nranks),tempdata,color=lambdacolors[ilam],linewidth=1)
ax.set_xlabel('rank')
ax.set_ylabel('R2')

ax = axes[1]
for ises,ses in enumerate(sessions):
    ax.plot(PCAdims,rankdata[ises,:],color='grey',linewidth=1)
ax.plot(PCAdims,np.nanmean(rankdata,axis=0),color='k',linewidth=1.5)
ax.scatter(PCAdims,np.nanmean(rankdata,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('PCAdim')
ax.set_xscale('log')
ax.set_ylabel('estimated rank')
ax.set_ylim([0,np.max(rankdata)+1])

ax = axes[2]
for ises,ses in enumerate(sessions):
    ax.plot(PCAdims,R2data[ises,:],color='grey',linewidth=1)
ax.plot(PCAdims,np.nanmean(R2data,axis=0),color='k',linewidth=1.5)
ax.scatter(PCAdims,np.nanmean(R2data,axis=0),s=100,marker='.',color=lambdacolors)
ax.set_xlabel('PCAdim')
ax.set_xscale('log')
ax.set_ylabel('R2 at optimal rank')
ax.set_ylim([0,np.max(R2data)+.05])
plt.tight_layout()
# plt.savefig(os.path.join(figdir,'RRR_PrePCA_lam5000_Rank_%dneurons.png' % nsampleneurons), format = 'png')
# plt.savefig(os.path.join(figdir,'RRR_PrePCA_lam0_Rank_%dneurons.png' % nsampleneurons), format = 'png')
plt.savefig(os.path.join(figdir,'RRR_PrePCA_lam0_perOri_Rank_%dneurons.png' % nsampleneurons), format = 'png')




#%%%


#%%

for ses in sessions:
    ses.respmat = np.nanmean(ses.tensor[:,:,idx_resp],axis=2)

#%%
from utils.tuning import compute_tuning_wrapper

sessions = compute_tuning_wrapper(sessions)

#%% Do RRR of V1 and PM labeled and unlabeled neurons
kfold           = 5
nmodelfits      = 2
nsampleneurons  = 50
nStim           = 16
idx_resp        = np.where((t_axis>=0) & (t_axis<=1.5))[0]

percs           = np.linspace(0,100,10)
cellfield       = 'event_rate'
cellfield       = 'OSI'
# cellfield       = 'tuning_var'
# cellfield       = 'noise_level'
data            = np.full((len(percs),nSessions,nStim),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different  sizes'):
    for iperc,perc in enumerate(percs):
        idx_areax           = np.where(np.all((ses.celldata['roi_name']=='V1',
                                ses.celldata[cellfield]>np.percentile(ses.celldata[cellfield],perc),	
                                ),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']=='PM',
                                ses.celldata[cellfield]>np.percentile(ses.celldata[cellfield],perc),	
                                ),axis=0))[0]

        if len(idx_areax)<nsampleneurons or len(idx_areay)<nsampleneurons:
            continue

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim

            #on tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            #subtract mean response across trials:
            X                   -= np.mean(X,axis=1,keepdims=True)
            Y                   -= np.mean(Y,axis=1,keepdims=True)

            # reshape to neurons x time points
            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            #zscore
            X                   = zscore(X,axis=0,nan_policy='omit')  #Z score activity for each neuron
            Y                   = zscore(Y,axis=0,nan_policy='omit')

            #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
            data[iperc,ises,istim],_,_ = RRR_wrapper(Y, X, 
                            nN=nsampleneurons,nK=None,lam=lam,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)
            # print(R2_cv)

#%%

plt.plot(percs,np.nanmean(data,axis=(1,2)),color='k',linewidth=2)

