# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.regress_lib import *
from utils.tuning import compute_tuning_wrapper
from utils.params import load_params
from utils.RRRlib import *

params = load_params()
figdir = os.path.join(params['figdir'],'RRR','Validation')

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches


#%% 
#     # ###  #####  #     #    #    #       ### ####### #######                                             
#     #  #  #     # #     #   # #   #        #       #  #                                                   
#     #  #  #       #     #  #   #  #        #      #   #                                                   
#     #  #   #####  #     # #     # #        #     #    #####                                               
 #   #   #        # #     # ####### #        #    #     #                                                   
  # #    #  #     # #     # #     # #        #   #      #                                                   
   #    ###  #####   #####  #     # ####### ### ####### #######    
                                             
######  #######  #####  ####### #     #  #####  ####### ######  #     #  #####  ####### ### ####### #     # 
#     # #       #     # #     # ##    # #     #    #    #     # #     # #     #    #     #  #     # ##    # 
#     # #       #       #     # # #   # #          #    #     # #     # #          #     #  #     # # #   # 
######  #####   #       #     # #  #  #  #####     #    ######  #     # #          #     #  #     # #  #  # 
#   #   #       #       #     # #   # #       #    #    #   #   #     # #          #     #  #     # #   # # 
#    #  #       #     # #     # #    ## #     #    #    #    #  #     # #     #    #     #  #     # #    ## 
#     # #######  #####  ####### #     #  #####     #    #     #  #####   #####     #    ### ####### #     # 


#%% 
session_list        = np.array([['LPE10919_2023_11_06']])

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,filter_noiselevel=True)

#%%  Load data properly:        
sessions[0].load_tensor(load_calciumdata=True,calciumversion=params['calciumversion'],keepraw=True)
# sessions[0].load_data(load_calciumdata=True,calciumversion=params['calciumversion'])
t_axis = sessions[0].t_axis

#%% Example reconstruction of activity in the target area:
#Perform RRR on raw calcium data: 
nN                  = 500
nM                  = 500
nranks              = 5

excerpt_length      = 1000
t_start             = np.random.choice(sessions[0].ts_F,1,replace=False)[0]
idx_T               = (sessions[0].ts_F>t_start) & (sessions[0].ts_F<t_start+excerpt_length)
ts                  = sessions[0].ts_F[idx_T]
nTs                 = np.sum(idx_T)

idx_areax           = np.where(np.all((sessions[0].celldata['roi_name']=='V1',
                        sessions[0].celldata['noise_level']<10),axis=0))[0]
idx_areay           = np.where(np.all((sessions[0].celldata['roi_name']=='PM',
                        sessions[0].celldata['noise_level']<10),axis=0))[0]

np.random.seed(9)
idx_areax_sub       = np.random.choice(idx_areax,nN,replace=False)
idx_areay_sub       = np.random.choice(idx_areay,nM,replace=False)

X                   = sessions[0].calciumdata.to_numpy()[np.ix_(idx_T,idx_areax_sub)]
Y                   = sessions[0].calciumdata.to_numpy()[np.ix_(idx_T,idx_areay_sub)]

X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
Y                   = zscore(Y,axis=0)

Y_hat_rr               = np.full((nTs,nM,nranks),np.nan)  

B_hat         = LM(Y,X, lam=params['lam'])

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
   #     #####  ######  #######  #####   #####     ####### ### #     # ####### 
  # #   #     # #     # #     # #     # #     #       #     #  ##   ## #       
 #   #  #       #     # #     # #       #             #     #  # # # # #       
#     # #       ######  #     #  #####   #####        #     #  #  #  # #####   
####### #       #   #   #     #       #       #       #     #  #     # #       
#     # #     # #    #  #     # #     # #     #       #     #  #     # #       
#     #  #####  #     # #######  #####   #####        #    ### #     # ####### 

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])
report_sessions(sessions)
# sessions = sessions[:10]

#%%  Load data properly:    
params['calciumversion'] = 'deconv'
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False,compute_respmat=True)

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
ax.set_xlim([-.5,2])
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('$R^2$')
sns.despine(fig=fig,top=True,right=True,trim=False,offset=3)
my_savefig(fig,figdir,'RRR_perf_across_time_deconv')

#%% Plotting, showing rank across time:
t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(1,1,figsize=(3.5*cm,3.5*cm))
ax = axes
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs):
    rank_toplot            = optim_rank[iapl].reshape(ntimebins,nSessions*nStim)
    handles.append(shaded_error(t_axis,rank_toplot.T,error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))

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
