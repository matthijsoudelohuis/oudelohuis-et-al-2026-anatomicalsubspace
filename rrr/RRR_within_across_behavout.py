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

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])
report_sessions(sessions)

#%%  Load data properly:    
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=True,compute_respmat=False)


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
my_savefig(fig,figdir,'RRR_R2Rank_WithinVSAcross_%dneurons_behavout' % nsampleneurons)

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
my_savefig(fig,figdir,'RRR_R2Rank_ratio_WithinVSAcross_%dneurons_behavout' % nsampleneurons)



