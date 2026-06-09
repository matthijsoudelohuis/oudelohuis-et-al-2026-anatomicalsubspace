# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
DEBUG = False

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import zscore,wilcoxon,ttest_rel
from sklearn.linear_model import RidgeCV

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.regress_lib import *
from utils.params import load_params
from utils.RRRlib import *

params = load_params()
params['nmodelfits'] = 10 if DEBUG else 100

figdir = os.path.join(params['figdir'],'RRR','WithinAcross')

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])
if DEBUG:
    sessions = sessions[:2]
    nSessions = len(sessions)
report_sessions(sessions)
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False)

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

# nsampleneurons      = 50
nsampleneurons      = 100
nranks              = 25
# nmodelfits          = 15 #number of times new neurons are resampled 
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
            
            # reshape to time points x neurons
            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
            R2_cv[iapl,ises,istim],optim_rank[iapl,ises,istim],_  = RRR_wrapper(Y, X, nN=nsampleneurons,nranks=nranks,nmodelfits=params['nmodelfits'])

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

#%% Are the within area predictive dimensions the same as the across?
# in the Semedo et al. 2019 paper they asked how the V1 and V2 predictive dimensions are related?
# Are the predictive dimensions for these target populations aligned or do they capture distinct activity
# fluctuations within the source V1 population? They examined
# how the V1-V2 communication subspace relates to the
# structure of activity within the source V1 population. Is V2 activity
# predicted by the most dominant fluctuations within V1?

#%%
def semedo_uncorrelated_subspace(B_bar, X_train, r):
    """
    Project X onto the subspace uncorrelated with the first r predictive dimensions.
    Implements Semedo et al. 2019

    B_bar : (p, m)  predictive X-directions (columns from B_OLS @ V from RRR SVD)
    X_train : (n, p)  source activity (training fold only — defines covariance)
    r : int           number of predictive dimensions to remove

    Returns Q : (p, p-r)  orthonormal basis of the uncorrelated subspace
    """
    Sigma = X_train.T @ X_train          # (p, p)  unnormalized covariance
    M = B_bar[:, :r].T @ Sigma           # (r, p)
    _, _, Vt = np.linalg.svd(M, full_matrices=True)  # Vt : (p, p)
    Q = Vt[r:, :].T                      # (p, p-r)  null space of M
    return Q


#%% 

sourceareas         = ['V1','PM']
targetareas         = ['V1','PM']
nsourceareas        = len(sourceareas)
ntargetareas        = len(targetareas)

rank_subspace       = 5
nrankstoremove      = 5
# nsampleneurons      = 50
nsampleneurons      = 50
nStim               = 16
idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
params['nmodelfits'] = 10

kf          = KFold(n_splits=params['kfold'],shuffle=True)

# shape: (nsourceareas, nrankstoremove, 4, nSessions, nStim, nmodelfits, kfold)
# condition axis (dim 2):
# cond 0: predict within, remove within dims
# cond 1: predict within, remove across dims
# cond 2: predict across,  remove within dims
# cond 3: predict across,  remove across dims

R2_removal = np.full((nsourceareas, nrankstoremove+1, 4, nSessions, nStim, params['nmodelfits'], params['kfold']), np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for within vs across populations'):
# for ises,ses in tqdm(enumerate(sessions[:2]),total=nSessions,desc='Fitting RRR model for within vs across populations'):
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    if np.sum((ses.celldata['roi_name']=='V1') & (ses.celldata['noise_level']<params['maxnoiselevel']))<(nsampleneurons*2):
        continue
    
    if np.sum((ses.celldata['roi_name']=='PM') & (ses.celldata['noise_level']<params['maxnoiselevel']))<(nsampleneurons*2):
        continue

    for isourcearea,sourcearea in enumerate(sourceareas):
        
        withinarea  = sourcearea                                  # within = same area as source
        acrossarea  = 'PM' if sourcearea == 'V1' else 'V1'       # across = the other area
        print(f'Source: {sourcearea}, within: {withinarea}, across: {acrossarea}')
        idx_areax   = np.where(np.all((ses.celldata['roi_name']==sourcearea,
                                ses.celldata['noise_level']<params['maxnoiselevel']
                                ),axis=0))[0]
        idx_areay   = np.where(np.all((ses.celldata['roi_name']==withinarea,  # Y = within target
                                ses.celldata['noise_level']<params['maxnoiselevel']
                                ),axis=0))[0]
        idx_areaz   = np.where(np.all((ses.celldata['roi_name']==acrossarea,  # Z = across target
                                ses.celldata['noise_level']<params['maxnoiselevel']
                                ),axis=0))[0]

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
        # for istim,stim in enumerate([0,4,7]): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
            for imf in range(params['nmodelfits']):
                idx_areax_split, idx_areay_split = np.array_split(np.random.permutation(idx_areax), 2)

                idx_areax_sub = np.random.choice(idx_areax_split,nsampleneurons,replace=False)
                idx_areay_sub = np.random.choice(idx_areay_split,nsampleneurons,replace=False)
                idx_areaz_sub = np.random.choice(idx_areaz,nsampleneurons,replace=False)

                #on tensor during the response:
                X            = sessions[ises].tensor[np.ix_(idx_areax_sub,idx_T,idx_resp)]
                Y            = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]
                Z            = sessions[ises].tensor[np.ix_(idx_areaz_sub,idx_T,idx_resp)]
                
                # reshape to time points x neurons
                X           = zscore(X.reshape(len(idx_areax_sub), -1).T, axis=0)   # (n, p)
                Y           = zscore(Y.reshape(len(idx_areay_sub), -1).T, axis=0)   # (n, q)
                Z           = zscore(Z.reshape(len(idx_areaz_sub), -1).T, axis=0)   # (n, q)

                ridge_alphas = np.logspace(-3, 3, 13)

                # ── Step 0: RRR — identify predictive dimensions from training data ─
                # OLS B_OLS then SVD of Y_hat gives the top predictive output modes V_m
                # in Y-space.  B_bar = B_OLS @ V_m are the predictive directions in X-space.
                B_within = LM(Y, X, lam=0)   # (p, q)
                B_across = LM(Z, X, lam=0)   # (p, q)

                _, _, Vt_w = np.linalg.svd(X @ B_within, full_matrices=False)  # (q, q)
                _, _, Vt_a = np.linalg.svd(X @ B_across, full_matrices=False)

                B_bar_within = B_within @ Vt_w[:rank_subspace].T   # (p, rank_subspace)
                B_bar_across = B_across @ Vt_a[:rank_subspace].T

                for r in range(nrankstoremove+1):
                    # ── Step 2: Semedo uncorrelated subspace ────────────────────────
                    Q_w = semedo_uncorrelated_subspace(B_bar_within, X, r)  # (p, p-r)
                    Q_a = semedo_uncorrelated_subspace(B_bar_across, X, r)  # (p, p-r)

                    if r == 0:
                        Q_w = np.eye(X.shape[1])
                        Q_a = np.eye(X.shape[1])

                    # ── Step 3: RidgeCV on X̂ = X @ Q, evaluate on held-out fold ────
                    for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                        X_tr, X_te = X[idx_train], X[idx_test]
                        Y_tr, Y_te = Y[idx_train], Y[idx_test]
                        Z_tr, Z_te = Z[idx_train], Z[idx_test]

                        for icond, (Q, Ytar_tr, Ytar_te) in enumerate([
                            (Q_w, Y_tr, Y_te),   # cond 0: predict within, remove within dims
                            (Q_a, Y_tr, Y_te),   # cond 1: predict within, remove across dims
                            (Q_w, Z_tr, Z_te),   # cond 2: predict across,  remove within dims
                            (Q_a, Z_tr, Z_te),   # cond 3: predict across,  remove across dims
                        ]):
                            # print(Q)
                            clf = RidgeCV(alphas=ridge_alphas, fit_intercept=False).fit(X_tr @ Q, Ytar_tr)
                            #score
                            R2_score = r2_score(Ytar_te, clf.predict(X_te @ Q))
                            # print("R2 Ridge model with rank {} is {}".format(r, R2_score))
                            R2_removal[isourcearea, r, icond, ises, istim, imf, ikf] = R2_score

#%% Plot like Semedo et al. 2019 Figure 6 (2x2: FF rows / within+across cols)
R2_toplot = np.nanmean(R2_removal, axis=(5, 6)) # Average over modelfits, kfolds:
R2_toplot = np.reshape(R2_toplot, (nsourceareas, nrankstoremove+1, 4, nSessions*nStim)) #reshape to datasets

normalize = True
# Normalize to performance without any removal
if normalize:
    R2_norm = R2_toplot / R2_toplot[:,0][:,np.newaxis]
else:
    R2_norm = R2_toplot

# R2_norm = R2_toplot / R2_toplot[:,0][:,np.newaxis]
# R2_norm = R2_toplot
# R2_norm shape: (nsrc, nranks, 4, nSessions)

dims_removed = np.arange(nrankstoremove+1)
subplot_cfgs = [
    (0, 3, 2, 'Source: V1\nPredict PM',  'PM dims',  'V1 dims'),   # FF, across target
    (0, 0, 1, 'Source: V1\nPredict V1',  'V1 dims',  'PM dims'),   # FF, within target
    (1, 3, 2, 'Source: PM\nPredict V1',  'V1 dims',  'PM dims'),   # FB, across target
    (1, 0, 1, 'Source: PM\nPredict PM',  'PM dims',  'V1 dims'),   # FB, within target
]

fig, axes = plt.subplots(2, 2, figsize=(10*cm, 9*cm), sharex=True, sharey=True)

for ax, (isrc, i_self, i_other, title, lbl_self, lbl_other) in zip(axes.flatten(), subplot_cfgs):

    data_self  = R2_norm[isrc, :, i_self,  :]   # (nranks, nSessions)
    data_other = R2_norm[isrc, :, i_other, :]

    # Mean and SEM across sessions
    n_ses = np.sum(~np.isnan(data_self[0]))
    mu_self   = np.nanmean(data_self,  axis=1)
    sem_self  = np.nanstd(data_self,   axis=1) / np.sqrt(n_ses)
    mu_other  = np.nanmean(data_other, axis=1)
    sem_other = np.nanstd(data_other,  axis=1) / np.sqrt(n_ses)

    # Self dims (filled circles, solid) — performance should drop
    ax.fill_between(dims_removed, mu_self - sem_self, mu_self + sem_self, alpha=0.25, color='k')
    ax.plot(dims_removed, mu_self, 'o-', color='k', markersize=3.5, linewidth=1.2, label=lbl_self)

    # Other dims (open circles, dashed) — performance should stay relatively high
    ax.fill_between(dims_removed, mu_other - sem_other, mu_other + sem_other, alpha=0.15, color='grey')
    ax.plot(dims_removed, mu_other, 'o--', color='grey', markersize=3.5, linewidth=1.2,
            markerfacecolor='white', markeredgecolor='grey', label=lbl_other)

    ax.axhline(y=0, color='grey', linestyle=':', linewidth=0.7)
    ax.set_title(title, fontsize=7)
    ax.set_xticks(dims_removed)

axes[0, 0].legend(frameon=False, fontsize=6, loc='upper right')
axes[1, 0].legend(frameon=False, fontsize=6, loc='upper right')

for ax in axes[1]:
    ax.set_xlabel('Predictive dims removed')
for ax in axes[:, 0]:
    ax.set_ylabel('Norm. performance')

# axes[0, 0].set_ylim([-0.15, 1.1])
axes[0, 0].set_xlim([-0.3, nrankstoremove + 0.3])
ax_nticks(axes[0, 0], 3)
# ax.set_yticks([0,0.5,1])
ax.set_xticks([0,1,3,5])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=2)
my_savefig(fig, figdir, 'RRR_Remove_Pred_Subspace_%dneurons_%dranks' % (nsampleneurons, nrankstoremove))

#%% Like Semedo Fig 6B/D: histograms of norm. performance at max dims removed (2x2)
R2_toplot = np.nanmean(R2_removal, axis=(5, 6)) # Average over modelfits, kfolds:
R2_toplot = np.reshape(R2_toplot, (nsourceareas, nrankstoremove+1, 4, nSessions*nStim)) #reshape to datasets

# Normalize to performance without any removal
R2_norm = R2_toplot / R2_toplot[:,0][:,np.newaxis]

r_plot  = nrankstoremove - 1   # index of last rank (e.g. 4 dims removed when nrankstoremove=5)
# r_plot  = nrankstoremove   # index of last rank (e.g. 4 dims removed when nrankstoremove=5)
bins    = np.arange(-0.2, 1, 0.05)

fig, axes = plt.subplots(2, 2, figsize=(10*cm, 9*cm), sharex=True, sharey=True)

for ax, (isrc, i_self, i_other, title, lbl_self, lbl_other) in zip(axes.flatten(), subplot_cfgs):

    data_self  = R2_norm[isrc, r_plot, i_self,  :]   # (nSessions,) — self dims removed
    data_other = R2_norm[isrc, r_plot, i_other, :]   # (nSessions,) — other dims removed

    # Filled bars: self dims removed → performance should collapse to ~0
    ax.hist(data_self,  bins=bins, color='k',    alpha=0.7,  label=lbl_self)
    # Outline bars: other dims removed → performance should stay relatively high
    ax.hist(data_other, bins=bins, color='grey', histtype='step', linewidth=1.2, label=lbl_other)

    # Downward triangles at the mean, placed at top of y-axis
    ymax = ax.get_ylim()[1]/0.9
    ax.plot(np.nanmean(data_self),  ymax, 'v', color='k',    markersize=5, clip_on=False)
    ax.plot(np.nanmean(data_other), ymax, 'v', color='grey', markersize=5, clip_on=False,
            markerfacecolor='white', markeredgecolor='grey')

    ax.axvline(x=0, color='grey', linestyle=':', linewidth=0.7)
    ax.set_title(title, fontsize=7)

axes[0, 0].legend(frameon=False, fontsize=6, loc='upper right')
axes[1, 0].legend(frameon=False, fontsize=6, loc='upper right')

for ax in axes[1]:
    ax.set_xlabel('Norm. performance')
for ax in axes[:, 0]:
    ax.set_ylabel('Sessions')

axes[0, 0].set_xlim([-0.3, 1])
axes[0, 0].set_xticks([0,0.25, 0.5, 0.75,1])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3)
my_savefig(fig, figdir, 'RRR_Remove_Pred_Subspace_hist_%dneurons_%ddims' % (nsampleneurons, r_plot + 1))

#%% Compute normalized R2 when removing predictive and non-predictive dimensions and print results:
# condition axis (dim 2):
# cond 0: predict within, remove within dims
# cond 1: predict within, remove across dims
# cond 2: predict across,  remove within dims
# cond 3: predict across,  remove across dims

data_self  = R2_norm[:, r_plot, [0,3],  :]   # — self-predictive dims removed
print('Normalized R2 when removing predictive dimensions: %.2f +/- %.2f' % (np.nanmean(data_self), np.nanstd(data_self)))
data_other = R2_norm[:, r_plot, [1,2], :]   # — other-predictive dims removed
print('Normalized R2 when removing non-predictive dimensions: %.2f +/- %.2f' % (np.nanmean(data_other), np.nanstd(data_other)))

# data_self  = R2_norm[:, r_plot, [0],  :]   # — self-predictive dims removed
# print('Normalized R2 when removing predictive dimensions: %.2f +/- %.2f' % (np.nanmean(data_self), np.nanstd(data_self)))
# data_other = R2_norm[:, r_plot, [2], :]   # — other-predictive dims removed
# print('Normalized R2 when removing non-predictive dimensions: %.2f +/- %.2f' % (np.nanmean(data_other), np.nanstd(data_other)))



#%% 
######  ####### ######  ######  #######  #####     #    ####### ####### ######  
#     # #       #     # #     # #       #     #   # #      #    #       #     # 
#     # #       #     # #     # #       #        #   #     #    #       #     # 
#     # #####   ######  ######  #####   #       #     #    #    #####   #     # 
#     # #       #       #   #   #       #       #######    #    #       #     # 
#     # #       #       #    #  #       #     # #     #    #    #       #     # 
######  ####### #       #     # #######  #####  #     #    #    ####### ######  


# #%%  Within to across area dimensionality comparison: 

# figdir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Interarea\\RRR\\WithinAcross\\')

# #%% 
# min_cells = 1500
# sessions,nSessions   = filter_sessions(protocols = 'GR',min_cells=min_cells,filter_areas=['V1','PM'])

# #%% 
# print('Number of cells in V1 and in PM:')
# for ises,ses in enumerate(sessions):
#     print('Session %d: %d cells in V1, %d cells in PM' % (ises,np.sum(ses.celldata['roi_name']=='V1'),np.sum(ses.celldata['roi_name']=='PM')))

# #%%  Load data properly:        
# calciumversion = 'dF'
# # calciumversion = 'deconv'

# for ises in range(nSessions):
#     sessions[ises].load_respmat(load_behaviordata=False, load_calciumdata=True,load_videodata=False,
#                                 calciumversion=calciumversion,keepraw=False)

# #%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
# arealabelpairs  = ['V1-V1',
#                     'PM-PM',
#                     'V1-PM',
#                     'PM-V1'
#                     ]
# alp_withinacross = ['Within',
#                    'Within',
#                    'Across',
#                    'Across'] 

# clrs_arealabelpairs = get_clr_area_pairs(arealabelpairs)
# narealabelpairs     = len(arealabelpairs)

# lam                 = 0
# nsampleneurons      = 100
# nranks              = 50
# nmodelfits          = 10 #number of times new neurons are resampled 
# kfold               = 5

# R2_cv               = np.full((narealabelpairs,nSessions),np.nan)
# optim_rank          = np.full((narealabelpairs,nSessions),np.nan)

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model for different population sizes'):
#     idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
#     if np.sum((ses.celldata['roi_name']=='V1') & (ses.celldata['noise_level']<params['maxnoiselevel']))<(nsampleneurons*2):
#         continue
    
#     if np.sum((ses.celldata['roi_name']=='PM') & (ses.celldata['noise_level']<params['maxnoiselevel']))<(nsampleneurons*2):
#         continue

#     for iapl, arealabelpair in enumerate(arealabelpairs):
        
#         alx,aly = arealabelpair.split('-')

#         idx_areax           = np.where(np.all((ses.celldata['roi_name']==alx,
#                                 ses.celldata['noise_level']<params['maxnoiselevel']
#                                 ),axis=0))[0]
#         idx_areay           = np.where(np.all((ses.celldata['roi_name']==aly,
#                                 ses.celldata['noise_level']<params['maxnoiselevel']
#                                 ),axis=0))[0]
    
#         if np.any(np.intersect1d(idx_areax,idx_areay)): #if interactions within one population:
#             if not np.array_equal(idx_areax, idx_areay): 
#                 print('Arealabelpair %s has partly overlapping neurons'%arealabelpair)
#             idx_areax, idx_areay = np.array_split(np.random.permutation(idx_areax), 2)

#         X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
#         Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T

#         R2_cv[iapl,ises],optim_rank[iapl,ises],_  = RRR_wrapper(Y, X, nN=nsampleneurons,nK=None,lam=0,nranks=nranks,kfold=kfold,nmodelfits=nmodelfits)

# #%% Plotting:
# clr = clrs_arealabelpairs[0]

# fig,axes = plt.subplots(1,2,figsize=(4.5,2.5))

# clrs = get_clr_areas(['V1','PM'])
# ax = axes[0]
# comps = [[0,3],[1,2]]
# for icomp,comp in enumerate(comps):
#     ax.scatter(R2_cv[comp[0],:],R2_cv[comp[1],:],s=120,edgecolor='w',marker='.',color=clrs[icomp])
#     # ax.scatter(R2_cv[comp[0],:],R2_cv[comp[1],:],s=80,alpha=0.8,marker='.',color=clrs[icomp])
#     ax_nticks(ax,3)
#     ax.set_xlabel('Within')
#     ax.set_ylabel('Across')
#     ax.set_title('R2')

#     _,pval = ttest_rel(R2_cv[comp[0],:],R2_cv[comp[1],:],nan_policy='omit')
#     ax.text(0.6,0.1*(1+icomp),'%sp=%.3f' % (get_sig_asterisks(pval),pval),transform=ax.transAxes,fontsize=10,color=clrs[icomp])
# ax.legend(['V1','PM'],frameon=False,fontsize=10)
# my_legend_strip(ax)
# ax.set_xlim([0,0.4])
# ax.set_ylim([0,0.4])
# ax.plot([0,0.4],[0,0.4],':',color='grey',linewidth=1)

# ax = axes[1]
# for icomp,comp in enumerate(comps):
#     ax.scatter(optim_rank[comp[0],:],optim_rank[comp[1],:],s=120,edgecolor='w',marker='.',color=clrs[icomp])
#     ax_nticks(ax,3)
#     ax.set_xlabel('Within')
#     ax.set_title('Rank')
#     _,pval = ttest_rel(optim_rank[comp[0],:],optim_rank[comp[1],:],nan_policy='omit')
#     ax.text(0.6,0.1*(1+icomp),'%sp=%.3f' % (get_sig_asterisks(pval),pval),transform=ax.transAxes,fontsize=10,color=clrs[icomp])
# ax.legend(['V1','PM'],frameon=False,fontsize=10)
# my_legend_strip(ax)
# ax.set_xlim([0,25])
# ax.set_ylim([0,25])
# ax.plot([0,25],[0,25],':',color='grey',linewidth=1)
# # ax.legend(['V1 (V1->V1 vs PM->V1','PM (PM->PM vs V1->PM'],frameon=False,fontsize=8)

# sns.despine(offset=3,top=True,right=True)
# plt.tight_layout()

# my_savefig(fig,figdir,'RRR_R2Rank_WithinVSAcross_%dneurons' % nsampleneurons,formats=['png'])

