# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import  os
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore, wilcoxon
import pickle
from datetime import datetime

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.params import load_params
from utils.tuning import compute_tuning_wrapper

#%% Load parameters and settings:
params = load_params()

# params['regress_behavout'] = True
params['regress_behavout'] = False
params['direction'] = 'FF'
params['direction'] = 'FB'
# params['direction'] = 'FF_AL'
# params['direction'] = 'FB_AL'

version = 'Joint_labeled_%s_%s' % (params['direction'],'behavout' if params['regress_behavout'] else 'original')

resultdir = os.path.join(params['resultdir'])
if not os.path.exists(resultdir):
    os.makedirs(resultdir)
datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savefilename = os.path.join(resultdir,'RRR_%s_%s' % (version,datetime_str))

#%% 
session_list        = np.array([
                                # ['LPE12223_2024_06_10'], #V1lab actually lower
                                ['LPE09830_2023_04_10'], #V1 labeled higher predictive than V1unl
                                # ['LPE10919_2023_11_06'],  #V1lab actually lower
                                # ['LPE12223_2024_06_08'], #V1lab actually lower
                                # ['LPE11998_2024_05_02'], # V1lab lower?
                                # ['LPE11622_2024_03_25'], #same
                                ['LPE09665_2023_03_14'], #V1lab higher
                                ['LPE10885_2023_10_23'], #V1lab much higher
                                ['LPE11086_2024_01_05'], #Really much higher, best session, first dimensions are more predictive.
                                # ['LPE11086_2024_01_10'], #Few v1 labeled cells, very noisy
                                # ['LPE11998_2024_05_10'], #
                                # ['LPE12013_2024_05_07'], #
                                # ['LPE11495_2024_02_28'], #
                                # ['LPE11086_2023_12_15'], #Same
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,
                                       min_lab_cells_V1=20,filter_noiselevel=False)

#%% Get all data 
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_noiselevel=False)
report_sessions(sessions)

#%% Wrapper function to load the tensor data
# params['calciumversion'] = 'deconv'
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=params['regress_behavout'],compute_respmat=True)

#%% Compute tuning metrics:
sessions = compute_tuning_wrapper(sessions)

#%% Do RRR of V1 and PM labeled and unlabeled neurons simultaneously
if params['direction'] =='FF': 
    sourcearealabelpairs = ['V1unl','V1unl','V1unl','V1lab']
    targetarealabelpair = 'PMunl'
    only_all_areas = np.array(['V1','PM'])
elif params['direction'] =='FB': 
    sourcearealabelpairs = ['PMunl','PMunl','PMunl','PMlab']
    targetarealabelpair = 'V1unl'
    only_all_areas = np.array(['V1','PM'])


#%% 
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 20
# nmodelfits          = 100
nmodelfits          = 50
nranks              = 20

params['nStim']     = 16

# idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
idx_resp            = np.where((t_axis>=-99) & (t_axis<=2))[0]

# Per-neuron alignment with shared and private subspaces for X3 (held-out unlabeled) and X4 (labeled)
alpha_unl           = np.full((Nsub,nSessions,params['nStim'],nmodelfits),np.nan) # shared alignment, X3
beta_unl            = np.full((Nsub,nSessions,params['nStim'],nmodelfits),np.nan) # private alignment, X3
alpha_lab           = np.full((Nsub,nSessions,params['nStim'],nmodelfits),np.nan) # shared alignment, X4
beta_lab            = np.full((Nsub,nSessions,params['nStim'],nmodelfits),np.nan) # private alignment, X4

np.random.seed(99) # for reproducibility of random subsampling
fixed_rank = None
fixed_rank = True
fixed_rank = 5

def _align(Q, X):
    """Fraction of each neuron's variance (columns of X) aligned with subspace Q.
    Q : (T, k) orthonormal columns
    X : (T, N) z-scored neuron activity
    Returns (N,) values in [0, 1]
    """
    proj = Q.T @ X                                    # (k, N)
    return np.sum(proj**2, axis=0) / np.sum(X**2, axis=0)

for ises,ses in enumerate(sessions):
    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    idx_areax1      = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[0],
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                # ses.celldata['tuning_var']>params['mintuningvar'],
                                idx_nearby
                                ),axis=0))[0]
    idx_areax2      = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[1],
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                # ses.celldata['tuning_var']>params['mintuningvar'],
                                idx_nearby
                                ),axis=0))[0]
    idx_areax3      = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[2],
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                # ses.celldata['tuning_var']>params['mintuningvar'],
                                idx_nearby
                                ),axis=0))[0]
    idx_areax4      = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[3],
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                # ses.celldata['tuning_var']>params['mintuningvar'],
                                idx_nearby
                                ),axis=0))[0]
    idx_areay       = np.where(np.all((ses.celldata['arealabel']==targetarealabelpair,
                                # ses.celldata['noise_level']<params['maxnoiselevel'],
                                # ses.celldata['tuning_var']>params['mintuningvar'],
                                idx_nearby
                                ),axis=0))[0]
    
    if len(idx_areax1)<Nsub*3 or len(idx_areax4)<Nsub or len(idx_areay)<Nsub: #skip exec if not enough neurons in one of the populations
        print('%d in %s, %d in %s, %d in %s' % (len(idx_areax4),sourcearealabelpairs[3],
                                                len(idx_areax1),sourcearealabelpairs[0],
                                                len(idx_areay),targetarealabelpair))
        continue

#     for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
    for istim,stim in tqdm(enumerate(np.unique(ses.trialdata['stimCond'])),total=params['nStim'],desc='Processing session %d/%d' % (ises+1,nSessions)):
        idx_T               = ses.trialdata['stimCond']==stim

        if fixed_rank:
            # rank_private = 4
            # rank_shared =  3
            rank_private = fixed_rank
            rank_shared =  fixed_rank
        else:
            X1                  = sessions[ises].tensor[np.ix_(idx_areax1,idx_T,idx_resp)]
            X2                  = sessions[ises].tensor[np.ix_(idx_areax2,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]

            # reshape to neurons x time points
            X1                  = X1.reshape(len(idx_areax1),-1).T
            X2                  = X2.reshape(len(idx_areax2),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T
            
            _,rank_private,_ = RRR_wrapper(X2,X1,nN=Nsub,lam=params['lam'],nranks=nranks,kfold=params['kfold'],
                                               nmodelfits=nmodelfits,fixed_rank=None)
            
            _,rank_shared,_ = RRR_wrapper(Y,X1,nN=Nsub,lam=params['lam'],nranks=nranks,kfold=params['kfold'],
                                               nmodelfits=nmodelfits,fixed_rank=None)
        
        # if rank_private <= rank_shared:
        # if rank_private >0:
            # print('Skipping session %d, stim %d, because private rank (%d) is smaller than shared rank (%d)' % (ises,istim,rank_private,rank_shared))
            # continue

        for imf in range(nmodelfits):
            idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
            idx_areax2_sub       = np.random.choice(np.setdiff1d(idx_areax2,idx_areax1_sub),Nsub,replace=False)
            idx_areax3_sub       = np.random.choice(np.setdiff1d(idx_areax3,[idx_areax1_sub,idx_areax2_sub]),Nsub,replace=False) # held-out unlabeled neurons for alignment
            idx_areax4_sub       = np.random.choice(idx_areax4,Nsub,replace=False)
            idx_areay_sub        = np.random.choice(idx_areay,Nsub,replace=False)
       
            X1                  = sessions[ises].tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
            X2                  = sessions[ises].tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]
            X3                  = sessions[ises].tensor[np.ix_(idx_areax3_sub,idx_T,idx_resp)]
            X4                  = sessions[ises].tensor[np.ix_(idx_areax4_sub,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay_sub,idx_T,idx_resp)]

            # reshape to neurons x time points
            X1                  = X1.reshape(len(idx_areax1_sub),-1).T
            X2                  = X2.reshape(len(idx_areax2_sub),-1).T
            X3                  = X3.reshape(len(idx_areax3_sub),-1).T
            X4                  = X4.reshape(len(idx_areax4_sub),-1).T
            Y                   = Y.reshape(len(idx_areay_sub),-1).T

            X1                  = zscore(X1,axis=0) #zscore the activity per neuron
            X2                  = zscore(X2,axis=0)
            X3                  = zscore(X3,axis=0)
            X4                  = zscore(X4,axis=0)
            Y                   = zscore(Y,axis=0)

            #RRR X1 to Y (Interareal RRR) — defines shared subspace
            B_shared            = LM(Y,X1, lam=params['lam'])
            Y_hat               = X1 @ B_shared

            #RRR X1 to X2 (intraareal RRR) — defines private subspace
            B_private           = LM(X2,X1, lam=params['lam'])
            X2_hat              = X1 @ B_private

            # ---- Shared subspace: left singular vectors of Y_hat ----
            # Y_hat = X1 @ B_shared has shape (T, Nsub); SVD gives T-space directions.
            # U columns are orthonormal by construction — no QR needed.
            U_share, _, _       = np.linalg.svd(Y_hat, full_matrices=False)
            Q_share             = U_share[:, :rank_shared]          # (T, rank_shared)

            # ---- Private subspace: left singular vectors of X2_hat, orthogonalized to shared ----
            U_priv, _, _        = np.linalg.svd(X2_hat, full_matrices=False)
            Z_priv_raw          = U_priv[:, :rank_private]          # (T, rank_private)
            Z_priv_orth         = Z_priv_raw - Q_share @ (Q_share.T @ Z_priv_raw)
            Q_priv, _           = np.linalg.qr(Z_priv_orth)        # (T, rank_private), orthonormal
            Q_priv              = Q_priv[:, :rank_private]

            # ---- Per-neuron alignment: fraction of variance in each subspace ----
            alpha_unl[:, ises, istim, imf] = _align(Q_share, X3)
            beta_unl[:,  ises, istim, imf] = _align(Q_priv,  X3)
            alpha_lab[:, ises, istim, imf] = _align(Q_share, X4)
            beta_lab[:,  ises, istim, imf] = _align(Q_priv,  X4)

#%% Selectivity index: fraction of explained variance (shared+private) that is shared
# s_n = alpha_n / (alpha_n + beta_n), per neuron per model fit
# Average across model fits and stimulus conditions to get one value per neuron per session

eps = 1e-10  # prevent division by zero for neurons fully off both subspaces
s_unl = alpha_unl / (alpha_unl + beta_unl + eps)   # (Nsub, nSessions, nStim, nmodelfits)
s_lab = alpha_lab / (alpha_lab + beta_lab + eps)    # (Nsub, nSessions, nStim, nmodelfits)

# Session-level summaries: mean across neurons, stimuli, and model fits
s_unl_ses = np.nanmean(s_unl, axis=(0, 2, 3))      # (nSessions,)
s_lab_ses = np.nanmean(s_lab, axis=(0, 2, 3))      # (nSessions,)

alpha_unl_ses = np.nanmean(alpha_unl, axis=(0, 2, 3))
alpha_lab_ses = np.nanmean(alpha_lab, axis=(0, 2, 3))
beta_unl_ses  = np.nanmean(beta_unl,  axis=(0, 2, 3))
beta_lab_ses  = np.nanmean(beta_lab,  axis=(0, 2, 3))

#%% Plot: session-level scatter (labeled vs unlabeled), one dot per session
figdir = os.path.join(params['figdir'],'RRR','PrivateShared')
cm = 1/2.54
set_plot_basic_config()

#%%
metrics   = [alpha_unl_ses, beta_unl_ses,  s_unl_ses]
metrics_l = [alpha_lab_ses, beta_lab_ses,  s_lab_ses]
labels    = [r'$\alpha$ (shared)', r'$\beta$ (private)', r'$s$ (selectivity)']

if params['direction'] == 'FB':
    figlabels = ['PM$_{ND}$','PM$_{V1}$']
elif params['direction'] == 'FF':
    figlabels = ['V1$_{ND}$','V1$_{PM}$']

fig, axes = plt.subplots(1, 3, figsize=(8*cm, 5*cm))
for ax, xu, xl, lbl, i in zip(axes, metrics, metrics_l, labels, range(len(labels))):
    # vmin = np.nanmin(np.concatenate([xu, xl])) * 0.9
    vmax = np.nanmax(np.concatenate([xu, xl])) * 1.1
    sns.stripplot(ax=ax, data=np.column_stack([xu, xl]), color='k', 
                  size=4)
    ax.plot(np.column_stack((np.zeros(len(xu)), np.ones(len(xu)))).T,
            np.column_stack([xu, xl]).T, color='k', lw=0.4)
    # ax.scatter(xu, xl, color='k', s=30, zorder=3)
    # ax.plot([0, vmax], [0, vmax], ':', color='grey', lw=1)
    ax.set_xticks([0, 1], labels=figlabels)
    ax.set_ylabel('')
    ax.set_title(lbl)
    # ax.set_xlim([0, vmax])
    if i < 2:
        ax.set_ylim([0, 0.25])
    if i == 2:
        ax.set_ylim([0.65, 0.9])
        # ax.axhline(0.5, color='grey', lw=1, ls='--')
    add_paired_wilcoxon_results(ax, xu, xl,pos=[0.5,0.95], fontsize=8)
    ax_nticks(ax,4)
sns.despine(offset=2, top=True, right=True)
plt.tight_layout()

if np.sum(~np.isnan(s_lab_ses - s_unl_ses)) >= 5: # only do stats if at least 5 sessions have valid data
    stat, p_sel = wilcoxon(s_lab_ses[~np.isnan(s_lab_ses)], s_unl_ses[~np.isnan(s_unl_ses)])
    stat, p_alpha = wilcoxon(alpha_lab_ses[~np.isnan(alpha_lab_ses)], alpha_unl_ses[~np.isnan(alpha_unl_ses)])
    stat, p_beta  = wilcoxon(beta_lab_ses[~np.isnan(beta_lab_ses)],   beta_unl_ses[~np.isnan(beta_unl_ses)])
    print('Selectivity index:  labeled=%.3f  unlabeled=%.3f  p=%s' % (np.nanmean(s_lab_ses), np.nanmean(s_unl_ses), p_sel))
    print('Alpha (shared):     labeled=%.3f  unlabeled=%.3f  p=%s' % (np.nanmean(alpha_lab_ses), np.nanmean(alpha_unl_ses), p_alpha))
    print('Beta  (private):    labeled=%.3f  unlabeled=%.3f  p=%s' % (np.nanmean(beta_lab_ses), np.nanmean(beta_unl_ses), p_beta))
# my_savefig(fig, figdir, 'PrivateShared_%s_%dsessions' % (version, nSessions))

#%% 
fig, axes = plt.subplots(1, 3, figsize=(18*cm, 6*cm))
for ax, xu, xl, lbl in zip(axes, metrics, metrics_l, labels):
    vmin = np.nanmin(np.concatenate([xu, xl])) * 0.9
    vmax = np.nanmax(np.concatenate([xu, xl])) * 1.1
    ax.scatter(xu, xl, color='k', s=30, zorder=3)
    ax.plot([0, vmax], [0, vmax], ':', color='grey', lw=1)
    ax.set_xlabel('Unlabeled (X3)')
    ax.set_ylabel('Labeled (X4)')
    ax.set_title(lbl)
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])
    ax_nticks(ax,4)
sns.despine(offset=2, top=True, right=True)
plt.tight_layout()
# my_savefig(fig, figdir, 'PrivateShared_alpha_beta_selectivity_%dsessions' % nSessions)

# #%%
# params['Nsub']     = Nsub
# params['nranks']    = nranks
# params['nmodelfits'] = nmodelfits
# params['nSessions'] = nSessions

# #%% Save the data:
# np.savez(savefilename + '.npz',
#          alpha_unl=alpha_unl,beta_unl=beta_unl,
#          alpha_lab=alpha_lab,beta_lab=beta_lab,
#          s_unl=s_unl,s_lab=s_lab,
#          sourcearealabelpairs=sourcearealabelpairs,
#          targetarealabelpair=targetarealabelpair,
#          allow_pickle=True)

# with open(savefilename +'_params' + '.txt', "wb") as myFile:
#     pickle.dump(params, myFile)

#%%