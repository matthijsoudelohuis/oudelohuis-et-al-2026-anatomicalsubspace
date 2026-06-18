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

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

only_all_areas = np.array(['V1','PM','AL'])

#%% Only one session
session_list        = np.array(['LPE12223_2024_06_10']) #GR
sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=only_all_areas,only_session_id=session_list)

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_noiselevel=False)
report_sessions(sessions)

#%% Load the data including behavior
[sessions,t_axis] = load_resid_tensor(sessions,params,load_behav=False)

#%% Parameters for decoding from size-matched populations of V1 and PM labeled and unlabeled neurons
arealabelpairs  = ['V1-PM','PM-V1']

clrs_arealabelpairs = get_clr_area_pairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

# nsampleneurons      = 100
# nranks_neural       = 25
nranks_neuralout    = 10
nmodelfits          = 15 #number of times new neurons are resampled for RRR
idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]

R2_cv               = np.full((narealabelpairs,nranks_neuralout,nSessions,params['nStim']),np.nan)
optim_rank          = np.full((narealabelpairs,nranks_neuralout,nSessions,params['nStim']),np.nan)

for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting RRR model'):
# for ises,ses in tqdm(enumerate(sessions[:2]),total=nSessions,desc='Fitting RRR model for within vs across populations'):
    idx_T               = np.ones(len(ses.trialdata['Orientation']),dtype=bool)
    for iapl, arealabelpair in enumerate(arealabelpairs):
        
        alx,aly = arealabelpair.split('-')

        idx_areax           = np.where(np.all((ses.celldata['roi_name']==alx,
                                ses.celldata['noise_level']<params['maxnoiselevel']
                                ),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==aly,
                                ses.celldata['noise_level']<params['maxnoiselevel']
                                ),axis=0))[0]
        idx_areaz           = np.where(np.all((ses.celldata['roi_name']=='AL',
                                ses.celldata['noise_level']<params['maxnoiselevel']
                                ),axis=0))[0]

        if len(idx_areax)<params['nsubnonlabeled'] or len(idx_areay)<params['nsubnonlabeled'] or len(idx_areaz)<params['nsubnonlabeled']:
            continue

        # B                   = np.concatenate((sessions[ises].tensor_vid,
        #                         sessions[ises].tensor_run),axis=0)

        # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over stimuli 
        for istim,stim in enumerate([0,4,7]): # loop over stimuli

            idx_areaz_sub = np.random.choice(idx_areaz,params['nsubnonlabeled'],replace=False)

            idx_T               = ses.trialdata['stimCond']==stim

            #on tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            Z                   = sessions[ises].tensor[np.ix_(idx_areaz_sub,idx_T,idx_resp)]
            
            # reshape to time points x neurons
            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T
            Z                   = Z.reshape(len(idx_areaz_sub),-1).T

            # #Get the behavioraldata in the same shape as the neural data:
            # Bstim                   = B[np.ix_(range(np.shape(B)[0]),idx_T,idx_resp)].reshape(np.shape(B)[0],-1).T
            # Bstim                   = zscore(Bstim,axis=0,nan_policy='omit')
            # Bstim                   = Bstim[:,~np.all(np.isnan(Bstim),axis=0)]
            
            #for each rank, regress out anything that can be behaviorally predicted
            for rankout in range(nranks_neuralout):
                if rankout>0:
                    X_orig,X_hat,X_out  = regress_out_cv(X=Z,Y=X,rank=rankout,lam=0,kfold=5)
                    Y_orig,Y_hat,Y_out  = regress_out_cv(X=Z,Y=Y,rank=rankout,lam=0,kfold=5)
                else:
                    X_out               = X
                    Y_out               = Y
                #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
                R2_cv[iapl,rankout,ises,istim],optim_rank[iapl,rankout,ises,istim],_  = RRR_wrapper(Y_out, X_out, nN=params['nsubnonlabeled'],nranks=params['nranks'],nmodelfits=nmodelfits)

#%% Plotting:
clr = clrs_arealabelpairs[0]
R2_toplot = np.reshape(R2_cv,(narealabelpairs,nranks_neuralout,nSessions*params['nStim']))
rank_toplot = np.reshape(optim_rank,(narealabelpairs,nranks_neuralout,nSessions*params['nStim']))

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))

clrs = get_clr_areas(['V1','PM'])
ax = axes
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs):
    handles.append(shaded_error(range(nranks_neuralout),R2_toplot[iapl].T,error='ci95',
                                color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
    ax_nticks(ax,3)
    for irank in range(nranks_neuralout-1):
        x = R2_toplot[iapl][irank]
        y = R2_toplot[iapl][irank+1]
        nas = np.logical_or(np.isnan(x), np.isnan(y))
        t,p = ttest_rel(x[~nas], y[~nas])
        p = p*nranks_neuralout #bonferonni correction
        print('Paired t-test: p=%.3f' % (p))
        ax.text((irank+1.5)/(nranks_neuralout),0.95-0.1*iapl,'%s' % get_sig_asterisks(p),rotation=45,
                transform=ax.transAxes,ha='center',va='center',fontsize=8,color=clrs_arealabelpairs[iapl]) #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

ax.set_ylabel('performance')
ax.legend(handles,['V1->PM','PM->V1'],frameon=False,loc='best',fontsize=6)
my_legend_strip(ax)
# ax.set_ylim([0,0.12])
ax.set_xticks(np.arange(0,nranks_neuralout))
ax.set_xticklabels(['orig']+['%d'%i for i in range(1,nranks_neuralout)])
ax.set_xlabel('rank neural out')
plt.tight_layout()
sns.despine(offset=3,top=True,right=True)
# my_savefig(fig,figdir,'RRR_Neuralout_AL_ranks_%dsessions' % nSessions)

#%% Quantify in percentage how much RRR performance was reduced due to neural variability that was shared: 
perfreduc = (R2_cv[:,-1,:]-R2_cv[:,0,:]) / R2_cv[:,0,:]
for iapl, arealabelpair in enumerate(arealabelpairs):
    print('%1.1f%% +- %1.1f%% reduction for %s' % (np.nanmean(perfreduc[iapl]*100),np.nanstd(perfreduc[iapl]*100), arealabelpairs[iapl]))
