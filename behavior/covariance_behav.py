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
######  ####### #     #    #    #     # ####### #     # ####### 
#     # #       #     #   # #   #     # #     # #     #    #    
#     # #       #     #  #   #  #     # #     # #     #    #    
######  #####   ####### #     # #     # #     # #     #    #    
#     # #       #     # #######  #   #  #     # #     #    #    
#     # #       #     # #     #   # #   #     # #     #    #    
######  ####### #     # #     #    #    #######  #####     #    

#%% Identify sessions to load
areas = ['V1','PM','AL']
# sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],only_all_areas=areas,filter_areas=areas,
#                                        min_lab_cells_V1=40,min_lab_cells_PM=40)

sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],filter_areas=areas)
report_sessions(sessions)

#%% Load the data including behavior
[sessions,t_axis] = load_resid_tensor(sessions,params,load_behav=True)

#%% Parameters for RRR size-matched populations of V1 and PM neurons
arealabelpairs  = ['V1-PM','PM-V1']

clrs_arealabelpairs = get_clr_area_pairs(arealabelpairs)
narealabelpairs     = len(arealabelpairs)

# nsampleneurons      = 100
# nranks_neural       = 25
nranks_behavout     = 5
nmodelfits          = 15 #number of times new neurons are resampled for RRR
idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]

R2_cv               = np.full((narealabelpairs,nranks_behavout,nSessions,params['nStim']),np.nan)
optim_rank          = np.full((narealabelpairs,nranks_behavout,nSessions,params['nStim']),np.nan)

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
        if len(idx_areax)<params['nsubnonlabeled'] or len(idx_areax)<params['nsubnonlabeled']:
            continue

        B                   = np.concatenate((sessions[ises].tensor_vid,
                                sessions[ises].tensor_run),axis=0)

        # for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over stimuli 
        for istim,stim in enumerate([0,4,7]): # loop over stimuli 
            idx_T               = ses.trialdata['stimCond']==stim

            #on tensor during the response:
            X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
            Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
            # reshape to time points x neurons
            X                   = X.reshape(len(idx_areax),-1).T
            Y                   = Y.reshape(len(idx_areay),-1).T

            #Get the behavioraldata in the same shape as the neural data:
            Bstim                   = B[np.ix_(range(np.shape(B)[0]),idx_T,idx_resp)].reshape(np.shape(B)[0],-1).T
            Bstim                   = zscore(Bstim,axis=0,nan_policy='omit')
            Bstim                   = Bstim[:,~np.all(np.isnan(Bstim),axis=0)]
            
            #for each rank, regress out anything that can be behaviorally predicted
            for rankout in range(nranks_behavout):
                if rankout>0:
                    X_orig,X_hat,X_out  = regress_out_cv(X=Bstim,Y=X,rank=rankout,lam=0,kfold=5)
                    Y_orig,Y_hat,Y_out  = regress_out_cv(X=Bstim,Y=Y,rank=rankout,lam=0,kfold=5)
                else:
                    X_out               = X
                    Y_out               = Y
                #OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS
                R2_cv[iapl,rankout,ises,istim],optim_rank[iapl,rankout,ises,istim],_  = RRR_wrapper(Y_out, X_out, nN=params['nsubnonlabeled'],nranks=params['nranks'],nmodelfits=nmodelfits)

#%% Plotting:
clr = clrs_arealabelpairs[0]
R2_toplot = np.reshape(R2_cv,(narealabelpairs,nranks_behavout,nSessions*params['nStim']))
rank_toplot = np.reshape(optim_rank,(narealabelpairs,nranks_behavout,nSessions*params['nStim']))

fig,axes = plt.subplots(1,1,figsize=(3*cm,4*cm))

clrs = get_clr_areas(['V1','PM'])
ax = axes
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs):
    handles.append(shaded_error(range(nranks_behavout),R2_toplot[iapl].T,error='ci95',
                                color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
    ax_nticks(ax,3)
    for irank in range(nranks_behavout-1):
        x = R2_toplot[iapl][irank]
        y = R2_toplot[iapl][irank+1]
        nas = np.logical_or(np.isnan(x), np.isnan(y))
        t,p = ttest_rel(x[~nas], y[~nas])
        p = p*nranks_behavout #bonferonni correction
        print('Paired t-test: p=%.3f' % (p))
        ax.text((irank+1.5)/(nranks_behavout),0.95-0.1*iapl,'%s' % get_sig_asterisks(p),rotation=45,
                transform=ax.transAxes,ha='center',va='center',fontsize=8,color=clrs_arealabelpairs[iapl]) #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

ax.set_ylabel('performance')
ax.legend(handles,['V1->PM','PM->V1'],frameon=False,loc='best',fontsize=6)
my_legend_strip(ax)
# ax.set_ylim([0,0.12])
ax.set_xticks(np.arange(0,nranks_behavout))
ax.set_xticklabels(['orig']+['%d'%i for i in range(1,nranks_behavout)])
ax.set_xlabel('rank behavior out')
plt.tight_layout()
sns.despine(offset=2,top=True,right=True)
my_savefig(fig,figdir,'RRR_Behavout_ranks_%dsessions' % nSessions)

#%% Quantify in percentage how much RRR performance was reduced due to behavioral variability that was shared: 
perfreduc = (R2_cv[:,-1,:]-R2_cv[:,0,:]) / R2_cv[:,0,:]
for iapl, arealabelpair in enumerate(arealabelpairs):
    print('%1.1f%% +- %1.1f%% reduction for %s' % (np.nanmean(perfreduc[iapl]*100),np.nanstd(perfreduc[iapl]*100), arealabelpairs[iapl]))

#%% 
   #    #       #           #####  #######  #####   #####  ### ####### #     #  #####  
  # #   #       #          #     # #       #     # #     #  #  #     # ##    # #     # 
 #   #  #       #          #       #       #       #        #  #     # # #   # #       
#     # #       #           #####  #####    #####   #####   #  #     # #  #  #  #####  
####### #       #                # #             #       #  #  #     # #   # #       # 
#     # #       #          #     # #       #     # #     #  #  #     # #    ## #     # 
#     # ####### #######     #####  #######  #####   #####  ### ####### #     #  #####  

# #%% 
# areas = ['V1','PM','AL']
# session_list        = np.array(['LPE12223_2024_06_10']) #GR
# sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list,filter_areas=areas)

# #%% 
# sessions,nSessions   = filter_sessions(protocols = 'GR')

#%% Get all data 
only_all_areas = np.array(['V1','PM'])
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,filter_areas=only_all_areas,
                                       min_lab_cells_V1=10,min_lab_cells_PM=10,filter_noiselevel=False)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_noiselevel=True)
report_sessions(sessions)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load the data including behavior
[sessions,t_axis] = load_resid_tensor(sessions,params,load_behav=True)

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
sns.despine(fig,top=True,right=True,offset=2)
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
sns.despine(fig,top=True,right=True,offset=2)
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

sns.despine(fig,top=True,right=True,offset=2)
# plt.tight_layout()
# my_savefig(fig,figdir,'VarianceExplained_V1PM_labeled_%dsessions' % nSessions)

#%% Plotting:
clrs_arealabels = ['grey','red','grey','red','grey','red']
fig,axes = plt.subplots(1,1,figsize=(4*cm,3.9*cm))
ax = axes
optim_rank = 5
data = np.nanmean(EV_pops[:,optim_rank],axis=(-3,-1))

for ialp,[ial0,ial1] in enumerate([[0,1],[2,3]]):
    xscatter = np.random.randn(nSessions)*0.1
    ax.scatter(xscatter+ial0,data[ial0,:].flatten(),s=5,color=clrs_arealabels[ial0],marker='.')
    ax.scatter(xscatter+ial1,data[ial1,:].flatten(),s=5,color=clrs_arealabels[ial1],marker='.')
    ax.plot([xscatter+ial0,xscatter+ial1],[data[ial0,:],data[ial1,:]],color='k',marker='',markersize=0,linewidth=0.4)
    ax.plot(ial0,np.nanmean(data[ial0,:]),color=clrs_arealabels[ial0],marker='o',markersize=5)
    ax.plot(ial1,np.nanmean(data[ial1,:]),color=clrs_arealabels[ial1],marker='o',markersize=5)

ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax_nticks(ax,4)
ax.set_ylabel('Variance explained (R$^2$)')

si                  = SimpleImputer()
data                = si.fit_transform(data)

#Statistical testing:
df = pd.DataFrame({'EV': data.flatten(),
                  'arealabel': np.repeat(arealabeled,nSessions),
                  'session': np.tile(np.arange(nSessions),narealabels)})
order = arealabeled #for statistical testing purposes
pairs = [('V1unl','V1lab'),('PMunl','PMlab')]

annotator = Annotator(ax, pairs, data=df, x="arealabel", y='EV', order=order)
annotator.configure(test='Wilcoxon', text_format='star', loc='inside',verbose=False,
                    line_offset_to_group=0.2, line_width=1,
                    comparisons_correction="Bonferroni",line_height=0, text_offset=-3,fontsize=9)
annotator.apply_and_annotate()

sns.despine(fig,top=True,right=True,offset=2)
ax.set_xticks(range(narealabels),arealabeled,rotation=45,fontsize=6)
my_savefig(fig,figdir,'VarianceExplained_V1PM_arealabeled_%dsessions' % nSessions)


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
sns.despine(fig,top=True,right=True,offset=2)
plt.tight_layout()
# my_savefig(fig,figdir,'Mean_CoVarianceExplained_V1PM_%dsessions' % nSessions)



#%%  #assign arealayerlabel

def reset_layer(sessions):
    for ses in sessions:
    # ses.reset_layer(splitdepth=250)

        layers = {
                'V1': {
                    'L2/3': (0, 250),
                    # 'L4': (250, 300),
                    # 'L5': (300, np.inf)
                    'L5': (250, np.inf)
                },
                'PM': {
                    'L2/3': (0, 250),
                    # 'L4': (250, 300),
                    # 'L5': (300, np.inf)
                    'L5': (250, np.inf)
                },
                'AL': {
                    'L2/3': (0, 250),
                    'L4': (250, 300),
                    'L5': (300, np.inf)
                },
                'RSP': {
                    'L2/3': (0, 300),
                    'L5': (300, np.inf)
                }
            }
        for roi, layerdict in layers.items():
            for layer, (mindepth, maxdepth) in layerdict.items():
                idx = ses.celldata[(ses.celldata['roi_name'] == roi) & (mindepth <= ses.celldata['depth']) & (ses.celldata['depth'] < maxdepth)].index
                ses.celldata.loc[idx, 'layer'] = layer
    
#%%

reset_layer(sessions)

#%% 
for ises,ses in enumerate(sessions):
    ses.celldata['arealayerlabel'] = ses.celldata['arealabel']  + ses.celldata['layer'] 
celldata = pd.concat([ses.celldata for ses in sessions])

#%% 
plt.figure()
plt.hist(celldata['depth'][celldata['arealayerlabel']=='V1unlL2/3'],bins=np.arange(0,600,10),color='blue')
plt.hist(celldata['depth'][celldata['arealayerlabel']=='V1unlL4'],bins=np.arange(0,600,10),color='red')
plt.hist(celldata['depth'][celldata['arealayerlabel']=='V1unlL5'],bins=np.arange(0,600,10),color='green')

#%%
plt.figure()
plt.hist(celldata['depth'][celldata['arealayerlabel']=='V1labL2/3'],bins=np.arange(0,600,10),color='blue')
plt.hist(celldata['depth'][celldata['arealayerlabel']=='V1labL4'],bins=np.arange(0,600,10),color='red')
plt.hist(celldata['depth'][celldata['arealayerlabel']=='V1labL5'],bins=np.arange(0,600,10),color='green')

#%%
plt.figure()
plt.hist(celldata['depth'][celldata['arealayerlabel']=='PMlabL2/3'],bins=np.arange(0,600,10),color='blue')
plt.hist(celldata['depth'][celldata['arealayerlabel']=='PMlabL4'],bins=np.arange(0,600,10),color='red')
plt.hist(celldata['depth'][celldata['arealayerlabel']=='PMlabL5'],bins=np.arange(0,600,10),color='green')

#%% Compute the variance and covariance explained by the behavior (layer-dependent)
arealayerlabeled         = np.array(['V1unlL2/3','V1labL2/3','PMunlL2/3','PMlabL2/3','PMunlL5','PMlabL5'])
narealayerlabels         = len(arealayerlabeled)

#Parameters:
nranks              = 20
nStim               = 16
filter_nearby       = True

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

#Explained (co)variance
EV_pops             = np.full((narealayerlabels,nranks,nStim,nSessions,params['kfold']),np.nan)
# EC_poppairs         = np.full((narealabelpairs,nranks,nStim,nSessions,params['kfold']),np.nan)

# for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Covariance by behavior, fitting models'):
for ises,ses in enumerate(sessions):

    if filter_nearby:
        idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

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

        si                  = SimpleImputer()
        Y                   = si.fit_transform(Y)
        X                   = si.fit_transform(X)

        kf = KFold(n_splits=params['kfold'],shuffle=True)
        for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
            X_train, X_test     = X[idx_train], X[idx_test]
            Y_train, Y_test     = Y[idx_train], Y[idx_test]

            B_hat_train         = LM(Y_train,X_train, lam=params['lam'])

            Y_hat_train         = X_train @ B_hat_train

            # decomposing and low rank approximation of A
            U, s, V = svds(Y_hat_train,k=nranks,which='LM')
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

            # Y_cov = np.cov(Y_train.T)

            for r in range(nranks):
                B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                #construct low rank subspace prediction
                Y_hat_test_rr   = X_test @ B_rrr
                #How much variance is explained for each of the populations?
                for ial,al in enumerate(arealayerlabeled):
                    idx_N           = np.where(np.all((
                                            # ses.celldata['arealabel']==al,
                                            ses.celldata['arealayerlabel']==al,
                                            ses.celldata['noise_level']<params['maxnoiselevel'],	
                                            idx_nearby),axis=0))[0]
                    if len(idx_N)>=10:
                        EV_pops[ial,r,istim,ises,ikf] = EV(Y_test[:,idx_N],Y_hat_test_rr[:,idx_N])
                
                # #How much covariance is explained for each of the population pairs?
                # Y_cov_rrr       = np.cov(Y_hat_test_rr.T)
                # for ialp,arealabelpair in enumerate(arealabelpairs):

                #     alx,aly             = arealabelpair.split('-')

                #     idx_areax           = np.where(np.all((ses.celldata['arealabel']==alx,
                #                             ses.celldata['noise_level']<params['maxnoiselevel'],
                #                             idx_nearby),axis=0))[0]
                #     idx_areay           = np.where(np.all((ses.celldata['arealabel']==aly,
                #                             ses.celldata['noise_level']<params['maxnoiselevel'],
                #                             idx_nearby
                #                             ),axis=0))[0]
                    
                #     EC_poppairs[ialp,r,istim,ises,ikf] = EV(Y_cov[np.ix_(idx_areax,idx_areay)],Y_cov_rrr[np.ix_(idx_areax,idx_areay)])

#%% Plotting:
clrs_arealabels = ['grey','red','grey','red','grey','red']
fig,axes = plt.subplots(1,1,figsize=(4*cm,3.9*cm))
ax = axes
data = np.nanmean(EV_pops,axis=-1)
optim_rank = 5
data = np.reshape(data[:,optim_rank,:,:],(narealayerlabels,-1))

for ial, arealabel in enumerate(arealayerlabeled):
    ax.scatter(np.random.randn(nSessions*nStim)*0.1+ial,data[ial,:].flatten(),s=5,color='k',marker='.')
    ax.plot(ial,np.nanmean(data[ial,:]),color=clrs_arealabels[ial],marker='o',markersize=5)

ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax_nticks(ax,4)
ax.set_xticks(range(narealayerlabels),arealayerlabeled,rotation=45,fontsize=6)
ax.set_ylabel('Variance explained (R$^2$)')

#Statistical testing:
df = pd.DataFrame({'EV': data.flatten(),
                  'arealayerlabel': np.repeat(arealayerlabeled,nSessions*nStim),
                  'session': np.tile(np.arange(nSessions*nStim),narealayerlabels)})
order = arealayerlabeled #for statistical testing purposes
pairs = [('V1unlL2/3','V1labL2/3'),('PMunlL2/3','PMlabL2/3'),('PMunlL5','PMlabL5')]

# For paired t-test: null out sessions where only one of the two conditions has data
df_pivot = df.pivot(index='session', columns='arealayerlabel', values='EV')
for lbl0, lbl1 in pairs:
    if lbl0 in df_pivot.columns and lbl1 in df_pivot.columns:
        mask = df_pivot[lbl0].isna() | df_pivot[lbl1].isna()
        df_pivot.loc[mask, [lbl0, lbl1]] = np.nan
df = df_pivot.reset_index().melt(id_vars='session', value_name='EV', var_name='arealayerlabel')

annotator = Annotator(ax, pairs, data=df, x="arealayerlabel", y='EV', order=order)
annotator.configure(test='Wilcoxon', text_format='star', loc='inside',verbose=False,
                    line_offset_to_group=0.2, line_width=1,
                    comparisons_correction="Bonferroni",line_height=0, text_offset=-3,fontsize=9)
                    # comparisons_correction=None,line_height=0, text_offset=-3,fontsize=9)
annotator.apply_and_annotate()

sns.despine(fig,top=True,right=True,offset=2)
# plt.tight_layout()
# my_savefig(fig,figdir,'VarianceExplained_V1PM_arealayerlabeled_%ddatasets' % (nStim*nSessions))

#%% Plotting:
clrs_arealabels = ['grey','red','grey','red','grey','red']
fig,axes = plt.subplots(1,1,figsize=(4*cm,3.9*cm))
ax = axes
optim_rank = 4
data = np.nanmean(EV_pops[:,optim_rank],axis=(-3,-1))

for ialp,[ial0,ial1] in enumerate([[0,1],[2,3],[4,5]]):
    xscatter = np.random.randn(nSessions)*0.1
    ax.scatter(xscatter+ial0,data[ial0,:].flatten(),s=5,color=clrs_arealabels[ial0],marker='.')
    ax.scatter(xscatter+ial1,data[ial1,:].flatten(),s=5,color=clrs_arealabels[ial1],marker='.')
    ax.plot([xscatter+ial0,xscatter+ial1],[data[ial0,:],data[ial1,:]],color='k',marker='',markersize=0,linewidth=0.4)
    ax.plot(ial0,np.nanmean(data[ial0,:]),color=clrs_arealabels[ial0],marker='o',markersize=5)
    ax.plot(ial1,np.nanmean(data[ial1,:]),color=clrs_arealabels[ial1],marker='o',markersize=5)

ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax_nticks(ax,4)
ax.set_ylabel('Variance explained (R$^2$)')

#Statistical testing:
df = pd.DataFrame({'EV': data.flatten(),
                  'arealayerlabel': np.repeat(arealayerlabeled,nSessions),
                  'session': np.tile(np.arange(nSessions),narealayerlabels)})
order = arealayerlabeled #for statistical testing purposes
pairs = [('V1unlL2/3','V1labL2/3'),('PMunlL2/3','PMlabL2/3'),('PMunlL5','PMlabL5')]

# For paired t-test: null out sessions where only one of the two conditions has data
df_pivot = df.pivot(index='session', columns='arealayerlabel', values='EV')
for lbl0, lbl1 in pairs:
    if lbl0 in df_pivot.columns and lbl1 in df_pivot.columns:
        mask = df_pivot[lbl0].isna() | df_pivot[lbl1].isna()
        df_pivot.loc[mask, [lbl0, lbl1]] = np.nan
df = df_pivot.reset_index().melt(id_vars='session', value_name='EV', var_name='arealayerlabel')

annotator = Annotator(ax, pairs, data=df, x="arealayerlabel", y='EV', order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='inside',verbose=False,
# annotator.configure(test='Wilcoxon', text_format='star', loc='inside',verbose=False,
                    line_offset_to_group=0.2, line_width=1,
                    comparisons_correction=None,line_height=0, text_offset=-3,fontsize=9)
annotator.apply_and_annotate()

#Statistical testing:
df = pd.DataFrame({'EV': data.flatten(),
                  'arealayerlabel': np.repeat(arealayerlabeled,nSessions),
                  'session': np.tile(np.arange(nSessions),narealayerlabels)})
pairs = [('V1unlL2/3','PMunlL2/3'),('PMunlL2/3','PMunlL5')]

# For paired test: null out sessions where only one of the unl labels has data
df_pivot = df.pivot(index='session', columns='arealayerlabel', values='EV')
for lbl0, lbl1 in pairs:
    if lbl0 in df_pivot.columns and lbl1 in df_pivot.columns:
        mask = df_pivot[lbl0].isna() | df_pivot[lbl1].isna()
        df_pivot.loc[mask, :] = np.nan #set session to NaN
df = df_pivot.reset_index().melt(id_vars='session', value_name='EV', var_name='arealayerlabel')

annotator = Annotator(ax, pairs, data=df, x="arealayerlabel", y='EV', order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='inside',verbose=False,
                    line_offset_to_group=0.2, line_width=1,
                    comparisons_correction=None,line_height=0, text_offset=-3,fontsize=9)
annotator.apply_and_annotate()

sns.despine(fig,top=True,right=True,offset=2)
ax.set_xticks(range(narealayerlabels),arealayerlabeled,rotation=45,fontsize=6)
my_savefig(fig,figdir,'VarianceExplained_V1PM_arealayerlabeled_%dsessions' % nSessions)
