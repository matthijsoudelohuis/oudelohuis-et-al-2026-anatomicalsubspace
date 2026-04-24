# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
import pickle

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.RRRlib import *
from utils.regress_lib import *
from params import load_params
from datetime import datetime

#%% Load parameters and settings:
params = load_params()

# params['regress_behavout'] = True
params['regress_behavout'] = False
params['direction'] = 'FF'
params['direction'] = 'FB'
# params['direction'] = 'FF_AL'
# params['direction'] = 'FB_AL'

version = 'Joint_labeled_%s_%s_spont' % (params['direction'],'behavout' if params['regress_behavout'] else 'original')

resultdir = os.path.join(params['resultdir'])
if not os.path.exists(resultdir):
    os.makedirs(resultdir)
datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savefilename = os.path.join(resultdir,'RRR_%s_%s' % (version,datetime_str))

#%% Do RRR of V1 and PM labeled and unlabeled neurons simultaneously
if params['direction'] =='FF': 
    sourcearealabelpairs = ['V1unl','V1unl','V1lab']
    targetarealabelpair = 'PMunl'
    only_all_areas = np.array(['V1','PM'])
elif params['direction'] =='FB': 
    sourcearealabelpairs = ['PMunl','PMunl','PMlab']
    targetarealabelpair = 'V1unl'
    only_all_areas = np.array(['V1','PM'])
elif params['direction'] =='FF_AL': 
    sourcearealabelpairs = ['V1unl','V1unl','V1lab']
    targetarealabelpair = 'ALunl'
    only_all_areas = np.array(['V1','PM','AL'])
elif params['direction'] =='FB_AL': 
    sourcearealabelpairs = ['PMunl','PMunl','PMlab']
    targetarealabelpair = 'ALunl'
    only_all_areas = np.array(['V1','PM','AL'])

#%% 
session_list        = np.array([
                                # ['LPE12223_2024_06_10'], #V1lab actually lower
                                ['LPE09830_2023_04_10'], #V1 labeled higher predictive than V1unl
                                # ['LPE10919_2023_11_06'],  #V1lab actually lower
                                # ['LPE12223_2024_06_08'], #V1lab actually lower
                                # ['LPE11622_2024_03_25'], #same
                                ['LPE09665_2023_03_14'], #V1lab higher
                                # ['LPE10885_2023_10_23'], #V1lab much higher
                                # ['LPE11086_2024_01_05'], #Really much higher, best session, first dimensions are more predictive.
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['SP'],only_session_id=session_list,
                                       filter_noiselevel=False)
report_sessions(sessions)

#%% Get all data 
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_all_areas=only_all_areas,min_lab_cells_V1=20,min_lab_cells_PM=20,filter_noiselevel=False)
sessions,nSessions   = filter_sessions(protocols = ['SP'],only_all_areas=only_all_areas,filter_noiselevel=False)
report_sessions(sessions)

#%% Wrapper function to load the tensor data, 
# [sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=params['regress_behavout'])
vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
                            ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

behavfields = np.array(['runspeed','diffrunspeed'])
si              = SimpleImputer()

for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=params['calciumversion'])
    
    neuraldata          = zscore(sessions[ises].calciumdata.to_numpy(),axis=0,nan_policy='omit')

    if params['regress_behavout']:
        B = np.empty((len(sessions[ises].ts_F),len(vidfields)+1))
        for ifie,field in enumerate(vidfields):
            B[:,ifie] = np.interp(x=sessions[ises].ts_F, xp=sessions[ises].videodata['ts'],
                                            # fp=sessions[ises].videodata['runspeed'])
                                            fp=sessions[ises].videodata[field])
        B[:,-1] = np.interp(x=sessions[ises].ts_F, xp=sessions[ises].behaviordata['ts'],
                                            fp=sessions[ises].behaviordata['runspeed'])

        B                   = si.fit_transform(B)
        B                   = zscore(B,axis=0,nan_policy='omit')

    if params['regress_behavout']:
        rank_behavout = 5

        areas = np.unique(sessions[ises].celldata['roi_name'])
        for area in areas:
            idx_N    = np.where(np.all((sessions[ises].celldata['roi_name']==area,
                                        sessions[ises].celldata['noise_level']<params['maxnoiselevel']),axis=0))[0]

            B                   = B[:,~np.all(np.isnan(B),axis=0)]
            _,_,neuraldata[:,idx_N]  = regress_out_cv(X=B,Y=neuraldata[:,idx_N],rank=np.min([rank_behavout,len(idx_N)-1]),
                                                lam=0,kfold=5)

#%% 
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 20
nranks              = 20 #number of ranks of RRR to be evaluated
nmodelfits          = 10
timechunck          = 180 #in seconds
sampleschunk        = int(timechunck * sessions[ises].sessiondata['fs'][0]) #in seconds

params['nStim']     = 15*60//timechunck #number of time chunks to split the data in for RRR (15 minutes of recording, split in 1 minute chunks)

R2_cv               = np.full((narealabelpairs+1,nSessions,params['nStim']),np.nan) #dim1: 3 = allneurons, V1unl, V1lab separately
optim_rank          = np.full((narealabelpairs+1,nSessions,params['nStim']),np.nan)
R2_ranks            = np.full((narealabelpairs+1,nSessions,params['nStim'],nranks,nmodelfits,params['kfold']),np.nan)

for ises,ses in enumerate(sessions):
    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(sessions[ises],radius=params['radius'])
    else:
        idx_nearby = np.ones(len(sessions[ises].celldata),dtype=bool)

    nsamples            = len(sessions[ises].calciumdata) #number of time points in the data
    idx_T_all           = np.arange(0,nsamples-1,sampleschunk,dtype=int)

    idx_areax1      = np.where(np.all((sessions[ises].celldata['arealabel']==sourcearealabelpairs[0],
                                sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
    idx_areax2      = np.where(np.all((sessions[ises].celldata['arealabel']==sourcearealabelpairs[1],
                                sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
    idx_areax3      = np.where(np.all((sessions[ises].celldata['arealabel']==sourcearealabelpairs[2],
                                sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
    idx_areay       = np.where(np.all((sessions[ises].celldata['arealabel']==targetarealabelpair,
                                            sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                            idx_nearby
                                            ),axis=0))[0]
    
    neuraldata          = sessions[ises].calciumdata.to_numpy()

    if len(idx_areax1)<Nsub*2 or len(idx_areax2)<Nsub*2 or len(idx_areax3)<Nsub or len(idx_areay)<narealabelpairs*Nsub: #skip exec if not enough neurons in one of the populations
        print('%d in %s, %d in %s' % (len(idx_areax3),sourcearealabelpairs[2],
                                                len(idx_areay),targetarealabelpair))
        continue

    for imf in tqdm(range(nmodelfits),total=nmodelfits,desc='Fitting RRR model for session %d/%d' % (ises+1,nSessions)):
        idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
        idx_areax2_sub       = np.random.choice(np.setdiff1d(idx_areax2,idx_areax1_sub),Nsub,replace=False)
        idx_areax3_sub       = np.random.choice(idx_areax3,Nsub,replace=False)
        idx_areay_sub        = np.random.choice(idx_areay,Nsub*narealabelpairs,replace=False)

        for istim in np.arange(len(idx_T_all)-1): # loop over time chunks 
            idx_T               = np.arange(idx_T_all[istim],idx_T_all[istim+1])

            X1                  = neuraldata[np.ix_(idx_T,idx_areax1_sub)]
            X2                  = neuraldata[np.ix_(idx_T,idx_areax2_sub)]
            X3                  = neuraldata[np.ix_(idx_T,idx_areax3_sub)]
            Y                   = neuraldata[np.ix_(idx_T,idx_areay_sub)]

            X1                  = zscore(X1,axis=0) #zscore the activity per neuron
            X2                  = zscore(X2,axis=0)
            X3                  = zscore(X3,axis=0)
            Y                   = zscore(Y,axis=0)

            X                   = np.concatenate((X1,X2,X3),axis=1) #use this as source to predict the activity in Y with RRR

            # OUTPUT: MAX PERF, OPTIM RANK, PERF FOR EACH RANK ACROSS FOLDS AND MODELFITS    
            R2_kfold    = np.zeros((params['kfold']))
            kf          = KFold(n_splits=params['kfold'],shuffle=True)
            for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
                X_train, X_test     = X[idx_train], X[idx_test]
                Y_train, Y_test     = Y[idx_train], Y[idx_test]

                #RRR X to Y
                B_hat_train         = LM(Y_train,X_train, lam=params['lam'])
                Y_hat_train         = X_train @ B_hat_train

                # decomposing and low rank approximation of Y_hat
                U, s, V = svds(Y_hat_train,k=nranks,which='LM')
                U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

                for r in range(nranks):
                    B_rrr           = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                    Y_hat_test_rr   = X_test @ B_rrr

                    R2_ranks[0,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)
                    
                    X_test_1 = copy.deepcopy(X_test)
                    X_test_1[:,Nsub:] = 0
                    Y_hat_test_rr   = X_test_1 @ B_rrr

                    R2_ranks[1,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)

                    X_test_2 = copy.deepcopy(X_test)
                    X_test_2[:,:Nsub] = 0
                    X_test_2[:,2*Nsub:] = 0
                    Y_hat_test_rr   = X_test_2 @ B_rrr

                    R2_ranks[2,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)

                    X_test_3 = copy.deepcopy(X_test)
                    X_test_3[:,:2*Nsub] = 0
                    Y_hat_test_rr   = X_test_3 @ B_rrr

                    R2_ranks[3,ises,istim,r,imf,ikf] = EV(Y_test,Y_hat_test_rr)

#%% Find best rank and cvR2 at this rank:
fixed_rank = None
for ises in range(nSessions):
    if np.any(~np.isnan(R2_ranks[0][ises])):
        for istim in range(params['nStim']):
            if fixed_rank is not None:
                rank = fixed_rank
                R2_cv[0,ises,istim] = np.nanmean(R2_ranks[0,ises,istim,rank,:,:])
                R2_cv[1,ises,istim] = np.nanmean(R2_ranks[1,ises,istim,rank,:,:])
                R2_cv[2,ises,istim] = np.nanmean(R2_ranks[2,ises,istim,rank,:,:])
                R2_cv[3,ises,istim] = np.nanmean(R2_ranks[3,ises,istim,rank,:,:])
            else:
                if not np.isnan(R2_ranks[0][ises][istim]).all():
                    R2_cv[0,ises,istim],optim_rank[0,ises,istim] = rank_from_R2(R2_ranks[0,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                    R2_cv[1,ises,istim],optim_rank[1,ises,istim] = rank_from_R2(R2_ranks[1,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                    R2_cv[2,ises,istim],optim_rank[2,ises,istim] = rank_from_R2(R2_ranks[2,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])
                    R2_cv[3,ises,istim],optim_rank[3,ises,istim] = rank_from_R2(R2_ranks[3,ises,istim,:,:,:].reshape([nranks,nmodelfits*params['kfold']]),nranks,nmodelfits*params['kfold'])

#%%
params['Nsub']     = Nsub
params['nranks']    = nranks
params['nmodelfits'] = nmodelfits
params['nSessions'] = nSessions

#%% Save the data:
np.savez(savefilename + '.npz',R2_cv=R2_cv,R2_ranks=R2_ranks,optim_rank=optim_rank,
         sourcearealabelpairs=sourcearealabelpairs,
         targetarealabelpair=targetarealabelpair,
         allow_pickle=True)

with open(savefilename +'_params' + '.txt', "wb") as myFile:
    pickle.dump(params, myFile)

#%%

figdir = os.path.join(params['figdir'],'RRR','Spontaneous','Labeling')

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches


#%% Show an example session:
clrs_arealabelpairs = ['grey','grey','red']
narealabelpairs = 3
fig, axes = plt.subplots(1,1,figsize=(6*cm,5*cm))
ax = axes
ises = 0
handles = []
# ax.plot(range(params['nranks']),np.nanmean(R2_ranks[0],axis=(0,1,3,4)),label='All neurons',color='grey')
# ax.plot(np.nanmean(R2_ranks[1][ises],axis=(0,2,3)),label=sourcearealabelpairs[0],color=clrs_arealabelpairs[0])
# ax.plot(np.nanmean(R2_ranks[2][ises],axis=(0,2,3)),label=sourcearealabelpairs[1],color=clrs_arealabelpairs[1])
# ax.plot(np.nanmean(R2_ranks[3][ises],axis=(0,2,3)),label=sourcearealabelpairs[2],color=clrs_arealabelpairs[2])
for iapl,apl in enumerate(sourcearealabelpairs):
    ymeantoplot = np.nanmean(R2_ranks[iapl+1][ises],axis=(0,2,3))
    yerrortoplot = np.nanstd(R2_ranks[iapl+1][ises],axis=(0,2,3)) / np.sqrt(nmodelfits)
    handles.append(shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[iapl],alpha=0.3))

leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Cross-validated R2')

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
# my_savefig(fig,figdir,'RRR_joint_cvR2_labunl_%s_ExampleSesion' % (version))


#%% Show the mean across sessions:
clrs_arealabelpairs = ['grey','grey','red']

nrankstoplot = 12
xposrank = 10
idxs = np.array([1,3])
meanranks = np.nanmean(optim_rank,axis=(-1,-2))
meanR2 = np.nanmean(R2_cv,axis=(-1,-2))
sourcearealabelpairs = np.array(sourcearealabelpairs)
fig, axes = plt.subplots(1,1,figsize=(5*cm,4.5*cm))
ax = axes
handles = []
# ax.plot(np.nanmean(R2_ranks[idxs[0]],axis=(0,1,3,4)),label=arealabeled_to_figlabels(sourcearealabelpairs[idxs[0]-1]),
#         color=clrs_arealabelpairs[idxs[0]-1],linewidth=2)
# ax.plot(np.nanmean(R2_ranks[idxs[1]],axis=(0,1,3,4)),label=arealabeled_to_figlabels(sourcearealabelpairs[idxs[1]-1]),
#         color=clrs_arealabelpairs[idxs[1]-1],linewidth=2)
ydata = np.nanmean(R2_ranks[idxs[0]],axis=(3,4))
ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
                            color=clrs_arealabelpairs[idxs[0]-1],alpha=0.3))
ydata = np.nanmean(R2_ranks[idxs[1]],axis=(3,4))
ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
                            color=clrs_arealabelpairs[idxs[1]-1],alpha=0.3))
for idx in idxs:
    ax.plot(meanranks[idx],meanR2[idx]+0.005,color=clrs_arealabelpairs[idx-1],marker='v',markersize=5)

leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs[idxs-1]),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel(r'Cross-validated R$^2$')

x = optim_rank[idxs[0],:]
y = optim_rank[idxs[1],:]
nas = np.logical_or(np.isnan(x), np.isnan(y))
t,p = ttest_rel(x[~nas], y[~nas])
print('Paired t-test (Rank): p=%.3f' % (p))
ax.plot(meanranks[idxs],np.repeat(np.nanmean(meanR2[idxs]),2)+0.007,linestyle='-',color='k',linewidth=2)
ax.text(np.nanmean(meanranks),np.nanmean(meanR2[idxs])+0.009,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

x = R2_cv[idxs[0],:]
y = R2_cv[idxs[1],:]
nas = np.logical_or(np.isnan(x), np.isnan(y))
t,p = ttest_rel(x[~nas], y[~nas])
print('Paired t-test (R2): p=%.3f' % (p))
ax.plot([xposrank,xposrank],meanR2[idxs],linestyle='-',color='k',linewidth=2)
ax.text(xposrank+0.5,np.nanmean(meanR2[idxs])+0.005,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

ax.set_xticks(np.arange(params['nranks'])[::3]+1)
ax.set_xlim([0,nrankstoplot])
ax.set_title(params['direction'])
plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
my_savefig(fig,figdir,'RRR_joint_R2_labunl_%s_%dsessions_spont' % (params['direction'],params['nSessions']))
