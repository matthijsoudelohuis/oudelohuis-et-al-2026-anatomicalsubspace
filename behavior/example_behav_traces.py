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

np.random.seed(42)
idx_areax_sub = np.random.choice(idx_areax,100,replace=False)
idx_areay_sub = np.random.choice(idx_areay,100,replace=False)

nrankstoplot = 1
#RRR latents:
X               = zscore(sessions[ises].calciumdata.iloc[:,idx_areax_sub],axis=0).to_numpy()
Y               = zscore(sessions[ises].calciumdata.iloc[:,idx_areay_sub],axis=0).to_numpy()
B_hat           = LM(Y,X, lam=0) #linear regression
Y_hat           = X @ B_hat #project X onto B_hat to obtain Y_hat
U, s, V         = svds(Y_hat,k=nrankstoplot,which='LM') #decompose Y_hat, get maximally predictive dimensions
U, s, V         = U[:, ::-1], s[::-1], V[::-1, :]
W               = B_hat @ V.T   # Predictive X-directions
Z               = X @ W   # Project X onto predictive dimensions
# Z               = Z.flatten() #remove redundant dim 
for irank in range(nrankstoplot):
    Z[:,irank]    = Z[:,irank]*np.sign(np.corrcoef(Z[:,irank],meanV1)[0,1]) #align sign with mean V1 rate


fig,ax = plt.subplots(1,1,figsize=(10*cm,5*cm)) #make the figure
data        = [sessions[ises].behaviordata['runspeed'],
               sessions[ises].videodata['pupil_area'],
               sessions[ises].videodata['motionenergy'],
               meanV1,
               meanPM,
               ]
for irank in range(nrankstoplot):
    data.append(Z[:,irank])
ts          = [
                sessions[ises].behaviordata['ts'],
                sessions[ises].videodata['ts'],
                sessions[ises].videodata['ts'],
                sessions[ises].ts_F,
               sessions[ises].ts_F,
               ]
for irank in range(nrankstoplot):
    ts.append(sessions[ises].ts_F)
# labels      = ['Run speed','Pupil area','Motion energy','Mean V1 Activity','Mean PM Activity','Subspace Latent 1']
labels      = ['Run speed','Pupil area','Motion energy','Mean V1 Activity',
               'Mean PM Activity']
for irank in range(nrankstoplot):
    labels.append('Latent %d' % (irank+1))

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
sns.despine(fig,top=True,right=True,offset=2)
# plt.tight_layout()
ax.add_artist(AnchoredSizeBar(ax.transData, 10,
                "10 Sec", loc='lower right', frameon=False))

# my_savefig(fig,figdir,'Example_Behavior_V1PM_%s' % sessions[ises].session_id)
