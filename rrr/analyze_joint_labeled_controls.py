# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.RRRlib import *
from utils.regress_lib import *
from params import load_params
from utils.corr_lib import filter_sharednan

params = load_params()
figdir = os.path.join(params['figdir'],'RRR','Labeling','FeedForward')
# figdir = os.path.join(params['figdir'],'RRR','Labeling','Feedback')
resultdir = params['resultdir']

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% Load the data:
version = 'FF_original'
filename = 'RRR_Joint_labeled_Controls_FF_original_2026-03-05_22-40-25'
filename = 'RRR_Joint_labeled_Controls_FF_original_2026-03-06_17-28-38'
# filename = 'RRR_Joint_labeled_Controls_FF_original_2026-03-06_23-31-25'
filename  = 'RRR_Joint_labeled_Controls_FF_original_2026-03-07_20-47-00'
# version = 'FB_original'
# filename = 'RRR_Joint_labeled_Controls_FB_original_2026-03-05_23-07-39'

#%%

valuematch_labels = np.array(['Cell radius','Noise level\n (Rupprecht et al. 2021)','Event rate','Tuning (gOSI)'])

#%% Save the data:
data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)

for key in data.keys():
    if key not in ['R2_ranks_neurons']:
        print(key)  
        exec(key+'=data[key]')

with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

nmodelfits = params['nmodelfits']
Nsub = params['Nsub']
narealabelpairs = 2

#%% Show figure for each of the arealabelpairs and each of the dataversions
for ivaluematching,valuematchfield in enumerate(valuematch_fields):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[ivaluematching],(narealabelpairs+1,params['nSessions']*params['nStim']))
    optim_rank_data         = np.reshape(optim_rank[ivaluematching],(narealabelpairs+1,params['nSessions']*params['nStim']))
    R2_ranks_data           = np.reshape(R2_ranks[ivaluematching],(narealabelpairs+1,params['nSessions']*params['nStim'],params['nranks'],nmodelfits,params['kfold']))
    if np.any(~np.isnan(R2_data)):
        for idx in np.array([[1,2]]):
            clrs        = ['grey','red']
            fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(sourcearealabelpairs)[idx-1],clrs)
            # my_savefig(fig,figdir,'RRR_cvR2_%s_%s_%dsessions' % (sourcearealabelpairs[idx[1]-1],version,params['nSessions']))


#%% Define the ratio of R2 between V1PM and V1ND

R2_ratio = R2_cv[:,2,:,:] / R2_cv[:,1,:,:]

#%% Define the ratio of R2 between V1PM and V1ND
#Only for dim 2-5

R2_ratio = np.nanmean(R2_ranks,axis=(-1,-2))
R2_ratio = np.diff(R2_ratio,axis=4)
R2_ratio = np.nansum(R2_ratio[:,:,:,:,np.arange(1,5)],axis=-1)
# R2_ratio = np.nansum(R2_ratio[:,:,:,:,np.arange(1,3)],axis=-1)

R2_ratio = R2_ratio[:,2,:,:] / R2_ratio[:,1,:,:]


#%% Make the figure of the ratio:
fig,axes = plt.subplots(1,1,sharex=True,sharey=True,figsize=(3*cm,3.6*cm))
ax = axes
ax.errorbar(x=range(params['nvaluefields']),y=np.nanmean(R2_ratio,axis=(1,2)),yerr=np.nanstd(R2_ratio,axis=(1,2))/np.sqrt(params['nSessions']*params['nStim']),
            color='red',marker='o',linestyle='',capsize=0)
# ax.errorbar(x=range(params['nvaluefields']),y=np.nanmean(ratiodata_FF_unlunl,axis=1),yerr=np.nanstd(ratiodata_FF_unlunl,axis=1)/np.sqrt(np.shape(ratiodata_FF_unlunl)[1]),
#             color='grey',marker='o',linestyle='-',capsize=0)
for ivaluematching in range(params['nvaluefields']):
    h,p = stats.ttest_1samp(R2_ratio[ivaluematching].flatten(),1,nan_policy='omit')
    ax.text(ivaluematching,np.nanmean(R2_ratio[ivaluematching])+0.05,get_sig_asterisks(p),rotation=0,ha='center',fontsize=9)
# ax.legend(['$V1_{PM}$ vs. $V1_{ND1}$'],
#           frameon=False,bbox_to_anchor=(1.08,0.8),fontsize=6)
# ax.set_title("Stratified sampling")
# ax.set_xlabel("Variable")
ax_nticks(ax,4)
ax.set_ylabel("Relative performance\n$V1_{PM}$ vs. $V1_{ND}$\n(Dimension 2-5)")
ax.axhline(y=1,color='k',linestyle='--')
ax.set_xticks(range(params['nvaluefields']))
ax.set_xticklabels(valuematch_labels,rotation=45,ha='right')
ax.set_xlim([-0.5,params['nvaluefields']-1+.25])
# plt.tight_layout()
sns.despine(fig=fig,trim=True)
my_savefig(fig,figdir,'RRR_cvR2_ratio_%s_controls_%dsessions' % (version,params['nSessions']))

#%% Define the ratio of R2 between V1PM and V1ND
Rank_ratio = optim_rank[:,2,:,:] / optim_rank[:,1,:,:]

#%% Make the figure of the ratio:
fig,axes = plt.subplots(1,1,sharex=True,sharey=True,figsize=(4*cm,4.6*cm))
ax = axes
ax.errorbar(x=range(params['nvaluefields']),y=np.nanmean(Rank_ratio,axis=(1,2)),yerr=np.nanstd(Rank_ratio,axis=(1,2))/np.sqrt(params['nSessions']*params['nStim']),
            color='red',marker='o',linestyle='',capsize=0)
for ivaluematching in range(params['nvaluefields']):
    h,p = stats.ttest_1samp(Rank_ratio[ivaluematching].flatten(),1,nan_policy='omit')
    ax.text(ivaluematching,np.nanmean(Rank_ratio[ivaluematching])+0.01,get_sig_asterisks(p),rotation=0,ha='center',fontsize=9)
ax.set_title("Stratified sampling")
ax.set_xlabel("Variable")
ax.set_ylim([0.98,my_ceil(ax.get_ylim()[1],2)])
ax_nticks(ax,4)
ax.set_ylabel("Relative rank\n$V1_{PM}$ vs. $V1_{ND}$")
ax.axhline(y=1,color='k',linestyle='--')
ax.set_xticks(range(params['nvaluefields']))
ax.set_xticklabels(valuematch_fields,rotation=45,ha='right')
ax.set_xlim([-0.5,params['nvaluefields']-1+.25])
# plt.tight_layout()
sns.despine(fig=fig,trim=True)
# my_savefig(fig,figdir,'RRR_rank_ratio_%s_controls_%dsessions' % (version,params['nSessions']))

