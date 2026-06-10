# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle

from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.params import load_params

params = load_params()
resultdir = params['resultdir']

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  #centimeters in inches

#%% Run block for FF 
version = 'FF_original'
filename = 'RRR_Separate_labeled_FF_original_2026-05-21_17-17-36'
exampleses = 12
figdir = os.path.join(params['figdir'],'RRR','Labeling','Feedforward')

#%% For FB:
version = 'FB_original'
filename = 'RRR_Separate_labeled_FB_original_2026-05-22_13-16-01'
exampleses = 1
figdir = os.path.join(params['figdir'],'RRR','Labeling','Feedback')

#%% For FF_AL:
version = 'FF_AL_original'
filename = 'RRR_Separate_labeled_FF_AL_original_2026-05-22_14-30-47'
exampleses = 4
figdir = os.path.join(params['figdir'],'RRR','Labeling','AL')

# version = 'FB_AL_original'
# filename = 'RRR_Separate_labeled_FB_AL_original_2026-05-22_14-45-21'

#%% Load the data:
data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)
sourcearealabelpairs = data['sourcearealabelpairs']
targetarealabelpair = data['targetarealabelpair']
R2_cv = data['R2_cv']
optim_rank = data['optim_rank']
R2_ranks = data['R2_ranks']

with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)
nmodelfits = params['nmodelfits']

#%% Show an example session:
clrs_arealabelpairs = ['grey','red']
nrankstoplot = 10
narealabelpairs = 2
fig, axes = plt.subplots(1,1,figsize=(5*cm,4.5*cm))
ax = axes
handles = []
for iapl,apl in enumerate(sourcearealabelpairs):
    ymeantoplot = np.nanmean(R2_ranks[iapl][exampleses],axis=(0,2,3))
    yerrortoplot = np.nanstd(R2_ranks[iapl][exampleses],axis=(0,2,3)) / np.sqrt(params['nmodelfits'])
    handles.append(shaded_error(np.arange(params['nranks']),ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[iapl],alpha=0.3))
    meanrank = np.nanmean(optim_rank[iapl][exampleses])
    meanr2 = np.nanmean(R2_cv[iapl][exampleses])
    ax.plot(meanrank,meanr2+0.005,color=clrs_arealabelpairs[iapl],marker='v',markersize=5)

leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Cross-validated R2')
ax.set_xlim([0,nrankstoplot])
# ax.set_xticks([0,1,5,10])
ax.set_xticks([1,4,7,10])
plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
# my_savefig(fig,figdir,'RRR_separate_cvR2_labunl_%s_ExampleSesion' % params['direction'])

#%% Show the mean across sessions:
xposrank = 10
idxs = np.array([0,1])
meanranks = np.nanmean(optim_rank,axis=(-1,-2))
meanR2 = np.nanmean(R2_cv,axis=(-1,-2))
R2_rank_datatoplot = np.nanmean(R2_ranks,axis=(4,5))

#Get only the sessions that have both populations:
idx_ses = np.all(~np.isnan(R2_ranks),axis=(0,2,3,4))

fig, axes = plt.subplots(1,1,figsize=(5*cm,4.5*cm))
ax = axes
handles = []

ydata = R2_rank_datatoplot[idxs[0]]
ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
                            color=clrs_arealabelpairs[idxs[0]-1],alpha=0.3))
ydata = R2_rank_datatoplot[idxs[1]]
# ydata = np.nanmean(R2_rank_datatoplot[idxs[1]])
ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
                            color=clrs_arealabelpairs[idxs[1]-1],alpha=0.3))
for idx in idxs:
    ax.plot(meanranks[idx],meanR2[idx]+0.005,color=clrs_arealabelpairs[idx-1],marker='v',markersize=5)

leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs[idxs-1]),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Cross-validated R2')

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

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
# my_savefig(fig,figdir,'RRR_joint_cvR2_labunl_%s_%dsessions' % (version,params['nSessions']))

#%% Show figure for each of the arealabelpairs and each of the dataversions
#Reshape stim x sessions:
R2_data                 = np.reshape(R2_cv,(narealabelpairs,params['nSessions']*params['nStim']))
optim_rank_data         = np.reshape(optim_rank,(narealabelpairs,params['nSessions']*params['nStim']))
R2_ranks_data           = np.reshape(R2_ranks,(narealabelpairs,params['nSessions']*params['nStim'],params['nranks'],nmodelfits,params['kfold']))

clrs        = ['grey','red']
fig         = plot_RRR_R2_arealabels_paired(R2_data,optim_rank_data,R2_ranks_data,np.array(sourcearealabelpairs),clrs)
my_savefig(fig,figdir,'RRR_cvR2_%s_%dsessions' % (version,params['nSessions']))

#%%
 #####  ####### #     # ####### ######  ####### #        #####  
#     # #     # ##    #    #    #     # #     # #       #     # 
#       #     # # #   #    #    #     # #     # #       #       
#       #     # #  #  #    #    ######  #     # #        #####  
#       #     # #   # #    #    #   #   #     # #             # 
#     # #     # #    ##    #    #    #  #     # #       #     # 
 #####  ####### #     #    #    #     # ####### #######  #####  
 
#%% Run block for FF 
version = 'FF_original'
filename = 'RRR_Separate_labeled_controls_FF_original_2026-06-09_18-31-07'
exampleses = 12
figdir = os.path.join(params['figdir'],'RRR','Labeling','Feedforward')

#%% For FB:
version = 'FB_original'
filename = 'RRR_Separate_labeled_controls_FB_original_2026-06-09_21-13-03'
exampleses = 1
figdir = os.path.join(params['figdir'],'RRR','Labeling','Feedback')

#%% Load the data:
data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)
print(data.files)
sourcearealabelpairs = data['sourcearealabelpairs']
narealabelpairs = len(sourcearealabelpairs)
alx1 = arealabeled_to_figlabels(sourcearealabelpairs[0].split('-')[0])
alx2 = arealabeled_to_figlabels(sourcearealabelpairs[1].split('-')[0])
targetarealabelpair = data['targetarealabelpair']
R2_cv = data['R2_cv']
optim_rank = data['optim_rank']
R2_ranks = data['R2_ranks']
valuematch_fields = data['valuematch_fields']
valuematch_labels = np.array(['Cell radius','Noise level\n (Rupprecht et al. 2021)','Event rate','Tuning (gOSI)'])

with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)
nmodelfits = params['nmodelfits']

#%% Show figure for each of the arealabelpairs and each of the dataversions
for ivaluematching,valuematchfield in enumerate(valuematch_fields):
    #Reshape stim x sessions:
    R2_data                 = np.reshape(R2_cv[ivaluematching],(narealabelpairs,params['nSessions']*params['nStim']))
    optim_rank_data         = np.reshape(optim_rank[ivaluematching],(narealabelpairs,params['nSessions']*params['nStim']))
    R2_ranks_data           = np.reshape(R2_ranks[ivaluematching],(narealabelpairs,params['nSessions']*params['nStim'],params['nranks'],nmodelfits,params['kfold']))
    if np.any(~np.isnan(R2_data)):
        for idx in np.array([[0,1]]):
            clrs        = ['grey','red']
            fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(sourcearealabelpairs)[idx-1],clrs)
            # my_savefig(fig,figdir,'RRR_cvR2_%s_%s_%dsessions' % (sourcearealabelpairs[idx[1]-1],version,params['nSessions']))

#%% Define the ratio of R2 between lab and unlab
R2_ratio = R2_cv[:,1,:,:] / R2_cv[:,0,:,:]

#Make the figure of the ratio:
fig,axes = plt.subplots(1,1,sharex=True,sharey=True,figsize=(3*cm,3.6*cm))
ax = axes
ax.errorbar(x=range(params['nvaluefields']),y=np.nanmean(R2_ratio,axis=(1,2)),yerr=np.nanstd(R2_ratio,axis=(1,2))/np.sqrt(params['nSessions']*params['nStim']),
            color='red',marker='o',linestyle='',capsize=0)
# ax.errorbar(x=range(params['nvaluefields']),y=np.nanmean(ratiodata_FF_unlunl,axis=1),yerr=np.nanstd(ratiodata_FF_unlunl,axis=1)/np.sqrt(np.shape(ratiodata_FF_unlunl)[1]),
#             color='grey',marker='o',linestyle='-',capsize=0)
for ivaluematching in range(params['nvaluefields']):
    h,p = stats.ttest_1samp(R2_ratio[ivaluematching].flatten(),1,nan_policy='omit')
    ax.text(ivaluematching,np.nanmean(R2_ratio[ivaluematching])+0.05,get_sig_asterisks(p),rotation=0,ha='center',fontsize=9)
ax_nticks(ax,4)
ax.set_ylabel("performance ratio\n%s/%s" % (alx2,alx1))
ax.axhline(y=1,color='k',linestyle='--')
ax.set_xticks(range(params['nvaluefields']))
ax.set_xticklabels(valuematch_labels,rotation=45,ha='right')
ax.set_xlim([-0.5,params['nvaluefields']-1+.25])
# plt.tight_layout()
sns.despine(fig=fig,trim=True)
my_savefig(fig,figdir,'perf_ratio_separate_%s_controls_%dsessions' % (version,params['nSessions']))

#%% Define the ratio of R2 between V1PM and V1ND
rank_ratio = optim_rank[:,1,:,:] / optim_rank[:,0,:,:]

#Make the figure of the ratio:
fig,axes = plt.subplots(1,1,sharex=True,sharey=True,figsize=(4*cm,4.6*cm))
ax = axes
ax.errorbar(x=range(params['nvaluefields']),y=np.nanmean(rank_ratio,axis=(1,2)),yerr=np.nanstd(rank_ratio,axis=(1,2))/np.sqrt(params['nSessions']*params['nStim']),
            color='red',marker='o',linestyle='',capsize=0)
for ivaluematching in range(params['nvaluefields']):
    h,p = stats.ttest_1samp(rank_ratio[ivaluematching].flatten(),1,nan_policy='omit')
    ax.text(ivaluematching,np.nanmean(rank_ratio[ivaluematching])+0.01,get_sig_asterisks(p),rotation=0,ha='center',fontsize=9)
ax.set_title("Stratified sampling")
ax.set_xlabel("Variable")
ax.set_ylim([0.98,my_ceil(ax.get_ylim()[1],2)])
ax_nticks(ax,4)
ax.set_ylabel("rank ratio\n%s/%s" % (alx2,alx1))
ax.axhline(y=1,color='k',linestyle='--')
ax.set_xticks(range(params['nvaluefields']))
ax.set_xticklabels(valuematch_fields,rotation=45,ha='right')
ax.set_xlim([-0.5,params['nvaluefields']-1+.25])
# plt.tight_layout()
sns.despine(fig=fig,trim=True)
my_savefig(fig,figdir,'rank_ratio_separate_%s_controls_%dsessions' % (version,params['nSessions']))

