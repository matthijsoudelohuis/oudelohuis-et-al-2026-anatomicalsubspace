# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
import numpy as np
from scipy.stats import zscore
import pickle

from loaddata.session_info import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.params import load_params
from datetime import datetime
from utils.tuning import compute_tuning_wrapper

#%% Load parameters and settings:
params = load_params()
figdir = os.path.join(params['figdir'],'RRR','DOC')
resultdir = params['resultdir']

#%% Plotting:
cm = 1/2.54
set_plot_basic_config()

#%% Load the data (Feedforward):
filename = 'RRR_doc_labeled_FF_2026-06-17_21-49-57'

#%% Load the data (Feedback):
filename = 'RRR_doc_labeled_FB_2026-06-18_00-13-25'

#%% Load the data:
data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)

for key in data.keys():
    print(key)
    exec(key+'=data[key]')

with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)


#%% Show the R2 for original and rotated:
t_ticks = np.array([-1,0,1,2])

x = np.arange(1,fixed_rank+1)

#Plotting the R2 of the original and DOC latents:
fig,axes = plt.subplots(2,2,figsize=(7.3*cm,6.3*cm),sharey=False,sharex=False)

plotcontrasts = np.array([[1,2],[1,3]])
ax = axes[0]

clrs = np.array([['grey','darkgrey'],['grey','red']])

for idata,data in enumerate([R2_ranks_orig,R2_ranks_doc]):
    for icontrast,contrast in enumerate(plotcontrasts):
        if params['direction'] =='FF' and icontrast == 0:
            figlabels = ['V1$_{ND1}$','V1$_{ND2}$']
        elif params['direction'] =='FF' and icontrast == 1:
            figlabels = ['V1$_{ND1}$','V1$_{PM}$']
        elif params['direction'] =='FB' and icontrast == 0:
            figlabels = ['PM$_{ND1}$','PM$_{ND2}$']
        elif params['direction'] =='FB' and icontrast == 1:
            figlabels = ['PM$_{ND1}$','PM$_{V1}$']

        ax = axes[icontrast,idata]
        handles = []

        for ial in range(2):
            datatoplot = np.nanmean(data[contrast[ial]][icontrast],axis=(3)).reshape(params['nSessions']*params['nStim'],fixed_rank) #average across modelfits
            handles.append(shaded_error(x,datatoplot,color=clrs[icontrast,ial],error='ci95',alpha=0.3,ax=ax))

        if idata == 1:
            ial = 0
            docdim = 0
            datatotest = np.nanmean(data[np.ix_([contrast[ial]],[icontrast],np.arange(params['nSessions']),range(params['nStim']),[docdim],range(params['nmodelfits']))]
                                    ,axis=(0,1,4,5)).flatten()
            tval,pval = stats.ttest_1samp(datatotest,0,nan_policy='omit')[:2]
            ax.text(docdim/fixed_rank,0.2,'p=%1.3f' % pval,transform=ax.transAxes)
            print('DOC latent %d is significantly nonzero for %s population (t=%1.1f,p=%1.2e)' % (docdim+1,sourcearealabelpairs[contrast[ial]-1],tval,pval))
            
            ial = 1
            docdim = fixed_rank-1
            datatotest = np.nanmean(data[np.ix_([contrast[ial]],[icontrast],np.arange(params['nSessions']),range(params['nStim']),[docdim],range(params['nmodelfits']))]
                                    ,axis=(0,1,4,5)).flatten()
            tval,pval = stats.ttest_1samp(datatotest,0,nan_policy='omit')[:2]
            ax.text(docdim/fixed_rank,0.2,'p=%1.3f' % stats.ttest_1samp(datatotest,0,nan_policy='omit')[1],transform=ax.transAxes)
            print('DOC latent %d is significantly nonzero for %s population (t=%1.1f,p=%1.2e)' % (docdim+1,sourcearealabelpairs[contrast[ial]-1],tval,pval))

        ax.legend(handles=handles,labels=figlabels,frameon=False,loc='upper center',fontsize=7)
        my_legend_strip(ax)
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_xlabel('latent')
        ax.set_ylabel('performance')
        ax_nticks(ax,4)
        ax.set_xticks(x)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,figdir,'DOC_latents_r2_%s' % params['direction'])

#%% Plotting the R2 for the first DOC latent as a scatter plot: 
fig, axes = plt.subplots(1,1,figsize=(3.5*cm,3.5*cm))
ax = axes

plotcontrasts = np.array([[1,2],[1,3]])
clrs = ['grey','red']
docdim = 0

datatotest = np.empty((2,params['nSessions']*params['nStim']))
for icontrast,contrast in enumerate(plotcontrasts):
    xdatatoplot = np.nanmean(data[np.ix_([contrast[0]],[icontrast],np.arange(params['nSessions']),range(params['nStim']),[docdim],range(params['nmodelfits']))]
                                  ,axis=(0,1,4,5)).flatten()
    ydatatoplot = np.nanmean(data[np.ix_([contrast[1]],[icontrast],np.arange(params['nSessions']),range(params['nStim']),[docdim],range(params['nmodelfits']))]
                                  ,axis=(0,1,4,5)).flatten()
    datatotest[icontrast,:] = ydatatoplot
    # xdatatoplot = np.nanmean(data[np.ix_([contrast[0]],[icontrast],np.arange(nSessions),range(params['nStim']),[docdim],range(params['nmodelfits']))]
    #                               ,axis=(0,1,3,4,5))
    # ydatatoplot = np.nanmean(data[np.ix_([contrast[1]],[icontrast],np.arange(nSessions),range(params['nStim']),[docdim],range(params['nmodelfits']))]
    #                               ,axis=(0,1,3,4,5))

    plotclip = my_ceil(np.nanpercentile([xdatatoplot,ydatatoplot],99),2)
    ax.scatter(xdatatoplot,ydatatoplot,color=clrs[icontrast],marker='.',s=8,alpha=0.5)


    # add_paired_wilcoxon_results(ax,xdatatoplot,ydatatoplot,pos=[0.8,0.4+icontrast*0.1],color=clrs[icontrast],fontsize=6)
add_paired_wilcoxon_results(ax,datatotest[0],datatotest[1],pos=[0.8,0.4+icontrast*0.1],color='k',fontsize=6)
# ax.legend(['V1$_{ND2}$ (vs V1$_{ND1}$)','V1$_{PM}$ (vs V1$_{ND1}$)'],loc='lower right',fontsize=6)
ax.legend(['V1$_{ND2}$','V1$_{PM}$'],loc='lower right',fontsize=6)
my_legend_strip(ax)
ax.plot([0,1],[0,1],color='k',linestyle='--',linewidth=0.5)
# add_paired_wilcoxon_results(ax,R2_cv[0,:],R2_cv[1,:],pos=[0.7,0.1])
ax.set_title('DOC vs V1$_{ND1}$\nlatent %d' % docdim)
ax.set_xlim([0,plotclip])
ax.set_ylim([0,plotclip])
ax.set_xlabel('performance')
ax.set_ylabel('performance')

# ax.set_xlim([0,my_ceil(np.nanmax(R2_cv),2)])
# ax.set_ylim([0,my_ceil(np.nanmax(R2_cv),2)])
ax.set_xticks(np.linspace(0,ax.get_xlim()[1],3))
ax.set_xticklabels(np.linspace(0,ax.get_xlim()[1],3),color='k')
ax.set_yticks(np.linspace(0,ax.get_ylim()[1],3))
ax.set_yticklabels(np.linspace(0,ax.get_ylim()[1],3),color='k')
sns.despine(fig=fig, top=True, right=True, offset = 2)
# my_savefig(fig,figdir,'DOC_Scatter_R2_latent_%d_%s' % (docdim,params['direction']))


#%% 











#%% Plotting the R2 of the latents across time:
# fig,axes = plt.subplots(3,4,figsize=(13,10),sharex=True,sharey=False)
fig,axes = plt.subplots(ncontrasts,fixed_rank,figsize=(12*cm,6*cm),sharex=True,sharey=True)
ax = axes
# plotcontrast = [1,2,3]
clrs = ['grey','black','red']
for icontrast in range(ncontrasts):
    for r in range(fixed_rank):
        ax = axes[icontrast,r]
        handles = []
        for ial in range(narealabelpairs):
            datatoplot = np.nanmean(R2_ranks_doc_t[:,icontrast],axis=(1,2,5))
            # datatoplot = np.nanmean(R2_ranks_doc_t[:,icontrast,0],axis=(1,4))
            # handles.append(ax.plot(t_axis[idx_resp],datatoplot[ial+1,:,r],color=clrs[ial])[0])
            handles.append(ax.plot(t_axis,datatoplot[ial+1,:,r],color=clrs[ial])[0])
        thickness = ax.get_ylim()[1]/20
        ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
            # ax.lege
        if icontrast == 0 and r == fixed_rank-1: 
            ax.legend(handles=handles,labels=list(arealabeled_to_figlabels(sourcearealabelpairs)),loc='best',bbox_to_anchor=(0.65, 0.5), bbox_transform=ax.transAxes)
            my_legend_strip(ax)
        if r == 0:
            ax.set_ylabel('R$^{2}$')
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')

# plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,figdir,'DOC_latents_r2_across_time_%s' % params['direction'])
# my_savefig(fig,figdir,'DOC_latents_r2_across_time_10dim')

#%% Plotting the R2 ratio of the latents across time:
fig,axes = plt.subplots(ncontrasts,fixed_rank,figsize=(12*cm,6*cm),sharex=True,sharey=True)
ax = axes
noise_constant = 1e-3
if params['direction'] =='FF':
    figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
elif params['direction'] =='FB':
    figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']

plotcontrasts = np.array([[1,2],[1,3]])
clrs = ['grey','red']
for icontrast in range(ncontrasts):
    for r in range(fixed_rank):
        ax = axes[icontrast,r]
        handles = []
        for iplotcontrast,plotcontrast in enumerate(plotcontrasts):

            data1 = np.nanmean(R2_ranks_doc_t[plotcontrast[0],icontrast],axis=(1,4))
            data2 = np.nanmean(R2_ranks_doc_t[plotcontrast[1],icontrast],axis=(1,4))
            datatoplot = (data2+noise_constant) / (data1+noise_constant)
            # handles.append(shaded_error(t_axis[idx_resp],datatoplot[:,:,r],color=clrs[iplotcontrast],ax=ax))
            handles.append(shaded_error(t_axis,datatoplot[:,:,r],color=clrs[iplotcontrast],ax=ax))

        thickness = ax.get_ylim()[1]/20
        ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
        ax.axhline(y=1,color='grey',linestyle='--')
        if icontrast == 0 and r == fixed_rank-1: 
            ax.legend(handles=handles,labels=figlabels,loc='best',bbox_to_anchor=(0.65, 0.5), bbox_transform=ax.transAxes)
            my_legend_strip(ax)
        if r == 0:
            ax.set_ylabel('R$^{2}$ ratio')
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (sec)')
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,figdir,'DOC_latents_r2_ratio_across_time_%s' % params['direction'])
# my_savefig(fig,figdir,'DOC_latents_r2_across_time_10dim')

#%%

#%% Plotting the R2 ratio of the latents across time:
# fig,axes = plt.subplots(ncontrasts,fixed_rank,figsize=(12*cm,6*cm),sharex=True,sharey=True)
ncomparisons = 4
# noise_constant = 1e-3
if params['direction'] =='FF':
    figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
elif params['direction'] =='FB':
    figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']
# figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']

if params['direction'] =='FF':
    figlabels = ['V1$_{ND2}$ in DOC vs V1$_{ND1}$',
                 'V1$_{PM}$ in DOC vs V1$_{ND1}$']
elif params['direction'] =='FB':
    figlabels = ['PM$_{ND2}$ in DOC vs PM$_{ND1}$',
                 'PM$_{V1}$ in DOC vs PM$_{ND1}$']
    
paneltitles = ['Max','Min']

clrs = ['grey','red']

# datatoplot = np.nanmean(R2_ranks_doc_t[plotcontrasts[icontrast,1],plotcontrasts[icontrast,1]-2],axis=(4))
# datatoplot = np.reshape(datatoplot,(nSessions*params['nStim'],nT,fixed_rank),order='C')
 
fig,axes = plt.subplots(1,2,figsize=(12*cm,6*cm),sharex=True,sharey=True)
for ir,r in enumerate([0,fixed_rank-1]):

# fig,axes = plt.subplots(1,fixed_rank,figsize=(12*cm,6*cm),sharex=True,sharey=True)
# paneltitles = ['Optimal','Suboptimal','Medium','','Worst']
# for ir,r in enumerate(np.arange(fixed_rank)):
    ax = axes[ir]
    handles = []
    
    datatotest = np.nanmean(R2_ranks_doc_t[np.ix_([2,3],[0,1],np.arange(nSessions),np.arange(params['nStim']),
                                                  np.arange(nT),[r],np.arange(params['nmodelfits']))],axis=(-1))
    
    datatotest = np.squeeze(datatotest)
    datatotest = np.reshape(datatotest,(2,2,nSessions*params['nStim'],nT),order='C')
    datatotest[datatotest<0] = np.nan

    for icontrast in range(ncontrasts):
        # datatoplot = np.nanmean(R2_ranks_doc_t[icontrast+2,icontrast,:,:,:,r],axis=(-1))
        # datatoplot = np.reshape(datatoplot,(nSessions*params['nStim'],nT),order='C')

        # datatoplot = np.reshape(datatotest[icontrast][icontrast],(nSessions*params['nStim'],nT),order='C')
        datatoplot = datatotest[icontrast][icontrast]

        handles.append(shaded_error(t_axis,datatoplot,error='sem',color=clrs[icontrast],ax=ax))
    
    idx_FF = np.where((t_axis > 0) & (t_axis < 1.3))[0]
    # h,p = stats.wilcoxon(datatotest[0,0,:,idx_FF],datatotest[1,1,:,idx_FF],nan_policy='omit')
    h,p = stats.wilcoxon(np.nanmean(datatotest[0][0][:,idx_FF],axis=1),
                         np.nanmean(datatotest[1][1][:,idx_FF],axis=1),nan_policy='omit')
    # plt.scatter(np.nanmean(datatotest[0][0][:,idx_FF],axis=1),
    #             np.nanmean(datatotest[1][1][:,idx_FF],axis=1),color='k',s=10)
    # plt.plot([0,0.004],[0,0.004],color='k',linestyle='--')
    # print(p)
    # if p < 0.05:
    #     ax.text(0.3,0.01,get_sig_asterisks(p),color='k')
    
    for it,t in enumerate(t_axis):
        # h,p = stats.ttest_rel(datatotest[0,0,:,it],datatotest[1,1,:,it],nan_policy='omit')
        h,p = stats.wilcoxon(datatotest[0,0,:,it],datatotest[1,1,:,it],nan_policy='omit')
        p = np.clip(p*ncomparisons,0,1) # correct for multiple comparisons
        if p < 0.05:
            # ax.text(t,0.01,get_sig_asterisks(p),color='k',fontsize=11,fontweight='bold')
            ax.text(t,0.01,'*',color='k',fontsize=11,fontweight='bold')

    if ir == 1:
        ax.legend(handles=handles,labels=figlabels,loc='best',bbox_to_anchor=(0.65, 0.5), bbox_transform=ax.transAxes)
        my_legend_strip(ax)
    ax.set_title(paneltitles[ir])
    thickness = ax.get_ylim()[1]/20
    ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
    # ax.axhline(y=1,color='grey',linestyle='--')
    # if icontrast == 0 and r == fixed_rank-1: 

    if r == 0:
        ax.set_ylabel('R$^{2}$ ratio')
        ax.set_ylabel('R$^{2}$')
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
# ax.set_xticklabels(t_ticks)
ax.set_xlim([-1,2])
ax.set_xlabel('Time (sec)')
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,figdir,'DOC_latents_r2_ratio_across_time_summary_%s' % params['direction'])

#%% Plotting the R2 of the latents across time:
# # fig,axes = plt.subplots(3,4,figsize=(13,10),sharex=True,sharey=False)
# fig,axes = plt.subplots(ncontrasts,fixed_rank,figsize=(12*cm,6*cm),sharex=True,sharey=True)
# ax = axes
# # plotcontrast = [1,2,3]
# clrs = ['grey','black','red']
# for icontrast in range(ncontrasts):
#     for r in range(fixed_rank):
#         ax = axes[icontrast,r]
#         handles = []
#         for ial in range(narealabelpairs):
#             datatoplot = np.nanmean(R2_ranks_doc_t[:,icontrast],axis=(1,2,5))
#             datatoplot = np.nanmean(R2_ranks_doc_t[:,icontrast,0],axis=(1,4))
#             handles.append(ax.plot(t_axis[idx_resp],datatoplot[ial+1,:,r],color=clrs[ial])[0])
#         thickness = ax.get_ylim()[1]/20
#         ax.fill_between([0,0.75], 0 - thickness/2, 0 + thickness/2, color='k', alpha=1)
#             # ax.lege
#         if icontrast == 0 and r == fixed_rank-1: 
#             ax.legend(handles=handles,labels=list(arealabeled_to_figlabels(sourcearealabelpairs)),loc='best',bbox_to_anchor=(0.65, 0.5), bbox_transform=ax.transAxes)
#             my_legend_strip(ax)
#         if r == 0:
#             ax.set_ylabel('R$^{2}$')
# ax_nticks(ax,3)
# ax.set_xticks(t_ticks)
# ax.set_xticklabels(t_ticks)
# ax.set_xlabel('Time (sec)')

# # plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True, offset = 3)
# figdir = os.path.join(params['figdir'],'RRR','DOC')
# my_savefig(fig,figdir,'DOC_latents_r2_across_time')

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