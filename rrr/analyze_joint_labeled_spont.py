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
from scipy import stats
import pickle

from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.params import load_params

params = load_params()
resultdir = params['resultdir']
figdir = os.path.join(params['figdir'],'RRR','Spontaneous','Labeling')

#%%  
version = 'FF_original'
FF_filename = 'RRR_Joint_labeled_FF_original_spont_2026-05-19_15-59-37'

version = 'FB_original'
FB_filename = 'RRR_Joint_labeled_FB_original_spont_2026-05-19_15-52-56'

#%% Load the data:
data = np.load(os.path.join(resultdir,FF_filename + '.npz'),allow_pickle=True)
for key in data.keys():
    exec(key+'_FF=data[key]')

with open(os.path.join(resultdir,FF_filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

data = np.load(os.path.join(resultdir,FB_filename + '.npz'),allow_pickle=True)
for key in data.keys():
    exec(key+'_FB=data[key]')

with open(os.path.join(resultdir,FB_filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

nmodelfits = params['nmodelfits']
Nsub = params['Nsub']

clrs_arealabelpairs = ['grey','grey','red']
narealabelpairs = 3

#%% Show an example session (FF)
clrs_arealabelpairs = ['grey','grey','red']
narealabelpairs = 3
nrankstoplot = 10
fig, ax = plt.subplots(1,1,figsize=(3.8*cm,3.7*cm))
ises = 4
handles = []
for iapl,apl in enumerate(sourcearealabelpairs_FF[1:]):
    ymeantoplot = np.nanmean(R2_ranks_FF[iapl+2][ises],axis=(0,2,3))
    yerrortoplot = np.nanstd(R2_ranks_FF[iapl+2][ises],axis=(0,2,3)) / np.sqrt(nmodelfits*params['nStim'])
    handles.append(shaded_error(np.arange(params['nranks']),ymeantoplot,yerrortoplot,ax=ax,
                                color=clrs_arealabelpairs[iapl+1],linewidth=1,alpha=0.3))

leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs_FF[1:]),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('rank')
ax.set_ylabel(r'performance (R$^2$)')
ax.set_title('FF')
ax.set_xticks(np.arange(params['nranks'])[::3]+1)
ax.set_xlim([0,nrankstoplot])

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
my_savefig(fig,figdir,'RRR_joint_cvR2_labunl_FF_ExampleSession%d' % (ises))

#%% Show an example session (FB)
fig, ax = plt.subplots(1,1,figsize=(3.8*cm,3.7*cm))
ises = 10
handles = []
for iapl,apl in enumerate(sourcearealabelpairs_FB[1:]):
    ymeantoplot = np.nanmean(R2_ranks_FB[iapl+2][ises],axis=(0,2,3))
    yerrortoplot = np.nanstd(R2_ranks_FB[iapl+2][ises],axis=(0,2,3)) / np.sqrt(nmodelfits*params['nStim'])
    handles.append(shaded_error(np.arange(params['nranks']),ymeantoplot,yerrortoplot,ax=ax,
                                color=clrs_arealabelpairs[iapl+1],linewidth=1,alpha=0.3))

leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs_FB[1:]),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('rank')
ax.set_ylabel(r'performance (R$^2$)')
ax.set_title('FB')
ax.set_xticks(np.arange(params['nranks'])[::3]+1)
ax.set_xlim([0,nrankstoplot])

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
my_savefig(fig,figdir,'RRR_joint_cvR2_labunl_FB_ExampleSession%d' % (ises))


# #%% Show the mean across sessions:
# clrs_arealabelpairs = ['grey','grey','red']

# nrankstoplot = 10
# xposrank = 8
# idxs = np.array([1,3])

# fig,axes = plt.subplots(1,2,figsize=(7*cm,3.5*cm),sharex=True,sharey=True)
# for idirec,(direc,optim_rank,R2_cv,R2_ranks,alps) in enumerate(zip(['FF','FB'],[optim_rank_FF,optim_rank_FB],[R2_cv_FF,R2_cv_FB],
#                                                                  [R2_ranks_FF,R2_ranks_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
#     # R2_toplot = np.reshape(data,(narealabelpairs+1,params['nSessions']*params['nStim'],params['nT']))

#     meanranks = np.nanmean(optim_rank,axis=(-1,-2))
#     meanR2 = np.nanmean(R2_cv,axis=(-1,-2))
#     alps = np.array(alps)
#     ax = axes[idirec]
#     handles = []
#     ydata = np.nanmean(R2_ranks[idxs[0]],axis=(3,4))
#     ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
#     handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
#                                 color=clrs_arealabelpairs[idxs[0]-1],alpha=0.3))
#     ydata = np.nanmean(R2_ranks[idxs[1]],axis=(3,4))
#     ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
#     handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
#                                 color=clrs_arealabelpairs[idxs[1]-1],alpha=0.3))
#     for idx in idxs:
#         ax.plot(meanranks[idx],meanR2[idx]+0.005,color=clrs_arealabelpairs[idx-1],marker='v',markersize=5)

#     leg = ax.legend(handles,arealabeled_to_figlabels(alps[idxs-1]),frameon=False)
#     my_legend_strip(ax)
#     ax.set_xlabel('Rank')
#     if idirec == 0:
#         ax.set_ylabel(r'Cross-validated R$^2$')

#     x = optim_rank[idxs[0],:]
#     y = optim_rank[idxs[1],:]
#     nas = np.logical_or(np.isnan(x), np.isnan(y))
#     t,p = ttest_rel(x[~nas], y[~nas])
#     print('Paired t-test (rank): p=%.3f' % (p))
#     ax.plot(meanranks[idxs],np.repeat(np.nanmean(meanR2[idxs]),2)+0.007,linestyle='-',color='k',linewidth=2)
#     ax.text(np.nanmean(meanranks),np.nanmean(meanR2[idxs])+0.009,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

#     x = R2_cv[idxs[0],:]
#     y = R2_cv[idxs[1],:]
#     nas = np.logical_or(np.isnan(x), np.isnan(y))
#     t,p = ttest_rel(x[~nas], y[~nas])
#     print('Paired t-test (performance): p=%.3f' % (p))
#     ax.plot([xposrank,xposrank],meanR2[idxs],linestyle='-',color='k',linewidth=2)
#     ax.text(xposrank+0.5,np.nanmean(meanR2[idxs])+0.005,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

#     ax.set_xticks(np.arange(params['nranks'])[::3]+1)
#     ax.set_xlim([0,nrankstoplot])
#     ax.set_title(direc)
# plt.tight_layout()
# sns.despine(fig=fig,trim=False,top=True,right=True)
# my_savefig(fig,figdir,'RRR_joint_R2_labunl_%dsessions_spont' % params['nSessions'])


#%% 
fig, axes = plt.subplots(1,1,figsize=(3.1*cm,3.9*cm))
contrasts   = np.array([[3,1],[2,1]])
contrasts   = np.array([[3,2],[1,2]])
clrs        = ['red','grey']
ntests = len(contrasts) * 2 #number of tests for bonferroni correction (FF/FB, 2 contrasts)
ax = axes
handles = []
noise_constant = 0
for idirec,(direc,optim_rank,R2_cv,R2_ranks,alps) in enumerate(zip(['FF','FB'],[optim_rank_FF,optim_rank_FB],[R2_cv_FF,R2_cv_FB],
                                                                 [R2_ranks_FF,R2_ranks_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
    # R2_cv[optim_rank==0] = np.nan
    # R2_cv[optim_rank<=1] = np.nan
    for icontrast,contrast in enumerate(contrasts):
        ratiodata = ((R2_cv[contrast[0]]+noise_constant) / (R2_cv[contrast[1]]+noise_constant)).flatten()
        ratiodata = np.nan_to_num(ratiodata,nan=np.nan,posinf=np.nan,neginf=np.nan)
        handle = ax.errorbar(x=idirec,y=np.nanmedian(ratiodata),yerr=2*np.nanstd(ratiodata)/np.sqrt(len(ratiodata)),
                                color=clrs[icontrast],capsize=3,elinewidth=1,marker='o',markersize=5)[0]
        h,p = stats.wilcoxon(ratiodata-1,nan_policy='omit')
        p = np.clip(p * ntests, 0, 1)
        if p>0.0001:
            print('%s vs %s: p = %.2f (Wilcoxon signed rank test, Bonferroni-corrected)' % (alps[contrast[0]-1],
                                            alps[contrast[1]-1],p))
        else:
            print('%s vs %s: p = %1.1e (Wilcoxon signed rank test, Bonferroni-corrected)' % (alps[contrast[0]-1],
                        alps[contrast[1]-1],p))

        if p < 0.05:
            ax.annotate(get_sig_asterisks(p),xy=(idirec,np.nanmean(ratiodata)+0.02),fontsize=8,ha='center',color='red')
        if idirec == 0:
            handles.append(handle)
ax.axhline(y=1,color='k',linestyle='--',linewidth=1)
# ax.legend(handles,['V1$_{PM}$/V1$_{ND}$','V1$_{ND}$/V1$_{ND}$'],frameon=False)
ax.set_xlim([-0.3,1.3])
ax.legend(handles,['XX/ND','ND/ND'],frameon=False)
my_legend_strip(ax)
# ax_nticks(ax,6)
ax.set_xticks([0,1],['FF','FB'])
ax.set_yticks([1,1.2,1.4])
ax.set_ylabel(r'performance ratio')

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True,offset=2)
# my_savefig(fig,figdir,'perf_ratio_labunl_spont_%dsessions' % params['nSessions'])


#%% Show the mean R2_cv across sessions:
clrs_arealabelpairs = ['grey','grey','red']
normalize = True
idxs = np.array([1,3])
fig,axes = plt.subplots(1,2,figsize=(6*cm,3.8*cm),sharex=False,sharey=False)
for idirec,(direc,optim_rank,R2_cv,R2_ranks,alps) in enumerate(zip(['FF','FB'],[optim_rank_FF,optim_rank_FB],[R2_cv_FF,R2_cv_FB],
                                                                 [R2_ranks_FF,R2_ranks_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
    ax = axes[idirec]
    if normalize:
        data = R2_cv[idxs]
        data -= data[0][np.newaxis,:,:]
        for iidx,idx in enumerate(idxs):
            sns.stripplot(x=iidx,y=data[iidx].flatten(),color=clrs_arealabelpairs[idx-1],ax=ax,size=3,alpha=0.2)
            sns.stripplot(x=iidx,y=np.nanmean(data[iidx],axis=-1),color=clrs_arealabelpairs[idx-1],ax=ax,size=4,alpha=1)

            # sns.stripplot(x=iidx,y=R2_cv[idx].flatten(),color=clrs_arealabelpairs[idx-1],ax=ax,size=3,alpha=0.2)
            # sns.stripplot(x=iidx,y=np.nanmean(R2_cv[idx],axis=-1),color=clrs_arealabelpairs[idx-1],ax=ax,size=4,alpha=1)
        # ax.plot(np.row_stack((np.zeros(params['nSessions']*params['nStim']),np.ones(params['nSessions']*params['nStim']))),
                # np.row_stack((R2_cv[idxs[0]].flatten(),R2_cv[idxs[1]].flatten())),color='k',alpha=0.3,linewidth=0.5)
        ax.plot(np.row_stack((np.zeros(params['nSessions']),np.ones(params['nSessions']))),
                np.row_stack((np.nanmean(data[0],axis=-1),np.nanmean(data[1],axis=-1))),color='k',alpha=0.3,linewidth=0.5)
        
    else:

        for iidx,idx in enumerate(idxs):
            sns.stripplot(x=iidx,y=R2_cv[idx].flatten(),color=clrs_arealabelpairs[idx-1],ax=ax,size=3,alpha=0.2)
            sns.stripplot(x=iidx,y=np.nanmean(R2_cv[idx],axis=-1),color=clrs_arealabelpairs[idx-1],ax=ax,size=4,alpha=1)
        # ax.plot(np.row_stack((np.zeros(params['nSessions']*params['nStim']),np.ones(params['nSessions']*params['nStim']))),
                # np.row_stack((R2_cv[idxs[0]].flatten(),R2_cv[idxs[1]].flatten())),color='k',alpha=0.3,linewidth=0.5)
        ax.plot(np.row_stack((np.zeros(params['nSessions']),np.ones(params['nSessions']))),
                np.row_stack((np.nanmean(R2_cv[idxs[0]],axis=-1),np.nanmean(R2_cv[idxs[1]],axis=-1))),color='k',alpha=0.3,linewidth=0.5)
        


    ax.set_xticks([0,1],alps[idxs-1])
    if idirec == 0:
        ax.set_ylabel(r'Cross-validated R$^2$')

    x = R2_cv[idxs[0],:]
    y = R2_cv[idxs[1],:]
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    t,p = ttest_rel(x[~nas], y[~nas])
    print('Paired t-test (R2): p=%.3f' % (p))

    add_paired_ttest_results(ax,R2_cv[idxs[0]].flatten(),R2_cv[idxs[1]].flatten(),color='k',pos=[0.5,0.9])
    ax.set_title(direc)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=2)
# my_savefig(fig,figdir,'RRR_joint_R2cv_%dsessions_spont' % params['nSessions'])


#%% Identify which dimensions are particularly enhanced in labeled cells:
noise_constant = 1e-4
diffmetric = 'ratio' #'difference'
clipval = 1e-4
pthr = 0.05 / (params['nranks']-1) #Bonferroni correction for multiple comparisons across ranks
pthr = 0.05 / 5 #Bonferroni correction for multiple comparisons across ranks
plotcontrasts = np.array([[1,2],[1,3]])
clrs = ['grey','red']
fig,axes = plt.subplots(1,2,figsize=(7*cm,3.5*cm),sharex=True,sharey=True)
for idirec,(direc,optim_rank,R2_cv,R2_ranks,alps) in enumerate(zip(['FF','FB'],[optim_rank_FF,optim_rank_FB],[R2_cv_FF,R2_cv_FB],
                                                                 [R2_ranks_FF,R2_ranks_FB],[sourcearealabelpairs_FF,sourcearealabelpairs_FB])):
    # R2_toplot = np.reshape(data,(narealabelpairs+1,params['nSessions']*params['nStim'],params['nT']))

    R2_toplot = np.diff(R2_ranks,axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
    R2_toplot = np.nanmean(R2_toplot,axis=(4,5))
    R2_toplot = np.reshape(R2_toplot,(narealabelpairs+1,params['nSessions']*params['nStim'],params['nranks']-1))
    # R2_toplot = np.clip(R2_toplot,clipval,np.inf) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
    R2_toplot = np.clip(R2_toplot,clipval,np.inf) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
    # R2_toplot[R2_toplot < 0] = np.nan

    if direc == 'FF': 
        figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
    elif direc == 'FB': 
        figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']

    ax = axes[idirec]
    handles = []
    for iplotcontrast,plotcontrast in enumerate(plotcontrasts):
        if diffmetric == 'ratio':
            R2_ratiodata = (R2_toplot[plotcontrast[1],:,:]+noise_constant) / (R2_toplot[plotcontrast[0],:,:]+noise_constant) #add a small constant to avoid division by zero

            # ymeantoplot = np.nanmean(data[2],axis=(0,1,3)) / (np.nanmean(data[1],axis=(0,1,3))+1e-3)
            # yerrortoplot = np.nanstd(data[2],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
            # ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)
            # yerrortoplot = (np.nanstd(data[2],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1],axis=(0,1,3))+noise_constant) / np.sqrt(params['nSessions']*nmodelfits)
        
            # ydata = (np.nanmean(data[2],axis=(3))+noise_constant) / (np.nanmean(data[1],axis=(3))+noise_constant)
            # ymeantoplot = np.nanmean(ydata,axis=(0,1))
            # yerrortoplot = np.nanstd(ydata,axis=(0,1)) / np.sqrt(params['nSessions']*params['nStim'])

        elif diffmetric == 'difference':
            ymeantoplot = np.nanmean(data[2] - data[1],axis=(0,1,3))
            yerrortoplot = np.nanstd(data[2] - data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
        handles.append(shaded_error(np.arange(params['nranks']-1)+1,R2_ratiodata,error='ci95',ax=ax,color=clrs[iplotcontrast],alpha=0.3))

        for r in range(nrankstoplot):
            # ydata = (data[2,:,:,r]+noise_constant) /  (data[1,:,:,r]+noise_constant)
            # ydata = (np.nanmean(data[2,:,:,r],axis=2)+noise_constant) /  (np.nanmean(data[1,:,:,r],axis=2)+noise_constant)
            # ydata = (np.nanmean(data[2,:,:,r],axis=2)) /  (np.nanmean(data[1,:,:,r],axis=2))
            # ydata = ydata.flatten()
            ydata = R2_ratiodata[:,r]
            h,p = stats.ttest_1samp(ydata,1,nan_policy='omit')
            if p<pthr:
                print('Rank %d is significantly enhanced in labeled cells (p=%.3f)' % (r+1,p))
                ax.text(r,1.05,get_sig_asterisks(p),ha='center',va='bottom',color=clrs[iplotcontrast],fontsize=10)

    # ax.legend(handles,['V1$_{ND}$/V1$_{ND}$','V1$_{PM}$/V1$_{ND}$'],frameon=False)
    ax.legend(handles,figlabels,frameon=False)
    my_legend_strip(ax)
    ax_nticks(ax,4)
    ax.set_xticks(np.arange(nrankstoplot)[::3]+1)
    ax.set_xlim([1,nrankstoplot])
    # ax.set_ylim([0.9,1.5])
    ax.set_xlabel('dimension')
    ax.set_ylabel('R$^{2}$ %s' % diffmetric)
    if diffmetric == 'ratio':
        ax.axhline(y=1,color='grey',linestyle='--')
    elif diffmetric == 'difference':
        ax.axhline(y=0,color='grey',linestyle='--')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=2)
my_savefig(fig,figdir,'RRR_joint_R2_ratio_%dsessions_spont' % params['nSessions'])


