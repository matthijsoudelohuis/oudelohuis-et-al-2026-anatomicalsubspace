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
figdir = os.path.join(params['figdir'],'RRR','Labeling','Separate')

#%% Run block for FF 
version = 'FF_original'
filename = 'RRR_Separate_labeled_FF_original_2026-05-21_17-17-36'
exampleses = 12

#%% For FB:
version = 'FB_original'
filename = 'RRR_Separate_labeled_FB_original_2026-05-22_13-16-01'
exampleses = 1

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

#%% Plot the performance across sessions as a function of rank:
alx1 = arealabeled_to_figlabels(sourcearealabelpairs[0].split('-')[0])
alx2 = arealabeled_to_figlabels(sourcearealabelpairs[1].split('-')[0])

R2_cvtoplot              = np.reshape(R2_cv,(narealabelpairs,params['nSessions']*params['nStim']))
optim_ranktoplot         = np.reshape(optim_rank,(narealabelpairs,params['nSessions']*params['nStim']))
R2_rankstoplot           = np.reshape(R2_ranks,(narealabelpairs,params['nSessions']*params['nStim'],params['nranks'],nmodelfits,params['kfold']))

nranks              = R2_rankstoplot.shape[2]
nSessions           = R2_cvtoplot.shape[1]

# filter only sessions for which both populations are present:
idx_ses = np.all(~np.isnan(R2_rankstoplot),axis=(0,2,3,4))
R2_rankstoplot[:,~idx_ses,:,:,:] = np.nan
R2_cvtoplot[:,~idx_ses] = np.nan
optim_ranktoplot[:,~idx_ses] = np.nan

#%% Plotting:
clrs_arealabels = ['grey','red']
fig,axes = plt.subplots(1,1,figsize=(3*cm,3.9*cm))
ax = axes
data = R2_cvtoplot

for ialp,[ial0,ial1] in enumerate([[0,1]]):
    xscatter = np.random.randn(nSessions)*0.1
    ax.scatter(xscatter+ial0,data[ial0,:].flatten(),s=5,color=clrs_arealabels[ial0],marker='.')
    ax.scatter(xscatter+ial1,data[ial1,:].flatten(),s=5,color=clrs_arealabels[ial1],marker='.')
    ax.plot([xscatter+ial0,xscatter+ial1],[data[ial0,:],data[ial1,:]],color='k',marker='',markersize=0,linewidth=0.1)
    ax.plot(ial0,np.nanmean(data[ial0,:]),color=clrs_arealabels[ial0],marker='o',markersize=5)
    ax.plot(ial1,np.nanmean(data[ial1,:]),color=clrs_arealabels[ial1],marker='o',markersize=5)

ax.set_ylim([0,my_ceil(ax.get_ylim()[1],2)])
ax_nticks(ax,4)
ax.set_ylabel('performance (R$^2$)')
ax.set_xticks([0,1],[alx1,alx2])
add_paired_ttest_results(ax,data[0,:],data[1,:],pos=[0.5,0.9])

# nas = np.logical_or(np.isnan(data[0,:]), np.isnan(data[1,:]))
# t,p = ttest_rel(data[0,:][~nas], data[1,:][~nas])

# print('Paired t-test: t=%1.2f,p=%.3e' % (t,p))
# sns.despine(fig,top=True,right=True,offset=2)
# my_savefig(fig,figdir,'perf_V1PM_arealabeled_paired_%s' % params['direction'])

#%%
fig,axes = plt.subplots(1,1,figsize=(4*cm,3.9*cm))
ax = axes
ax.hist(R2_cv[0].flatten(),bins=np.arange(0,0.1,0.005),color='grey',alpha=0.6)
ax.hist(R2_cv[1].flatten(),bins=np.arange(0,0.1,0.005),color='red',alpha=0.6)
ax.plot(np.nanmean(R2_cv[0].flatten()),40,'v',color='grey')
ax.plot(np.nanmean(R2_cv[1].flatten()),40,'v',color='red')
ax.set_xlabel('performance (R2)')
ax.set_ylabel('# datasets')
# ax.axvline(1,linestyle='--',linewidth=1,color='black')
sns.despine(fig,top=True,right=True,offset=2)
my_savefig(fig,figdir,'perf_hist_%s' % params['direction'])

#%% 
# perf_ratio = (np.nanmean(R2_cv,axis=2)[1].flatten() / np.nanmean(R2_cv,axis=2)[0].flatten())
# perf_ratio = (np.nanmean(R2_cv,axis=1)[1].flatten() / np.nanmean(R2_cv,axis=1)[0].flatten())
perf_ratio = (R2_cv[1].flatten() / R2_cv[0].flatten())

fig,axes = plt.subplots(1,1,figsize=(4*cm,3.9*cm))
ax = axes
ax.hist(perf_ratio,bins=np.arange(0.5,1.8,0.1),color='red',alpha=0.6)
ax.plot(np.nanmean(perf_ratio),40,'v',color='red')
ax.set_xlabel('performance ratio')
ax.set_ylabel('# datasets')
ax.axvline(1,linestyle='--',linewidth=1,color='black')
sns.despine(fig,top=True,right=True,offset=2)
my_savefig(fig,figdir,'perf_ratio_hist_%s' % params['direction'])

# t,p = stats.ttest_1samp(perf_ratio[~np.isnan(perf_ratio)],1)
# print('1 sample t-test: t=%1.2f,p=%.3e' % (t,p))

t,p = stats.wilcoxon(perf_ratio[~np.isnan(perf_ratio)]-1)
print('p=%.3e, wilcoxon test' % p)

#%% How much more predictive are labeled neurons on average:
perf_ratio = (R2_cv[1].flatten() / R2_cv[0].flatten())*100-100
print('%1.1f%% +- %1.1f%% more predictive' % (np.nanmean(perf_ratio),np.nanstd(perf_ratio)))

#%%
fig,axes = plt.subplots(1,1,figsize=(4*cm,3.9*cm))
ax = axes
ax.hist(optim_rank[0].flatten(),bins=np.arange(0,8,1)+0.5,color='grey',alpha=0.6)
ax.hist(optim_rank[1].flatten(),bins=np.arange(0,8,1)+0.5,color='red',alpha=0.6)
ax.plot(np.nanmean(optim_rank[0].flatten()),40,'v',color='grey')
ax.plot(np.nanmean(optim_rank[1].flatten()),40,'v',color='red')
ax.set_xlabel('rank')
ax.set_ylabel('# datasets')
# ax.axvline(1,linestyle='--',linewidth=1,color='black')
sns.despine(fig,top=True,right=True,offset=2)
my_savefig(fig,figdir,'rank_hist_%s' % params['direction'])

#%% 
rank_ratio = (optim_rank[1].flatten() - optim_rank[0].flatten())

fig,axes = plt.subplots(1,1,figsize=(4*cm,3.9*cm))
ax = axes
binedges = np.arange(-2,3,1)+0.5
ax.hist(rank_ratio,bins=binedges,color='red',alpha=0.6)
# binedges = 
ax.plot(np.nanmean(rank_ratio),40,'v',color='red')
ax.set_xticks(np.arange(-2,3))
ax.set_xlabel('rank difference')
ax.set_ylabel('# datasets')
ax.axvline(0,linestyle='--',linewidth=1,color='black')
sns.despine(fig,top=True,right=True,offset=2)
my_savefig(fig,figdir,'rank_diff_hist_%s' % params['direction'])

t,p = stats.wilcoxon(rank_ratio[~np.isnan(rank_ratio)])
print('p=%.3e, wilcoxon test' % p)
# print('p=%.3f, wilcoxon test' % p)

# nas = np.logical_or(np.isnan(optim_rank[0].flatten()), np.isnan(optim_rank[1].flatten()))
# t,p = ttest_rel(optim_rank[0].flatten()[~nas], optim_rank[1].flatten()[~nas])

# print('Paired t-test: t=%1.2f,p=%.3e' % (t,p))

#%%
fig, axes = plt.subplots(1,3,figsize=(12*cm,4*cm))

ax = axes[0]
handles = []
for iapl, arealabelpair in enumerate(sourcearealabelpairs):
    handles.append(shaded_error(np.arange(nranks),meanrankdata[iapl,:,:],color=clrs_arealabelpairs[iapl],
                                alpha=0.25,error='sem',ax=ax))

ax.legend(handles,[alx1,alx2],frameon=False,loc='lower right')
my_legend_strip(ax)
ax.set_xlabel('rank')
ax.set_ylabel('performance')
# ax.set_yticks(np.arange(0,0.3,0.05))
ax.set_ylim([0,axlim])
ax.set_yticks(np.linspace(0,axlim,3))

# ax.set_yticks(np.arange(0,0.3,0.025))
ax.set_xlim([0,nranks])
ax.set_xticks(np.arange(0,nranks,5))
ax.set_xlim([0,nrankstoplot])

ax=axes[1]
R2_toplot = R2_cv.copy()
plotclip = my_ceil(np.nanpercentile(R2_toplot,99),2)
R2_toplot = np.clip(R2_toplot,0,plotclip)
ax.scatter(R2_toplot[0,:],R2_toplot[1,:],color=clrs_arealabelpairs[0],marker='o',s=8,alpha=0.25)

ax.plot([0,1],[0,1],color='k',linestyle='--',linewidth=0.5)
ax.set_xlabel(alx1,color=clrs_arealabelpairs[0])
ax.set_ylabel(alx2,color=clrs_arealabelpairs[1])
add_paired_ttest_results(ax,R2_cv[0,:],R2_cv[1,:],pos=[0.7,0.1])
# add_paired_wilcoxon_results(ax,R2_cv[0,:],R2_cv[1,:],pos=[0.7,0.1])
ax.set_title('performance')
ax.set_xlim([0,plotclip])
ax.set_ylim([0,plotclip])
# ax.set_xlim([0,my_ceil(np.nanmax(R2_cv),2)])
# ax.set_ylim([0,my_ceil(np.nanmax(R2_cv),2)])
ax.set_xticks(np.linspace(0,ax.get_xlim()[1],3))
ax.set_xticklabels(np.linspace(0,ax.get_xlim()[1],3),color=clrs_arealabelpairs[0])
ax.set_yticks(np.linspace(0,ax.get_ylim()[1],3))
ax.set_yticklabels(np.linspace(0,ax.get_ylim()[1],3),color=clrs_arealabelpairs[1])

ax=axes[2]

#Show the estimated ranks for all sessions and stimuli:
nrankstoplot = my_ceil(np.nanmax(optim_rank),0)+1
xdata = optim_rank[0,:].flatten()
ydata = optim_rank[1,:].flatten()
xdata,ydata = filter_sharednan(xdata,ydata)
ax.hist2d(xdata, ydata, bins=[np.arange(-0.5, nrankstoplot+1.5, 1), 
                            np.arange(-0.5, nrankstoplot+1.5, 1)], cmap='Blues',cmin=1)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('#datasets', rotation=270, labelpad=10)
ax.plot([0, nrankstoplot], [0, nrankstoplot], color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel(alx1,color=clrs_arealabelpairs[0])
ax.set_ylabel(alx2,color=clrs_arealabelpairs[1])
add_paired_ttest_results(ax,optim_rank[0,:],optim_rank[1,:],pos=[0.7,0.1])
ax.set_xlim([0,my_ceil(np.nanmax(optim_rank),0)+1])
ax.set_ylim([0,my_ceil(np.nanmax(optim_rank),0)+1])
ticks = np.arange(0,nrankstoplot+1,2)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks,color=clrs_arealabelpairs[0])
ax.set_yticks(ticks)
ax.set_yticklabels(ticks,color=clrs_arealabelpairs[1])
ax.set_title('rank')

sns.despine(top=True,right=True,offset=2)
fig.tight_layout()




# #%% Show the mean across sessions:
# xposrank = 10
# idxs = np.array([0,1])
# meanranks = np.nanmean(optim_rank,axis=(-1,-2))
# meanR2 = np.nanmean(R2_cv,axis=(-1,-2))
# R2_rank_datatoplot = np.nanmean(R2_ranks,axis=(4,5))

# #Get only the sessions that have both populations:
# idx_ses = np.all(~np.isnan(R2_ranks),axis=(0,2,3,4))

# fig, axes = plt.subplots(1,1,figsize=(5*cm,4.5*cm))
# ax = axes
# handles = []

# ydata = R2_rank_datatoplot[idxs[0]]
# ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
# handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
#                             color=clrs_arealabelpairs[idxs[0]],alpha=0.3))
# ydata = R2_rank_datatoplot[idxs[1]]

# ydata = np.transpose(ydata,(2,0,1)).reshape(params['nranks'],-1)
# handles.append(shaded_error(np.arange(params['nranks']),ydata.T,ax=ax,error='sem',
#                             color=clrs_arealabelpairs[idxs[1]],alpha=0.3))
# for idx in idxs:
#     ax.plot(meanranks[idx],meanR2[idx]+0.005,color=clrs_arealabelpairs[idx],marker='v',markersize=5)

# leg = ax.legend(handles,arealabeled_to_figlabels(sourcearealabelpairs[idxs]),frameon=False)
# my_legend_strip(ax)
# ax.set_xlabel('Rank')
# ax.set_ylabel('Cross-validated R2')

# x = optim_rank[idxs[0],:]
# y = optim_rank[idxs[1],:]
# nas = np.logical_or(np.isnan(x), np.isnan(y))
# t,p = ttest_rel(x[~nas], y[~nas])
# print('Paired t-test (Rank): p=%.3f' % (p))
# ax.plot(meanranks[idxs],np.repeat(np.nanmean(meanR2[idxs]),2)+0.007,linestyle='-',color='k',linewidth=2)
# ax.text(np.nanmean(meanranks),np.nanmean(meanR2[idxs])+0.009,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

# x = R2_cv[idxs[0],:]
# y = R2_cv[idxs[1],:]
# nas = np.logical_or(np.isnan(x), np.isnan(y))
# t,p = ttest_rel(x[~nas], y[~nas])
# print('Paired t-test (R2): p=%.3f' % (p))
# ax.plot([xposrank,xposrank],meanR2[idxs],linestyle='-',color='k',linewidth=2)
# ax.text(xposrank+0.5,np.nanmean(meanR2[idxs])+0.005,'%s' % get_sig_asterisks(p,return_ns=True),ha='center',va='center',color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

# ax.set_xticks(np.arange(params['nranks'])[::3]+1)
# ax.set_xlim([0,nrankstoplot])

# plt.tight_layout()
# sns.despine(fig=fig,trim=False,top=True,right=True)
# # my_savefig(fig,figdir,'RRR_joint_cvR2_labunl_%s_%dsessions' % (version,params['nSessions']))

# #%% Show figure for each of the arealabelpairs and each of the dataversions
# #Reshape stim x sessions:
# R2_data                 = np.reshape(R2_cv,(narealabelpairs,params['nSessions']*params['nStim']))
# optim_rank_data         = np.reshape(optim_rank,(narealabelpairs,params['nSessions']*params['nStim']))
# R2_ranks_data           = np.reshape(R2_ranks,(narealabelpairs,params['nSessions']*params['nStim'],params['nranks'],nmodelfits,params['kfold']))

# clrs        = ['grey','red']
# fig         = plot_RRR_R2_arealabels_paired(R2_data,optim_rank_data,R2_ranks_data,np.array(sourcearealabelpairs),clrs)
# my_savefig(fig,figdir,'RRR_cvR2_%s_%dsessions' % (version,params['nSessions']))

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
filename = 'RRR_Separate_labeled_controls_FF_original_2026-06-10_15-49-20'
figdir = os.path.join(params['figdir'],'RRR','Labeling','Feedforward')

#%% For FB:
version = 'FB_original'
filename = 'RRR_Separate_labeled_controls_FB_original_2026-06-10_20-17-02'
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
valuematch_labels = np.array(['cell radius','noise level','event rate','tuning (gOSI)'])
# valuematch_labels = np.array(['Cell radius','Noise level\n (Rupprecht et al. 2021)',
#                               'Event rate','Tuning (gOSI)','Tuning (EV)'])

with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)
nmodelfits = params['nmodelfits']

#%% Remove tuning variance because it is complicated metric:
R2_cv = R2_cv[:4]
optim_rank = optim_rank[:4]
R2_ranks = R2_ranks[:4]
valuematch_fields = valuematch_fields[:4]
params['nvaluefields'] = 4

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
for ivaluematching in range(params['nvaluefields']):
    h,p = stats.ttest_1samp(R2_ratio[ivaluematching].flatten(),1,nan_policy='omit')
    ax.text(ivaluematching,np.nanmean(R2_ratio[ivaluematching])+0.05,get_sig_asterisks(p),rotation=0,ha='center',fontsize=9)
ax_nticks(ax,4)
ax.set_ylabel("performance ratio\n%s/%s" % (alx2,alx1))
ax.axhline(y=1,color='k',linestyle='--')
ax.set_xticks(range(params['nvaluefields']))
ax.set_xticklabels(valuematch_labels,rotation=45,ha='right')
ax.set_xlim([-0.5,params['nvaluefields']-1+.25])
sns.despine(fig=fig,trim=False)
my_savefig(fig,figdir,'perf_ratio_separate_%s_controls_%dsessions' % (version,params['nSessions']))

#%% Define the ratio of R2 between V1PM and V1ND
rank_ratio = optim_rank[:,1,:,:] / optim_rank[:,0,:,:]

#Make the figure of the ratio:
fig,axes = plt.subplots(1,1,sharex=True,sharey=True,figsize=(3*cm,3.6*cm))
ax = axes
ax.errorbar(x=range(params['nvaluefields']),y=np.nanmean(rank_ratio,axis=(1,2)),yerr=np.nanstd(rank_ratio,axis=(1,2))/np.sqrt(params['nSessions']*params['nStim']),
            color='red',marker='o',linestyle='',capsize=0)
for ivaluematching in range(params['nvaluefields']):
    h,p = stats.ttest_1samp(rank_ratio[ivaluematching].flatten(),1,nan_policy='omit')
    ax.text(ivaluematching,np.nanmean(rank_ratio[ivaluematching])+0.01,get_sig_asterisks(p),rotation=0,ha='center',fontsize=9)
ax.set_ylim([0.98,my_ceil(ax.get_ylim()[1],2)])
ax_nticks(ax,4)
ax.set_ylabel("rank ratio\n%s/%s" % (alx2,alx1))
ax.axhline(y=1,color='k',linestyle='--')
ax.set_xticks(range(params['nvaluefields']))
ax.set_xticklabels(valuematch_labels,rotation=45,ha='right')
ax.set_xlim([-0.5,params['nvaluefields']-1+.25])
sns.despine(fig=fig,trim=False)
my_savefig(fig,figdir,'rank_ratio_separate_%s_controls_%dsessions' % (version,params['nSessions']))

