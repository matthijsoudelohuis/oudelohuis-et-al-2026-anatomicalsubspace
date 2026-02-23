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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy import stats
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pickle

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
# from utils.corr_lib import *
from utils.RRRlib import *
from utils.regress_lib import *
from utils.pair_lib import value_matching
from utils.psth import compute_tensor
from params import load_params
from utils.corr_lib import filter_sharednan

params = load_params()
# figdir = os.path.join(params['figdir'],'RRR','Labeling','FeedForward')
figdir = os.path.join(params['figdir'],'RRR','Labeling','Feedback')
resultdir = params['resultdir']

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% Load the data:
version = 'FF_original'
filename = 'RRR_Joint_labeled_FF_original_2026-02-19_18-05-04'

# version = 'FF_behavout'
# filename = 'RRR_Joint_labeled_FF_behavout_2026-02-20_02-00-03'

# version = 'FB_original'
# filename = 'RRR_Joint_labeled_FB_original_2026-02-19_21-42-16'

# version = 'FB_behavout'
# filename = 'RRR_Joint_labeled_FB_behavout_2026-02-20_06-11-04'

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
dim_method = params['dim_method']


#%% Show an example session:
clrs_arealabelpairs = ['grey','grey','red']
narealabelpairs = 3
fig, axes = plt.subplots(1,1,figsize=(6*cm,5*cm))
ax = axes
ises = 8
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
my_savefig(fig,figdir,'RRR_joint_cvR2_labunl_%s_ExampleSesion' % (version))


#%% 
clrs_arealabelpairs = ['grey','grey','red']
narealabelpairs = 3
fig, axes = plt.subplots(1,1,figsize=(6*cm,5*cm))
# ax = axes[0]
ax = axes
# ax.plot(range(params['nranks']),np.nanmean(R2_ranks[0],axis=(0,1,3,4)),label='All neurons',color='grey')
ax.plot(np.nanmean(R2_ranks[1],axis=(0,1,3,4)),label=sourcearealabelpairs[0],color=clrs_arealabelpairs[0])
ax.plot(np.nanmean(R2_ranks[2],axis=(0,1,3,4)),label=sourcearealabelpairs[1],color=clrs_arealabelpairs[1])
ax.plot(np.nanmean(R2_ranks[3],axis=(0,1,3,4)),label=sourcearealabelpairs[2],color=clrs_arealabelpairs[2])
leg = ax.legend(frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_ylabel('Cross-validated R2')

plt.tight_layout()
sns.despine(fig=fig,trim=False,top=True,right=True)
my_savefig(fig,figdir,'RRR_joint_cvR2_labunl_%s_%dsessions' % (version,params['nSessions']))


#%% Show figure for each of the arealabelpairs and each of the dataversions
#Reshape stim x sessions:
R2_data                 = np.reshape(R2_cv,(narealabelpairs+1,params['nSessions']*params['nStim']))
optim_rank_data         = np.reshape(optim_rank,(narealabelpairs+1,params['nSessions']*params['nStim']))
R2_ranks_data           = np.reshape(R2_ranks,(narealabelpairs+1,params['nSessions']*params['nStim'],params['nranks'],nmodelfits,params['kfold']))
if np.any(~np.isnan(R2_data)):
    # for idx in np.array([[0,1],[2,3]]):
    for idx in np.array([[1,3]]):
    # for idx in np.array([[2,3]]):
    # for idx in np.array([[0,3]]):
        # clrs        = ['grey',get_clr_area_labeled([sourcearealabelpairs[idx[1]].split('-')[0]])]
        clrs        = ['grey','red']
        fig         = plot_RRR_R2_arealabels_paired(R2_data[idx],optim_rank_data[idx],R2_ranks_data[idx],np.array(sourcearealabelpairs)[idx-1],clrs)
        my_savefig(fig,figdir,'RRR_cvR2_%s_%s_%dsessions' % (sourcearealabelpairs[idx[1]-1],version,params['nSessions']))

#%% Identify which dimensions are particularly enhanced in labeled cells:
data = np.nanmean(R2_ranks,axis=(5)) #average across kfolds
data = np.diff(data,axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

diffmetric = 'ratio' #'difference'
# diffmetric = 'difference' #'difference'
noise_constant = 1e-3
# noise_constant = 0
nrankstoplot = 12
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm),sharey=True,sharex=True)
ax = axes
handles = []
pthr = 0.05 / (params['nranks']-1) #Bonferroni correction for multiple comparisons across ranks
if diffmetric == 'ratio':
    # ymeantoplot = np.nanmean(data[2],axis=(0,1,3)) / (np.nanmean(data[1],axis=(0,1,3))+1e-3)
    # yerrortoplot = np.nanstd(data[2],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
    ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)
    yerrortoplot = (np.nanstd(data[2],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1],axis=(0,1,3))+noise_constant) / np.sqrt(params['nSessions']*nmodelfits)
# 
    # ydata = (np.nanmean(data[2],axis=(3))+noise_constant) / (np.nanmean(data[1],axis=(3))+noise_constant)
    # ymeantoplot = np.nanmean(ydata,axis=(0,1))
    # yerrortoplot = np.nanstd(ydata,axis=(0,1)) / np.sqrt(params['nSessions']*params['nStim'])

elif diffmetric == 'difference':
    ymeantoplot = np.nanmean(data[2] - data[1],axis=(0,1,3))
    yerrortoplot = np.nanstd(data[2] - data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
handles.append(shaded_error(np.arange(params['nranks']-1)+1,ymeantoplot,yerrortoplot,ax=ax,color='black',alpha=0.3))

for r in range(nrankstoplot):
    # ydata = (data[2,:,:,r]+noise_constant) /  (data[1,:,:,r]+noise_constant)
    # ydata = (np.nanmean(data[2,:,:,r],axis=2)+noise_constant) /  (np.nanmean(data[1,:,:,r],axis=2)+noise_constant)
    ydata = (np.nanmean(data[2,:,:,r],axis=2)) /  (np.nanmean(data[1,:,:,r],axis=2))
    ydata = ydata.flatten()

    # ydata = np.nanmean(ydata,axis=2).flatten()
    h,p = stats.ttest_1samp(ydata,1,nan_policy='omit')
    if p<pthr:
        print('Rank %d is significantly enhanced in unlabeled cells (p=%.3f)' % (r+1,p))
        ax.text(r,1.05,'*',ha='center',va='bottom',color='black',fontsize=10)
    # ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)

if diffmetric == 'ratio':
    ymeantoplot = (np.nanmean(data[3],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)
    yerrortoplot = (np.nanstd(data[3],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1],axis=(0,1,3))+noise_constant) / np.sqrt(params['nSessions']*nmodelfits)
    # ymeantoplot = np.nanmean(data[3],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3))
    # yerrortoplot = np.nanstd(data[3],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)

    # ydata = (np.nanmean(data[3],axis=(3))+noise_constant) / (np.nanmean(data[1],axis=(3))+noise_constant)
    # ymeantoplot = np.nanmean(ydata,axis=(0,1))
    # yerrortoplot = np.nanstd(ydata,axis=(0,1)) / np.sqrt(params['nSessions']*params['nStim'])

elif diffmetric == 'difference':
    ymeantoplot = np.nanmean(data[3] - data[1],axis=(0,1,3))
    yerrortoplot = np.nanstd(data[3] - data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
handles.append(shaded_error(np.arange(params['nranks']-1)+1,ymeantoplot,yerrortoplot,ax=ax,color='red',alpha=0.3))

for r in range(nrankstoplot):
    # ydata = (data[3,:,:,r]+noise_constant) /  (data[1,:,:,r]+noise_constant)
    # ydata = data[3,:,:,r] /  data[1,:,:,r]
    # ydata = np.nanmean(ydata,axis=2).flatten()
    ydata = (np.nanmean(data[3,:,:,r],axis=2)) /  (np.nanmean(data[1,:,:,r],axis=2))

    # ydata = (np.nanmean(data[3,:,:,r],axis=2)+noise_constant) /  (np.nanmean(data[1,:,:,r],axis=2)+noise_constant)
    # ydata = (np.nanmean(data[3,:,:,r],axis=2)) /  (np.nanmean(data[1,:,:,r],axis=2))
    ydata = ydata.flatten()

    h,p = stats.ttest_1samp(ydata,1,nan_policy='omit')
    if p<pthr:
        print('Rank %d is significantly enhanced in labeled cells (p=%.3f)' % (r+1,p))
        ax.text(r+1,1.2,'*',ha='center',va='bottom',color='red',fontsize=10)
    # ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)

# ax.legend(handles,['V1$_{ND}$/V1$_{ND}$','V1$_{PM}$/V1$_{ND}$'],frameon=False)
ax.legend(handles,['PM$_{ND}$/PM$_{ND}$','PM$_{V1}$/PM$_{ND}$'],frameon=False)
my_legend_strip(ax)
ax_nticks(ax,4)
ax.set_xticks(np.arange(nrankstoplot)[::3]+1)
ax.set_xlim([1,nrankstoplot])
ax.set_ylim([0.9,1.25])
ax.set_xlabel('dimension')
ax.set_ylabel('R$^{2}$ %s' % diffmetric)
if diffmetric == 'ratio':
    ax.axhline(y=1,color='grey',linestyle='--')
elif diffmetric == 'difference':
    ax.axhline(y=0,color='grey',linestyle='--')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'RRR_R2_%s_rank_noiseconstant_%s_%dsessions' % (diffmetric,version,params['nSessions']))
# my_savefig(fig,figdir,'RRR_unique_cvR2_V1lab_V1unl_V1unl_%dneurons' % Nsub)

#%% Are the dimensions which are enhanced in labeled cells unique or express in unlabeled cells as well?
params['nrankstoplot'] = 4
r2data = np.nanmean(R2_ranks,axis=(5)) #average across kfolds
r2data = np.diff(r2data[:,:,:,:params['nrankstoplot']+1,:],axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
nbins = 20
r2lim = 0.05
bins = np.linspace(0,r2lim,nbins)
fig,axes = plt.subplots(1,3,figsize=(12*cm,4.5*cm),sharey=True,sharex=True)
ax = axes[0]
xdata = r2data[1].flatten()
ydata = r2data[3].flatten()
xdata,ydata = filter_sharednan(xdata,ydata)
histdata1 = np.histogram2d(xdata,ydata,bins=bins)[0].T
histdata1 /= np.sum(histdata1)
ax.pcolor(bins,bins,np.log(histdata1),cmap='viridis')
ax.set_ylabel('R2 (lab)')
ax.set_xlabel('R2 (unl)')
ax.set_title('lab-unl')

ax = axes[1]
ydata = r2data[2].flatten()
xdata = r2data[1].flatten()
xdata,ydata = filter_sharednan(xdata,ydata)
histdata2 = np.histogram2d(xdata,ydata,bins=bins)[0].T
histdata2 /= np.sum(histdata2)
ax.pcolor(bins,bins,np.log(histdata2),cmap='viridis')
ax.set_xlabel('R2 (unl)')
ax.set_ylabel('R2 (unl)')
ax.set_title('Unl-unl')

ax = axes[2]
diffdata = histdata1 - histdata2
# Apply gaussian filter
sigma = [1, 1]
diffdata = sp.ndimage.filters.gaussian_filter(diffdata, sigma, mode='constant')
vmin,vmax = -np.abs(np.nanpercentile(diffdata,99.5)),np.abs(np.nanpercentile(diffdata,99.5)) #diffdata.max()),diffdata.max()
ax.pcolor(bins,bins,diffdata,cmap='bwr',vmin=vmin,vmax=vmax)
ax.set_title('Difference')
ax.plot([0,r2lim],[0,r2lim],color='grey',linestyle='--')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'R2_2dhist_%s_%dsessions' % (version,params['nSessions']))

#%% Are the dimensions which are enhanced in labeled cells unique or express in unlabeled cells as well?
r2data = np.nanmean(R2_ranks,axis=(5)) #average across kfolds
r2data = np.diff(r2data,axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
nbins = 20
r2lim = 0.05
bins = np.linspace(0,r2lim,nbins)
fig,axes = plt.subplots(1,params['nrankstoplot'],figsize=(params['nrankstoplot']*4*cm,4.5*cm),sharey=True,sharex=True)

vmin,vmax = -.005,.005
sigma = [1, 1]

for r in range(params['nrankstoplot']):
    ax = axes[r]
    rankdata = r2data[:,:,:,r,:]
    xdata = rankdata[1].flatten()
    ydata = rankdata[3].flatten()
    xdata,ydata = filter_sharednan(xdata,ydata)
    histdata1 = np.histogram2d(xdata,ydata,bins=bins)[0].T
    histdata1 /= np.sum(histdata1)

    ydata = rankdata[2].flatten()
    xdata = rankdata[1].flatten()
    xdata,ydata = filter_sharednan(xdata,ydata)
    histdata2 = np.histogram2d(xdata,ydata,bins=bins)[0].T
    histdata2 /= np.sum(histdata2)

    diffdata = histdata1 - histdata2
    # Apply gaussian filter
    diffdata = sp.ndimage.filters.gaussian_filter(diffdata, sigma, mode='constant')
    # vmin,vmax = -np.abs(np.nanpercentile(diffdata,99.5)),np.abs(np.nanpercentile(diffdata,99.5)) #diffdata.max()),diffdata.max()
    ax.pcolor(bins,bins,diffdata,cmap='bwr',vmin=vmin,vmax=vmax)
    ax.set_title('Dimension %d' % (r+1))
    ax.plot([0,r2lim],[0,r2lim],color='grey',linestyle='--')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'R2_2dhist_perrank_%s_%dsessions' % (version,params['nSessions']))


#%% Which neurons are well predicted? Is this distribution Gaussian, or skewed?
data = np.nanmean(R2_ranks_neurons,axis=(6)) #average across kfolds
# data = np.nanmean(R2_ranks_neurons,axis=(3,6)) #average across stim and kfolds
bins = np.arange(-0.1,0.6,0.02)
clrs = sns.color_palette('viridis',params['nranks'])
fig,axes = plt.subplots(1,1,figsize=(5*cm,5*cm),sharey=True,sharex=True)
ax = axes
for i in range(3):
    # sns.histplot(data=data[i].flatten(),bins=20,kde=True,ax=ax,color=clrs_arealabelpairs[i])
    sns.histplot(data=data[i+1].flatten(),bins=bins,ax=ax,color=clrs_arealabelpairs[i],fill=False,
                 stat='probability',cumulative=False,element='step')
ax.legend(sourcearealabelpairs,frameon=False)
ax.set_xlabel('Crossvalidated R2 per neuron')
ax.set_yscale('log')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=0)
my_savefig(fig,figdir,'cvR2_pertargetneuron_%s_%dsessions' % (version,params['nSessions']))

#%% Are specific neurons much better predicted by labeled cells? Is this distribution Gaussian, or skewed?
data = np.nanmean(R2_ranks_neurons,axis=(6)) #average across kfolds
# data = np.nanmean(R2_ranks_neurons,axis=(3,6)) #average across kfolds
bins = np.arange(-0.1,0.7,0.02)
clrs = sns.color_palette('viridis',params['nranks'])
fig,axes = plt.subplots(1,1,figsize=(5*cm,5*cm),sharey=True,sharex=True)
ax = axes
sns.histplot(data=(data[2] - data[1]).flatten(),bins=bins,ax=ax,color='grey',fill=False,
                stat='probability',cumulative=False,element='step')
sns.histplot(data=(data[3] - data[1]).flatten(),bins=bins,ax=ax,color='red',fill=False,
                stat='probability',cumulative=False,element='step')
ax.legend(['unl-unl','lab-unl'],frameon=False)
sns.despine(fig=fig,top=True,right=True,offset=0)
ax.set_yscale('log')
ax.set_xlabel('Crossvalidated R2 per neuron')
my_savefig(fig,figdir,'cvR2_pertargetneuron_labunldiff_%s_' % version)

#%% Are the dimensions which are enhanced in labeled cells unique or express in unlabeled cells as well?
r2data = np.nanmean(R2_ranks_neurons,axis=(3,6)) #average across kfolds
r2data = np.diff(r2data,axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

#%%
nbins = 20
r2lim = my_ceil(np.nanpercentile(r2data,99),2)
bins = np.linspace(0,r2lim,nbins)
fig,axes = plt.subplots(1,params['nrankstoplot'],figsize=(params['nrankstoplot']*4*cm,4.5*cm),sharey=True,sharex=True)

for r in range(params['nrankstoplot']):
    ax = axes[r]
    rankdata = r2data[:,:,:,r,:]
    xdata = rankdata[1].flatten()
    ydata = rankdata[3].flatten()
    xdata,ydata = filter_sharednan(xdata,ydata)
    histdata1 = np.histogram2d(xdata,ydata,bins=bins)[0].T
    histdata1 /= np.sum(histdata1)

    ydata = rankdata[2].flatten()
    xdata = rankdata[1].flatten()
    xdata,ydata = filter_sharednan(xdata,ydata)
    histdata2 = np.histogram2d(xdata,ydata,bins=bins)[0].T
    histdata2 /= np.sum(histdata2)

    diffdata = histdata1 - histdata2
    # Apply gaussian filter
    sigma = [1, 1]
    diffdata = sp.ndimage.filters.gaussian_filter(diffdata, sigma, mode='constant')
    vmin,vmax = -np.abs(np.nanpercentile(diffdata,99.5)),np.abs(np.nanpercentile(diffdata,99.5)) #diffdata.max()),diffdata.max()
    ax.pcolor(bins,bins,diffdata,cmap='bwr',vmin=vmin,vmax=vmax)
    ax.set_title('Dimension %d' % (r+1))
    ax.plot([0,r2lim],[0,r2lim],color='grey',linestyle='--')
    ax_nticks(ax,3)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'R2_neuron_2dhist_perrank_%s_%dsessions' % (version,params['nSessions']))


#%% Show the correlation between R2 predicted by labeled and unlabeled neurons:
data = np.nanmean(R2_ranks_neurons,axis=(3,6)) #average across stimuli/kfolds
data = np.diff(data,axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

# plotlims = [-0.1,my_ceil(np.nanmax([xdata,ydata]),1)]
plotlims = [0,my_ceil(np.nanmax(data[1:]),2)]
bintickres = 0.1
params['nrankstoplot'] = 5
clrs = sns.color_palette('Reds_r',params['nrankstoplot'])

fig,axes = plt.subplots(1,params['nrankstoplot']-1,figsize=(params['nrankstoplot']*3.2*cm,4*cm),sharey=True,sharex=True)
for r in range(params['nrankstoplot']-1):
    ax = axes[r]
    xdatatoplot = data[1,:,:,r,:].flatten()
    ydatatoplot = data[2,:,:,r,:].flatten()
    # ax.scatter(xdatatoplot,ydatatoplot,alpha=0.3,color=clrs[r],s=5)
    ax.scatter(xdatatoplot,ydatatoplot,alpha=0.7,color='grey',s=3)

    xdatatoplot = data[1,:,:,r,:].flatten()
    ydatatoplot = data[3,:,:,r,:].flatten()
    ax.scatter(xdatatoplot,ydatatoplot,alpha=0.7,color=clrs[r],s=3)

    ax.set_title('Rank %d' % (r+1))
    if r==0:
        ax.set_xlabel('R2 unl ')
        ax.set_ylabel('R2 unl/lab ')

    ax.legend(['unl-unl','lab-unl'],frameon=False)
    my_legend_strip(ax)
    ax.set_xlim(plotlims)
    ax.set_ylim(plotlims)
    ax.set_xticks(np.arange(plotlims[0],plotlims[1]+bintickres,bintickres))
    ax.set_yticks(np.arange(plotlims[0],plotlims[1]+bintickres,bintickres))
    # ax_nticks(ax[r],5)
    ax.plot(plotlims,plotlims,'--',linewidth=1,color='grey')
    xdatatoplot = xdatatoplot[~np.isnan(ydatatoplot)]
    ydatatoplot = ydatatoplot[~np.isnan(ydatatoplot)]
    print(np.corrcoef(xdatatoplot,ydatatoplot)[0,1])

plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,figdir,'RRR_unique_cvR2_%dneurons_%dsessions' % (Nsub,params['nSessions']))

#%%





#%% Plot the fraction of output weights (onto target area) that have a positive projection onto firing rate for each rank:
data = frac_pos_weight_out #take the maximum across all kfolds
ymeantoplot = np.nanmean(data,axis=(0,1,3,4)) #mean across sessions and stim and modelfits
# yerrortoplot = np.nanstd(data,axis=(0,1,3,4)) / np.sqrt(params['nSessions']*nmodelfits)
yerrortoplot = np.nanstd(data,axis=(0,1,3,4))# / np.sqrt(params['nSessions']*nmodelfits)

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm),sharey=True,sharex=True)
ax = axes
shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,color='blue',alpha=0.3)
ax_nticks(ax,4)
ax.axhline(y=0.5,color='grey',linestyle='--')
ax.set_xticks(np.arange(params['nranks'])[::3]+1)
# ax.set_xlim([1,10])
ax.set_xlabel('dimension')
ax.set_ylabel('Frac. pos. projection')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'Frac_pos_weightsout_%s_%dsessions' % (version,params['nSessions']))

#%% Are those dimensions that are enhanced in V1lab, dimensions that are leading to positive projections?
r2data = np.nanmean(R2_ranks,axis=(5)) #average across kfolds
r2data = np.diff(r2data[:,:,:,:params['nrankstoplot']+1,:],axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

weightdata = np.nanmean(frac_pos_weight_out[:,:,:params['nrankstoplot'],:,:],axis=(4)) #average across kfolds

fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharey=True,sharex=True)
ax = axes[0]
xdata = (r2data[3] - r2data[1]).flatten()
ydata = weightdata.flatten()
xdata,ydata = filter_sharednan(xdata,ydata)
sns.regplot(x=xdata,y=ydata,marker="o",color='red',ax=ax,scatter_kws={'s':5, 'facecolors': 'black', 'edgecolors': 'None'})
ax.text(0.7,0.1,'r=%1.2f' % np.corrcoef(xdata.flatten(),ydata.flatten())[0,1],color='red',transform=ax.transAxes)
ax.set_ylabel('Frac. pos. target weights')
ax.set_xlabel('Diff. R2')
ax.set_title('Lab-unl')
ax = axes[1]
xdata = (r2data[2] - r2data[1]).flatten()
ydata = weightdata.flatten()
xdata,ydata = filter_sharednan(xdata,ydata)
sns.regplot(x=xdata.flatten(),y=ydata.flatten(),marker="o",color='blue',ax=ax,scatter_kws={'s':5, 'facecolors': 'black', 'edgecolors': 'None'})
ax.text(0.7,0.1,'r=%1.2f' % np.corrcoef(xdata.flatten(),ydata.flatten())[0,1],color='blue',transform=ax.transAxes)
ax.set_title('Unl-unl')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'Corr_pos_weight_%s_%dsessions' % (version,params['nSessions']))

##

# WEIGHTS IN 


#%% Plot the fraction of output weights (onto target area) that have a positive projection onto firing rate for each rank:
data = frac_pos_weight_in #take the maximum across all kfolds
ymeantoplot = np.nanmean(data[0],axis=(0,1,3,4)) #mean across sessions and stim and modelfits
# yerrortoplot = np.nanstd(data,axis=(0,1,3,4)) / np.sqrt(params['nSessions']*nmodelfits)
yerrortoplot = np.nanstd(data[0],axis=(0,1,3,4)) / np.sqrt(params['nSessions'])

fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharey=True,sharex=True)
ax = axes[0]
shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,color='blue',alpha=0.3)
ax_nticks(ax,4)
ax.axhline(y=0.5,color='grey',linestyle='--')
ax.set_xticks(np.arange(params['nranks'])[::3]+1)
# ax.set_xlim([1,10])
ax.set_xlabel('dimension')
ax.set_ylabel('Frac. pos. source weights')

ax = axes[1]
data = frac_pos_weight_in #take the maximum across all kfolds
handles = []
for i in range(narealabelpairs):
    ymeantoplot = np.nanmean(data[i+1],axis=(0,1,3,4)) #mean across sessions and stim and modelfits
    yerrortoplot = np.nanstd(data[i+1],axis=(0,1,3,4)) / np.sqrt(params['nSessions'])

    handles.append(shaded_error(np.arange(params['nranks'])+1,ymeantoplot,yerrortoplot,ax=ax,color=clrs_arealabelpairs[i],alpha=0.3))
ax.axhline(y=0.5,color='grey',linestyle='--')
ax.legend(handles,sourcearealabelpairs,frameon=False)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'Frac_pos_weightsin_%s_%dsessions' % (version,params['nSessions']))


#%% Are those dimensions that are enhanced in V1lab, dimensions that are resulting from positive weights of labeled cells?
r2data = np.nanmean(R2_ranks,axis=(5)) #average across kfolds
r2data = np.diff(r2data,axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

topfrac = 10 #show the top 10% latents with largest difference in R2 between labeled and unlabeled

weightdata = np.nanmean(weights_in,axis=(6)) #average across kfolds
bins = np.linspace(-1,1,50)
bins = np.linspace(-.2,.2,100)
# bins = np.linspace(-1,1,50)
fig,axes    = plt.subplots(1,params['nrankstoplot'],figsize=(params['nrankstoplot']*4*cm,4*cm),sharey=True,sharex=True)
for r in range(params['nrankstoplot']):
    ax          = axes[r]
    xdata       = r2data[3] - r2data[1]
    xdata       = xdata[:,:,r,:]
    xdata = xdata.flatten()

    ydata = weightdata[2]
    ydata = ydata[:,:,:,r,:]
    ydata = np.reshape(ydata,(Nsub,-1))

    assert xdata.shape[0] == ydata.shape[1]
    idx_top = np.where(xdata > np.nanpercentile(xdata,100-topfrac))[0]
    histdata = ydata[:,idx_top].flatten()
    histdata = histdata[~np.isnan(histdata)]
    # ax.hist(histdata,bins=bins,color='red',alpha=0.25,density=True,
            # cumulative=True,fill=False)
    sns.histplot(histdata,bins=bins,color='red',alpha=1,cumulative=True,
                stat='probability',ax=ax,element='step',fill=False)
    idx_top = np.where(xdata < np.nanpercentile(xdata,topfrac))[0]
    histdata = ydata[:,idx_top].flatten()
    histdata = histdata[~np.isnan(histdata)]
    sns.histplot(histdata,bins=bins,color='grey',alpha=1,cumulative=True,
                stat='probability',ax=ax,element='step',fill=False)
    # ax.hist(histdata,bins=bins,color='blue',alpha=0.25,density=True,cumulative=True,fill=False)

plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'hist_weight_in_%s_%dsessions' % (version,params['nSessions']))

# #%% Are those dimensions that are enhanced in V1lab, dimensions that are associated with positive source weights?
# r2data = np.nanmean(R2_ranks,axis=(5)) #average across kfolds
# r2data = np.diff(r2data[:,:,:,:params['nrankstoplot']+1,:],axis=3) #take the difference between rank r and r+1 (uniquely explained variance by rank r)

# weightdata = np.nanmean(weights_in[:,:,:params['nrankstoplot'],:,:],axis=(4)) #average across kfolds

# fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharey=True,sharex=True)
# ax = axes[0]
# xdata = (r2data[3] - r2data[1]).flatten()
# ydata = weightdata.flatten()
# xdata,ydata = filter_sharednan(xdata,ydata)
# sns.regplot(x=xdata,y=ydata,marker="o",color='red',ax=ax,scatter_kws={'s':5, 'facecolors': 'black', 'edgecolors': 'None'})
# ax.text(0.7,0.1,'r=%1.2f' % np.corrcoef(xdata.flatten(),ydata.flatten())[0,1],color='red',transform=ax.transAxes)
# ax.set_ylabel('Frac. pos. target weights')
# ax.set_xlabel('Diff. R2')
# ax.set_title('Lab-unl')
# ax = axes[1]
# xdata = (r2data[2] - r2data[1]).flatten()
# ydata = weightdata.flatten()
# xdata,ydata = filter_sharednan(xdata,ydata)
# sns.regplot(x=xdata.flatten(),y=ydata.flatten(),marker="o",color='blue',ax=ax,scatter_kws={'s':5, 'facecolors': 'black', 'edgecolors': 'None'})
# ax.text(0.7,0.1,'r=%1.2f' % np.corrcoef(xdata.flatten(),ydata.flatten())[0,1],color='blue',transform=ax.transAxes)
# ax.set_title('Unl-unl')
# plt.tight_layout()
# sns.despine(fig=fig,top=True,right=True,offset=3)
# # my_savefig(fig,figdir,'Corr_pos_weight_%s_%dsessions' % (version,params['nSessions']))



#%% 
 #####  ####### #     # ######   #####  #######    #     # #######  #####  #     #    #    #     # ###  #####  #     #  #####  
#     # #     # #     # #     # #     # #          ##   ## #       #     # #     #   # #   ##    #  #  #     # ##   ## #     # 
#       #     # #     # #     # #       #          # # # # #       #       #     #  #   #  # #   #  #  #       # # # # #       
 #####  #     # #     # ######  #       #####      #  #  # #####   #       ####### #     # #  #  #  #   #####  #  #  #  #####  
      # #     # #     # #   #   #       #          #     # #       #       #     # ####### #   # #  #        # #     #       # 
#     # #     # #     # #    #  #     # #          #     # #       #     # #     # #     # #    ##  #  #     # #     # #     # 
 #####  #######  #####  #     #  #####  #######    #     # #######  #####  #     # #     # #     # ###  #####  #     #  #####  

#%% How much of the variance in the source area is aligned with the predictive subspace:
fig, axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
# ax.plot(range(params['nranks']),np.nanmean(R2_ranks[0],axis=(0,1,3,4)),label='All neurons',color='grey')
ax.plot(np.nanmean(R2_sourcealigned[1],axis=(0,1,3,4)),label=sourcearealabelpairs[0],color=clrs_arealabelpairs[0])
ax.plot(np.nanmean(R2_sourcealigned[2],axis=(0,1,3,4)),label=sourcearealabelpairs[1],color=clrs_arealabelpairs[1])
ax.plot(np.nanmean(R2_sourcealigned[3],axis=(0,1,3,4)),label=sourcearealabelpairs[2],color=clrs_arealabelpairs[2])
leg = ax.legend(frameon=False,fontsize=6)
my_legend_strip(ax)
ax.set_xlabel('Rank')
ax.set_title('Source variance along\npredictive dimensions')
ax.set_xticks(np.arange(params['nranks'])[::3]+1)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'RRR_source_aligned_R2_%sversion_%dsessions' % (version,params['nSessions']))


#%% Are subpopulations that are more predictive lower dimensional?
fig, axes = plt.subplots(1,1,figsize=(5*cm,4*cm))
rank = 4
ax = axes
xdata = source_dim[0]
ydata = R2_ranks[0,:,:,rank,:,:].mean(axis=3)
xdata,ydata = filter_sharednan(xdata,ydata)
sns.regplot(x=xdata.flatten(),y=ydata.flatten(),marker="o",color='blue',ax=ax,scatter_kws={'s':5, 'facecolors': 'black', 'edgecolors': 'None'})
ax.set_xlabel(dim_method)
ax.set_ylabel('R2')
ax.text(0.1,0.05,'r=%1.2f' % np.corrcoef(xdata.flatten(),ydata.flatten())[0,1],color='blue',transform=ax.transAxes)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'RRR_dimensionality_%s_targetprediction_%s_%dsessions' % (dim_method,version,params['nSessions']))

#%% Are subpopulations that are more predictive lower dimensional?
fig, axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
for i in range(3):
    meantoplot = np.nanmean(source_dim[i+1])
    errortoplot = np.nanstd(source_dim[i+1])/np.sqrt(params['nSessions'])
    ax.errorbar(x=i,y=meantoplot,yerr=errortoplot,color=clrs_arealabelpairs[i],marker='o',linestyle='None')
ax.set_xticks([0,1,2],labels=sourcearealabelpairs)
ax.set_xlabel('Source area')
ax.set_ylabel(dim_method)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'RRR_dimensionality_%s_%s_subpopulations_%dsessions' % (dim_method,version,params['nSessions']))

#%% Is the lower dimensionality of labeled population responsible for the increased predictive accuracy?
fig, axes = plt.subplots(1,1,figsize=(5*cm,4*cm))
rank = 4
ax = axes
for i in range(3):
    xdata = source_dim[i+1]
    ydata = R2_ranks[i+1,:,:,rank,:,:].mean(axis=3)
    sns.regplot(x=xdata.flatten(),y=ydata.flatten(),marker="o",color=clrs_arealabelpairs[i],
                ax=ax,scatter_kws={'s':5, 'facecolors': clrs_arealabelpairs[i], 'edgecolors': 'None'})
# ax.legend(sourcearealabelpairs,frameon=False)
ax.set_xlabel(dim_method)
ax.set_ylabel('R2')
ax_nticks(ax,3)
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'RRR_dimensionality_%s_subpopulations_PMprediction_%dsessions' % (dim_method,params['nSessions']))

numobs = len(source_dim[1].flatten())
df = pd.DataFrame({'dim': source_dim[2].flatten(),
                   'R2': R2_ranks[np.ix_([2],range(params['nSessions']),range(params['nStim']),[rank],range(nmodelfits),range(params['kfold']))].mean(axis=5).flatten(),
                   'idx': np.arange(numobs),
                   'source': np.repeat(sourcearealabelpairs[1],numobs)})
df = pd.concat((df,pd.DataFrame({'dim': source_dim[3].flatten(),
                   'R2': R2_ranks[np.ix_([3],range(params['nSessions']),range(params['nStim']),[rank],range(nmodelfits),range(params['kfold']))].mean(axis=5).flatten(),
                   'idx': np.arange(numobs),
                   'source': np.repeat(sourcearealabelpairs[2],numobs)})))
df.dropna().reset_index()
# from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
import statsmodels.api as sm

model = ols('R2 ~ source', data=df).fit()
#summarise model
model.summary()
# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model = ols('R2 ~ source + dim', data=df).fit()
#summarise model
model.summary()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

#%% Do PCA of populations to see spectrum:
sourcearealabelpairs = ['V1unl','V1lab']

clrs_arealabelpairs = get_clr_area_labeled(sourcearealabelpairs)
narealabelpairs     = len(sourcearealabelpairs)

Nsub                = 20
nmodelfits          = 5
# dim_method          = 'parallel_analysis'
# dim_method          = 'participation_ratio'
# dim_method          = 'pca_shuffle'

idx_resp            = np.where((t_axis>=params['tresp_start']) & (t_axis<=params['tresp_end']))[0]
ntimebins           = len(idx_resp)

source_spectrum    = np.full((2,2,Nsub,params['nSessions'],params['nStim'],nmodelfits),np.nan)
source_dim          = np.full((2,params['nSessions'],params['nStim'],nmodelfits),np.nan)

from sklearn.decomposition import FactorAnalysis as FA
from sklearn.decomposition import PCA as PCA
# PCA = PCA(n_components=Nsub)
for ises,ses in enumerate(sessions):
    if params['filter_nearby']:
        idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
    else:
        idx_nearby = np.ones(len(ses.celldata),dtype=bool)

    idx_areax1      = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[0],
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
    idx_areax2      = np.where(np.all((ses.celldata['arealabel']==sourcearealabelpairs[1],
                                ses.celldata['noise_level']<params['maxnoiselevel'],
                                idx_nearby),axis=0))[0]
    
    if len(idx_areax1)<Nsub or len(idx_areax2)<Nsub: #skip exec if not enough neurons in one of the populations
        continue

    for imf in tqdm(range(nmodelfits),total=nmodelfits,desc='Fitting RRR model for session %d/%d' % (ises+1,params['nSessions'])):
        idx_areax1_sub       = np.random.choice(idx_areax1,Nsub,replace=False)
        idx_areax2_sub       = np.random.choice(idx_areax2,Nsub,replace=False)

        # idx_areax1_sub       = idx_areax1
        # idx_areax2_sub       = idx_areax2

        for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            idx_T               = ses.trialdata['stimCond']==stim
       
            X1                  = ses.tensor[np.ix_(idx_areax1_sub,idx_T,idx_resp)]
            X2                  = ses.tensor[np.ix_(idx_areax2_sub,idx_T,idx_resp)]

            # reshape to neurons x time points
            X1                  = X1.reshape(len(idx_areax1_sub),-1).T
            X2                  = X2.reshape(len(idx_areax2_sub),-1).T

            X1                  = zscore(X1,axis=0)
            X2                  = zscore(X2,axis=0)

            # FA = FA(n_components=10)
            # where 𝐲 is a q-dimensional vector containing the observed residuals at a given time point, L is the 𝑞×𝑚 loading matrix that defines the relationship between the m-dimensional (𝑚<𝑞) latent variable 𝐳 and 𝐲, μ is a q-dimensional vector and Ψ is a 𝑞×𝑞 diagonal matrix. We estimated the dimensionality of the latent variable 𝐳 in two steps: (1) we found the number of dimensions 𝑚𝑝⁢𝑒⁢𝑎⁢𝑘 that maximized the cross-validated log-likelihood of the observed residuals; (2) we fitted a FA model with 𝑚𝑝⁢𝑒⁢𝑎⁢𝑘 dimensions and chose m, using the eigenvalue decomposition, as the smallest dimensionality that captured 95% of the variance in the shared covariance matrix 𝐿⁢𝐿𝑇. This procedure provides more robust estimates of the FA model dimensionality (Williamson et al., 2016).
            # FA.fit(X1)
            # FA.

            # for i,data in enumerate([X1,X2]):
            #     source_dim[i,ises,istim,imf] = estimate_dimensionality(data,method=dim_method)

            for i,data in enumerate([X1,X2]):
                X_shuffled = my_shuffle(data, method='random')
                pca_original = PCA(n_components=Nsub).fit(data)
                pca_shuffled = PCA(n_components=Nsub).fit(X_shuffled)
                source_spectrum[i,0,:,ises,istim,imf] = pca_original.explained_variance_ / np.sum(pca_original.explained_variance_)
                source_spectrum[i,1,:,ises,istim,imf] = pca_shuffled.explained_variance_ / np.sum(pca_shuffled.explained_variance_)

#%% Is the lower dimensionality of labeled population responsible for the increased predictive accuracy?
fig, axes = plt.subplots(1,1,figsize=(5*cm,4*cm))
ax = axes
handles = []
for i in range(2):
    ymeantoplot = np.nanmean(source_spectrum[i][0],axis=(1,2,3))
    yerrortoplot = np.nanstd(source_spectrum[i][0],axis=(1,2,3)) / np.sqrt(Nsub*params['nSessions']*nmodelfits)
    handles.append(shaded_error(x=np.arange(Nsub),y=ymeantoplot,yerror=yerrortoplot,ax=ax,color=clrs_arealabelpairs[i]))

for i in range(2):
    ymeantoplot = np.nanmean(source_spectrum[i][1],axis=(1,2,3))
    yerrortoplot = np.nanstd(source_spectrum[i][1],axis=(1,2,3)) / np.sqrt(Nsub*params['nSessions']*nmodelfits)
    handles.append(shaded_error(x=np.arange(Nsub),y=ymeantoplot,yerror=yerrortoplot,ax=ax,color='grey'))

ax.legend(handles,('Unl','Lab','Shuf'),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Component')
ax.set_ylabel('Fraction of variance')
ax_nticks(ax,4)
# ax.set_yscale('log')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'PCA_Spectrum_V1unl_V1lab_%d_sessions' % (params['nSessions']))

#%% Is the lower dimensionality of labeled population responsible for the increased predictive accuracy?
fig, axes = plt.subplots(1,1,figsize=(5*cm,4*cm))
ax = axes
handles = []
for i in range(2):
    ymeantoplot = np.nanmean(source_spectrum[i][0],axis=(1,2,3)) - np.nanmean(source_spectrum[i][1],axis=(1,2,3))
    yerrortoplot = np.nanstd(source_spectrum[i][0],axis=(1,2,3)) / np.sqrt(Nsub*params['nSessions']*nmodelfits)
    handles.append(shaded_error(x=np.arange(Nsub),y=ymeantoplot,yerror=yerrortoplot,ax=ax,color=clrs_arealabelpairs[i]))

ax.axhline(y=0,color='grey',linestyle='--')
ax.legend(handles,('Unl','Lab','Shuf'),frameon=False)
my_legend_strip(ax)
ax.set_xlabel('Component')
ax.set_ylabel('Fraction of variance')
ax_nticks(ax,4)
# ax.set_yscale('log')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
# my_savefig(fig,figdir,'PCA_Spectrum_V1unl_V1lab_shuffle_%d_sessions' % (params['nSessions']))

#%% 
######  ####### #     #    #    #     # ####### #     # ####### 
#     # #       #     #   # #   #     # #     # #     #    #    
#     # #       #     #  #   #  #     # #     # #     #    #    
######  #####   ####### #     # #     # #     # #     #    #    
#     # #       #     # #######  #   #  #     # #     #    #    
#     # #       #     # #     #   # #   #     # #     #    #    
######  ####### #     # #     #    #    #######  #####     #    

#%% Comparison original and behavout:


#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% Load the data:
version = 'FB_original'
filename = 'RRR_Joint_labeled_FB_original_2026-02-19_21-42-16'

data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)

for key in data.keys():
    if key in ['R2_cv','R2_ranks','optim_rank']:
        print(key)  
        exec(key+'_orig'+'=data[key]')

version = 'FB_behavout'
filename = 'RRR_Joint_labeled_FB_behavout_2026-02-20_06-11-04'

data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)

for key in data.keys():
    if key in ['R2_cv','R2_ranks','optim_rank']:
        print(key)  
        exec(key+'_behavout'+'=data[key]')


with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

#%% Reshape data to stretch stim x sessions
nversions = 2
R2_cv = np.concatenate((R2_cv_orig[:,np.newaxis,:,:],R2_cv_behavout[:,np.newaxis,:,:]),axis=1)
R2_cv_2         = np.reshape(R2_cv,(narealabelpairs+1,nversions,params['nSessions']*params['nStim']))
optim_rank = np.concatenate((optim_rank_orig[:,np.newaxis],optim_rank_behavout[:,np.newaxis]),axis=1)
optim_rank_2    = np.reshape(optim_rank,(narealabelpairs+1,nversions,params['nSessions']*params['nStim']))
R2_ranks = np.concatenate((R2_ranks_orig[:,np.newaxis],R2_ranks_behavout[:,np.newaxis]),axis=1)
R2_ranks_2      = np.reshape(R2_ranks,(narealabelpairs+1,nversions,params['nSessions']*params['nStim'],
                                       params['nranks'],params['nmodelfits'],params['kfold']))
dataversions = ['Original','Behavout']

#%% Plot the R2 for each of the arealabelpairs and each of the dataversions
#Residual variance explained goes down with behavior or brainwide activity regressed out: 
fig,axes = plt.subplots(1,1,sharex=True,sharey=True,figsize=(4*cm,5*cm))
# ax = axes[0]
ax = axes
# ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[0],axis=1),yerr=np.nanstd(R2_cv_2[0],axis=1)/np.sqrt(ndatasets),color=clrs_arealabelpairs[0])
ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[0],axis=(1)),yerr=np.nanstd(R2_cv_2[0],axis=(1))/np.sqrt(params['nSessions']),
            color='k',marker='o',linestyle='-',capsize=0)
ax.set_ylabel("Performance $R^2$")
ax.axhline(y=0,color='k',linestyle='--')
# ax.set_title('FF')
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
# ax.set_ylim([-5,20])
ax.set_xlim([-0.25,1.25])

h,p = stats.ttest_rel(R2_cv_2[0][0],R2_cv_2[0][1],nan_policy='omit')
add_stat_annotation(ax, 0,1, 0.11, p, h=0.01,color='grey',fontsize=10)

plt.tight_layout()
sns.despine(fig=fig,trim=True)
my_savefig(fig,figdir,'RRR_cvR2_FB_diffversions_%dsessions' % (params['nSessions']))

#%% Plot the R2 for each of the arealabelpairs and each of the dataversions
# #Residual variance explained goes down with behavior or brainwide activity regressed out: 
ndatasets = params['nSessions']*params['nStim']
fig,axes = plt.subplots(1,2,sharex=True,sharey=False,figsize=(5,4))
ax = axes[0]
for ialp,alp in enumerate(sourcearealabelpairs):
    ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[ialp+1],axis=1),yerr=np.nanstd(R2_cv_2[ialp+1],axis=1)/np.sqrt(ndatasets),color=clrs_arealabelpairs[ialp])
# ax.errorbar(x=range(nversions),y=np.nanmean(R2_cv_2[1],axis=1),yerr=np.nanstd(R2_cv_2[1],axis=1)/np.sqrt(ndatasets),color=clrs_arealabelpairs[1])
ax.set_ylabel("Performance $R^2$")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FB')

ax = axes[1]
for ialp,alp in enumerate(sourcearealabelpairs):
    ax.errorbar(x=range(nversions),y=np.nanmean(optim_rank_2[ialp+1],axis=1),yerr=np.nanstd(optim_rank_2[ialp+1],axis=1)/np.sqrt(ndatasets),color=clrs_arealabelpairs[ialp])
# ax.errorbar(x=range(nversions),y=np.nanmea
ax.set_ylabel("Rank")
ax.axhline(y=0,color='k',linestyle='--')

plt.tight_layout()
sns.despine(fig=fig,trim=True)
# my_savefig(fig,figdir,'RRR_cvR2_FF_FB_labeled_diffversions_%dsessions' % (nSessions))


#%% Define the ratio of R2 between V1PM and V1ND
ratiodata_FB_unlunl       = (R2_cv_2[2] / R2_cv_2[1])*100-100 #Unl1 / Unl2
ratiodata_FB_labunl       = (R2_cv_2[3] / R2_cv_2[1])*100-100 #Lab / Unl1

#%% Make the figure of the ratio:
fig,axes = plt.subplots(1,1,sharex=True,sharey=True,figsize=(4*cm,5*cm))
ax = axes
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FB_labunl,axis=1),yerr=np.nanstd(ratiodata_FB_labunl,axis=1)/np.sqrt(np.shape(ratiodata_FB_labunl)[1]),
            color='red',marker='o',linestyle='-',capsize=0)
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FB_unlunl,axis=1),yerr=np.nanstd(ratiodata_FB_unlunl,axis=1)/np.sqrt(np.shape(ratiodata_FB_unlunl)[1]),
            color='grey',marker='o',linestyle='-',capsize=0)

# ax.legend(['$V1_{PM}$ vs. $V1_{ND1}$','$V1_{ND1}$ vs. $V1_{ND2}$'],
ax.legend(['$PM_{V1}$ vs. $PM_{ND1}$','$PM_{ND1}$ vs. $PM_{ND2}$'],
# ax.legend(['$V1_{ND1}$ vs. $V1_{ND2}$','$V1_{PM}$ vs. $V1_{ND1}$'],
          frameon=False,bbox_to_anchor=(1.08,0.8),fontsize=6)
# my_legend_strip(ax)
# ax.set_ylabel("Relative performance in %\n$V1_{PM}$ vs. $V1_{ND}$")
ax.set_ylabel("Relative performance (%)")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FB')

# for it,(ix,iy) in enumerate(np.array([[0,1],[0,2],[1,2]])):
for it,(ix,iy) in enumerate(np.array([[0,1]])):
    h,p = stats.ttest_rel(ratiodata_FB_unlunl[ix],ratiodata_FB_unlunl[iy],nan_policy='omit')
    add_stat_annotation(ax, ix,iy, 15, p, h=1,color='grey')
    h,p = stats.ttest_rel(ratiodata_FB_labunl[ix],ratiodata_FB_labunl[iy],nan_policy='omit')
    add_stat_annotation(ax, ix,iy, 18, p, h=1,color='red')
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
# ax.set_ylim([-5,20])
ax.set_xlim([-0.25,1.25])
plt.tight_layout()
sns.despine(fig=fig,trim=True)
my_savefig(fig,figdir,'RRR_cvR2_ratio_FB_diffversions_%dsessions' % (params['nSessions']))

#%% Define the ratio of R2 between V1PM and V1ND
ratiodata_FB_unlunl       = (optim_rank_2[2] / optim_rank_2[1])*100-100 #Unl1 / Unl2
ratiodata_FB_labunl       = (optim_rank_2[3] / optim_rank_2[1])*100-100 #Lab / Unl1

#%% Make the figure of the ratio:
fig,axes = plt.subplots(1,1,sharex=True,sharey=True,figsize=(4*cm,5*cm))
ax = axes
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FB_labunl,axis=1),yerr=np.nanstd(ratiodata_FB_labunl,axis=1)/np.sqrt(np.shape(ratiodata_FB_labunl)[1]),
            color='red',marker='o',linestyle='-',capsize=0)
ax.errorbar(x=range(nversions),y=np.nanmean(ratiodata_FB_unlunl,axis=1),yerr=np.nanstd(ratiodata_FB_unlunl,axis=1)/np.sqrt(np.shape(ratiodata_FB_unlunl)[1]),
            color='grey',marker='o',linestyle='-',capsize=0)

ax.legend(['$V1_{PM}$ vs. $V1_{ND1}$','$V1_{ND1}$ vs. $V1_{ND2}$'],
# ax.legend(['$V1_{ND1}$ vs. $V1_{ND2}$','$V1_{PM}$ vs. $V1_{ND1}$'],
          frameon=False,bbox_to_anchor=(1.08,0.8),fontsize=6)
# my_legend_strip(ax)
# ax.set_ylabel("Relative performance in %\n$V1_{PM}$ vs. $V1_{ND}$")
ax.set_ylabel("Relative rank (%)")
ax.axhline(y=0,color='k',linestyle='--')
ax.set_title('FB')

# for it,(ix,iy) in enumerate(np.array([[0,1],[0,2],[1,2]])):
for it,(ix,iy) in enumerate(np.array([[0,1]])):
    h,p = stats.ttest_rel(ratiodata_FB_unlunl[ix],ratiodata_FB_unlunl[iy],nan_policy='omit')
    add_stat_annotation(ax, ix,iy, 15, p, h=1,color='grey')
    h,p = stats.ttest_rel(ratiodata_FB_labunl[ix],ratiodata_FB_labunl[iy],nan_policy='omit')
    add_stat_annotation(ax, ix,iy, 18, p, h=1,color='red')
ax.set_xticks(range(nversions))
ax.set_xticklabels(dataversions)
ax.set_ylim([-5,20])
ax.set_xlim([-0.25,1.25])
plt.tight_layout()
sns.despine(fig=fig,trim=True)
my_savefig(fig,figdir,'RRR_Rank_ratio_FB_diffversions_%dsessions' % (params['nSessions']))

