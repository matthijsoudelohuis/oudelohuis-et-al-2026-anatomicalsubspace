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
figdir = os.path.join(params['figdir'],'RRR','Labeling','AL')
resultdir = params['resultdir']

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% Load the data:
version = 'FF_AL_original'
filename = 'RRR_Joint_labeled_FF_AL_original_2026-02-20_14-50-51'

version = 'FF_AL_original'
filename = 'RRR_Joint_labeled_FF_AL_behavout_2026-02-20_16-02-43'

version = 'FB_AL_original'
filename = 'RRR_Joint_labeled_FB_AL_original_2026-02-22_21-21-50'

version = 'FB_AL_behavout'
filename = 'RRR_Joint_labeled_FB_AL_behavout_2026-02-22_19-33-50'

#%% Load the data:
data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)

for key in data.keys():
    print(key)
    if key not in ['R2_ranks_neurons','weights_in']:
        exec(key+'=data[key]')

with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

nmodelfits = 100
Nsub = 25
dim_method = 'PCA_shuffle'

#%% 
clrs_arealabelpairs = ['grey','grey','red']
narealabelpairs = 3
fig, axes = plt.subplots(1,1,figsize=(6*cm,5*cm))
# ax = axes[0]
ax = axes
ax.plot(range(params['nranks']),np.nanmean(R2_ranks[0],axis=(0,1,3,4)),label='All neurons',color='grey')
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
    # for idx in np.array([[1,3]]):
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
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm),sharey=True,sharex=True)
ax = axes
handles = []
if diffmetric == 'ratio':
    # ymeantoplot = np.nanmean(data[2],axis=(0,1,3)) / (np.nanmean(data[1],axis=(0,1,3))+1e-3)
    # yerrortoplot = np.nanstd(data[2],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
    ymeantoplot = (np.nanmean(data[2],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)
    yerrortoplot = (np.nanstd(data[2],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1],axis=(0,1,3))+noise_constant) / np.sqrt(params['nSessions']*nmodelfits)
elif diffmetric == 'difference':
    ymeantoplot = np.nanmean(data[2] - data[1],axis=(0,1,3))
    yerrortoplot = np.nanstd(data[2] - data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
handles.append(shaded_error(np.arange(params['nranks']-1)+1,ymeantoplot,yerrortoplot,ax=ax,color='black',alpha=0.3))

if diffmetric == 'ratio':
    ymeantoplot = (np.nanmean(data[3],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1],axis=(0,1,3))+noise_constant)
    yerrortoplot = (np.nanstd(data[3],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1],axis=(0,1,3))+noise_constant) / np.sqrt(params['nSessions']*nmodelfits)
    # ymeantoplot = np.nanmean(data[3],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3))
    # yerrortoplot = np.nanstd(data[3],axis=(0,1,3)) / np.nanmean(data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
elif diffmetric == 'difference':
    ymeantoplot = np.nanmean(data[3] - data[1],axis=(0,1,3))
    yerrortoplot = np.nanstd(data[3] - data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)

handles.append(shaded_error(np.arange(params['nranks']-1)+1,ymeantoplot,yerrortoplot,ax=ax,color='red',alpha=0.3))
ax.legend(handles,['unl-unl','lab-unl'],frameon=False)
my_legend_strip(ax)
ax_nticks(ax,4)
ax.set_xticks(np.arange(params['nranks']-1)[::3]+1)
ax.set_xlim([1,10])
ax.set_xlabel('dimension')
ax.set_ylabel('R2 %s' % diffmetric)
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
    # vmin,vmax = -np.abs(np.nanpercentile(diffdata,99.5)),np.abs(np.nanpercentile(diffdata,99.5)) #diffdata.max()),diffdata.max()
    ax.pcolor(bins,bins,diffdata,cmap='bwr',vmin=vmin,vmax=vmax)
    ax.set_title('Dimension %d' % (r+1))
    ax.plot([0,r2lim],[0,r2lim],color='grey',linestyle='--')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
my_savefig(fig,figdir,'R2_2dhist_perrank_%s_%dsessions' % (version,params['nSessions']))
