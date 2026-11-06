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
figdir = os.path.join(params['figdir'],'RRR','Labeling','Falsenegatives')

#%%  
version = 'FF_original'
FF_filename = 'RRR_Joint_labeled_control_falsenegative_FF_2026-11-02_19-04-08'

version = 'FB_original'
FB_filename = 'RRR_Joint_labeled_control_falsenegative_FB_2026-11-03_10-24-25'

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

#%%
binedges = np.arange(-0.005,0.04,0.00025)
handles = []
fig,ax = plt.subplots(1,1,figsize=(3.7*cm,3.5*cm))
counts,bins = np.histogram(np.nanmean(R2_ranks_bysourceneurons_FF[:Nsub],axis=-1).flatten(),bins=binedges)
# counts,bins = np.histogram(np.nanmean(R2_ranks_bysourceneurons_FF[:Nsub],axis=(2,4).flatten(),bins=binedges)
counts = counts / counts.sum()
handles.append(ax.stairs(counts, bins,color='grey',linewidth=0.5,fill=False,alpha=1,edgecolor='grey'))
ax.stairs(counts, bins,color='grey',linewidth=0.5,fill=True,alpha=0.5,edgecolor='grey')
# ax.stairs(counts, bins,color='black',linewidth=0.5,fill=True,alpha=0.5,edgecolor='black')

counts,bins = np.histogram(np.nanmean(R2_ranks_bysourceneurons_shuf_FF,axis=-1).flatten(),bins=binedges)
counts = counts / counts.sum()
# ax.stairs(counts, bins,color='black',linewidth=0.5,fill=True,alpha=0.5,edgecolor='black')
handles.append(ax.stairs(counts, bins,color='black',linewidth=0.5,fill=False,alpha=1,edgecolor='black'))
ax.set_yscale('log')
ax.legend(handles,['V1ND source neurons','shuffle'],frameon=False,fontsize=5)
my_legend_strip(ax)
ax.set_ylabel('Probability (log scale)')
ax.set_xlabel('R2 contribution of source \nneuron to target neurons')
sns.despine(fig=fig, top=True, right=True, offset = 2)
my_savefig(fig,figdir,'R2_per_source_neuron_FF')

print('Number of neurons: %d' % np.sum(~np.isnan(np.nanmean(R2_ranks_bysourceneurons_FF[:Nsub],axis=(2,4)).flatten())))

#%%
handles = []
fig,ax = plt.subplots(1,1,figsize=(3.7*cm,3.5*cm))
counts,bins = np.histogram(np.nanmean(R2_ranks_bysourceneurons_FB[:Nsub],axis=-1).flatten(),bins=binedges)
counts = counts / counts.sum()
handles.append(ax.stairs(counts, bins,color='grey',linewidth=0.5,fill=False,alpha=1,edgecolor='grey'))
ax.stairs(counts, bins,color='grey',linewidth=0.5,fill=True,alpha=0.5,edgecolor='grey')
# ax.stairs(counts, bins,color='black',linewidth=0.5,fill=True,alpha=0.5,edgecolor='black')

counts,bins = np.histogram(np.nanmean(R2_ranks_bysourceneurons_shuf_FB,axis=-1).flatten(),bins=binedges)
counts = counts / counts.sum()
# ax.stairs(counts, bins,color='black',linewidth=0.5,fill=True,alpha=0.5,edgecolor='black')
handles.append(ax.stairs(counts, bins,color='black',linewidth=0.5,fill=False,alpha=1,edgecolor='black'))
ax.set_yscale('log')
ax.legend(handles,['PMND source neurons','shuffle'],frameon=False,fontsize=5)
my_legend_strip(ax)
ax.set_ylabel('Probability (log scale)')
ax.set_xlabel('R2 contribution of source \nneuron to target neurons')
sns.despine(fig=fig, top=True, right=True, offset = 2)
my_savefig(fig,figdir,'R2_per_source_neuron_FB')

print('Number of neurons: %d' % np.sum(~np.isnan(np.nanmean(R2_ranks_bysourceneurons_FB[:Nsub],axis=(2,4)).flatten())))

#%%
alpha = 0.01
r2_thr = np.nanpercentile(np.nanmean(R2_ranks_bysourceneurons_shuf_FF,axis=-1).flatten(), 100-alpha*100)

frac_sig_FF = np.sum(np.nanmean(R2_ranks_bysourceneurons_FF[:Nsub],axis=-1)>r2_thr,axis=0) / Nsub
frac_sig_FF[frac_sig_FF==0] = np.nan
frac_sig_FF = np.nanmean(frac_sig_FF,axis=-1)

frac_sig_FB = np.sum(np.nanmean(R2_ranks_bysourceneurons_FB[:Nsub],axis=-1)>r2_thr,axis=0) / Nsub
frac_sig_FB[frac_sig_FB==0] = np.nan
frac_sig_FB = np.nanmean(frac_sig_FB,axis=-1)

fig,ax = plt.subplots(1,1,figsize=(2.1*cm,3.7*cm))
sns.barplot(x=0,y=frac_sig_FF.flatten(),color='grey',edgecolor='black',alpha=0.5)
sns.barplot(x=1,y=frac_sig_FB.flatten(),color='grey',edgecolor='black',alpha=0.5)
# ax.errorbar(0,y=frac_sig.flatten(),color='grey',edgecolor='black',alpha=0.5)
# ax.bar(np.arange(nSessions*params['nStim']),frac_sig.flatten(),color='grey',edgecolor='black',alpha=0.5)
ax.set_ylabel('Fraction unlabeled source neurons\nwith significant contribution')
ax.set_xticks([0,1])
ax.set_xticklabels(['V1ND','PMND'])

sns.despine(fig=fig, top=True, right=True, offset = 2)
my_savefig(fig,figdir,'Sig_R2_per_source_neuron_FF_FB')
print('FF: %d datasets' % np.sum(~np.isnan(frac_sig_FF.flatten())))
print('FB: %d datasets' % np.sum(~np.isnan(frac_sig_FB.flatten())))

#%% 
fig,ax = plt.subplots(1,1,figsize=(3.7*cm,3.5*cm))
R2_source = np.nanmean(R2_ranks_bysourceneurons_FF[:Nsub],axis=(2,4)).flatten()

Fchan2_source  = Fchan2_sourceneurons_FF[:Nsub].flatten()
frac_red_source  = frac_of_ROI_red_sourceneurons_FF[:Nsub].flatten()

sns.regplot(x=Fchan2_source,y=R2_source,scatter_kws={'s':0.2,'alpha':1,'color':'grey'},line_kws={'color':'black'},ax=ax)
xdata,ydata = filter_sharednan(Fchan2_source,R2_source)
ax.set_xlabel('tdTomato fluorescence')
ax.set_ylabel('R2 per source neuron')
ax_nticks(ax,4)
ax.text(0.05,0.9,'r=%.2f,p=%.2f' % (np.corrcoef(xdata,ydata)[0,1], stats.pearsonr(xdata,ydata)[1]),transform=ax.transAxes)
sns.despine(fig=fig, top=True, right=True, offset = 2)
my_savefig(fig,figdir,'R2_vs_Fluorescence_per_source_neuron_FF')

print('Number of neurons: %d' % np.sum(~np.isnan(xdata)))

#%% 
fig,ax = plt.subplots(1,1,figsize=(3.7*cm,3.5*cm))
R2_source = np.nanmean(R2_ranks_bysourceneurons_FB[:Nsub],axis=(2,4)).flatten()

Fchan2_source  = Fchan2_sourceneurons_FB[:Nsub].flatten()
frac_red_source  = frac_of_ROI_red_sourceneurons_FB[:Nsub].flatten()

sns.regplot(x=Fchan2_source,y=R2_source,scatter_kws={'s':0.2,'alpha':1,'color':'grey'},line_kws={'color':'black'},ax=ax)
xdata,ydata = filter_sharednan(Fchan2_source,R2_source)
ax.set_xlabel('tdTomato fluorescence')
ax.set_ylabel('R2 per source neuron')
ax_nticks(ax,4)
ax.text(0.05,0.9,'r=%.2f,p=%.2g' % (np.corrcoef(xdata,ydata)[0,1], stats.pearsonr(xdata,ydata)[1]),transform=ax.transAxes)
sns.despine(fig=fig, top=True, right=True, offset = 2)
my_savefig(fig,figdir,'R2_vs_Fluorescence_per_source_neuron_FB')
print('Number of neurons: %d' % np.sum(~np.isnan(xdata)))

#%% 
# handles = []
fig,ax = plt.subplots(1,1,figsize=(3.7*cm,3.5*cm))
R2_source = np.nanmean(R2_ranks_bysourceneurons_FF[:Nsub],axis=(2,4)).flatten()

Fchan2_source  = Fchan2_sourceneurons_FF[:Nsub].flatten()
frac_red_source  = frac_of_ROI_red_sourceneurons_FF[:Nsub].flatten()

sns.regplot(x=frac_red_source,y=R2_source,scatter_kws={'s':0.2,'alpha':1,'color':'grey'},line_kws={'color':'black'},ax=ax)
ax.set_xlabel('overlap suite2p/cellpose masks')
ax.set_ylabel('R2 per source neuron')
ax.set_xlim([-0.01,0.5])
xdata,ydata = filter_sharednan(frac_red_source,R2_source)
ax.text(0.05,0.9,'r=%.2f,p=%.2f' % (np.corrcoef(xdata,ydata)[0,1], stats.pearsonr(xdata,ydata)[1]),transform=ax.transAxes)
sns.despine(fig=fig, top=True, right=True, offset = 2)
my_savefig(fig,figdir,'R2_vs_overlap_per_source_neuron_FF')

#%%
fig,ax = plt.subplots(1,1,figsize=(3.7*cm,3.5*cm))
R2_source = np.nanmean(R2_ranks_bysourceneurons_FB[:Nsub],axis=(2,4)).flatten()

Fchan2_source  = Fchan2_sourceneurons_FB[:Nsub].flatten()
frac_red_source  = frac_of_ROI_red_sourceneurons_FB[:Nsub].flatten()

sns.regplot(x=frac_red_source,y=R2_source,scatter_kws={'s':0.2,'alpha':1,'color':'grey'},line_kws={'color':'black'},ax=ax)
ax.set_xlabel('overlap suite2p/cellpose masks')
ax.set_ylabel('R2 per source neuron')
ax.set_xlim([-0.01,0.5])
xdata,ydata = filter_sharednan(frac_red_source,R2_source)
ax.text(0.05,0.9,'r=%.2f,p=%.2f' % (np.corrcoef(xdata,ydata)[0,1], stats.pearsonr(xdata,ydata)[1]),transform=ax.transAxes)
sns.despine(fig=fig, top=True, right=True, offset = 2)
my_savefig(fig,figdir,'R2_vs_overlap_per_source_neuron_FB')
