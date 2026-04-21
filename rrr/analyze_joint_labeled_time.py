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
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy import stats
import pickle
from statsmodels.stats.anova import AnovaRM

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
filename = 'RRR_time_Joint_labeled_FF_original_2026-03-31_16-17-42'

# version = 'FF_behavout'
# filename = 'RRR_Joint_labeled_FF_behavout_2026-02-20_02-00-03'

version = 'FB_original'
filename = 'RRR_time_Joint_labeled_FB_original_2026-04-02_22-50-15'
# filename = 'RRR_time_Joint_labeled_FB_original_2026-04-03_12-05-05'

# version = 'FB_behavout'
# filename = 'RRR_Joint_labeled_FB_behavout_2026-02-20_06-11-04'

#%% Load the data:
data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)

for key in data.keys():
    if key not in ['R2_ranks_neurons','weights_in']:
    # if key in ['R2_ranks','R2_cv','optim_rank']:
        print(key)  
        exec(key+'=data[key]')

with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

nmodelfits = params['nmodelfits']
Nsub = params['Nsub']

clrs_arealabelpairs = ['grey','grey','red']
narealabelpairs = 3

#%%
t_axis = np.array([-1.        , -0.8131887 , -0.62637741, -0.43956611, -0.25275482,
       -0.06594352,  0.12086777,  0.30767907,  0.49449036,  0.68130166,
        0.86811295,  1.05492425,  1.24173554,  1.42854684,  1.61535813,
        1.80216943,  1.98898072,  2.17579202])
idx_resp            = np.where((t_axis>=-99) & (t_axis<=99))[0]
nT                  = len(idx_resp)
params['idx_resp'] = idx_resp
params['nT'] = nT
params['t_axis'] = t_axis

#%% Show an example session:
ises = 16
# ises = 13
# ises = 1

R2_toplot = np.reshape(R2_cv[:,ises,:,:],(narealabelpairs+1,params['nStim'],params['nT']))
t_ticks = np.array([-1,0,1,2])
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
handles = []

plotcontrast = np.array([1,3])
clrs = ['grey','red']

for iapl,apl in enumerate(plotcontrast):
    handles.append(shaded_error(params['t_axis'],R2_toplot[apl,:,:],error='sem',color=clrs[iapl],alpha=0.3,ax=ax))

ymin = 0.00
ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/15
ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=list(arealabeled_to_figlabels(sourcearealabelpairs[plotcontrast-1])),loc='best')
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (s)')
ax.set_ylabel('R$^{2}$')
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,figdir,'RRR_joint_time_%s_example_session_%d' % (version, ises))

#%% Plotting the mean across time across sessions: 
R2_toplot = np.reshape(R2_cv,(narealabelpairs+1,params['nSessions']*params['nStim'],params['nT']))

t_ticks = np.array([-1,0,1,2])

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
handles = []
plotcontrast = np.array([1,3])
clrs = ['grey','red']

for iapl,apl in enumerate(plotcontrast):
    handles.append(shaded_error(params['t_axis'],R2_toplot[apl,:,:],error='sem',color=clrs[iapl],alpha=0.3,ax=ax))

ymin = 0.01
ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/15
ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=list(arealabeled_to_figlabels(sourcearealabelpairs[plotcontrast-1])),loc='best')
# ax.legend(handles=handles,loc='best')
my_legend_strip(ax)
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (s)')
ax.set_ylabel('R$^{2}$')

# plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,figdir,'RRR_joint_time_%s' % (version))

#%% Plot the ratio across time across sessions: 
R2_toplot = np.reshape(R2_cv,(narealabelpairs+1,params['nSessions']*params['nStim'],params['nT']))
R2_toplot = np.clip(R2_toplot,np.nanpercentile(R2_toplot,1),np.nanpercentile(R2_toplot,99)) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
plotcontrasts = np.array([[1,2],[1,3]])
noise_constant = 1e-5
ymin = 0.9

if params['direction'] == 'FF': 
    figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
elif params['direction'] == 'FB': 
    figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']
clrs = ['grey','red']

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
handles = []
for iplotcontrast,plotcontrast in enumerate(plotcontrasts):
    R2_ratio = (R2_toplot[plotcontrast[1],:,:]+noise_constant) / (R2_toplot[plotcontrast[0],:,:]+noise_constant) #add a small constant to avoid division by zero
    handles.append(shaded_error(params['t_axis'],R2_ratio,error='ci95',color=clrs[iplotcontrast],alpha=0.3,ax=ax))
ax.axhline(y=1,color='grey',linestyle='--')
ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/15
ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=figlabels,loc='best')
my_legend_strip(ax)
ax.set_xlim([-1,2])
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (s)')
ax.set_ylabel('R$^{2}$ ratio')
sns.despine(fig=fig, top=True, right=True, offset = 3)
# my_savefig(fig,figdir,'RRR_joint_time_ratio_%s' % (version))

#%% Identify which dimensions are particularly enhanced at which timepoints in labeled cells:
data = np.nanmean(R2_ranks,axis=(6)) #average across kfolds
data = np.diff(data,axis=4) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
# data = np.clip(data,np.nanpercentile(data,1),np.nanpercentile(data,99)) #clip negative values to zero for better visualization of ratios (since negative values can be very close to zero and lead to extreme ratios)

diffmetric = 'ratio' #'difference'
noise_constant = 1e-5
nrankstoplot = 4

fig,axes = plt.subplots(1,nrankstoplot,figsize=(16*cm,4*cm),sharey=True,sharex=True)
ax = axes
for r in range(nrankstoplot):
    ax = axes[r]
    handles = []
    pthr = 0.05 / (params['nranks']-1) #Bonferroni correction for multiple comparisons across ranks
    if diffmetric == 'ratio':
        ymeantoplot = (np.nanmean(data[2,:,:,:,r,:],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1,:,:,:,r,:],axis=(0,1,3))+noise_constant)
        yerrortoplot = (np.nanstd(data[2,:,:,:,r,:],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1,:,:,:,r,:],axis=(0,1,3))+noise_constant) / np.sqrt(params['nSessions']*nmodelfits)
    # elif diffmetric == 'difference':
    #     ymeantoplot = np.nanmean(data[2] - data[1],axis=(0,1,3))
        # yerrortoplot = np.nanstd(data[2] - data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
    handles.append(shaded_error(params['t_axis'],ymeantoplot,yerrortoplot,ax=ax,color='black',alpha=0.3))
    ax.axhline(y=1,color='grey',linestyle='--')

    # for r in range(nrankstoplot):
    #     ydata = (np.nanmean(data[2,:,:,r],axis=2)) /  (np.nanmean(data[1,:,:,r],axis=2))
    #     ydata = ydata.flatten()
    #     h,p = stats.ttest_1samp(ydata,1,nan_policy='omit')
    #     if p<pthr:
    #         print('Rank %d is significantly enhanced in unlabeled cells (p=%.3f)' % (r+1,p))
    #         ax.text(r,1.05,'*',ha='center',va='bottom',color='black',fontsize=10)

    if diffmetric == 'ratio':
        ymeantoplot = (np.nanmean(data[3,:,:,:,r,:],axis=(0,1,3))+noise_constant) / (np.nanmean(data[1,:,:,:,r,:],axis=(0,1,3))+noise_constant)
        yerrortoplot = (np.nanstd(data[3,:,:,:,r,:],axis=(0,1,3))+noise_constant) / (np.nanstd(data[1,:,:,:,r,:],axis=(0,1,3))+noise_constant) / np.sqrt(params['nSessions']*nmodelfits)
    # elif diffmetric == 'difference':
    #     ymeantoplot = np.nanmean(data[2] - data[1],axis=(0,1,3))
        # yerrortoplot = np.nanstd(data[2] - data[1],axis=(0,1,3)) / np.sqrt(params['nSessions']*nmodelfits)
    handles.append(shaded_error(params['t_axis'],ymeantoplot,yerrortoplot,ax=ax,color='red',alpha=0.3))

    if r==0:
        ax.legend(handles,['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND}$'],frameon=False)
        my_legend_strip(ax)
    ax_nticks(ax,4)
    ax.set_xlim([-1,2])
    ax_nticks(ax,3)
    ax.set_xticks(t_ticks)
    ax.set_xticklabels(t_ticks)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('R$^{2}$ %s' % diffmetric)
    ax.set_title('Dimension %d' % (r+1))
if diffmetric == 'ratio':
    ax.axhline(y=1,color='grey',linestyle='--')
elif diffmetric == 'difference':
    ax.axhline(y=0,color='grey',linestyle='--')
plt.tight_layout()
sns.despine(fig=fig,top=True,right=True,offset=3)
# my_savefig(fig,figdir,'RRR_R2_time_%s_rank_noiseconstant_%s_%dsessions' % (diffmetric,version,params['nSessions']))

#%% Plot the ratio across time across sessions: 
if params['direction'] == 'FF':
    rankstoaverage = np.array([1,2,3,4])
elif params['direction'] == 'FB':
    rankstoaverage = np.array([0,1,2,3,4])

# data = R2_ranks #average across kfolds
data = np.diff(R2_ranks,axis=4) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
data = np.nanmean(data[:,:,:,:,rankstoaverage],axis=(4,5,6)) #average across ranks selected

R2_toplot = np.reshape(data,(narealabelpairs+1,params['nSessions']*params['nStim'],params['nT']))
# R2_toplot = np.clip(R2_toplot,np.nanpercentile(R2_toplot,10),np.nanpercentile(R2_toplot,99.8)) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
# R2_toplot = np.clip(R2_toplot,np.nanpercentile(R2_toplot,8),np.nanpercentile(R2_toplot,99)) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
# R2_toplot = np.clip(R2_toplot,np.nanpercentile(R2_toplot,3),np.nanpercentile(R2_toplot,99)) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
R2_toplot = np.clip(R2_toplot,1e-4,np.nanpercentile(R2_toplot,99)) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)

plotcontrasts = np.array([[1,2],[1,3]])
ymin = 0.9
if params['direction'] == 'FF': 
    figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
elif params['direction'] == 'FB': 
    figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']
clrs = ['grey','red']

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
handles = []
for iplotcontrast,plotcontrast in enumerate(plotcontrasts):
    R2_ratio = (R2_toplot[plotcontrast[1],:,:]) / (R2_toplot[plotcontrast[0],:,:]) #add a small constant to avoid division by zero
    handles.append(shaded_error(params['t_axis'],R2_ratio,error='ci95',color=clrs[iplotcontrast],alpha=0.3,ax=ax))
ax.axhline(y=1,color='grey',linestyle='--')
ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
thickness = ax.get_ylim()[1]/15
ax.fill_between([0,0.75], ymin - thickness/2, ymin + thickness/2, color='k', alpha=1)
ax.legend(handles=handles,labels=figlabels,loc='best')
my_legend_strip(ax)
ax.set_xlim([-1,2])
ax_nticks(ax,3)
ax.set_xticks(t_ticks)
ax.set_xticklabels(t_ticks)
ax.set_xlabel('Time (s)')
ax.set_ylabel('R$^{2}$ ratio')
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,figdir,'RRR_joint_time_ratio_sigranks_%s' % (version))
# my_savefig(fig,figdir,'RRR_R2_time_%s_rank_noiseconstant_%s_%dsessions' % (diffmetric,version,params['nSessions']))


#%%  
version = 'FF_original'
FF_filename = 'RRR_time_Joint_labeled_FF_original_2026-03-31_16-17-42'

version = 'FB_original'
FB_filename = 'RRR_time_Joint_labeled_FB_original_2026-04-02_22-50-15'

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

#%% Plot the ratio across time across sessions: 
FF_rankstoaverage = np.array([1,2,3,4])
FB_rankstoaverage = np.array([0,1,2,3,4])
clipval = 1e-4
# data = R2_ranks #average across kfolds
FF_data = np.diff(R2_ranks_FF,axis=4) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
FF_data = np.nanmean(FF_data[:,:,:,:,FF_rankstoaverage],axis=(4,5,6)) #average across ranks selected
FF_data = np.reshape(FF_data,(narealabelpairs+1,params['nSessions']*params['nStim'],params['nT']))
FF_data = np.clip(FF_data,clipval,np.nanpercentile(FF_data,100)) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
FF_data = (FF_data[3,:,:]) / (FF_data[1,:,:])

FB_data = np.diff(R2_ranks_FB,axis=4) #take the difference between rank r and r+1 (uniquely explained variance by rank r)
FB_data = np.nanmean(FB_data[:,:,:,:,FB_rankstoaverage],axis=(4,5,6)) #average across ranks selected
FB_data = np.reshape(FB_data,(narealabelpairs+1,params['nSessions']*params['nStim'],params['nT']))
FB_data = np.clip(FB_data,clipval,np.nanpercentile(FB_data,100)) #clip negative R2 values to zero for better visualization of ratios (since negative R2 values can be very close to zero and lead to extreme ratios)
FB_data = (FB_data[3,:,:]) / (FB_data[1,:,:])

#%%
idx_resp = (params['t_axis']>=0) & (params['t_axis']<=1)
FF_data_rsp = np.nanmean(FF_data[:,idx_resp],axis=1)
FB_data_rsp = np.nanmean(FB_data[:,idx_resp],axis=1)
idx_iti = (params['t_axis']>=-1) & (params['t_axis']<=0)
FF_data_iti = np.nanmean(FF_data[:,idx_iti],axis=1)
FB_data_iti = np.nanmean(FB_data[:,idx_iti],axis=1)

# df = pd.DataFrame({'FF_rsp':FF_data_rsp,'FF_iti':FF_data_iti,'FB_rsp':FB_data_rsp,'FB_iti':FB_data_iti})
df = pd.DataFrame({'FF_iti':FF_data_iti,'FF_rsp':FF_data_rsp,'FB_iti':FB_data_iti,'FB_rsp':FB_data_rsp})
df = df.melt(var_name='direc_window', value_name='ratio')
df['dataset'] = np.tile(np.arange(params['nSessions']*params['nStim']),4)
df['direction'] = df['direc_window'].apply(lambda x: x[:2])
df['timewindow'] = df['direc_window'].apply(lambda x: x[3:])

ymin = 0.9
# if params['direction'] == 'FF': 
#     figlabels = ['V1$_{ND1}$/V1$_{ND2}$','V1$_{PM}$/V1$_{ND1}$']
# elif params['direction'] == 'FB': 
#     figlabels = ['PM$_{ND1}$/PM$_{ND2}$','PM$_{V1}$/PM$_{ND1}$']
# clrs = ['grey','red']

clr_palette = ['grey','blue','grey','blue']
fig,ax = plt.subplots(1,1,figsize=(4*cm,4*cm))
# sns.barplot(data=df,x='direc_window',y='ratio',hue='direc_window',palette=clr_palette,alpha=0.3,ax=ax,errorbar='se')
sns.pointplot(data=df,x='direc_window',y='ratio',hue='direc_window',palette=clr_palette,alpha=0.7,ax=ax,errorbar=('ci', 95),capsize=0)

df_totest = pd.DataFrame({'FF_rsp':FF_data_rsp,'FF_iti':FF_data_iti,'FB_rsp':FB_data_rsp,'FB_iti':FB_data_iti})
df_totest.dropna(inplace=True)
df_totest = df_totest.melt(var_name='direc_window', value_name='ratio')
df_totest['dataset'] = np.tile(np.arange(int(len(df_totest)/4)),4)
df_totest['direction'] = df_totest['direc_window'].apply(lambda x: x[:2])
df_totest['timewindow'] = df_totest['direc_window'].apply(lambda x: x[3:])

# Conduct the repeated measures ANOVA
aov = AnovaRM(data=df_totest,
              depvar='ratio',
              subject='dataset',
              within=['direction', 'timewindow'])
res = aov.fit()
print(res.summary())
restable = res.anova_table
testlabel = ['direction','timewindow','Interaction']
for i in range(3):
    ax.text(0.5,0.95-0.07*i,'%s%s: F(%d,%d)=%1.2f, p=%1.3f' % (get_sig_asterisks(restable['Pr > F'][i]),testlabel[i],restable['Num DF'][i],restable['Den DF'][i],restable['F Value'][i],restable['Pr > F'][i])
            ,transform=plt.gca().transAxes,fontsize=5,ha='center',va='center')

ax.axhline(y=1,color='grey',linestyle='--')
ax.set_ylim([ymin,my_ceil(ax.get_ylim()[1],2)])
ax.set_ylabel('R$^{2}$ ratio')
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,figdir,'RRR_FF_FB_joint_time_ratio_sigranks')

#%%
fig,ax = plt.subplots(1,1,figsize=(3.5*cm,3.5*cm))
ax.scatter(FF_data,FB_data,5,color='black',alpha=0.5)
h,p = stats.spearmanr(FF_data.flatten(),FB_data.flatten(),nan_policy='omit')
ax.text(0.7,0.7,'%sr=%1.2f' % (get_sig_asterisks(p),h),transform=ax.transAxes,ha='center',va='center',fontsize=10,color='k')
ax.axhline(y=1,color='grey',linestyle='--')
ax.axvline(x=1,color='grey',linestyle='--')
ax_nticks(ax,4)
ax.set_xlabel('FF ratio')
ax.set_ylabel('FB ratio')
sns.despine(fig=fig, top=True, right=True, offset = 3)
my_savefig(fig,figdir,'RRR_FF_FB_time_anticorrelation_sigranks')
