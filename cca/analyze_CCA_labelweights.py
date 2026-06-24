# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os, math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.params import load_params

params = load_params()
figdir = os.path.join(params['figdir'],'CCA')
resultdir = os.path.join(params['resultdir'])

#%% Plotting parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% 
# areas       = ['V1','PM']
# nareas      = len(areas)

# #%% Load example sessions:
# session_list        = np.array([['LPE09665_2023_03_14'], #V1lab higher
#                                 ['LPE10885_2023_10_23'], #V1lab much higher
#                                 ])
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list,only_all_areas=areas,filter_areas=areas)

# #%% 
# # sessions,nSessions   = filter_sessions(protocols = 'GR',only_all_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)
# sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],filter_areas=areas,min_lab_cells_V1=20,min_lab_cells_PM=20)

# #%% Wrapper function to load the tensor data, 
# [sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False)
# # sessions = load_resid_tensor(sessions,behavout=True)

#%% Load the data:
filename = 'CCA_labeled_2026-05-05_17-42-11'

#%% Load the data:
data = np.load(os.path.join(resultdir,filename + '.npz'),allow_pickle=True)
for key in data.keys():
    exec(key+'=data[key]')

with open(os.path.join(resultdir,filename + '_params' + '.txt'), "rb") as myFile:
    params = pickle.load(myFile)

#%% 
 #####   #####     #       #     #    #   ######  #     #    #     # ####### ###  #####  #     # #######  #####  
#     # #     #   # #      #     #   ##   #     # ##   ##    #  #  # #        #  #     # #     #    #    #     # 
#       #        #   #     #     #  # #   #     # # # # #    #  #  # #        #  #       #     #    #    #       
#       #       #     #    #     #    #   ######  #  #  #    #  #  # #####    #  #  #### #######    #     #####  
#       #       #######     #   #     #   #       #     #    #  #  # #        #  #     # #     #    #          # 
#     # #     # #     #      # #      #   #       #     #    #  #  # #        #  #     # #     #    #    #     # 
 #####   #####  #     #       #     ##### #       #     #     ## ##  ####### ###  #####  #     #    #     #####  

# Are the weights higher for V1lab or PMlab than unlabeled neurons?
clrs_arealabels = get_clr_area_labeled(arealabels)
params['nStim'] = 16

#%% 
fig,axes = plt.subplots(1,1,figsize=(4*cm,3.5*cm),sharex=True,sharey=True)

varversion = 'stim'
# varversion = 'session'
ax = axes
ialdata = np.nanmean(cancorr_CCA[:,:,:,:],axis=(-1,-2))
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(params['nSessions']*params['nStim'])

mindim = 5
# mindim = np.where(meantoplot<0.1)[0][0]-1

ax.errorbar(range(params['n_components']),meantoplot,yerr=errortoplot,label='all',fmt='o-',markerfacecolor='w',
        elinewidth=1,markersize=6,markeredgecolor='blue',color='grey')

ax.set_xlabel('Dimension')
ax.set_ylabel('Canonical Correlation\n(held-out test data)')

ax.set_yticks(np.arange(0,1.1,0.2))
ax.set_xticks(np.arange(0,params['n_components']+5,5),np.arange(0,params['n_components']+5,5)+1)
# ax.set_xlim
# ax.axhline(0.1,linestyle='--',color='k')
sns.despine(top=True,right=True,offset=2,trim=True)
my_savefig(fig,figdir,'CCA_V1PM_labeled_testcorr_%dsessions' % (params['nSessions']))

#%% 
fig,axes = plt.subplots(1,2,figsize=(9*cm,4.5*cm),sharex=True,sharey=False)
dimstoplot = 12
varversion = 'stim'
# varversion = 'session'
ypos = 0.135
ax = axes[0]
for ial,al in enumerate(arealabels[:2]):
    
    if varversion == 'session':
        ialdata = np.nanmean(weights_CCA[:,ial,:,:,:],axis=(-1,-2))
    elif varversion == 'stim':
        ialdata = np.nanmean(weights_CCA[:,ial,:,:,:],axis=(-1))
        ialdata = ialdata.reshape(params['n_components'],params['nSessions']*params['nStim'])
    meantoplot = np.nanmean(ialdata,axis=1)
    errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

    ax.errorbar(np.arange(1,params['n_components']+1),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=6,color=clrs_arealabels[ial])
ialdatatotest = np.nanmean(weights_CCA[:,[0,1],:,:,:],axis=(-1))
ialdatatotest = ialdatatotest.reshape(params['n_components'],2,params['nSessions']*params['nStim'])

for icomp in np.arange(0,mindim):
    # h,pval = stats.ranksums(ialdatatotest[icomp,0,:],ialdatatotest[icomp,1,:],nan_policy='omit',alternative='less')
    h,pval = stats.ttest_rel(ialdatatotest[icomp,0,:],ialdatatotest[icomp,1,:],nan_policy='omit',alternative='less')
    # pval = pval*np.sqrt(params['n_components']) #(to correct for multiple comparisons)
    pval = pval*params['n_components'] #(to correct for multiple comparisons)
    if pval < 0.05:
        ax.text(icomp+1,ypos,get_sig_asterisks(pval),color='k',fontsize=8,rotation=45,ha='left')
ax.axvline(x=mindim,color='grey',linestyle='--')
# ax.text(mindim+0.5,ax.get_ylim()[1],'CCA Dim',fontsize=8)
ax.set_xlabel('Dimension')
ax.set_ylabel('|Loadings|')
ax.legend(frameon=False,loc='lower center')
ax.set_title('V1')
# ax_nticks(ax,4)

ax = axes[1]
ypos = 0.147
for ial,al in enumerate(arealabels[2:]):
    if varversion == 'session':
        ialdata = np.nanmean(weights_CCA[:,ial+2,:,:,:],axis=(-1,-2))
    elif varversion == 'stim':
        ialdata = np.nanmean(weights_CCA[:,ial+2,:,:,:],axis=(-1))
        ialdata = ialdata.reshape(params['n_components'],params['nSessions']*params['nStim'])
    meantoplot = np.nanmean(ialdata,axis=1)
    errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

    ax.errorbar(np.arange(1,params['n_components']+1),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=6,color=clrs_arealabels[ial+2])
    # ax.errorbar(range(params['n_components']),meantoplot,yerr=errortoplot,label=al,color=clrs_arealabels[ial+2])

ialdatatotest = np.nanmean(weights_CCA[:,[2,3],:,:,:],axis=(-1))
ialdatatotest = ialdatatotest.reshape(params['n_components'],2,params['nSessions']*params['nStim'])
for icomp in np.arange(0,mindim):
    # h,pval = stats.ranksums(ialdatatotest[icomp,0,:],ialdatatotest[icomp,1,:],nan_policy='omit',alternative='less')
    h,pval  = stats.ttest_rel(ialdatatotest[icomp,0,:],ialdatatotest[icomp,1,:],nan_policy='omit',alternative='less')
    pval     = pval*params['n_components'] #(to correct for multiple comparisons)
    # print(pval)
    if pval < 0.05:
        ax.text(icomp+1,ypos,get_sig_asterisks(pval),color='k',fontsize=8,rotation=45,ha='left')

ax.axvline(x=mindim,color='grey',linestyle='--')
# ax.text(mindim+0.5,ax.get_ylim()[1],'CCA Dim',fontsize=8)
ax.set_xlabel('Dimension')
ax.set_title('PM')
ax.set_ylabel('|Loadings|')
ax_nticks(ax,5)
ax.set_xticks(np.arange(1,params['n_components']+5,5),np.arange(1,params['n_components']+5,5))
ax.set_xlim([0.5,dimstoplot+0.5])
ax.legend(frameon=False,loc='lower right')
sns.despine(top=True,right=True,offset=2,trim=False)
plt.tight_layout()
my_savefig(fig,figdir,'CCA_V1PM_labeled_loadings_%dsessions' % (params['nSessions']))
# my_savefig(fig,figdir,'CCA_V1PM_labeled_weights_v2_%dsessions' % nSessions,formats=['png'])

#%% 
varversion = 'stim'
# varversion = 'session'
yoffset = 0.05
hlinepos = 1

# yoffset = 0.00001
# hlinepos = 0

fig,axes = plt.subplots(1,2,figsize=(8.5*cm,3.5*cm),sharex=True,sharey=True)

ax = axes[0]
if varversion == 'session':
    ialdata = np.nanmean(weights_CCA[:,1,:,:,:] - weights_CCA[:,0,:,:,:],axis=(-1,-2))
    ialdata = np.nanmean(weights_CCA[:,1,:,:,:] / weights_CCA[:,0,:,:,:],axis=(-1,-2))
elif varversion == 'stim':
    ialdata = np.nanmean(weights_CCA[:,1,:,:,:] - weights_CCA[:,0,:,:,:],axis=(-1))
    ialdata = np.nanmean(weights_CCA[:,1,:,:,:] / weights_CCA[:,0,:,:,:],axis=(-1))
    ialdata = ialdata.reshape(params['n_components'],params['nSessions']*params['nStim'])
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

ax.errorbar(range(params['n_components']),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=get_clr_areas(['V1']))
for icomp in range(params['n_components']):
    ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='greater')
    # ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='two-sided')
    pval = pval*params['n_components'] #(to correct for multiple comparisons)
    # print(pval)
    if pval < 0.05:
        ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+yoffset,'*',color='k',markersize=8)
# ax.errorbar(range(params['n_components']),meantoplot,yerr=errortoplot,label=al,color=get_clr_areas(['V1']))
ax.axvline(x=mindim,color='grey',linestyle='--')
# ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
ax.set_ylabel(r'$\Delta$|Loadings|   (Lab-Unl)')
ax.set_xlabel('Dimension')
ax.set_title('V1')
ax.axhline(y=hlinepos,color='k',linestyle='--')

ax = axes[1]
if varversion == 'session':
    # ialdata = np.nanmean(weights_CCA[:,3,:,:,:] - weights_CCA[:,2,:,:,:],axis=(-1,-2))
    ialdata = np.nanmean(weights_CCA[:,3,:,:,:] / weights_CCA[:,2,:,:,:],axis=(-1,-2))
elif varversion == 'stim':
    # ialdata = np.nanmean(weights_CCA[:,3,:,:,:] - weights_CCA[:,2,:,:,:],axis=(-1))
    ialdata = np.nanmean(weights_CCA[:,3,:,:,:] / weights_CCA[:,2,:,:,:],axis=(-1))
    ialdata = ialdata.reshape(params['n_components'],params['nSessions']*params['nStim'])
meantoplot = np.nanmean(ialdata,axis=1)
errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

ax.errorbar(range(params['n_components']),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            elinewidth=1,markersize=8,color=get_clr_areas(['PM']))
for icomp in range(params['n_components']):
    ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='greater')
    pval = pval*params['n_components'] #(to correct for multiple comparisons)

    # ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='two-sided')
    if pval < 0.05:
        ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+yoffset,'*',color='k',markersize=8)

ax.axvline(x=mindim,color='grey',linestyle='--')
# ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
ax.set_xlabel('Dimension')
ax.set_title('PM')
ax.axhline(y=hlinepos,color='k',linestyle='--')

ax_nticks(ax,5)
ax.set_xticks(np.arange(0,params['n_components']+5,5),np.arange(0,params['n_components']+5,5)+1)

sns.despine(top=True,right=True,offset=2,trim=True)
# my_savefig(fig,figdir,'CCA_V1PM_labeled_ratioloadings_%dsessions_%s' % (params['nSessions'],params['varversion']))
# my_savefig(fig,figdir,'CCA_V1PM_labeled_deltaloadings_%dsessions_%s' % (params['nSessions'],params['varversion']),formats=['png'])


#%% 
 #####   #####     #       ######  ####### ######     ######  ####### ######     ######     #    ### ######  
#     # #     #   # #      #     # #       #     #    #     # #     # #     #    #     #   # #    #  #     # 
#       #        #   #     #     # #       #     #    #     # #     # #     #    #     #  #   #   #  #     # 
#       #       #     #    ######  #####   ######     ######  #     # ######     ######  #     #  #  ######  
#       #       #######    #       #       #   #      #       #     # #          #       #######  #  #   #   
#     # #     # #     #    #       #       #    #     #       #     # #          #       #     #  #  #    #  
 #####   #####  #     #    #       ####### #     #    #       ####### #          #       #     # ### #     # 

#%%
clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
fig, axes = plt.subplots(1,1,figsize=(4*cm,4*cm))

ax = axes
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs):
    # ax.plot(np.arange(n_components),np.nanmean(CCA_corrtest[iapl,:,:,:],axis=(1,2)),
            # color=clrs_arealabelpairs[iapl],linewidth=2)
    iapldata = CCA_corrtest[iapl,:,:,:].reshape(params['n_components'],-1)
    handles.append(shaded_error(x=np.arange(params['n_components']),
                                # y=np.nanmean(CCA_corrtest_norm[iapl,:,:,:],axis=(1)).T,
                                y=iapldata.T,
                                error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
ax.set_xticks(np.arange(0,params['n_components']+5,5))
ax.set_xticklabels(np.arange(0,params['n_components']+5,5)+1)
ax.set_ylim([0,my_ceil(np.nanmax(np.nanmean(CCA_corrtest,axis=(2,3))),1)])
ax.set_yticks([0,ax.get_ylim()[1]/2,ax.get_ylim()[1]])
ax.set_xlabel('CCA Dimension')
ax.set_ylabel('Correlation')
ax.legend(handles,arealabelpair_to_figlabel(arealabelpairs),loc='upper right',frameon=False,fontsize=6)
sns.despine(top=True,right=True,offset=2,trim=True)
my_savefig(fig,figdir,'CCA_V1PM_pops_labeled_testcorr_%dsessions' % (params['nSessions']))

#%%
fig, axes = plt.subplots(1,1,figsize=(4*cm,4*cm))

ax = axes
CCA_corrtest_norm = CCA_corrtest / CCA_corrtest[0,0,:,:][np.newaxis,np.newaxis,:,:]
# CCA_corrtest_norm = CCA_corrtest / CCA_corrtest[:,0,:,:][:,np.newaxis,:,:]

for iapl, arealabelpair in enumerate(arealabelpairs):
    ax.plot(np.arange(params['n_components']),np.nanmean(CCA_corrtest_norm[iapl,:,:,:],axis=(1,2)),
            color=clrs_arealabelpairs[iapl],linewidth=2)
ax.set_xticks(np.arange(0,params['n_components']+5,5))
ax.set_xticklabels(np.arange(0,params['n_components']+5,5)+1)
ax.set_ylim([0,my_ceil(np.nanmax(np.nanmean(CCA_corrtest_norm,axis=(2,3))),1)])
ax.set_yticks([0,ax.get_ylim()[1]/2,ax.get_ylim()[1]])
ax.set_xlabel('CCA Dimension')
ax.set_ylabel('Correlation')
ax.legend(arealabelpairs,loc='upper right',frameon=False,fontsize=9)
sns.despine(top=True,right=True,offset=2,trim=True)
# my_savefig(fig,figdir,'CCA_V1PM_pops_labeled_testcorr_normdim1_%dsessions_%s' % (nSessions,varversion),formats=['png'])

#%%
fig, axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
# CCA_corrtest_norm = CCA_corrtest / CCA_corrtest[0,:,:,:][np.newaxis,:,:,:]
CCA_corrtest_norm = CCA_corrtest - CCA_corrtest[0,:,:,:][np.newaxis,:,:,:]
handles = []
for iapl, arealabelpair in enumerate(arealabelpairs):
    # ax.plot(np.arange(n_components),np.nanmean(CCA_corrtest_norm[iapl,:,:,:],axis=(1,2)),
            # color=clrs_arealabelpairs[iapl],linewidth=2)
    iapldata = CCA_corrtest_norm[iapl,:,:,:].reshape(params['n_components'],-1)
    handles.append(shaded_error(x=np.arange(params['n_components']),
                                # y=np.nanmean(CCA_corrtest_norm[iapl,:,:,:],axis=(1)).T,
                                y=iapldata.T,
                                error='sem',color=clrs_arealabelpairs[iapl],alpha=0.3,ax=ax))
    for icomp in range(0,mindim):
        # ttest,pval = stats.ttest_1samp(iapldata[icomp],0,nan_policy='omit',alternative='greater')
        ttest,pval = stats.ttest_1samp(iapldata[icomp],0,nan_policy='omit',alternative='greater')
        if pval < 0.05:
            ax.plot(icomp,0.02+0.03 + iapl*0.006,'*',color=clrs_arealabelpairs[iapl],markersize=6)

# ax.errorbar(range(params['n_components']),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
            # elinewidth=1,markersize=8,color=get_clr_areas(['PM']))

ax.set_xticks(np.arange(0,params['n_components']+5,5))
ax.set_xticklabels(np.arange(0,params['n_components']+5,5)+1)
# ax.set_ylim([0,my_ceil(np.nanmax(np.nanmean(CCA_corrtest_norm,axis=(2,3))),1)])
ax.set_yticks([-0.025,0,0.025,0.05])
ax.set_ylim([-0.025,0.075])
ax.set_xlim([-0.5,dimstoplot+0.5])
ax.set_xlabel('CCA Dimension')
ax.set_ylabel(u'Δ Correlation')
ax.legend(handles,arealabelpair_to_figlabel(arealabelpairs),loc='upper right',frameon=False,fontsize=6)
my_legend_strip(ax)
sns.despine(top=True,right=True,offset=2,trim=False)
my_savefig(fig,figdir,'CCA_V1PM_pops_labeled_testcorr_divUnl_%dsessions' % (params['nSessions']))

#%% 














# Deprecated below:
# 



#%% 

 #####  ####### #     # ####### ######  ####### #        #####  
#     # #     # ##    #    #    #     # #     # #       #     # 
#       #     # # #   #    #    #     # #     # #       #       
#       #     # #  #  #    #    ######  #     # #        #####  
#       #     # #   # #    #    #   #   #     # #             # 
#     # #     # #    ##    #    #    #  #     # #       #     # 
 #####  ####### #     #    #    #     # ####### #######  #####  

#%% Dimensionality of the labeled and unlabeled populations: (estimated using PCA)

# #%% Are the weights higher for V1lab or PMlab than unlabeled neurons?
# n_components        = 20
# nmodelfits          = 10
# arealabels          = np.array(['V1unl', 'V1lab', 'PMunl', 'PMlab'])
# pca_ev              = np.full((params['n_components'],len(arealabels),params['nSessions'],params['nStim'],params['nmodelfits']),np.nan)
# pca                 = PCA(n_components=params['n_components'])
# minsampleneurons    = 20

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

#     if params['filter_nearby']:
#         idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
#     else:
#         idx_nearby = np.ones(len(ses.celldata),dtype=bool)

#     nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
#                                           ses.celldata['noise_level']<maxnoiselevel,
#                                           idx_nearby),axis=0)) for i in arealabels])
#     if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
#         continue
#     nsampleneurons = 20

#     for ial, al in enumerate(arealabels):
#         idx_area           = np.where(np.all((ses.celldata['arealabel']==al,
#                                 ses.celldata['noise_level']<params['maxnoiselevel'],
#                                 idx_nearby),axis=0))[0]
#         for imf in range(nmodelfits):
#             idx_area_sub = np.random.choice(idx_area,nsampleneurons,replace=False)
#             for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
#                 idx_T               = ses.trialdata['stimCond']==stim
                
#                 #on residual tensor during the response:
#                 X                   = sessions[ises].tensor[np.ix_(idx_area_sub,idx_T,idx_resp)]
#                 X                   -= np.mean(X,axis=1,keepdims=True)
#                 X                   = X.reshape(len(idx_area_sub),-1).T
#                 X                   = zscore(X,axis=0,nan_policy='omit')  #Z score activity for each neuron

#                 pca                 = PCA(n_components=np.min([nsampleneurons,n_components]))
#                 pca.fit(X)
#                 pca_ev[:np.min([nsampleneurons,n_components]),ial,ises,istim,imf] = pca.explained_variance_ratio_

# #%% Show PCA dimensionality: 

# fig,ax = plt.subplots(1,1,figsize=(4,3))
# for ial,al in enumerate(arealabels):
#     # ax.plot(np.nanmean(pca_ev[:,ial,:,:,:],axis=(-1,-2,-3)),label=al,color=clrs_arealabels[ial])
#     ax.plot(np.nanmean(np.cumsum(pca_ev[:,ial,:,:,:],axis=0),axis=(-1,-2,-3)),label=al,color=clrs_arealabels[ial])

# ax.set_xlim([0,n_components])
# ax.set_ylim([0,1])
# ax.legend(frameon=False,loc='best')
# ax.set_xlabel('PC Dimension')
# ax.set_ylabel('Explained variance')
# sns.despine(top=True,right=True,offset=2,trim=True)
# plt.tight_layout()
# # my_savefig(fig,figdir,'PCA_V1PM_labeled_GRGN_%dsessions' % (nSessions),formats=['png'])

# #%% Now control for the dimensionality: 

# #%% 
# def equalize_eigenspectra(X, Y, mode='geometric'):
#     # Center the data
#     Xc = X - np.mean(X, axis=0)
#     Yc = Y - np.mean(Y, axis=0)

#     # Compute SVD (PCA basis)
#     Ux, Sx, VxT = np.linalg.svd(Xc, full_matrices=False)
#     Uy, Sy, VyT = np.linalg.svd(Yc, full_matrices=False)

#     # Variances (squared singular values)
#     var_x = Sx**2
#     var_y = Sy**2

#     # Compute target eigenspectrum
#     if mode == 'geometric':
#         target_var = np.sqrt(var_x * var_y)
#     elif mode == 'arithmetic':
#         target_var = 0.5 * (var_x + var_y)
#     else:
#         raise ValueError("Unsupported mode. Choose 'geometric' or 'arithmetic'.")

#     # New singular values to match target variance
#     Sx_new = np.sqrt(target_var)
#     Sy_new = np.sqrt(target_var)

#     # Reconstruct matrices with matched eigenspectra
#     X_new = (Ux * Sx_new) @ VxT
#     Y_new = (Uy * Sy_new) @ VyT

#     # Re-add the original mean
#     X_new += np.mean(X, axis=0)
#     Y_new += np.mean(Y, axis=0)

#     return X_new, Y_new

# #%% 

# # Generate example data
# np.random.seed(0)
# M, N = 100, 20

# # X with strong signal
# X1 = np.random.randn(M, N) @ np.diag(np.linspace(5, 1, N))

# # X2 with more uniform variance
# X2 = np.random.randn(M, N)

# # Compute eigenspectra before
# def get_eigenspectrum(A):
#     Ac = A - np.mean(A, axis=0)
#     _, S, _ = np.linalg.svd(Ac, full_matrices=False)
#     return S**2

# var_X1_before = get_eigenspectrum(X1)
# var_X2_before = get_eigenspectrum(X2)

# # Equalize eigenspectra
# X1_eq, X2_eq = equalize_eigenspectra(X1, X2)

# # Compute eigenspectra after
# var_X1_after = get_eigenspectrum(X1_eq)
# var_X2_after = get_eigenspectrum(X2_eq)

# # Plotting
# plt.figure(figsize=(10, 6))

# plt.plot(var_X1_before, label='X Before', marker='o')
# plt.plot(var_X2_before, label='Y Before', marker='o')
# plt.plot(var_X1_after, label='X After', marker='x')
# plt.plot(var_X2_after, label='Y After', marker='x')

# plt.title('Eigenspectra Before and After Equalization')
# plt.xlabel('Principal Component')
# plt.ylabel('Variance (Eigenvalue)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# #%% 

# minsampleneurons    = 10
# pca                 = PCA(n_components=n_components)
# ev_thr              = 0.8
# nmodelfits          = 1

# model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)
# # model_CCA           = PLSCanonical(n_components=n_components,scale = False, max_iter = 1000)
# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
# # for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

#     if params['filter_nearby']:
#         idx_nearby  = filter_nearlabeled(ses,radius=params['radius'])
#     else:
#         idx_nearby = np.ones(len(ses.celldata),dtype=bool)

#     nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
#                                           ses.celldata['noise_level']<maxnoiselevel,
#                                           idx_nearby),axis=0)) for i in arealabels])
    
#     if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
#         continue

#     idx_N_all = np.empty(len(arealabels),dtype=object)
#     for ial, al in enumerate(arealabels):
#         idx_N_all[ial]           = np.where(np.all((ses.celldata['arealabel']==al,
#                                 ses.celldata['noise_level']<maxnoiselevel,	
#                                 idx_nearby),axis=0))[0]
    
#     for imf in range(nmodelfits):
        
#         idx_areax1           = np.random.choice(idx_N_all[0],nsampleneurons,replace=False)
#         idx_areax2           = np.random.choice(idx_N_all[1],nsampleneurons,replace=False)
#         idx_areay1           = np.random.choice(idx_N_all[2],nsampleneurons,replace=False)
#         idx_areay2           = np.random.choice(idx_N_all[3],nsampleneurons,replace=False)

#         for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
#         # for istim,stim in enumerate([np.unique(ses.trialdata['stimCond'])[0]]): # loop over orientations 
#             idx_T               = ses.trialdata['stimCond']==stim
        
#             X1              = sessions[ises].tensor[np.ix_(idx_areax1,idx_T,idx_resp)]
#             X2              = sessions[ises].tensor[np.ix_(idx_areax2,idx_T,idx_resp)]
#             Y1              = sessions[ises].tensor[np.ix_(idx_areay1,idx_T,idx_resp)]
#             Y2              = sessions[ises].tensor[np.ix_(idx_areay2,idx_T,idx_resp)]

#             X1              -= np.mean(X1,axis=1,keepdims=True)
#             X2              -= np.mean(X2,axis=1,keepdims=True)
#             Y1              -= np.mean(Y1,axis=1,keepdims=True)
#             Y2              -= np.mean(Y2,axis=1,keepdims=True)

#             X1              = X1.reshape(len(idx_areax1),-1).T
#             X2              = X2.reshape(len(idx_areax2),-1).T
#             Y1              = Y1.reshape(len(idx_areay1),-1).T
#             Y2              = Y2.reshape(len(idx_areay2),-1).T

#             X1              = zscore(X1,axis=0,nan_policy='omit')  #Z score activity for each neuron
#             X2              = zscore(X2,axis=0,nan_policy='omit')
#             Y1              = zscore(Y1,axis=0,nan_policy='omit')
#             Y2              = zscore(Y2,axis=0,nan_policy='omit')

#             [X1, X2]    = equalize_eigenspectra(X1, X2, mode='geometric')
#             [Y1, Y2]    = equalize_eigenspectra(Y1, Y2, mode='geometric')

#             #on residual tensor during the response:
#             X                   = np.concatenate((X1,X2),axis=1)
#             Y                   = np.concatenate((Y1,Y2),axis=1)
            
#             # Fit CCA MODEL:
#             model_CCA.fit(X,Y)
            
#             weights_CCA[:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_loadings_[:nsampleneurons,:]),axis=0)
#             weights_CCA[:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_loadings_[nsampleneurons:,:]),axis=0)

#             weights_CCA[:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_loadings_[:nsampleneurons,:]),axis=0)
#             weights_CCA[:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_loadings_[nsampleneurons:,:]),axis=0)

# #%% 

# #%% Plot the results: 
# varversion = 'stim'
# # varversion = 'session'

# fig,axes = plt.subplots(1,2,figsize=(6,2.5),sharex=True,sharey=True)

# ax = axes[0]
# if varversion == 'session':
#     ialdata = np.nanmean(weights_CCA[:,1,:,:,:] - weights_CCA[:,0,:,:,:],axis=(-1,-2))
# elif varversion == 'stim':
#     ialdata = np.nanmean(weights_CCA[:,1,:,:,:] - weights_CCA[:,0,:,:,:],axis=(-1))
#     ialdata = ialdata.reshape(n_components,nSessions*nStim)
    
# meantoplot = np.nanmean(ialdata,axis=1)
# errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

# ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
#             elinewidth=1,markersize=8,color=get_clr_areas(['V1']))
# for icomp in range(n_components):
#     ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='greater')
#     # ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='two-sided')
#     # pval = pval*n_components (to correct for multiple comparisons)
#     # print(pval)
#     if pval < 0.05:
#         ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.002,'*',color='k',markersize=8)
# # ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,color=get_clr_areas(['V1']))
# ax.axvline(x=mindim,color='grey',linestyle='--')
# ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
# ax.set_ylabel(r'$\Delta$|Loadings|   (Lab-Unl)')
# ax.set_xlabel('Dimension')
# ax.set_title('V1')
# ax.axhline(y=0,color='k',linestyle='--')

# ax = axes[1]
# if varversion == 'session':
#     ialdata = np.nanmean(weights_CCA[:,3,:,:,:] - weights_CCA[:,2,:,:,:],axis=(-1,-2))
# elif varversion == 'stim':
#     ialdata = np.nanmean(weights_CCA[:,3,:,:,:] - weights_CCA[:,2,:,:,:],axis=(-1))
#     ialdata = ialdata.reshape(n_components,nSessions*nStim)
# meantoplot = np.nanmean(ialdata,axis=1)
# errortoplot = np.nanstd(ialdata,axis=1)/np.sqrt(ialdata.shape[1])

# ax.errorbar(range(n_components),meantoplot,yerr=errortoplot,label=al,fmt='o-',markerfacecolor='w',
#             elinewidth=1,markersize=8,color=get_clr_areas(['PM']))
# for icomp in range(n_components):
#     ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='greater')
#     # ttest,pval = stats.ttest_1samp(ialdata[icomp],0,nan_policy='omit',alternative='two-sided')
#     if pval < 0.05:
#         ax.plot(icomp,meantoplot[icomp]+errortoplot[icomp]+0.002,'*',color='k',markersize=8)

# ax.axvline(x=mindim,color='grey',linestyle='--')
# ax.text(mindim+0.5,0.02,'CCA Dim',fontsize=8)
# ax.set_xlabel('Dimension')
# ax.set_title('PM')
# ax.axhline(y=0,color='k',linestyle='--')

# ax_nticks(ax,5)
# ax.set_xticks(np.arange(0,n_components+5,5),np.arange(0,n_components+5,5)+1)

# sns.despine(top=True,right=True,offset=2,trim=True)
# my_savefig(fig,figdir,'CCA_V1PM_labeled_deltaweights_%dsessions_%s_equaleigenspectrum' % (nSessions,varversion),formats=['png'])


# #%% 
# ######  #######  #####  ######  #######  #####   #####     ######  ####### #     #    #    #     # 
# #     # #       #     # #     # #       #     # #     #    #     # #       #     #   # #   #     # 
# #     # #       #       #     # #       #       #          #     # #       #     #  #   #  #     # 
# ######  #####   #  #### ######  #####    #####   #####     ######  #####   ####### #     # #     # 
# #   #   #       #     # #   #   #             #       #    #     # #       #     # #######  #   #  
# #    #  #       #     # #    #  #       #     # #     #    #     # #       #     # #     #   # #   
# #     # #######  #####  #     # #######  #####   #####     ######  ####### #     # #     #    #    

# #%% 

# #%% 
# session_list        = np.array([['LPE12223_2024_06_10'], #GR
#                                 ['LPE10919_2023_11_06']]) #GR
# # session_list        = np.array([['LPE09665','2023_03_21'], #GR
#                                 # ['LPE10919','2023_11_06']]) #GR

# sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

# #%%  Load data properly with behavior as well as tensor:     
# ## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
# ## Parameters for temporal binning
# t_pre       = -1    #pre s
# t_post      = 1.9     #post s
# binsize     = 0.2
# calciumversion = 'dF'
# vidfields = np.concatenate((['videoPC_%d'%i for i in range(30)],
#                             ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)

# behavfields = np.array(['runspeed','diffrunspeed'])

# for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
#     sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
#                                 calciumversion=calciumversion)
#     [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
#                                  t_pre, t_post, method='binmean',binsize=binsize)
#     [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
#                                  t_pre, t_post, method='binmean',binsize=binsize)
#     sessions[ises].behaviordata['diffrunspeed'] = np.diff(sessions[ises].behaviordata['runspeed'],prepend=0)
#     [sessions[ises].tensor_run,t_axis] = compute_tensor(sessions[ises].behaviordata[behavfields], sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'], 
#                                  t_pre, t_post, method='binmean',binsize=binsize)
#     delattr(sessions[ises],'calciumdata')
#     delattr(sessions[ises],'behaviordata')
#     delattr(sessions[ises],'videodata')

# #%% 

# #%% Are the weights higher for V1lab or PMlab than unlabeled neurons?
# n_components        = 20
# nStim               = 16
# nmodelfits          = 10
# minsampleneurons    = 10
# maxnoiselevel       = 20
# filter_nearby       = True
# kFold               = 5
# idx_resp            = np.where((t_axis>=0) & (t_axis<=1.5))[0]

# arealabels          = np.array(['V1unl', 'V1lab', 'PMunl', 'PMlab'])
# weights_CCA         = np.full((2,n_components,len(arealabels),nSessions,nStim,nmodelfits),np.nan)

# #%% Fit:
# model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)
# si                  = SimpleImputer()

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
# # for ises,ses in tqdm(enumerate([sessions[0]]),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions

#     if filter_nearby:
#         idx_nearby  = filter_nearlabeled(ses,radius=25)
#     else:
#         idx_nearby = np.ones(len(ses.celldata),dtype=bool)

#     nsampleneurons      = np.min([np.sum(np.all((ses.celldata['arealabel']==i,
#                                           ses.celldata['noise_level']<maxnoiselevel,
#                                           idx_nearby),axis=0)) for i in arealabels])
    
#     if nsampleneurons<minsampleneurons: #skip session if less than minsampleneurons in either population
#         continue

#     idx_N_all = np.empty(len(arealabels),dtype=object)
#     for ial, al in enumerate(arealabels):
#         idx_N_all[ial]           = np.where(np.all((ses.celldata['arealabel']==al,
#                                 ses.celldata['noise_level']<maxnoiselevel,	
#                                 idx_nearby),axis=0))[0]
#     for imf in range(nmodelfits):
#         idx_areax           = np.concatenate((np.random.choice(idx_N_all[0],nsampleneurons,replace=False),
#                                                 np.random.choice(idx_N_all[1],nsampleneurons,replace=False)))
#         idx_areay           = np.concatenate((np.random.choice(idx_N_all[2],nsampleneurons,replace=False),
#                                                 np.random.choice(idx_N_all[3],nsampleneurons,replace=False)))
#         assert len(idx_areax)==2*nsampleneurons and len(idx_areay)==2*nsampleneurons

#         for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
#             idx_T               = ses.trialdata['stimCond']==stim
            
#              #on residual tensor during the response:
#             X                   = sessions[ises].tensor[np.ix_(idx_areax,idx_T,idx_resp)]
#             Y                   = sessions[ises].tensor[np.ix_(idx_areay,idx_T,idx_resp)]
            
#             X                   -= np.nanmean(X,axis=1,keepdims=True)
#             Y                   -= np.nanmean(Y,axis=1,keepdims=True)

#             X                   = X.reshape(len(idx_areax),-1).T
#             Y                   = Y.reshape(len(idx_areay),-1).T

#             X                   = zscore(X,axis=0,nan_policy='omit')  #Z score activity for each neuron
#             Y                   = zscore(Y,axis=0,nan_policy='omit')

#             #Get behavioral matrix: 
#             B                   = np.concatenate((sessions[ises].tensor_vid[np.ix_(range(np.shape(sessions[ises].tensor_vid)[0]),idx_T,idx_resp)],
#                                         sessions[ises].tensor_run[np.ix_(range(np.shape(sessions[ises].tensor_run)[0]),idx_T,idx_resp)]),axis=0)
#             B                   = B.reshape(np.shape(B)[0],-1).T
#             B                   = zscore(B,axis=0,nan_policy='omit')

#             X       = si.fit_transform(X)
#             Y       = si.fit_transform(Y)
#             B       = si.fit_transform(B)

#             for irbh in range(2):   
#                 if irbh:
#                     _,_,X,_,_ = regress_out_behavior_modulation(ses,B,X,rank=6,lam=0,perCond=False)
#                     _,_,Y,_,_ = regress_out_behavior_modulation(ses,B,Y,rank=6,lam=0,perCond=False)

#                 # Y2,Y_hat_rr,Y_out,rank,EV = regress_out_behavior_modulation(ses,B,Y,rank=10,lam=0,perCond=False)

#                 # plt.imshow(Y,aspect='auto',vmin=-0.5,vmax=0.5)
#                 # plt.imshow(Y2,aspect='auto',vmin=-0.5,vmax=0.5)
#                 # plt.imshow(Y_out,aspect='auto',vmin=-0.5,vmax=0.5)
#                 # plt.imshow(Y_hat_rr,aspect='auto',vmin=-0.5,vmax=0.5)
#                 # plt.imshow(B,aspect='auto',vmin=-0.5,vmax=0.5)
#                 # print(EV)

#                 # Fit CCA MODEL:
#                 model_CCA.fit(X,Y)
                
#                 # weights_CCA[irbh,:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[:nsampleneurons,:]),axis=0)
#                 # weights_CCA[irbh,:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[nsampleneurons:,:]),axis=0)

#                 # weights_CCA[irbh,:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[:nsampleneurons,:]),axis=0)
#                 # weights_CCA[irbh,:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[nsampleneurons:,:]),axis=0)

#                 weights_CCA[irbh,:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[:nsampleneurons,:]),axis=0)
#                 weights_CCA[irbh,:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[nsampleneurons:,:]),axis=0)

#                 weights_CCA[irbh,:,2,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[:nsampleneurons,:]),axis=0)
#                 weights_CCA[irbh,:,3,ises,istim,imf] = np.mean(np.abs(model_CCA.y_weights_[nsampleneurons:,:]),axis=0)

# #%% 
# plt.scatter(model_CCA.x_weights_,model_CCA.x_loadings_)

# # weights_CCA[irbh,:,0,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[:nsampleneurons,:]),axis=0)
# # weights_CCA[irbh,:,1,ises,istim,imf] = np.mean(np.abs(model_CCA.x_weights_[nsampleneurons:,:]),axis=0)

# #%% 
# # sumdim = [0,2,3,4,5,6]
# sumdim = 5

# datatoplot1 = weights_CCA[:,:,1,:,:,:] - weights_CCA[:,:,0,:,:,:]
# datatoplot2 = weights_CCA[:,:,3,:,:,:] - weights_CCA[:,:,2,:,:,:]

# datatoplot1 = weights_CCA[:,:,1,:,:,:] / weights_CCA[:,:,0,:,:,:]
# datatoplot2 = weights_CCA[:,:,3,:,:,:] / weights_CCA[:,:,2,:,:,:]

# # datatoplot = np.diff(weights_CCA,axis=2)[:,:,[0,2],:,:,:] #Get the difference between labeled and unlabeled
# datatoplot = np.concatenate((datatoplot1[:,:,np.newaxis,:,:,:],datatoplot2[:,:,np.newaxis,:,:,:]),axis=2) #Get the difference between labeled and unlabeled

# datatoplot = np.nanmean(datatoplot,axis=(-1)) #average modelfits
# # datatoplot = np.nansum(datatoplot[:,:sumdim,:,:,:],axis=1) #sum weight diff for first sumdim dims
# datatoplot = np.nanmean(datatoplot[:,:sumdim,:,:,:],axis=1) #sum weight diff for first sumdim dims
# datatoplot = np.reshape(datatoplot,(2,2,nSessions*nStim)) #stretch sessions * stim
# # Dim1: with/wo behavior, dim2: v1/PM, dim3: datasets
# datatoplot[datatoplot == 0] = np.nan

# N       = datatoplot.shape[2]

# fig,axes = plt.subplots(1,2,figsize=(4,2.5),sharex=True,sharey=True)

# for iarea,area in enumerate(areas):
#     ax = axes[iarea]
#     for irbh in range(2):
#         ax.scatter(np.zeros(N)+np.random.randn(N)*0.1+irbh,datatoplot[irbh,iarea,:],s=8,color=clrs_arealabels[iarea])
#         ax.errorbar(irbh,np.nanmean(datatoplot[irbh,iarea,:]),yerr=np.nanstd(datatoplot[irbh,iarea,:])/np.sqrt(N),label='Orig',fmt='o',markerfacecolor='k',
#                     elinewidth=2,markersize=8,color=clrs_arealabels[iarea])
#     t,p = stats.ttest_rel(datatoplot[0,iarea,:],datatoplot[1,iarea,:],nan_policy='omit')
#     # add_stat_annotation(ax, 0, 1, 0.15, p, h=None)
#     add_stat_annotation(ax, 0, 1, 1.15, p, h=None)

#     print('With vs Without behavior (%s): t=%1.3f, p=%1.3f' % (area,t,p))
#     # add_star(ax,p)
#         # ax.errorbar(1,np.nanmean(datatplot[0,0,:]),yerr=np.nanstd(datatplot[1,0,:])/np.sqrt(N),label='MinBeh',fmt='o',markerfacecolor='k',
#                     # elinewidth=2,markersize=8,color=clrs_arealabels[0])

# # ax.errorbar(1,np.nanmean(PMlabdiff),yerr=np.nanstd(PMlabdiff)/np.sqrt(N),label='PM',fmt='o',markerfacecolor='k',
#             # elinewidth=2,markersize=8,color=clrs_arealabels[2])
#     ax.set_xticks([0,1])
#     ax.set_xticklabels(['Orig','-Beh(RRR)'])
#     if irbh==0:
#         ax.set_ylabel('|Weight|   (Lab-Unl)')
#     # ax.set_ylim([-0.1,0.25])
#     ax.set_title(area)
#     # ax.axhline(y=0,color='k',linestyle='--')
#     ax.axhline(y=1,color='k',linestyle='--')
# sns.despine(top=True,right=True,offset=2,trim=True)
# #  ax.scatter(1,PMlabdiff,s=20,color='k')
# plt.tight_layout()
# # my_savefig(fig,figdir,'CCA_V1PM_labeled_sumdeltaweights_behav_%dsessions_%s' % (nSessions,varversion),formats=['png'])

