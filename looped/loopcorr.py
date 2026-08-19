# -*- coding: utf-8 -*-
"""
This script analyzes correlations in a multi-area calcium imaging
dataset with labeled projection neurons. 
Matthijs Oude Lohuis, 2022-2026, Champalimaud Center, Lisbon
"""

#%% ###################################################
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

from utils.params import load_params
from loaddata.session_info import *
from loaddata.get_data_folder import get_local_drive
from utils.pair_lib import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.corr_lib import *
from utils.tuning import compute_tuning_wrapper

#%% Plotting and parameters:
params  = load_params()
params['multcomp_method'] = 'holm'
figdir = os.path.join(params['figdir'],'NoiseCorrelations')

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
# session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
# sessions,nSessions   = filter_sessions(protocols = 'GR')
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'])
sessiondata          = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
report_sessions(sessions)

#%%  Load data properly:
for ises in range(nSessions):
    sessions[ises].load_respmat()

#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_anatomical_distance(sessions)

#%% ########################### Compute tuning metrics: ###################################
sessions = compute_tuning_wrapper(sessions)

#%% ########################## Compute signal and noise correlations: ###################################
sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filter_stationary=True)
# sessions = compute_signal_noise_correlation(sessions,uppertriangular=False,filter_stationary=False)

#%% 
 #####  ####### ######  ######     #     #    #    ####### ######  ### #     # 
#     # #     # #     # #     #    ##   ##   # #      #    #     #  #   #   #  
#       #     # #     # #     #    # # # #  #   #     #    #     #  #    # #   
#       #     # ######  ######     #  #  # #     #    #    ######   #     #    
#       #     # #   #   #   #      #     # #######    #    #   #    #    # #   
#     # #     # #    #  #    #     #     # #     #    #    #    #   #   #   #  
 #####  ####### #     # #     #    #     # #     #    #    #     # ### #     # 

#%% Plot example noise correlation matrix:
ises                = 0 #which session
cmap                = 'rocket'

#Plot: 
arealabeled     = np.array(['V1unl','V1lab','PMunl','PMlab'])
clrs_arealabels = ['grey','red','grey','red']
al_fig          = arealabeled_to_figlabels(arealabeled)

idx_sort       = np.argsort(sessions[ises].celldata['arealabel'])[::-1]
al_sorted      = sessions[ises].celldata['arealabel'][idx_sort]

corrdata_sort      = copy.deepcopy(sessions[ises].noise_corr)
corrdata_sort      = corrdata_sort[idx_sort,:]
corrdata_sort      = corrdata_sort[:,idx_sort]

vmin,vmax       = np.nanpercentile(corrdata_sort,15),np.nanpercentile(corrdata_sort,90)

fig,ax = plt.subplots(1,1,figsize=(5*cm,4*cm))
im = ax.imshow(corrdata_sort,vmin=vmin,vmax=vmax,cmap=cmap)
ax.set_yticks([])
for ial,arealabel in enumerate(arealabeled):
    start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
    ax.plot([-5,-5],[start,stop],color=clrs_arealabels[ial],linestyle='-',linewidth=3)
    labeltext = '%s\nn=%d' % (al_fig[ial],stop-start)
    ax.text(-85,(start+stop)/2,labeltext,color=clrs_arealabels[ial],
               rotation=0,ha='right',va='center')
for ial,arealabel in enumerate(arealabeled):
    start,stop = np.where(al_sorted==arealabel)[0][0],np.where(al_sorted==arealabel)[0][-1]
    ax.plot([start,stop],[-5,-5],color=clrs_arealabels[ial],linestyle='-',linewidth=3)
    ax.text((start+stop)/2,-85,al_fig[ial],color=clrs_arealabels[ial],
               rotation=90,ha='center',va='bottom')
ax.set_xticks([0,np.shape(corrdata_sort)[0]-1])
ax.set_xticks([])
ax.set_yticks([])
cb = fig.colorbar(im,ax=ax,shrink=0.3,location='right',label='Noise\nCorrelation',aspect=10)
ax.yaxis.set_label_position("right")
plt.tight_layout()
my_savefig(fig,figdir,'CorrMatrix_V1PM_%s' % sessions[ises].session_id)

#%% 

#     # ###  #####  #######     #####  ####### ######  ######  
#     #  #  #     #    #       #     # #     # #     # #     # 
#     #  #  #          #       #       #     # #     # #     # 
#######  #   #####     #       #       #     # ######  ######  
#     #  #        #    #       #       #     # #   #   #   #   
#     #  #  #     #    #       #     # #     # #    #  #    #  
#     # ###  #####     #        #####  ####### #     # #     # 


def my_shuffle(data,method='random',axis=0):
    data = copy.deepcopy(data)
    if method == 'random':
        if axis == 0:
            for icol in range(data.shape[1]):
                data[:,icol] = np.random.permutation(data[:,icol])
        elif axis == 1:
            for irow in range(data.shape[0]):
                data[irow,:] = np.random.permutation(data[irow,:])
        elif axis is None:
            rng = np.random.default_rng()
            orig_size = data.shape
            data = np.random.permutation(data.ravel()).reshape(orig_size)

    elif method == 'circular':
        if axis == 0:
            for icol in range(data.shape[1]):
                data[:,icol] = np.roll(data[:,icol],shift=np.random.randint(0,data.shape[0]))
        elif axis == 1:
            for irow in range(data.shape[0]):
                data[irow,:] = np.roll(data[irow,:],shift=np.random.randint(0,data.shape[1])) 
    else:
        raise ValueError('method should be "random" or "circular"')
    return data

def corr_shuffle(sessions,method='random'):
    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing shuffled noise correlations: '):
        if hasattr(sessions[ises],'respmat'):
            data                                = my_shuffle(sessions[ises].respmat,axis=1,method=method)
            sessions[ises].corr_shuffle         = np.corrcoef(data)
            [N,K]                               = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            np.fill_diagonal(sessions[ises].corr_shuffle,np.nan)
    return sessions

#%%
# np.random.seed(0)
# sessions = corr_shuffle(sessions,method='random')

#%% Compute distribution of pairwise correlations across sessions conditioned on area pairs:

areapairs           = ['V1-V1','PM-PM','V1-PM']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

corr_type           = 'noise_corr'
layerpair           = ' '

bincenters,histcorr,meancorr,varcorr,fraccorr = hist_corr_areas_labeling(sessions,corr_type=corr_type,projpairs=projpairs,
                                                    areapairs=areapairs,layerpairs=layerpair,minNcells=params['minnneurons'],
                                                    # areapairs=areapairs,layerpairs=layerpair,minNcells=10,
                                                    filternear=True)

# bincenters_sh,histcorr_sh,meancorr_sh,varcorr_sh,fraccorr_sh = hist_corr_areas_labeling(sessions,corr_type='corr_shuffle',projpairs=projpairs,
#                                                                             noise_thr=params['maxnoiselevel'],
#                                                     # areapairs=[areapair],layerpairs=' ',minNcells=params['minnneurons'],
#                                                     areapairs=areapairs,layerpairs=layerpair,minNcells=params['minnneurons'],
#                                                     valuematching=None,filternear=True)

#%% print number of pairs:
npairs = np.zeros((len(areapairs),len(projpairs),nSessions))
for ises,ses in enumerate(sessions):
    # npairs[ises] = np.sum(~np.isnan(ses.noise_corr))/2

    nearfilter      = filter_nearlabeled(sessions[ises],radius=params['radius'])
    nearfilter      = np.meshgrid(nearfilter,nearfilter)
    nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
    corrdata        = sessions[ises].noise_corr
    for iap,areapair in enumerate(areapairs):
        for ipp,projpair in enumerate(projpairs):
                    
            signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<params['maxnoiselevel'],sessions[ises].celldata['noise_level']<params['maxnoiselevel'])
            signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

            areafilter      = filter_2d_areapair(sessions[ises],areapair)

            projfilter      = filter_2d_projpair(sessions[ises],projpair)

            nanfilter       = ~np.isnan(corrdata)

            proxfilter      = ~(sessions[ises].distmat_xy<10)

            cellfilter      = np.all((signalfilter,areafilter,
                                    projfilter,proxfilter,nanfilter,nearfilter),axis=0)

            npairs[iap,ipp,ises] = np.sum(cellfilter)

#%% Quantification of number of sessions and pairs for the interarea labeled situation:
ialp =2 #interarea
ilp = 0 #no layer specificity
ipp = 3 #lab-lab pair
print('%d/%d sessions with V1lab-PMlab populations'
        % (np.sum(~np.any(np.isnan(histcorr[:,:,ialp,ilp,ipp]),axis=0)),nSessions))

areapairs = ['V1unl-PMunl','V1unl-PMlab','V1lab-PMunl','V1lab-PMlab']
for ipp,projpair in enumerate(areapairs):
    print('%.1f +/- %.1f pairs per session for %s' % (np.nanmean(npairs[iap,ipp,:]),np.nanstd(npairs[iap,ipp,:]),projpair))

#%% 
#     # ### ####### #     # ### #     #       #    ######  #######    #    
#  #  #  #     #    #     #  #  ##    #      # #   #     # #         # #   
#  #  #  #     #    #     #  #  # #   #     #   #  #     # #        #   #  
#  #  #  #     #    #######  #  #  #  #    #     # ######  #####   #     # 
#  #  #  #     #    #     #  #  #   # #    ####### #   #   #       ####### 
#  #  #  #     #    #     #  #  #    ##    #     # #    #  #       #     # 
 ## ##  ###    #    #     # ### #     #    #     # #     # ####### #     # 

#%%
pairs = [
            ('V1-V1','PM-PM'),
            ('PM-PM','V1-PM'),
            ('V1-V1','V1-PM'),
         ] #for statistics

clrs_area_labelpairs = ['#818181',
                        '#818181',
                        '#818181']

areapairs           = ['V1-V1','PM-PM','V1-PM']

df                  = pd.DataFrame(data=meancorr[:,:,0,0],columns=areapairs)

fig,axes = plt.subplots(1,1,figsize=(2*cm,4*cm))
ax                  = axes
sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs,
            err_kws={'color': 'k','linewidth': 1})#,labels=legendlabels_upper_tri)
sns.stripplot(ax=ax,data=df,legend=False,color='black',size=1)

pvals = np.full((len(pairs)),np.nan)
for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    pvals[ipair]  = stats.wilcoxon(df.iloc[:,idx_1],df.iloc[:,idx_2])[1]

pvals = multipletests(pvals,method=params['multcomp_method'])[1]
for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    if pvals[ipair]:
        offset = ipair*0.005 + 0.03
        ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
        ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)

ax.set_ylabel('mean NC')
ax_nticks(ax,4)
sns.despine(fig=fig, top=True, right=True,offset=2)
ax.set_xticks(np.arange(len(areapairs)),labels=areapairs,rotation=90)
plt.tight_layout()
my_savefig(fig,figdir,'Noisecorr_Areas_%s_%dSessions' % (corr_type,nSessions))

#%%
projpairs_areas = [['V1unl-V1unl','V1unl-V1lab','V1lab-V1lab'],
             ['PMunl-PMunl','PMunl-PMlab','PMlab-PMlab']]

statpairs_areas = [[('V1unl-V1unl','V1lab-V1lab'),
         ('V1unl-V1unl','V1unl-V1lab'),
         ('V1unl-V1lab','V1lab-V1lab')],
           [('PMunl-PMunl','PMunl-PMlab'),
         ('PMunl-PMunl','PMlab-PMlab'),
         ('PMunl-PMlab','PMlab-PMlab'),
         ]] #for statistics

clrs_projpairs      = get_clr_labelpairs(['unl-unl','unl-lab','lab-lab'])

for iarea,area in enumerate(['V1','PM']):
    projpairs_areas[iarea]
    pairs = statpairs_areas[iarea]

    df                  = pd.DataFrame(data=meancorr[:,iarea,0,[0,1,3]],columns=projpairs_areas[iarea])

    fig,axes = plt.subplots(1,1,figsize=(2.5*cm,4*cm))
    ax                  = axes
    sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_projpairs,
                err_kws={'color': 'k','linewidth': 1})
    sns.stripplot(ax=ax,data=df,legend=False,color='black',size=1)
    # sns.lineplot(ax=ax,data=df.T,legend=False,lw=0.1,dashes=False,palette=['grey','grey'])

    pvals = np.full((len(pairs)),np.nan)
    for ipair,pair in enumerate(pairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        pvals[ipair]  = stats.wilcoxon(df.iloc[:,idx_1],df.iloc[:,idx_2],nan_policy='omit')[1]
        print('%s vs %s, p=%2.2g' % (pair[0],pair[1],pvals[ipair]))
    print('Wilcoxon signed rank test, n=%d sessions' % np.sum(~np.isnan(df.iloc[:,idx_1])))

    for ipair,pair in enumerate(pairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        if pvals[ipair]:
            offset = ipair*0.01 + 0.07
            ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
            ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                    get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)

    ax.set_ylabel('mean NC')
    ax.set_title('within %s' % area)
    ax_nticks(ax,4)
    sns.despine(fig=fig, top=True, right=True,offset=2)
    ax.set_xticks(np.arange(3),labels=arealabelpair_to_figlabel(projpairs_areas[iarea]),rotation=45,fontsize=5)

    my_savefig(fig,figdir,'Noisecorr_Area_%s_%s_%dSessions' % (area,corr_type,nSessions))

#%% 

### #     # ####### ####### ######     #    ######  #######    #    
 #  ##    #    #    #       #     #   # #   #     # #         # #   
 #  # #   #    #    #       #     #  #   #  #     # #        #   #  
 #  #  #  #    #    #####   ######  #     # ######  #####   #     # 
 #  #   # #    #    #       #   #   ####### #   #   #       ####### 
 #  #    ##    #    #       #    #  #     # #    #  #       #     # 
### #     #    #    ####### #     # #     # #     # ####### #     # 

#%%
ises = 19
# ises = 11
iap = 2

areaprojpairs = projpairs.copy()
for ipp,projpair in enumerate(projpairs):
    areaprojpairs[ipp]       = areapairs[iap].split('-')[0] + projpair.split('-')[0] + '-' + areapairs[iap].split('-')[1] + projpair.split('-')[1] 
areaprojpairs = arealabelpair_to_figlabel(areaprojpairs)
clrs_projpairs = get_clr_labelpairs(projpairs)

fig,axes = plt.subplots(1,1,figsize=(3.5*cm,3.5*cm),sharex=True,sharey=True)
ax = axes
ax.plot(bincenters,histcorr_sh[:,ises,iap,0,0],color='k',lw=0.7)
for ipp,projpair in enumerate(projpairs):
    ax.plot(bincenters,histcorr[:,ises,iap,0,ipp],color=clrs_projpairs[ipp],lw=0.7)
ax.set_xlim([-0.2,0.4])
ax.legend(['shuffle']  + areaprojpairs,fontsize=5)
my_legend_strip(ax)
ax.set_xlabel('NC')
ax.set_ylabel('Density (a.u)')
sns.despine(fig=fig,top=True,right=True)
plt.tight_layout()
my_savefig(fig,figdir,'Histcorr_Proj_%s_%s' % (areapairs[iap],corr_type))


#%%
areapairs = ['V1unl-PMunl','V1unl-PMlab','V1lab-PMunl','V1lab-PMlab']
statpairs = [('V1unl-PMunl','V1lab-PMunl'),
            ('V1unl-PMunl','V1unl-PMlab'),
            ('V1unl-PMunl','V1lab-PMlab'),
            ('V1unl-PMlab','V1lab-PMunl'),
            ('V1unl-PMlab','V1lab-PMlab'),
            ('V1lab-PMunl','V1lab-PMlab'),
            ] #for statistics

clrs_area_labelpairs = ['#929491',
                        "#C1707E",
                        "#DB7624",
                        '#C81D1D']
normalize = False

for data,title in zip([meancorr,varcorr],['mean','sd']):
    df                  = pd.DataFrame(data=data[:,2,0,:],columns=areapairs)
    if normalize:
        df = df.sub(df['V1unl-PMunl'],axis=0)
    fig,axes = plt.subplots(1,1,figsize=(2.5*cm,4*cm))
    ax                  = axes
    sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs,
                err_kws={'color': 'k','linewidth': 1})
    sns.stripplot(ax=ax,data=df,legend=False,color='black',size=1)
    # sns.lineplot(ax=ax,data=df.T,legend=False,lw=0.1,dashes=False,palette=['grey','grey'])
    pvals = np.full((len(statpairs)),np.nan)
    for ipair,pair in enumerate(statpairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        pvals[ipair] = stats.wilcoxon(df.iloc[:,idx_1],df.iloc[:,idx_2],nan_policy='omit')[1]

    pvals = multipletests(pvals,method=params['multcomp_method'])[1]
    for ipair,pair in enumerate(statpairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        print('%s vs %s, p=%2.2g' % (pair[0],pair[1],pvals[ipair]))
        if pvals[ipair]<0.05:
            offset = ipair*0.005 + 0.07
            ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
            ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                    get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)
    print('Wilcoxon signed rank test with bonferroni-holm multiple comparison correction, n=%d sessions' % np.sum(~np.isnan(df.iloc[:,idx_1])))

    ax.set_ylabel('%s NC' % (title))
    ax.set_title('across V1-PM')
    ax_nticks(ax,4)
    sns.despine(fig=fig, top=True, right=True,offset=2)
    ax.set_xticks(np.arange(4),labels=arealabelpair_to_figlabel(areapairs),rotation=45,fontsize=5)
    if title=='sd':
        ax.set_ylim([0.04,ax.get_ylim()[1]])
    my_savefig(fig,figdir,'%s_Noisecorr_Arealabeled_%dSessions' % (title,nSessions))


#%%

 #####  ###  #####      #####  ####### ######  ######  
#     #  #  #     #    #     # #     # #     # #     # 
#        #  #          #       #     # #     # #     # 
 #####   #  #  ####    #       #     # ######  ######  
      #  #  #     #    #       #     # #   #   #   #   
#     #  #  #     #    #     # #     # #    #  #    #  
 #####  ###  #####      #####  ####### #     # #     # 


#%% Compute distribution of pairwise correlations across sessions conditioned on area pairs:

areapairs           = ['V1-V1','PM-PM','V1-PM']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

corr_type           = 'sig_corr'
layerpair           = ' '

bincenters,histcorr,meancorr,varcorr,fraccorr = hist_corr_areas_labeling(sessions,corr_type=corr_type,projpairs=projpairs,
                                                    areapairs=areapairs,layerpairs=layerpair,minNcells=params['minnneurons'],
                                                    filternear=True)

#%%
areapairs = ['V1unl-PMunl','V1unl-PMlab','V1lab-PMunl','V1lab-PMlab']
statpairs = [('V1unl-PMunl','V1lab-PMunl'),
            ('V1unl-PMunl','V1unl-PMlab'),
            ('V1unl-PMunl','V1lab-PMlab'),
            ('V1unl-PMlab','V1lab-PMunl'),
            ('V1unl-PMlab','V1lab-PMlab'),
            ('V1lab-PMunl','V1lab-PMlab'),
            ] #for statistics

clrs_area_labelpairs = ['#929491',
                        "#C1707E",
                        "#DB7624",
                        '#C81D1D']
normalize = False

# for data,title in zip([meancorr,varcorr],['mean','sd']):
for data,title in zip([meancorr],['mean']):
    df                  = pd.DataFrame(data=data[:,2,0,:],columns=areapairs)
    if normalize:
        df = df.sub(df['V1unl-PMunl'],axis=0)
    fig,axes = plt.subplots(1,1,figsize=(2.5*cm,4*cm))
    ax                  = axes
    sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs,
                err_kws={'color': 'k','linewidth': 1})
    sns.stripplot(ax=ax,data=df,legend=False,color='black',size=1)
    # sns.lineplot(ax=ax,data=df.T,legend=False,lw=0.1,dashes=False,palette=['grey','grey'])
    pvals = np.full((len(statpairs)),np.nan)
    for ipair,pair in enumerate(statpairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        pvals[ipair] = stats.wilcoxon(df.iloc[:,idx_1],df.iloc[:,idx_2],nan_policy='omit')[1]

    pvals = multipletests(pvals,method=params['multcomp_method'])[1]
    for ipair,pair in enumerate(statpairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        print('%s vs %s, p=%2.2g' % (pair[0],pair[1],pvals[ipair]))
        if pvals[ipair]<0.05:
            offset = ipair*0.005 + 0.07
            ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
            ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                    get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)
    print('Wilcoxon signed rank test with bonferroni-holm multiple comparison correction, n=%d sessions' % np.sum(~np.isnan(df.iloc[:,idx_1])))

    ax.set_ylabel('%s SC' % (title))
    ax.set_title('across V1-PM')
    ax_nticks(ax,4)
    sns.despine(fig=fig, top=True, right=True,offset=2)
    ax.set_xticks(np.arange(4),labels=arealabelpair_to_figlabel(areapairs),rotation=45,fontsize=5)
    if title=='sd':
        ax.set_ylim([0.04,ax.get_ylim()[1]])
    my_savefig(fig,figdir,'%s_Sigcorr_Arealabeled_%dSessions' % (title,nSessions))


# #%%
# areapairs = ['V1unl-PMunl','V1unl-PMlab','V1lab-PMunl','V1lab-PMlab']
# statpairs = [('V1unl-PMunl','V1lab-PMunl'),
#             ('V1unl-PMunl','V1unl-PMlab'),
#             ('V1unl-PMunl','V1lab-PMlab'),
#             ('V1unl-PMlab','V1lab-PMunl'),
#             ('V1unl-PMlab','V1lab-PMlab'),
#             ('V1lab-PMunl','V1lab-PMlab'),
#             ] #for statistics

# clrs_area_labelpairs = ['#818181',
#                                 "#FA9CBB",
#                                 "#E6A77E",
#                                 '#FF4C4D']

# for data,title in zip([meancorr,varcorr],['Mean','SD']):
#     df                  = pd.DataFrame(data=data[:,0,0,:],columns=areapairs)
#     fig,axes = plt.subplots(1,1,figsize=(2*cm,4*cm))
#     ax                  = axes
#     sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs,
#                 err_kws={'color': 'k','linewidth': 1})
#     sns.stripplot(ax=ax,data=df,legend=False,color='black',size=1)

#     pvals = np.full((len(statpairs)),np.nan)
#     for ipair,pair in enumerate(statpairs):
#         idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
#         pvals[ipair] = stats.ttest_rel(df.iloc[:,idx_1],df.iloc[:,idx_2],nan_policy='omit')[1]

#     pvals = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
#     for ipair,pair in enumerate(statpairs):
#         idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
#         if pvals[ipair]:
#             offset = ipair*0.01 + 0.07
#             ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
#             ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
#                     get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)

#     ax.set_ylabel('%s Signal correlation' % (title))
#     ax.set_title(area)
#     ax_nticks(ax,4)
#     sns.despine(fig=fig, top=True, right=True,offset=2)
#     ax.set_xticks(np.arange(4),labels=arealabelpair_to_figlabel(areapairs),rotation=90)

#     my_savefig(fig,figdir,'%s_Sigcorr_Arealabeled_%dSessions' % (title,nSessions))



























#%% 

####### ######     #     #####      #####  ###  #####  
#       #     #   # #   #     #    #     #  #  #     # 
#       #     #  #   #  #          #        #  #       
#####   ######  #     # #           #####   #  #  #### 
#       #   #   ####### #                #  #  #     # 
#       #    #  #     # #     #    #     #  #  #     # 
#       #     # #     #  #####      #####  ###  #####  

#%%
areapairs           = ['V1-V1','PM-PM','V1-PM']

histdata    = np.cumsum(histcorr,axis=0)/100 #get cumulative distribution
histmean    = np.nanmean(histdata,axis=1) #get mean across sessions
histerror   = np.nanstd(histdata,axis=1) / np.sqrt(nSessions) #compute SEM

histdata_sh  = np.cumsum(histcorr_sh,axis=0)/100 #get cumulative distribution
histmean_sh = np.nanmean(histdata_sh,axis=1) #get mean across sessions
histerror_sh = np.nanstd(histdata_sh,axis=1) / np.sqrt(nSessions) #compute SEM
histmean_sh = np.nanmean(histmean_sh,axis=tuple(np.arange(1,np.ndim(histmean_sh))))
histerror_sh = np.nanmean(histerror_sh,axis=tuple(np.arange(1,np.ndim(histerror_sh))))

fraccorr = np.full(np.shape(fraccorr),np.nan)
histmean    = np.nanmean(histdata,axis=1) #get mean across sessions

for iap,areapair in enumerate(areapairs): #show for each projection identity pair:
    for ipp,projpair in enumerate(projpairs): #show for each projection identity pair:
        for ises in range(nSessions):
            tempdata = histdata_sh[:,ises,iap,:,ipp].squeeze()
            if not np.isnan(tempdata).any():
                thr_min     = np.where(tempdata>=params['alpha_corrshuf'])[0][0] #get threshold)
                thr_max     = np.where(tempdata>=(1-params['alpha_corrshuf']))[0][0] #get threshold)

                fraccorr[0,ises,iap,0,ipp] = histdata[thr_min,ises,iap,0,ipp] #get threshold)
                fraccorr[1,ises,iap,0,ipp] = 1-histdata[thr_max,ises,iap,0,ipp] #get threshold)

#%% 
iap = 2
areapair = areapairs[iap]

if areapair=='V1-PM':
    test_indices = np.array([[0,1],[0,2],[1,2],[2,3],[0,3],[1,3]])
else: 
    test_indices = np.array([[0,1],[0,3],[1,3]])

fig,axes = plt.subplots(1,2,figsize=(6*cm,3.5*cm),sharex=True,sharey=False)
for isign, sign in enumerate(['neg','pos']):
    ax = axes[isign]
    sns.stripplot(fraccorr[isign,:,iap,0,:].squeeze(),ax=ax,legend=False,
                  palette=clrs_projpairs,
                  color='black',
                    s=2)
    sns.barplot(fraccorr[isign,:,iap,0,:].squeeze(),ax=ax,legend=False,estimator='mean',alpha=0.3,
                palette=clrs_projpairs,errorbar=('ci',95))

    # sns.scatterplot(fraccorr_sh[isign].squeeze().T,ax=ax,legend=False,palette=np.repeat('grey',nSessions),markers='o')
    # sns.barplot(fraccorr_sh[isign].squeeze(),ax=ax,legend=False,estimator='mean',palette=clrs_projpairs,errorbar=('ci',95))

    pvals = np.empty(len(test_indices))
    for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
        data1 = fraccorr[isign,:,iap,0,ix]
        data2 = fraccorr[isign,:,iap,0,iy]
        pvals[itest] = stats.ttest_rel(data1,data2,nan_policy='omit')[1]

    pvals = multipletests(pvals,alpha=0.05,method=params['method_multcomp'])[1]
    for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
        yloc = np.nanmean([data1,data2])
        if pvals[itest]<0.05:
            ax.plot([ix,iy],np.repeat(yloc,2)+0.1+0.025*itest,'k-',linewidth=1)
            ax.text(np.mean([ix,iy]),yloc+0.1+0.025*itest,get_sig_asterisks(pvals[itest]),fontsize=8) #
    ax_nticks(ax,5)
# ax.set_ylim([0,np.nanpercentile(fraccorr,100)])
axes[0].set_title('Fraction sign. neg.')
axes[1].set_title('Fraction sign. pos.')
ax.set_xticks(np.arange(len(projpairs)),areaprojpairs)
# ax.set_ylim([0,1])
sns.despine(fig=fig,top=True,right=True,trim=True,offset=2)
axes[0].set_xticklabels(areaprojpairs,rotation=45)
axes[1].set_xticklabels(areaprojpairs,rotation=45)
my_savefig(fig,figdir,'FracCorr_%s_%s' % (areapair,corr_type))

#%% 
iap = 2
areapair = areapairs[iap]

if areapair=='V1-PM':
    test_indices = np.array([[0,1],[0,2],[1,2],[2,3],[0,3],[1,3]])
else: 
    test_indices = np.array([[0,1],[0,3],[1,3]])

fig,axes = plt.subplots(1,1,figsize=(3.5*cm,3.5*cm),sharex=True,sharey=False)
ax = axes
modcorr = np.sum(fraccorr,axis=0)
sns.stripplot(modcorr[:,iap,0,:].squeeze(),ax=ax,legend=False,
                palette=clrs_projpairs,
                color='black',
                s=2)
sns.barplot(modcorr[:,iap,0,:].squeeze(),ax=ax,legend=False,estimator='mean',alpha=0.3,
            palette=clrs_projpairs,errorbar=('ci',95))

# sns.scatterplot(fraccorr_sh[isign].squeeze().T,ax=ax,legend=False,palette=np.repeat('grey',nSessions),markers='o')
# sns.barplot(fraccorr_sh[isign].squeeze(),ax=ax,legend=False,estimator='mean',palette=clrs_projpairs,errorbar=('ci',95))

pvals = np.empty(len(test_indices))
for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
    data1 = fraccorr[isign,:,iap,0,ix]
    data2 = fraccorr[isign,:,iap,0,iy]
    pvals[itest] = stats.ttest_rel(data1,data2,nan_policy='omit')[1]

pvals = multipletests(pvals,alpha=0.05,method=params['method_multcomp'])[1]
for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
    yloc = np.nanmean([data1,data2])
    if pvals[itest]<0.05:
        ax.plot([ix,iy],np.repeat(yloc,2)+0.1+0.025*itest,'k-',linewidth=1)
        ax.text(np.mean([ix,iy]),yloc+0.1+0.025*itest,get_sig_asterisks(pvals[itest]),fontsize=8) #
ax_nticks(ax,5)
# ax.set_ylim([0,np.nanpercentile(fraccorr,100)])
axes[0].set_title('Fraction sign. neg.')
axes[1].set_title('Fraction sign. pos.')
ax.set_xticks(np.arange(len(projpairs)),areaprojpairs)
# ax.set_ylim([0,1])
sns.despine(fig=fig,top=True,right=True,trim=True,offset=2)
axes[0].set_xticklabels(areaprojpairs,rotation=45)
axes[1].set_xticklabels(areaprojpairs,rotation=45)
# my_savefig(fig,figdir,'FracCorr_%s_%s' % (areapair,corr_type))


######     #    ######  ### #     #  #####      #####  ####### #     # ####### ######  ####### #       
#     #   # #   #     #  #  #     # #     #    #     # #     # ##    #    #    #     # #     # #       
#     #  #   #  #     #  #  #     # #          #       #     # # #   #    #    #     # #     # #       
######  #     # #     #  #  #     #  #####     #       #     # #  #  #    #    ######  #     # #       
#   #   ####### #     #  #  #     #       #    #       #     # #   # #    #    #   #   #     # #       
#    #  #     # #     #  #  #     # #     #    #     # #     # #    ##    #    #    #  #     # #       
#     # #     # ######  ###  #####   #####      #####  ####### #     #    #    #     # ####### ####### 

#%% Get distribution of pairwise correlations across sessions conditioned on area pairs:
areapairs           = ['V1-PM']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
layerpair           = ' '

corr_type           = 'noise_corr'
params['min_nearbycells'] = 2

#%% Bootstrapped comparison of correlations and significant correlations with other area: 
# The distribution of correlations is compared to the loop correlation distribution.
# The fraction of significantly positive and negative as well. 
radii           = np.arange(10,200,20)
nradii          = len(radii)

fraccorr_radius = np.full((2,nradii,nSessions,len(projpairs)),np.nan)

for irad,radius in enumerate(radii):
    print(radius)
    # df_mean,df_frac     = mean_corr_areas_labeling(sessions,corr_type=corr_type,
    #                                             absolute=True,filternear=True,
    #                                             minNcells=params['minnneurons'],radius=radius,
    #                                             maxnoiselevel=params['maxnoiselevel'])

    bincenters,histcorr,meancorr,varcorr,fraccorr = hist_corr_areas_labeling(sessions,corr_type=corr_type,projpairs=projpairs,
                                                                            noise_thr=params['maxnoiselevel'],
                                                    areapairs=areapairs,layerpairs=layerpair,minNcells=5,
                                                    filternear=True,radius=radius,min_nearbycells=params['min_nearbycells'])

    bincenters_sh,histcorr_sh,meancorr_sh,varcorr_sh,fraccorr_sh = hist_corr_areas_labeling(sessions,corr_type='corr_shuffle',projpairs=projpairs,
                                                                            noise_thr=params['maxnoiselevel'],
                                                    areapairs=areapairs,layerpairs=layerpair,minNcells=5,
                                                    filternear=True,radius=radius,min_nearbycells=params['min_nearbycells'])

    histdata    = np.cumsum(histcorr,axis=0)/100 #get cumulative distribution
    histmean    = np.nanmean(histdata,axis=1) #get mean across sessions
    histerror   = np.nanstd(histdata,axis=1) / np.sqrt(nSessions) #compute SEM

    histdata_sh  = np.cumsum(histcorr_sh,axis=0)/100 #get cumulative distribution
    # histmean_sh = np.nanmean(histdata_sh,axis=1) #get mean across sessions
    # histmean_sh = np.nanmean(histmean_sh,axis=tuple(np.arange(1,np.ndim(histmean_sh))))

    for ipp,projpair in enumerate(projpairs): #show for each projection identity pair:
        for ises in range(nSessions):
            tempdata = histdata_sh[:,ises,0,0,ipp].squeeze()
            if not np.isnan(tempdata).any():
                thr_min     = np.where(tempdata>=params['alpha_corrshuf'])[0][0] #get threshold)
                thr_max     = np.where(tempdata>=(1-params['alpha_corrshuf']))[0][0] #get threshold)

                fraccorr_radius[0,irad,ises,ipp] = histdata[thr_min,ises,iap,0,ipp] #get threshold)
                fraccorr_radius[1,irad,ises,ipp] = 1-histdata[thr_max,ises,iap,0,ipp] #get threshold)

fraccorr_mod = np.sum(fraccorr_radius,axis=0)

#%% Plot as a function of radius:
areapairs = ['V1unl-PMunl','V1unl-PMlab','V1lab-PMunl','V1lab-PMlab']
statpairs = [('V1unl-PMunl','V1lab-PMunl'),
            ('V1unl-PMunl','V1unl-PMlab'),
            ('V1unl-PMunl','V1lab-PMlab'),
            ('V1unl-PMlab','V1lab-PMunl'),
            ('V1unl-PMlab','V1lab-PMlab'),
            ('V1lab-PMunl','V1lab-PMlab'),
            ] #for statistics

clrs_projpairs = ['#818181',
                                "#FA9CBB",
                                "#E6A77E",
                                '#FF4C4D']

test_indices = np.array([[0,1],[0,2],[1,2],[2,3],[0,3],[1,3]])
test_indices = np.array([[0,1],[0,2],[0,3]])
# test_indices = np.array([[0,3]])

for i,(data,modlabel) in enumerate(zip([fraccorr_mod,fraccorr_radius[0],fraccorr_radius[1]],['mod','neg','pos'])):
    # data = fraccorr_radius[0]
    # data = fraccorr_radius[1]

    handles = []
    fig,ax = plt.subplots(1,1,figsize=(5*cm,3.5*cm))
    for ipp,projpair in enumerate(projpairs):
        handles.append(shaded_error(x=radii,y=data[:,:,ipp].T,error='sem',color=clrs_projpairs[ipp],
                                    alpha=0.25,ax=ax,linewidth=1))

    ax.set_xlabel('Radius (um)')
    ax.set_ylabel('Fraction sign. correlated')
    for irad,radius in enumerate(radii):

        pvals = np.empty(len(test_indices))
        for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
            data1 = data[irad,:,ix]
            data2 = data[irad,:,iy]
            pvals[itest] = stats.ttest_rel(data1,data2,nan_policy='omit')[1]

        pvals = multipletests(pvals,alpha=0.05,method=params['method_multcomp'])[1]
        for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
            data1 = data[irad,:,ix]
            data2 = data[irad,:,iy]
            yloc = np.nanmax([np.nanmean(data1),np.nanmean(data2)])
            if pvals[itest]<0.05:
                ax.text(radius,yloc,get_sig_asterisks(pvals[itest]),fontsize=5,fontweight='bold',
                        ha='center',va='bottom',color=clrs_projpairs[iy],rotation=45) 
                
    ax.legend(handles,arealabelpair_to_figlabel(areapairs),loc='best',bbox_to_anchor=(0.8,0.4),reverse=True)
    my_legend_strip(ax)
    ax_nticks(ax,4)
    ax.set_title('%s' % modlabel)
    ax.set_xticks(np.arange(40,200+40,40))
    sns.despine(fig=fig, top=True, right=True,offset=2)
    my_savefig(fig,figdir,'Frac_Sig_NC_%s_AreaLabeled_Radii_%dSessions' % (modlabel,nSessions))

#%% 











######  ####### ######  ######  #######  #####     #    ####### ####### ######  
#     # #       #     # #     # #       #     #   # #      #    #       #     # 
#     # #       #     # #     # #       #        #   #     #    #       #     # 
#     # #####   ######  ######  #####   #       #     #    #    #####   #     # 
#     # #       #       #   #   #       #       #######    #    #       #     # 
#     # #       #       #    #  #       #     # #     #    #    #       #     # 
######  ####### #       #     # #######  #####  #     #    #    ####### ######  





# #%%
# fig,axes = plt.subplots(1,1,figsize=(5*cm,5*cm),sharex=True,sharey=True)
# ax = axes
# sns.stripplot(meancorr.squeeze(),ax=ax,legend=False,
#                 palette=clrs_projpairs,
#                 color='black',
#                 s=3)
# sns.barplot(meancorr.squeeze(),ax=ax,legend=False,estimator='mean',alpha=0.3,
#             palette=clrs_projpairs,errorbar=('ci',95))

# pvals = np.empty(len(test_indices))
# for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
#     data1 = meancorr[:,0,0,ix]
#     data2 = meancorr[:,0,0,iy]
#     pvals[itest] = stats.ttest_rel(data1,data2,nan_policy='omit')[1]

# pvals = multipletests(pvals,alpha=0.05,method=params['method_multcomp'])[1]
# for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
#     yloc = np.nanmean([data1,data2])
#     if pvals[itest]<0.05:
#         ax.plot([ix,iy],np.repeat(yloc,2)+0.06+0.015*itest,'k-',linewidth=1)
#         ax.text(np.mean([ix,iy]),yloc+0.06+0.015*itest,get_sig_asterisks(pvals[itest]),fontsize=9) #
# ax_nticks(ax,5)
# ax.set_ylabel('Mean. noise correlation')
# ax.set_xticks(np.arange(len(projpairs)),areaprojpairs)
# # plt.tight_layout()
# # ax.set_ylim([0,1])
# sns.despine(fig=fig,top=True,right=True,trim=True,offset=2)
# my_savefig(fig,figdir,'MeanCorr_%s_%s' % (areapair,corr_type))

# #%%
# fig,axes = plt.subplots(1,1,figsize=(5*cm,5*cm),sharex=True,sharey=True)
# ax = axes
# sns.stripplot(varcorr.squeeze(),ax=ax,legend=False,
#                 palette=clrs_projpairs,
#                 color='black',jitter=0.15,
#                 s=3)
# sns.barplot(varcorr.squeeze(),ax=ax,legend=False,estimator='mean',alpha=0.3,
#             palette=clrs_projpairs,errorbar=('ci',90))

# pvals = np.empty(len(test_indices))
# for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
#     data1 = varcorr[:,0,0,ix]
#     data2 = varcorr[:,0,0,iy]
#     pvals[itest] = stats.ttest_rel(data1,data2,nan_policy='omit')[1]

# pvals = multipletests(pvals,alpha=0.05,method=params['method_multcomp'])[1]
# for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
#     yloc = np.nanmean([data1,data2])
#     if pvals[itest]<0.05:
#         ax.plot([ix,iy],np.repeat(yloc,2)+0.06+0.015*itest,'k-',linewidth=1)
#         ax.text(np.mean([ix,iy]),yloc+0.06+0.015*itest,get_sig_asterisks(pvals[itest]),fontsize=9) #
# ax_nticks(ax,5)
# # ax.set_ylim([0,np.nanpercentile(varcorr,99)])
# ax.set_ylabel('Std. noise correlation')
# ax.set_xticks(np.arange(len(projpairs)),areaprojpairs)
# # plt.tight_layout()
# # ax.set_ylim([0,1])
# sns.despine(fig=fig,top=True,right=True,trim=True,offset=2)

# my_savefig(fig,figdir,'StdCorr_%s_%s' % (areapair,corr_type))

# #%%
# fig,axes = plt.subplots(1,1,figsize=(5*cm,5*cm),sharex=True,sharey=True)
# ax = axes
# varcorr -= varcorr[:,:,:,0][:,:,:,None]
# sns.stripplot(varcorr.squeeze(),ax=ax,legend=False,
#                 palette=clrs_projpairs,
#                 color='black',jitter=0.15,
#                 s=3)
# sns.barplot(varcorr.squeeze(),ax=ax,legend=False,estimator='mean',alpha=0.3,
#             palette=clrs_projpairs,errorbar=('ci',90))

# pvals = np.empty(len(test_indices))
# for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
#     data1 = varcorr[:,0,0,ix]
#     data2 = varcorr[:,0,0,iy]
#     pvals[itest] = stats.ttest_rel(data1,data2,nan_policy='omit')[1]

# pvals = multipletests(pvals,alpha=0.05,method=params['method_multcomp'])[1]
# for itest,(ix,iy) in enumerate(zip(test_indices[:,0],test_indices[:,1])):
#     yloc = np.nanmean([data1,data2])
#     if pvals[itest]<0.05:
#         ax.plot([ix,iy],np.repeat(yloc,2)+0.06+0.015*itest,'k-',linewidth=1)
#         ax.text(np.mean([ix,iy]),yloc+0.06+0.015*itest,get_sig_asterisks(pvals[itest]),fontsize=9) #
# ax_nticks(ax,5)
# # ax.set_ylim([0,np.nanpercentile(varcorr,99)])
# ax.set_ylabel('Std. noise correlation')
# ax.set_xticks(np.arange(len(projpairs)),areaprojpairs)
# # plt.tight_layout()
# # ax.set_ylim([0,1])
# sns.despine(fig=fig,top=True,right=True,trim=True,offset=2)

# my_savefig(fig,figdir,'StdCorr_Norm_%s_%s' % (areapair,corr_type))

