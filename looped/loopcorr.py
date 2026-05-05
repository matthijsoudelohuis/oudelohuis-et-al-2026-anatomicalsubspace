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

os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from params import load_params
from loaddata.session_info import *
from loaddata.get_data_folder import get_local_drive
from utils.pair_lib import *
from utils.plot_lib import * #get all the fixed color schemes

# savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\Looping\\')

from utils.corr_lib import *
from utils.tuning import compute_tuning_wrapper
# from utils.shuffle_lib import my_shuffle, corr_shuffle
# from utils.gain_lib import * 
# 
savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\Looping\\NoiseCorrelations')

#%% Plotting and parameters:
params  = load_params()
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches
params['method_multcomp'] = 'bonferroni'

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
# session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR')
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
my_savefig(fig,savedir,'CorrMatrix_V1PM_%s' % sessions[ises].session_id)

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
np.random.seed(0)
sessions = corr_shuffle(sessions,method='random')

#%% Plot distribution of pairwise correlations across sessions conditioned on area pairs:

areapairs           = ['V1-V1','PM-PM','V1-PM']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']

corr_type           = 'noise_corr'
# areapair            = 'V1-PM'
# areapair           = 'V1-V1'
# areapair           = 'PM-PM'
layerpair = ' '

bincenters,histcorr,meancorr,varcorr,fraccorr = hist_corr_areas_labeling(sessions,corr_type=corr_type,projpairs=projpairs,
                                                                            noise_thr=params['maxnoiselevel'],
                                                    # areapairs=[areapair],layerpairs=layerpair,minNcells=params['minnneurons'],
                                                    areapairs=areapairs,layerpairs=layerpair,minNcells=params['minnneurons'],
                                                    valuematching=None,filternear=True,radius=params['radius'])

bincenters_sh,histcorr_sh,meancorr_sh,varcorr_sh,fraccorr_sh = hist_corr_areas_labeling(sessions,corr_type='corr_shuffle',projpairs=projpairs,
                                                                            noise_thr=params['maxnoiselevel'],
                                                    # areapairs=[areapair],layerpairs=' ',minNcells=params['minnneurons'],
                                                    areapairs=areapairs,layerpairs=layerpair,minNcells=params['minnneurons'],
                                                    valuematching=None,filternear=True,radius=params['radius'])

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
print('%d/%d sessions with V1lab-PMlab populations'
        % (np.sum(~np.any(np.isnan(histcorr[:,:,2,0,-1]),axis=0)),nSessions))

iap = 2
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

areapairs = ['V1-V1','PM-PM','V1-PM']

df                  = pd.DataFrame(data=meancorr[:,:,0,0],columns=areapairs)

fig,axes = plt.subplots(1,1,figsize=(2*cm,4*cm))
ax                  = axes
sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs,
            err_kws={'color': 'k','linewidth': 1})#,labels=legendlabels_upper_tri)
sns.stripplot(ax=ax,data=df,legend=False,color='black',size=1)

pvals = np.full((len(pairs)),np.nan)
for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    pvals[ipair]  = stats.ttest_rel(df.iloc[:,idx_1],df.iloc[:,idx_2])[1]

pvals = multipletests(pvals,alpha=0.05,method=params['method_multcomp'])[1]
for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    if pvals[ipair]:
        offset = ipair*0.005 + 0.03
        ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
        ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)

ax.set_ylabel('Noise correlation')
ax_nticks(ax,4)
sns.despine(fig=fig, top=True, right=True,offset=1)
ax.set_xticks(np.arange(len(areapairs)),labels=areapairs,rotation=90)
plt.tight_layout()

# my_savefig(fig,savedir,'Noisecorr_Areas_%s_%dSessions' % (corr_type,nSessions))

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

fig,axes = plt.subplots(1,2,figsize=(5,3.5),sharey=True)
for iarea,area in enumerate(['V1','PM']):
    projpairs_areas[iarea]
    pairs = statpairs_areas[iarea]

    df                  = pd.DataFrame(data=meancorr[:,iarea,0,[0,1,3]],columns=projpairs_areas[iarea])

    fig,axes = plt.subplots(1,1,figsize=(2*cm,4*cm))
    ax                  = axes
    sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_projpairs,
                err_kws={'color': 'k','linewidth': 1})
    sns.stripplot(ax=ax,data=df,legend=False,color='black',size=1)

    pvals = np.full((len(pairs)),np.nan)
    for ipair,pair in enumerate(pairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        pvals[ipair]  = stats.ttest_rel(df.iloc[:,idx_1],df.iloc[:,idx_2],nan_policy='omit')[1]

    pvals = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
    for ipair,pair in enumerate(pairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        if pvals[ipair]:
            offset = ipair*0.01 + 0.07
            ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
            ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                    get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)

    ax.set_ylabel('Noise correlation')
    ax.set_title('within %s' % area)
    ax_nticks(ax,4)
    sns.despine(fig=fig, top=True, right=True,offset=1)
    ax.set_xticks(np.arange(3),labels=arealabelpair_to_figlabel(projpairs_areas[iarea]),rotation=90)

    my_savefig(fig,savedir,'Noisecorr_Area_%s_%s_%dSessions' % (area,corr_type,nSessions))

#%% 

### #     # ####### ####### ######     #    ######  #######    #    
 #  ##    #    #    #       #     #   # #   #     # #         # #   
 #  # #   #    #    #       #     #  #   #  #     # #        #   #  
 #  #  #  #    #    #####   ######  #     # ######  #####   #     # 
 #  #   # #    #    #       #   #   ####### #   #   #       ####### 
 #  #    ##    #    #       #    #  #     # #    #  #       #     # 
### #     #    #    ####### #     # #     # #     # ####### #     # 

#%%
ises = 11
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
ax.set_xlabel('Noise Correlation')
ax.set_ylabel('Density (a.u)')
sns.despine(fig=fig,top=True,right=True)
plt.tight_layout()
my_savefig(fig,savedir,'Histcorr_Proj_%s_%s' % (areapairs[iap],corr_type))



#%%
areapairs = ['V1unl-PMunl','V1unl-PMlab','V1lab-PMunl','V1lab-PMlab']
statpairs = [('V1unl-PMunl','V1lab-PMunl'),
            ('V1unl-PMunl','V1unl-PMlab'),
            ('V1unl-PMunl','V1lab-PMlab'),
            ('V1unl-PMlab','V1lab-PMunl'),
            ('V1unl-PMlab','V1lab-PMlab'),
            ('V1lab-PMunl','V1lab-PMlab'),
            ] #for statistics

clrs_area_labelpairs = ['#818181',
                                "#FA9CBB",
                                "#E6A77E",
                                '#FF4C4D']
normalize = False
for data,title in zip([meancorr,varcorr],['Mean','SD']):
    df                  = pd.DataFrame(data=data[:,2,0,:],columns=areapairs)
    if normalize:
        df = df.sub(df['V1unl-PMunl'],axis=0)
    fig,axes = plt.subplots(1,1,figsize=(2*cm,4*cm))
    ax                  = axes
    sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs,
                err_kws={'color': 'k','linewidth': 1})
    sns.stripplot(ax=ax,data=df,legend=False,color='black',size=1)
    # sns.lineplot(ax=ax,data=df.T,legend=False,lw=0.3,color='k')
    pvals = np.full((len(statpairs)),np.nan)
    for ipair,pair in enumerate(statpairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        pvals[ipair] = stats.ttest_rel(df.iloc[:,idx_1],df.iloc[:,idx_2],nan_policy='omit')[1]

    pvals = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
    for ipair,pair in enumerate(statpairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        if pvals[ipair]:
            offset = ipair*0.01 + 0.07
            ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
            ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                    get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)

    ax.set_ylabel('%s Noise correlation' % (title))
    ax.set_title(area)
    ax_nticks(ax,4)
    sns.despine(fig=fig, top=True, right=True,offset=1)
    ax.set_xticks(np.arange(4),labels=arealabelpair_to_figlabel(areapairs),rotation=90)

    my_savefig(fig,savedir,'%s_Noisecorr_Arealabeled_%dSessions' % (title,nSessions))

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
sns.despine(fig=fig,top=True,right=True,trim=True,offset=1)
axes[0].set_xticklabels(areaprojpairs,rotation=45)
axes[1].set_xticklabels(areaprojpairs,rotation=45)
my_savefig(fig,savedir,'FracCorr_%s_%s' % (areapair,corr_type))

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
sns.despine(fig=fig,top=True,right=True,trim=True,offset=1)
axes[0].set_xticklabels(areaprojpairs,rotation=45)
axes[1].set_xticklabels(areaprojpairs,rotation=45)
# my_savefig(fig,savedir,'FracCorr_%s_%s' % (areapair,corr_type))

#%%

 #####  ###  #####      #####  ####### ######  ######  
#     #  #  #     #    #     # #     # #     # #     # 
#        #  #          #       #     # #     # #     # 
 #####   #  #  ####    #       #     # ######  ######  
      #  #  #     #    #       #     # #   #   #   #   
#     #  #  #     #    #     # #     # #    #  #    #  
 #####  ###  #####      #####  ####### #     # #     # 


#%% Plot distribution of pairwise correlations across sessions conditioned on area pairs:

areapairs           = ['V1-PM']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
corr_type           = 'sig_corr'
layerpair = ' '

bincenters,histcorr,meancorr,varcorr,fraccorr = hist_corr_areas_labeling(sessions,corr_type=corr_type,projpairs=projpairs,
                                                                            noise_thr=params['maxnoiselevel'],
                                                    areapairs=areapairs,layerpairs=layerpair,minNcells=params['minnneurons'],
                                                    valuematching=None,filternear=True,radius=params['radius'])

#%%
areapairs = ['V1unl-PMunl','V1unl-PMlab','V1lab-PMunl','V1lab-PMlab']
statpairs = [('V1unl-PMunl','V1lab-PMunl'),
            ('V1unl-PMunl','V1unl-PMlab'),
            ('V1unl-PMunl','V1lab-PMlab'),
            ('V1unl-PMlab','V1lab-PMunl'),
            ('V1unl-PMlab','V1lab-PMlab'),
            ('V1lab-PMunl','V1lab-PMlab'),
            ] #for statistics

clrs_area_labelpairs = ['#818181',
                                "#FA9CBB",
                                "#E6A77E",
                                '#FF4C4D']

for data,title in zip([meancorr,varcorr],['Mean','SD']):
    df                  = pd.DataFrame(data=data[:,0,0,:],columns=areapairs)
    fig,axes = plt.subplots(1,1,figsize=(2*cm,4*cm))
    ax                  = axes
    sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs,
                err_kws={'color': 'k','linewidth': 1})
    sns.stripplot(ax=ax,data=df,legend=False,color='black',size=1)

    pvals = np.full((len(statpairs)),np.nan)
    for ipair,pair in enumerate(statpairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        pvals[ipair] = stats.ttest_rel(df.iloc[:,idx_1],df.iloc[:,idx_2],nan_policy='omit')[1]

    pvals = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
    for ipair,pair in enumerate(statpairs):
        idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
        if pvals[ipair]:
            offset = ipair*0.01 + 0.07
            ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
            ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                    get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)

    ax.set_ylabel('%s Signal correlation' % (title))
    ax.set_title(area)
    ax_nticks(ax,4)
    sns.despine(fig=fig, top=True, right=True,offset=1)
    ax.set_xticks(np.arange(4),labels=arealabelpair_to_figlabel(areapairs),rotation=90)

    my_savefig(fig,savedir,'%s_Sigcorr_Arealabeled_%dSessions' % (title,nSessions))



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
    sns.despine(fig=fig, top=True, right=True,offset=3)
    my_savefig(fig,savedir,'Frac_Sig_NC_%s_AreaLabeled_Radii_%dSessions' % (modlabel,nSessions))

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
# sns.despine(fig=fig,top=True,right=True,trim=True,offset=1)
# my_savefig(fig,savedir,'MeanCorr_%s_%s' % (areapair,corr_type))

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
# sns.despine(fig=fig,top=True,right=True,trim=True,offset=1)

# my_savefig(fig,savedir,'StdCorr_%s_%s' % (areapair,corr_type))

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
# sns.despine(fig=fig,top=True,right=True,trim=True,offset=1)

# my_savefig(fig,savedir,'StdCorr_Norm_%s_%s' % (areapair,corr_type))











#%%
plt.rcParams['axes.spines.right']   = True
plt.rcParams['axes.spines.top']     = True


fig         = plt.figure(figsize=(10*cm, 6*cm))
gspec       = fig.add_gridspec(nrows=2, ncols=3)

histdata    = np.cumsum(histcorr,axis=0)/100 #get cumulative distribution
histmean    = np.nanmean(histdata,axis=1) #get mean across sessions
histerror   = np.nanstd(histdata,axis=1) / np.sqrt(len(ses)) #compute SEM

histdata_sh  = np.cumsum(histcorr_sh,axis=0)/100 #get cumulative distribution
histmean_sh = np.nanmean(histdata_sh,axis=1) #get mean across sessions
histerror_sh = np.nanstd(histdata_sh,axis=1) / np.sqrt(len(ses)) #compute SEM
histmean_sh = np.nanmean(histmean_sh,axis=tuple(np.arange(1,np.ndim(histmean_sh))))
histerror_sh = np.nanmean(histerror_sh,axis=tuple(np.arange(1,np.ndim(histerror_sh))))

ax0         = fig.add_subplot(gspec[:2, :2]) #bigger subplot for the cum dist

xpos = bincenters[np.where(np.nanmean(histmean,axis=3).squeeze()<0.1)[0][-1]]
axins1 = ax0.inset_axes([0.05, 0.25, 0.3, 0.4],xlim=([xpos-0.05,xpos+0.025]),ylim=[0,0.2],xticklabels=[], yticklabels=[])
ax0.indicate_inset_zoom(axins1, edgecolor="black")
axins1.tick_params(axis='both', which='both', length=0)
for axis in ['top','bottom','left','right']:
    axins1.spines[axis].set_color('gray')
    axins1.spines[axis].set_linewidth(1)

xpos = bincenters[np.where(np.nanmean(histmean,axis=3).squeeze()>0.9)[0][0]]
axins2 = ax0.inset_axes([0.65, 0.25, 0.3, 0.4],xlim=([xpos-0.05,xpos+0.05]),ylim=[0.8,1],xticklabels=[], yticklabels=[])
ax0.indicate_inset_zoom(axins2, edgecolor="gray")
axins2.tick_params(axis='both', which='both', length=0)
for axis in ['top','bottom','left','right']:
    axins2.spines[axis].set_color('gray')
    axins2.spines[axis].set_linewidth(1)

handles = []
for ipp,projpair in enumerate(projpairs): #show for each projection identity pair:
    ax0.plot(bincenters,np.squeeze(histmean[:,0,0,ipp]),color=clrs_projpairs[ipp],linewidth=0.3)
    axins1.plot(bincenters,np.squeeze(histmean[:,0,0,ipp]),color=clrs_projpairs[ipp])
    axins2.plot(bincenters,np.squeeze(histmean[:,0,0,ipp]),color=clrs_projpairs[ipp])
    # plot triangle for mean:
    ax0.plot(np.nanmean(meancorr[:,0,0,ipp],axis=None),0.9+ipp/50,'v',color=clrs_projpairs[ipp],markersize=5)

handles.append(shaded_error(x=bincenters,y=histmean_sh,
                    yerror=histerror_sh,ax=ax0,color='k',linewidth=1))
axins1.plot(bincenters,histmean_sh,color='k')
axins2.plot(bincenters,histmean_sh,color='k')  

ax0.set_xlabel('Correlation')
ax0.set_ylabel('Cumulative Fraction')
ax0.legend(handles=handles,labels=areaprojpairs,frameon=False,loc='upper left')
ax0.set_xlim([-0.25,0.35])
ax0.set_xlim([-0.15,0.25])
ax0.set_xlim([-0.1,0.20])
if zscoreflag:
    ax0.set_xlim([-2,2])
ax0.axvline(0,linewidth=0.5,linestyle=':',color='k') #add line at zero for ref
ax0.set_ylim([0,1])
# ax0.set_ylim([0,0.15])
ax0.set_title('%s %s' % (areapair,corr_type))

#  Now show a heatmap of the meancorr data averaged over sessions (first dimension). 
#  Between each projpair a paired t-test is done of the mean across sesssions and if significant a line is 
#  drawn from the center of that entry of the heatmap and other one with an asterisk on top of the line. 
#  For subplot 3 the same is done but then with varcorr.
data        = np.squeeze(np.nanmean(meancorr[:,0,:,:],axis=0))
data        = np.reshape(data,(2,2))

xlabels     = [areapair.split('-')[1] + 'unl',areapair.split('-')[1] + 'lab'] 
ylabels     = [areapair.split('-')[0] + 'unl',areapair.split('-')[0] + 'lab'] 
xlocs        = np.array([0,1,0,1])
ylocs        = np.array([0,0,1,1])
if areapair=='V1-PM':
    test_indices = np.array([[0,1],[0,2],[1,2],[2,3],[0,3],[1,3]])
else: 
    test_indices = np.array([[0,1],[0,3],[1,3]])

ax1 = fig.add_subplot(gspec[0, 2])
pcm = ax1.imshow(data,cmap='hot',vmin=my_floor(np.min(data)-0.002,2),vmax=my_ceil(np.max(data),2))
ax1.set_xticks([0,1],labels=xlabels)
ax1.xaxis.tick_top()
ax1.set_yticks([0,1],labels=ylabels)
ax1.set_title('Mean')
fig.colorbar(pcm, ax=ax1)

for ix,iy in zip(test_indices[:,0],test_indices[:,1]):
    data1 = meancorr[:,0,0,ix]
    data2 = meancorr[:,0,0,iy]
    pval = stats.ttest_rel(data1,data2,nan_policy='omit')[1]
    # pval = stats.ttest_rel(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
    # pval = stats.wilcoxon(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
    # pval = pval * 3 #bonferroni correction
    # print(pval)
    if pval<0.05:
        ax1.plot([xlocs[ix],xlocs[iy]],[ylocs[ix],ylocs[iy]],'k-',linewidth=1)
        ax1.text(np.mean([xlocs[ix],xlocs[iy]])-0.15,np.mean([ylocs[ix],ylocs[iy]]),get_sig_asterisks(pval),
                            weight='bold') #

# Now the same but for the std of the pairwise correlations:
data        = np.squeeze(np.nanmean(varcorr[:,0,:,:],axis=0))
data        = np.reshape(data,(2,2))

ax2 = fig.add_subplot(gspec[1, 2])
pcm = ax2.imshow(data,cmap='hot',vmin=my_floor(np.min(data)-0.002,2),vmax=my_ceil(np.max(data),2))
ax2.set_xticks([0,1],labels=xlabels)
ax2.xaxis.tick_top()
ax2.set_yticks([0,1],labels=ylabels)
ax2.set_title('Std')
fig.colorbar(pcm, ax=ax2)

for ix,iy in zip(test_indices[:,0],test_indices[:,1]):
    data1 = varcorr[:,0,0,ix]
    data2 = varcorr[:,0,0,iy]
    pval = stats.ttest_rel(data1,data2,nan_policy='omit')[1]
    # pval = stats.wilcoxon(data1[~np.isnan(data1) & ~np.isnan(data2)],data2[~np.isnan(data1) & ~np.isnan(data2)])[1]
    # pval = pval * 6 #bonferroni correction
    # print(pval)
    if pval<0.05:
        ax2.plot([xlocs[ix],xlocs[iy]],[ylocs[ix],ylocs[iy]],'k-',linewidth=1)
        ax2.text(np.mean([xlocs[ix],xlocs[iy]])-0.15,np.mean([ylocs[ix],ylocs[iy]]),get_sig_asterisks(pval),
                            weight='bold')

plt.tight_layout()
# fig.savefig(os.path.join(savedir,'HistCorr','Histcorr_Proj_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')




#%% #########################################################################################
# Contrast: across areas, layers and projection pairs:
areapairs           = ['V1-V1','PM-PM','V1-PM']
layerpairs          = ['L2/3-L2/3','L2/3-L5','L5-L2/3','L5-L5']
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
#If you override any of these with input to the deltarf bin function as ' ', then these pairs will be ignored

clrs_areapairs      = get_clr_area_pairs(areapairs)
clrs_layerpairs     = get_clr_layerpairs(layerpairs)
clrs_projpairs      = get_clr_labelpairs(projpairs)

# clrs_area_labelpairs = get_clr_area_labelpairs(areapairs+projpairs)
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% #####################################################################################################
# DELTA ANATOMICAL DISTANCE :
# #######################################################################################################

#%% Define the areapairs:
areapairs       = ['V1-V1','PM-PM']
clrs_areapairs  = get_clr_area_pairs(areapairs)

#%% Compute pairwise correlations as a function of pairwise anatomical distance ###################################################################
# for corr_type in ['noise_corr','sig_corr','noise_corr']:
for corr_type in ['noise_corr']:
    [binmean,binedges] = bin_corr_distance(sessions,areapairs,corr_type=corr_type)

    #Make the figure per protocol:
    fig = plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type=corr_type)
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'Corr_anatomicaldist_Protocols_' % (corr_type) + '.pdf'), format = 'pdf')

#%% #########################################################################################
protocols           = ['GR','GN']
# protocols           = ['IM']
ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]

areapairs           = ['V1-V1','PM-PM']
layerpairs          = ' '
projpairs           = ['unl-unl','unl-lab','lab-lab']
clrs_projpairs      = get_clr_labelpairs(projpairs)

corr_type           = 'noise_corr'
# corr_type           = 'noise_corr'
# corr_thr            = 0.025 #thr in percentile of total corr for significant pos or neg

[bincenters_2d,bin_2d_mean,bin_2d_count,bin_dist_mean,bin_dist_count,bincenters_dist,
bin_angle_cent_mean,bin_angle_cent_count,bin_angle_surr_mean,
bin_angle_surr_count,bincenters_angle] = bin_corr_deltaxy(ses,onlysameplane=False,
                        areapairs=areapairs,layerpairs=layerpairs,projpairs=projpairs,
                        method='mean',filtersign=None,corr_type=corr_type,binresolution=25,noise_thr=100)

#%% Plot:
binidx = bincenters_dist>10
fig,axes = plt.subplots(1,2,sharey=True,figsize=(6.5,3.5))
for iap,areapair in enumerate(['V1','PM']):
    ax = axes[iap]
    dim12label = 'XY (um)'
    handles = []
    for ipp,projpair in enumerate(projpairs):
        bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.5
        handles.append(shaded_error(x=bincenters_dist[binidx],y=bin_dist_mean[binidx,iap,0,ipp],yerror=bin_dist_error[binidx,iap,0,ipp],
                        ax = ax,color=clrs_projpairs[ipp],label=projpair))
        # ax.plot(bincenters_dist,bin_dist_mean[:,iap,0,ipp],color=clrs_projpairs[ipp],label=projpair)
    ax.set_title(areapair)
    if iap==0:
        ax.set_ylabel('Correlation')
    ax.legend(handles=handles,labels=projpairs,frameon=False)
    ax.set_xlabel(u'Δ %s' % dim12label)   
    ax.set_yticks(np.arange(0,0.051,0.01))
    ax.set_xlim([-10,500])
    # ax.set_ylim([0.0,0.023])
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'MeanCorr','DistXY_MeanCorr_WithinArea_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'MeanCorr','DistXY_MeanCorr_WithinArea_%s_PCA1_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')

#%% ########################################################################################################
# ##################### Noise correlations within and across areas: ########################################
# ##########################################################################################################

dfses = mean_corr_areas_labeling([sessions[0]],corr_type='noise_corr',absolute=True,minNcells=100)[0]
clrs_area_labelpairs = get_clr_area_labelpairs(list(dfses.columns))

#%% Compute the variance across trials for each cell:
# for ses in sessions:
#     if ses.sessiondata['protocol'][0]=='GR':
#         resp_meanori,respmat_res        = mean_resp_gr(ses)
#     elif ses.sessiondata['protocol'][0]=='GN':
#         resp_meanori,respmat_res        = mean_resp_gn(ses)
#     ses.celldata['noise_variance']  = np.var(respmat_res,axis=1)

#%% Plot distribution of pairwise correlations across sessions conditioned on area pairs:
protocols           = ['GR','GN']
# protocols           = ['IM']
# corr_type           = 'noise_corr'
corr_type           = 'noise_corr'
projpairs           = ['unl-unl','unl-lab','lab-lab']
clrs_projpairs      = get_clr_labelpairs(projpairs)

ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]

bincenters,_,meancorrV1,varcorrV1,fraccorrV1 = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=False,projpairs=projpairs,noise_thr=20,
                                                    areapairs=['V1-V1'],layerpairs=' ',minNcells=10)
bincenters,_,meancorrPM,varcorrPM,fraccorrPM = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=False,projpairs=projpairs,noise_thr=20,
                                                    areapairs=['PM-PM'],layerpairs=' ',minNcells=10)

#%% Plot within area mean pairwise correlations:
#combine V1 and PM, and filter out duplicate pair (unl-lab is same as lab-unl):
# meancorr            = np.stack([meancorrV1[:,:,:,[0,1,3]],meancorrPM[:,:,:,[0,1,3]]],axis=0) 
meancorr            = np.stack([meancorrV1,meancorrPM],axis=0)
# meancorr             = np.stack([fraccorrV1[0,:,:,:,:],fraccorrPM[0,:,:,:,:]],axis=0)
# meancorr             = np.stack([fraccorrV1[1,:,:,:,:],fraccorrPM[1,:,:,:,:]],axis=0)

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

fig,axes = plt.subplots(1,2,figsize=(5,3.5),sharey=True)
# for isign,sign in enumerate(['pos','neg']):
for iarea,area in enumerate(['V1','PM']):
    df                  = pd.DataFrame(data=meancorr[iarea,:,:,:,:].squeeze(),columns=projpairs_areas[iarea])
    df                  = df.dropna(axis=0,thresh=2).reset_index(drop=True) #drop occasional missing data

    ax                  = axes[iarea]
    if df.any(axis=None):
        ax.scatter(np.arange(3),df.mean(axis=0),marker='o',s=15,color='k')
        ax.plot(np.arange(3),df.mean(axis=0),linestyle='-',color='k')
        sns.stripplot(ax=ax,data=df,color='grey',size=3,palette=clrs_projpairs,jitter=0.15)
        ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
        # annotator = Annotator(ax, statpairs_areas[iarea], data=df,order=list(df.columns))
        # annotator.configure(test='Wilcoxon', text_format='star', loc='inside',line_height=0,text_offset=-0.5,fontsize=7,	
        #                     line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=0,
        #                     correction_format='replace')
        # annotator.apply_and_annotate()
        ax.set_ylabel('Correlation')
        # ax.set_title('%s' % '_'.join(protocols),fontsize=12)
        ax.set_ylim([my_floor(df.min(axis=None)*0.9,3),my_ceil(df.max(axis=None)*1.1,3)])
        # ax.set_ylim([0,my_ceil(df.max(axis=None)*1.2,2)])
    ax.set_title('%s' % (area),fontsize=12)
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_WithinArea_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')

#%% Plot distribution of pairwise correlations across sessions conditioned on area pairs:
protocols           = ['GR','GN']
# protocols           = ['IM']
# corr_type           = 'noise_corr'
corr_type           = 'noise_corr'
projpairs           = ['unl-unl','unl-lab','lab-unl','lab-lab']
clrs_projpairs      = get_clr_labelpairs(projpairs)

ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]

bincenters,_,meancorr,varcorr,fraccorr = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=False,projpairs=projpairs,noise_thr=20,
                                                    areapairs=['V1-PM'],layerpairs=' ',minNcells=10)

#%% 
dfses = mean_corr_areas_labeling([sessions[0]],corr_type=corr_type,absolute=True,minNcells=1000)[0]

statpairs = [('V1unl-PMunl','V1lab-PMunl'),
         ('V1unl-PMunl','V1unl-PMlab'),
         ('V1unl-PMunl','V1lab-PMlab'),
         ('V1unl-PMlab','V1lab-PMunl'),
         ('V1unl-PMlab','V1lab-PMlab'),
         ('V1lab-PMunl','V1lab-PMlab'),
         ] #for statistics

fig,axes = plt.subplots(1,1,figsize=(3.5,3.5))
df                  = pd.DataFrame(data=meancorr[:,:,:,:].squeeze(),columns=dfses.columns[-4:])
df                  = df.dropna(axis=0,thresh=2).reset_index(drop=True) #drop occasional missing data
df                  = df.fillna(df.mean()) #interpolate occasional missing data

ax                  = axes
if df.any(axis=None):
    ax.scatter(np.arange(4),df.mean(axis=0),marker='o',s=15,color='k')
    ax.plot(np.arange(4),df.mean(axis=0),linestyle='-',color='k')
    sns.stripplot(ax=ax,data=df,color='grey',size=3,palette=clrs_projpairs)
    ax.set_xticklabels(labels=df.columns,rotation=60,fontsize=8)
    # annotator = Annotator(ax, statpairs, data=df,order=list(df.columns))
    # annotator.configure(test='Wilcoxon', text_format='star', loc='inside',line_height=0,text_offset=3,fontsize=8,	
    #                     line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
    #                     correction_format='replace')
    # annotator.apply_and_annotate()
    ax.set_ylabel('Correlation')
    ax.set_title('%s' % '_'.join(protocols),fontsize=12)
    ax.set_ylim([my_floor(df.min(axis=None)*0.95,3),my_ceil(df.max(axis=None)*1.4,2)])
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_InterArea_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')

#%% 
dfses = mean_corr_areas_labeling([sessions[0]],corr_type='noise_corr',absolute=True,minNcells=1000)[0]

statpairs = [('V1unl-PMunl','V1lab-PMunl'),
         ('V1unl-PMunl','V1unl-PMlab'),
         ('V1unl-PMunl','V1lab-PMlab'),
         ('V1unl-PMlab','V1lab-PMunl'),
         ('V1unl-PMlab','V1lab-PMlab'),
         ('V1lab-PMunl','V1lab-PMlab'),
         ] #for statistics

fig,axes = plt.subplots(1,2,figsize=(6.5,3.5))
for isign,sign in enumerate(['pos','neg']):
    df                  = pd.DataFrame(data=fraccorr[isign,:,:,:,:].squeeze(),columns=dfses.columns[-4:])
    df                  = df.dropna(axis=0,thresh=2).reset_index(drop=True) #drop occasional missing data
    df                  = df.fillna(df.mean()) #interpolate occasional missing data

    ax                  = axes[isign]
    if df.any(axis=None):
        ax.scatter(np.arange(4),df.mean(axis=0),marker='o',s=15,color='k')
        ax.plot(np.arange(4),df.mean(axis=0),linestyle='-',color='k')
        sns.stripplot(ax=ax,data=df,color='grey',size=3,palette=clrs_projpairs,jitter=0.15)
        ax.set_xticklabels(labels=df.columns,rotation=60,fontsize=8)
        # annotator = Annotator(ax, statpairs, data=df,order=list(df.columns))
        # annotator.configure(test='Wilcoxon', text_format='star', loc='inside',line_height=0,text_offset=2,fontsize=8,	
        #                     line_width=1,verbose=True,
        #                     correction_format='replace')
        # annotator.apply_and_annotate()
        ax.set_ylabel('Frac. correlated pairs')
        # ax.set_title('%s' % '_'.join(protocols),fontsize=12)
        ax.set_ylim([my_floor(df.min(axis=None)*0.9,1),my_ceil(df.max(axis=None)*1.4,1)])
    ax.set_title('%s' % (sign),fontsize=12)
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_PosNeg_InterArea_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')

#%%  Find session with largest effect:
# areapairs           = ['V1-PM']
# corr_type           = 'noise_corr'
# ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]
# bincenters,histcorr,meancorr,varcorr = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=False,projpairs=projpairs,noise_thr=20,
                                                    # areapairs=['V1-PM'],layerpairs=' ',minNcells=10)

sesidx = np.argmax(np.nanvar(meancorr[:,0,0,:],axis=-1))
sesidx = np.nanargmax(meancorr[:,0,0,3]-meancorr[:,0,0,0])
print('Session with largest difference in mean %s by labeling across areas is:\n%s' % (corr_type,ses[sesidx].sessiondata['session_id'][0]))

sesidx = np.argmax(np.nanvar(varcorr[:,0,0,:],axis=-1))
# sesidx = np.nanargmax(meancorr[:,0,0,3]-meancorr[:,0,0,0])

print('Session with largest difference in %s variance by labeling across areas is:\n%s' % (corr_type,ses[sesidx].sessiondata['session_id'][0]))


#%% Show correlation matrix for this session.
# sort by area and labeling identity:
# sesidx = 8
ses = sessions[sesidx]
corr_type = 'noise_corr'
data = getattr(ses,corr_type)
arealabels = ['V1unl','V1lab','PMunl','PMlab']

# sort rows and columns by area and label:
sortidx     = np.flip(np.argsort(ses.celldata['arealabel']))
sortlabels  = np.flip(np.sort(ses.celldata['arealabel']))

# sort rows and columns by area and label, and sort by mean correlation to all other cells:
avgcorrcells = np.nanmean(data,axis=0)
sortidx     = np.flip(np.lexsort((avgcorrcells,ses.celldata['arealabel'])))
sortlabels  = ses.celldata['arealabel'][sortidx]

data        = data[sortidx,:][:,sortidx]
# data        = data[sortidx,:]#[:,sortidx]

fig,ax = plt.subplots(figsize=(6,6))
pcm = ax.imshow(data,cmap='bwr',clim=(-my_ceil(np.nanpercentile(data,95),2),my_ceil(np.nanpercentile(data,95),2)))
# pcm = ax.imshow(data,cmap='bwr',clim=(-my_ceil(np.nanpercentile(data,95),2),my_ceil(np.nanpercentile(data,95),2)))
for al in arealabels:
    ax.axhline(y=np.where(sortlabels==al)[0][0],color='k',linestyle='-',linewidth=0.5)
    ax.axvline(x=np.where(sortlabels==al)[0][0],color='k',linestyle='-',linewidth=0.5)
ax.set_xticks([]); ax.set_yticks([])
# fig.colorbar(pcm, ax=ax)

cb = fig.colorbar(pcm, ax=ax,shrink=0.3)
cb.set_label('Correlation',fontsize=10,loc='center')
cb.set_ticks([cb.vmin,0,cb.vmax])
# fig.savefig(os.path.join(savedir,'CorrMat','CorrMat_%s_%s' % (corr_type,sessions[sesidx].sessiondata['session_id'][0]) + '.png'), format = 'png')



#%% Plot mean vs standard deviation for labeling across areapairs:
# Umakantha et al. 2023: might signal different population activity fluctuations that are shared

areapairs           = ['V1-V1','PM-PM','V1-PM']
zscoreflag      = True
circres         = 0.25
tickres         = 0.2
lim             = 1.7

# for corr_type in ['noise_corr','sig_corr','noise_corr']:
for corr_type in ['noise_corr']:
    fig,axes = plt.subplots(1,3,figsize=(9,3))
    for iap,areapair in enumerate(areapairs):
        ax                  = axes[iap]
        ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'].isin(protocols))[0]]
        
        bincenters,histcorr,meancorr,varcorr,_ = hist_corr_areas_labeling(ses,corr_type=corr_type,filternear=True,projpairs=projpairs,
                                                            areapairs=[areapair],layerpairs=' ',minNcells=10,zscore=zscoreflag)

        for ipp,projpair in enumerate(projpairs):
            # ax.scatter(meancorr[:,0,0,ipp],varcorr[:,0,0,ipp],c=clrs_projpairs[ipp],s=4,alpha=0.7)
            ax.errorbar(np.nanmean(meancorr[:,0,0,ipp]),np.nanmean(varcorr[:,0,0,ipp]),
                        np.nanstd(meancorr[:,0,0,ipp]) / np.sqrt(len(ses)),np.nanstd(varcorr[:,0,0,ipp])/ np.sqrt(len(ses)),
                        ecolor=clrs_projpairs[ipp],elinewidth=1,capsize=3)
        ax.set_xlabel('Mean')
        ax.set_ylabel('Std')

        ax.set_xticks(np.arange(0,lim,tickres))
        ax.set_yticks(np.arange(0,lim,tickres))
        ax.set_xlim([0,lim])
        ax.set_ylim([0,lim])
        # ax.set_xlim([0,my_ceil(np.nanmax(varcorr),2)])
        # ax.set_ylim([0,my_ceil(np.nanmax(varcorr),2)])
        ax.set_title(areapair)

        for radius in np.arange(0,lim*2,circres):
            Drawing_uncolored_circle = plt.Circle( (0, 0), radius, linestyle=':',fill=False)
            ax.add_artist(Drawing_uncolored_circle)
        # ax0.legend(frameon=False,loc='upper left',fontsize=8)
        # ax0.set_xlim([-0.5,0.5])
        # ax0.set_ylim([0,1.1])
    plt.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdScatter_Z_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdMean_Z_PCA1_%s_%s' % (corr_type,'_'.join(protocols)) + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdScatter_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanStdScatter_%s_%s_%s' % (areapair,corr_type,'_'.join(protocols)) + '.pdf'), format = 'pdf')
        
# clrs_areapairs      = get_clr_area_pairs(areapairs)
# clrs_layerpairs     = get_clr_layerpairs(layerpairs)
# clrs_projpairs      = get_clr_labelpairs(projpairs)

#%% Plot mean absolute correlation across sessions conditioned on area pairs and per protocol:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

fig,axes = plt.subplots(1,3,figsize=(12,4),sharex=True,sharey='row')
# for iprot,prot in enumerate(['GR','GN','IM']):
    # ax                  = axes[iprot]
    # ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'] == prot)[0]]
df_mean,df_frac     = mean_corr_areas_labeling(sessions,corr_type='noise_corr',absolute=True,filternear=True,minNcells=10)
df                  = df_mean
df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
df                  = df.fillna(df.mean()) #interpolate occasional missing data

if df.any(axis=None):
    sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
    ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
    # annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
    # annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=-5,text_offset=0, 
    #                     line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
    #                     # line_width=1,comparisons_correction=None,verbose=False,
    #                     correction_format='replace')
    # annotator.apply_and_annotate()
    ax.set_ylabel('Correlation')
plt.suptitle('%s' % (corr_type),fontsize=12)
plt.tight_layout()

#%% Plot mean correlation across sessions conditioned on area pairs and per protocol for pos and neg separately:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

for corr_type in ['noise_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(2,3,figsize=(12,6),sharex=True,sharey='row')
    # for iprot,prot in enumerate(['GR','GN','IM']):
    for iprot,prot in enumerate(['SP']):
        for isign,sign in enumerate(['pos','neg']):
            ax                  = axes[isign,iprot]
            ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'] == prot)[0]]
            df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,filtersign=sign,filternear=True,minNcells=10)
            df                  = df_mean
            df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
            df                  = df.fillna(df.mean()) #interpolate occasional missing data
            # df                  = df.dropna() #drop sessions with occasional missing data

            if df.any(axis=None):
                sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
                if isign==1:
                    ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
                else: ax.set_xticks([])
                if isign==1: 
                    ax.invert_yaxis()
                annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
                annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=0.05,text_offset=0, 
                                    line_width=0.5,comparisons_correction='Benjamini-Hochberg',verbose=False,fontsize=7,
                                    correction_format='replace')
                annotator.apply_and_annotate()
                if isign==1:
                    ax.invert_yaxis()
                ax.set_ylabel('%s correlation' % sign)
                ax.set_title('%s' %(prot),fontsize=12)
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','MeanCorr_sigOnly_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')

#%% Plot fraction of correlated units across sessions conditioned on area pairs and per protocol:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

for corr_type in ['noise_corr','sig_corr','noise_corr']:
    fig,axes = plt.subplots(2,3,figsize=(12,6),sharex=True,sharey='row')
    for iprot,prot in enumerate(['GR','GN','IM']):
        for isign,sign in enumerate(['pos','neg']):
            ax                  = axes[isign,iprot]
            ses                 = [sessions[ises] for ises in np.where(sessiondata['protocol'] == prot)[0]]
            df_mean,df_frac     = mean_corr_areas_labeling(ses,corr_type=corr_type,filtersign=sign,filternear=True,minNcells=10)
            df                  = df_frac
            df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
            df                  = df.fillna(df.mean()) #interpolate occasional missing data
            
            if df.any(axis=None):
                sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs)#,labels=legendlabels_upper_tri)
                if isign==1:
                    ax.set_xticklabels(labels=df.columns,rotation=90,fontsize=8)
                else: ax.set_xticks([])

                annotator = Annotator(ax, pairs, data=df,order=list(df.columns))
                annotator.configure(test='t-test_paired', text_format='star', loc='inside',line_height=0,line_offset_to_group=-5,text_offset=0, 
                                    line_width=1,comparisons_correction='Benjamini-Hochberg',verbose=False,
                                    correction_format='replace')
                annotator.apply_and_annotate()
                ax.set_ylabel('Fraction of %s correlated units' % sign)
                ax.set_title('%s' %(prot),fontsize=12)
            # ax.set_ylim([0,1])
    plt.suptitle('%s' % (corr_type),fontsize=12)
    plt.tight_layout()
    # fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')
    fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_stationary_Labeling_Areas_perProtocol_%s' % corr_type + '.png'), format = 'png')
    fig.savefig(os.path.join(savedir,'MeanCorr','FracCorr_dF_stationary_Labeling_Areas_perProtocol_%s' % corr_type + '.pdf'), format = 'pdf')

#%%















#%% Plot mean absolute correlation across sessions conditioned on area pairs:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

corr_type           = 'noise_corr'
absolute            = False
abslabel            = 'Abs' if absolute else ''
df_mean,df_frac     = mean_corr_areas_labeling(sessions,corr_type=corr_type,
                                               absolute=absolute,filternear=False,
                                               minNcells=params['minnneurons'],radius=params['radius'],maxnoiselevel=params['maxnoiselevel'])

#%% 
pairs = [
            ('V1unl-V1unl','PMunl-PMunl'),
            ('PMunl-PMunl','V1unl-PMunl'),
            ('V1unl-V1unl','V1unl-PMunl'),
         ] #for statistics

clrs_area_labelpairs = ['#818181',
                        '#818181',
                        '#818181',
                        ]

df                  = df_mean[['V1unl-V1unl','PMunl-PMunl','V1unl-PMunl']]
df                  = df.dropna().reset_index(drop=True) #drop occasional missing data
df                  = df.fillna(df.mean()) #interpolate occasional missing data
arealabelpairs      = arealabelpair_to_figlabel(df.columns)

fig,axes = plt.subplots(1,1,figsize=(2*cm,4*cm))
ax                  = axes
sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs,
            err_kws={'color': 'k','linewidth': 1})#,labels=legendlabels_upper_tri)
sns.stripplot(ax=ax,data=df,legend=False,color='black',size=0.5)

pvals = np.full((len(pairs)),np.nan)
for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    pvals[ipair]  = stats.ttest_rel(df.iloc[:,idx_1],df.iloc[:,idx_2])[1]

pvals = multipletests(pvals,alpha=0.05,method=params['method_multcomp'])[1]
for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    if pvals[ipair]:
        offset = ipair*0.003 + 0.01
        ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
        ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)

ax.set_ylabel('%s Noise correlation' % (abslabel))
ax_nticks(ax,4)
sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticks(np.arange(len(arealabelpairs)),labels=arealabelpairs,rotation=90)
plt.tight_layout()

my_savefig(fig,savedir,'%sNoisecorr_Areas_%s_%dSessions' % (abslabel,corr_type,nSessions))

#%% 
pairs = [
            ('V1unl-V1unl','V1unl-V1lab'),
            ('V1unl-V1unl','V1lab-V1lab'),
            ('V1unl-V1lab','V1lab-V1lab'),

            ('PMunl-PMunl','PMunl-PMlab'),
            ('PMunl-PMunl','PMlab-PMlab'),
            ('PMunl-PMlab','PMlab-PMlab'),

            ('V1unl-PMunl','V1lab-PMunl'),
            ('V1unl-PMunl','V1unl-PMlab'),
            ('V1unl-PMunl','V1lab-PMlab'),
            ('V1unl-PMlab','V1lab-PMunl'),
            ('V1unl-PMlab','V1lab-PMlab'),
            ('V1lab-PMunl','V1lab-PMlab'),
         ] #for statistics

clrs_area_labelpairs = ['#818181',
                                "#D69393",
                                '#FF4C4D',
                                '#818181',
                                "#D69393",
                                '#FF4C4D',
                                '#818181',
                                "#FA9CBB",
                                "#E6A77E",
                                '#FF4C4D',
                                ]

df                  = df_mean
df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
arealabelpairs      = arealabelpair_to_figlabel(df.columns)

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax                  = axes
sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs,
            err_kws={'color': 'k','linewidth': 0.5})#,labels=legendlabels_upper_tri)
sns.stripplot(ax=ax,data=df,legend=False,color='black',size=0.5)

pvals = np.full((len(pairs)),np.nan)
for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    pvals[ipair]  = stats.ttest_rel(df.iloc[:,idx_1],df.iloc[:,idx_2])[1]

print(pvals[-1])
pvals = multipletests(pvals,alpha=0.05,method='fdr_bh')[1]
print(pvals[-1])

for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    if pvals[ipair]:
        offset = ipair*0.01 + 0.03
        ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
        ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)

ax.set_ylabel('Abs. noise correlation')
ax_nticks(ax,4)
sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticks(np.arange(len(arealabelpairs)),labels=arealabelpairs,rotation=90)
plt.tight_layout()

# my_savefig(fig,savedir,'Noisecorr_AreaLabeled_%dSessions' % (nSessions))

#%% 

pairs = [
            ('V1unl-V1unl','V1unl-V1lab'),
            ('V1unl-V1unl','V1lab-V1lab'),
            ('V1unl-V1lab','V1lab-V1lab'),

            ('PMunl-PMunl','PMunl-PMlab'),
            ('PMunl-PMunl','PMlab-PMlab'),
            ('PMunl-PMlab','PMlab-PMlab'),

            ('V1unl-PMunl','V1lab-PMunl'),
            ('V1unl-PMunl','V1unl-PMlab'),
            ('V1unl-PMunl','V1lab-PMlab'),
            ('V1unl-PMlab','V1lab-PMunl'),
            ('V1unl-PMlab','V1lab-PMlab'),
            ('V1lab-PMunl','V1lab-PMlab'),
         ] #for statistics

clrs_area_labelpairs = ['#818181',
                                "#D69393",
                                '#FF4C4D',
                                '#818181',
                                "#D69393",
                                '#FF4C4D',
                                '#818181',
                                "#FA9CBB",
                                "#E6A77E",
                                '#FF4C4D',
                                ]

df                  = df_mean
df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
arealabelpairs      = arealabelpair_to_figlabel(df.columns)

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax                  = axes
sns.barplot(ax=ax,data=df,estimator="mean",errorbar='se',palette=clrs_area_labelpairs,
            err_kws={'color': 'k','linewidth': 0.5})#,labels=legendlabels_upper_tri)
sns.stripplot(ax=ax,data=df,legend=False,color='black',size=0.5)

pvals = np.full((len(pairs)),np.nan)
for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    pvals[ipair]  = stats.ttest_rel(df.iloc[:,idx_1],df.iloc[:,idx_2])[1]

print(pvals[-1])
pvals = multipletests(pvals,alpha=0.05,method='fdr_bh')[1]
print(pvals[-1])

for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    if pvals[ipair]:
        offset = ipair*0.01 + 0.03
        ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
        ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)

ax.set_ylabel('Abs. noise correlation')
ax_nticks(ax,4)
sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticks(np.arange(len(arealabelpairs)),labels=arealabelpairs,rotation=90)
plt.tight_layout()

# my_savefig(fig,savedir,'Noisecorr_AreaLabeled_%dSessions' % (nSessions))



#%% Plot mean absolute correlation across sessions conditioned on area pairs:
sessiondata    = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

corr_type = 'sig_corr'
df_mean,df_frac     = mean_corr_areas_labeling(sessions,corr_type=corr_type,
                                               absolute=False,filternear=True,
                                               minNcells=params['minnneurons'],radius=params['radius'],maxnoiselevel=params['maxnoiselevel'])

#%% 
df                  = df_mean
# df                  = df.dropna(axis=0,thresh=8).reset_index(drop=True) #drop occasional missing data
df                  = df.dropna().reset_index(drop=True) #drop occasional missing data
df                  = df.fillna(df.mean()) #interpolate occasional missing data
arealabelpairs      = arealabelpair_to_figlabel(df.columns)

fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax                  = axes
sns.barplot(ax=ax,data=df,estimator="median",errorbar='se',palette=clrs_area_labelpairs,
            err_kws={'color': 'k','linewidth': 0.5})#,labels=legendlabels_upper_tri)
sns.stripplot(ax=ax,data=df,legend=False,color='black',size=0.5)

pvals = np.full((len(pairs)),np.nan)
for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    pvals[ipair]  = stats.ttest_rel(df.iloc[:,idx_1],df.iloc[:,idx_2])[1]

pvals = multipletests(pvals,alpha=0.05,method='fdr_bh')[1]

for ipair,pair in enumerate(pairs):
    idx_1,idx_2 = df.columns.get_loc(pair[0]),df.columns.get_loc(pair[1])
    if pvals[ipair]:
        offset = ipair*0.005 + 0.01
        ax.plot([idx_1,idx_2],[df.iloc[:,idx_1].mean()+offset,df.iloc[:,idx_1].mean()+offset],color='k',lw=0.5)
        ax.text(np.mean([idx_1,idx_2]),df.iloc[:,idx_1].mean()+offset+0.0025,
                get_sig_asterisks(pvals[ipair],return_ns=True),color='k',ha='center',va='center',fontsize=5)

ax.set_ylabel('Signal correlation')
ax.set_ylim(np.nanpercentile(df.values,1),np.nanpercentile(df.values,98))
ax_nticks(ax,4)

sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticks(np.arange(len(arealabelpairs)),labels=arealabelpairs,rotation=90)
# plt.tight_layout()

my_savefig(fig,savedir,'Sigcorr_AreaLabeled_%s_%dSessions' % (corr_type,nSessions))

#%% Show as a function of different radius from looped cells:

corr_type = 'noise_corr'

