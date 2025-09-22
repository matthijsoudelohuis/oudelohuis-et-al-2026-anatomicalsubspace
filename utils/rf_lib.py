# -*- coding: utf-8 -*-
"""
Set of function used for analysis of mouse behavior in visual navigation task
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025
"""

import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
import seaborn as sns
from tqdm import tqdm
from scipy import ndimage
from scipy.stats import zscore,ttest_rel,multivariate_normal
from scipy.sparse.linalg import svds
from sklearn.metrics import r2_score
from tqdm import tqdm

from utils.rf_lib import *
from utils.imagelib import load_natural_images
from utils.pair_lib import *
from utils.tuning import mean_resp_image
from utils.plot_lib import * #get all the fixed color schemes
from loaddata.session import Session

from utils.RRRlib import *

def plot_rf_plane(celldata,r2_thr=0,rf_type='Fneu'):
    
    areas           = np.sort(celldata['roi_name'].unique())[::-1]
    # vars            = ['rf_azimuth','rf_elevation']
    vars            = ['rf_az_' + rf_type,'rf_el_' + rf_type]

    fig,axes        = plt.subplots(2,len(areas),figsize=(5*len(areas),10))
    if 'rf_az_' + rf_type in celldata:
        for i in range(len(vars)): #for azimuth and elevation
            for j in range(len(areas)): #for areas
                
                idx_area    = celldata['roi_name']==areas[j]
                idx_sig     = celldata['rf_r2_' + rf_type] > r2_thr
                idx         = np.logical_and(idx_area,idx_sig)
                if np.any(celldata[idx][vars[i]]):
                    if vars[i]=='rf_az_' + rf_type:
                        sns.scatterplot(data = celldata[idx],x='yloc',y='xloc',hue_norm=(-135,135),
                                    hue=vars[i],ax=axes[i,j],palette='gist_rainbow',size=9,edgecolor="none")
                    elif vars[i]=='rf_el_' + rf_type:
                        sns.scatterplot(data = celldata[idx],x='yloc',y='xloc',hue_norm=(-16.7,50.2),
                                    hue=vars[i],ax=axes[i,j],palette='gist_rainbow',size=9,edgecolor="none")

                box = axes[i,j].get_position()
                axes[i,j].set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])  # Shrink current axis's height by 10% on the bottom
                axes[i,j].set_xlabel('')
                axes[i,j].set_ylabel('')
                axes[i,j].set_xticks([])
                axes[i,j].set_yticks([])
                axes[i,j].set_xlim([0,512])
                axes[i,j].set_ylim([0,512])
                axes[i,j].set_title(areas[j] + ' - ' + vars[i],fontsize=15)
                axes[i,j].set_facecolor("black")
                axes[i,j].invert_yaxis()

                if vars[i]=='rf_az_' + rf_type:
                    norm = plt.Normalize(-135,135)
                elif vars[i]=='rf_el_' + rf_type:
                    norm = plt.Normalize(-16.7,50.2)
                sm = plt.cm.ScalarMappable(cmap="gist_rainbow", norm=norm)
                sm.set_array([])

                if np.any(celldata[idx][vars[i]]):
                    axes[i,j].get_legend().remove()
                    # Remove the legend and add a colorbar (optional)
                    axes[i,j].figure.colorbar(sm,ax=axes[i,j],pad=0.02,label=vars[i])
        plt.suptitle(celldata['session_id'][0])
        plt.tight_layout()

    return fig

def plot_rf_screen(celldata,r2_thr=0,rf_type='Fneu'):
    
    areas           = np.sort(celldata['roi_name'].unique())[::-1]
    clr_areas       = get_clr_areas(areas)
    fig,ax          = plt.subplots(1,1,figsize=(6,2))
    x, y            = np.mgrid[-135:135:.1, -16.7:50.2:.1]
    data            = np.dstack((x, y))
    nNeurons        = len(celldata)

    # # Compute z for all neurons at once
    # means = np.array([celldata['rf_az_' + rf_type], celldata['rf_el_' + rf_type]]).T
    # covs = np.array([[celldata['rf_sx_' + rf_type]**2, np.zeros(nNeurons)], [np.zeros(nNeurons), celldata['rf_sy_' + rf_type]**2]]).T
    # idx = np.logical_and(~np.isnan(celldata['rf_sx_' + rf_type]),
    #                      celldata['rf_r2_' + rf_type]> r2_thr)

    nNeurons = len(celldata)
    for i in range(nNeurons):
    # for i in range(50):
        if not np.isnan(celldata['rf_sx_' + rf_type][i]) and celldata['rf_r2_' + rf_type][i] > r2_thr:
            mean        = [celldata['rf_az_' + rf_type][i], celldata['rf_el_' + rf_type][i]]
            cov         = [[celldata['rf_sx_' + rf_type][i]**2, 0], [0, celldata['rf_sy_' + rf_type][i]**2]]
            rv          = multivariate_normal(mean, cov)
            z           = rv.pdf(data)
            peak_pdf    = rv.pdf(mean) # Compute peak PDF value
            # Set level to 0.606 * peak value for one standard deviation
            contour_level = 0.606 * peak_pdf

            plt.contour(x, y, z, levels=[contour_level], colors=get_clr_areas([celldata['roi_name'][i]]), 
                        linestyles='solid',linewidths=0.25,alpha=0.5)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])  # Shrink current axis's height by 10% on the bottom
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axvline(x=45,color = 'black', linestyle = '--')
    ax.axvline(x=-45,color = 'black', linestyle = '--')
    ax.set_xlim([-135,135])
    ax.set_ylim([-16.7,50.2])
    ax.legend(labels=areas,loc='center left', bbox_to_anchor=(1, 0.5))        # Put a legend next to current axis
    ax.set_title(celldata['session_id'][0])
    plt.tight_layout()

    return fig

def plot_RF_frac(sessions,rf_type,r2_thr):
    areas   = ['V1','PM']

    rf_frac = np.empty((len(sessions),len(areas)))
    for iarea in range(len(areas)):    # iterate over sessions
        for ises in range(len(sessions)):    # iterate over sessions
            idx = sessions[ises].celldata['roi_name'] == areas[iarea]
            if 'rf_r2_' + rf_type  in sessions[ises].celldata:
                rf_frac[ises,iarea] = np.sum(sessions[ises].celldata['rf_r2_' + rf_type][idx]>r2_thr) / np.sum(idx)
        print('%2.1f +- %2.1f %% neurons with RF in area %s\n'  % (np.mean(rf_frac[:,iarea])*100,np.std(rf_frac[:,iarea])*100,areas[iarea]))
    fig,ax = plt.subplots(figsize=(3,3))
    # sns.scatterplot(rf_frac.T,color='black',s=50)
    sns.stripplot(rf_frac,s=6,jitter=0.1,palette=get_clr_areas(areas),ax=ax)
    plt.xlim([-0.5,1.5])
    plt.ylim([0,1])
    plt.xticks([0,1],labels=areas)
    plt.xlabel('Area')
    plt.ylabel('Fraction receptive fields')
    plt.title(rf_type)
    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True, offset = 3)
    # ax.get_legend().remove()

    return fig,rf_frac

def interp_rf(sessions,rf_type='Fneu',r2_thr=0.2,reg_alpha=1):

    for ises,ses in enumerate(sessions):
        if 'rf_r2_' + rf_type in ses.celldata:
            areas           = np.sort(ses.celldata['roi_name'].unique())[::-1]
            # vars            = ['rf_azimuth','rf_elevation']
            vis_dims        = ['rf_az_' + rf_type,'rf_el_' + rf_type]

            # if show_fit:
            #     fig,axes        = plt.subplots(2,2,figsize=(5*len(areas),10))

            r2 = np.empty((len(sessions),2,2))

            ses.celldata[vis_dims[0] + '_interp'] = '' 
            ses.celldata[vis_dims[1] + '_interp'] = '' 
            ses.celldata['rf_r2_' + rf_type + '_interp'] = 0

            for idim,dim in enumerate(vis_dims): #for azimuth and elevation
                for iarea,area in enumerate(areas): #for areas
                    
                    idx_area    = ses.celldata['roi_name']==area
                    idx_sig     = ses.celldata['rf_r2_' + rf_type] > r2_thr
                    idx_nan     = ~np.isnan(ses.celldata['rf_az_' + rf_type])
                    idx         = np.all((idx_area,idx_sig,idx_nan),axis=0) 

                    areadf      = ses.celldata[idx] #.dropna()
                    X           = np.array([areadf['xloc'],areadf['yloc']])
                    y           = np.array(areadf[dim])

                    # reg         = LinearRegression().fit(X.T, y)
                    # reg         = Ridge(alpha=1)
                    reg         = Lasso(alpha=reg_alpha)
                    reg         = reg.fit(X.T, y)
                    
                    # plt.scatter(y,reg.predict(X.T))
                    # weights     = np.abs(-np.log10(ses.celldata[idx]['rf_r2_' + rf_type]))
                    # # Fit weighted least squares regression model
                    # X = sm.add_constant(X)
                    # # reg = sm.WLS(y, X.T, weights=weights)
                    # reg = sm.WLS(y, X.T, weights=weights)
                    # results = reg.fit()

                    # r2[ises,idim,iarea]     = r2_score(y,reg.predict(X.T))
                    r2[ises,idim,iarea]     = r2_score(y,reg.predict(X.T))

                    if r2[ises,idim,iarea]>r2_thr:
                        # ses.celldata.loc[ses.celldata[idx].index,vis_dims[i]] = reg.predict(ses.celldata.loc[ses.celldata[idx].index,['xloc','yloc']].to_numpy())
                        ses.celldata.loc[idx_area,dim + '_interp'] = reg.predict(ses.celldata.loc[idx_area,['xloc','yloc']].to_numpy())
    
    return r2


def smooth_rf(sessions,r2_thr=0.2,radius=50,mincellsFneu=10,rf_type='Fneu'):

    # for ses in sessions:
    for ses in tqdm(sessions,total=len(sessions),desc= 'Smoothed interpolation of missing RF: '):
        if 'rf_az_' + rf_type in ses.celldata:
            ses.celldata['rf_az_Fsmooth']          = np.nan
            ses.celldata['rf_el_Fsmooth']          = np.nan
            ses.celldata['rf_sx_Fsmooth']          = np.nan
            ses.celldata['rf_sy_Fsmooth']          = np.nan
            ses.celldata['rf_r2_Fsmooth']          = np.nan
            
            for iN in range(len(ses.celldata)):
                
                idx_near_Fneu = np.all((ses.distmat_xy[iN,:] < radius,
                                   ses.celldata['rf_r2_' + rf_type]>r2_thr,
                                   ~np.isnan(ses.celldata['rf_az_' + rf_type])),axis=0)
                if np.sum(idx_near_Fneu)>mincellsFneu:
                    # idx_near = np.logical_and(ses.distmat_xy[iN,:] < radius,idx_RF)
                    # ses.celldata.loc[iN,'rf_az_Fsmooth']    = np.average(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_az_Fneu'],
                                                                    # weights=np.abs(-np.log10(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_r2_Fneu'])))

                    # ses.celldata.loc[iN,'rf_el_Fsmooth']    = np.average(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_el_Fneu'],
                                                                    # weights=np.abs(-np.log10(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_r2_Fneu'])))

                    ses.celldata.loc[iN,'rf_az_Fsmooth']    = np.nanmedian(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_az_' + rf_type])
                    ses.celldata.loc[iN,'rf_el_Fsmooth']    = np.nanmedian(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_el_' + rf_type])
                    ses.celldata.loc[iN,'rf_r2_Fsmooth']    = 1

    return sessions

def exclude_outlier_rf(sessions,rf_thr_V1=20,rf_thr_PM=40):
    # Filter out neurons with receptive fields that are too far from the local neuropil receptive field:
    #radius specifies cortical distance of neuropil to include for local rf center
    #rf_thr specifies cutoff of deviation from local rf center to be excluded

    for ses in tqdm(sessions,total=len(sessions),desc= 'Setting outlier RFs to NaN: '):
        if 'rf_az_F' in ses.celldata and 'rf_az_Fsmooth' in ses.celldata:
            idx_V1 = ses.celldata['roi_name']=='V1'
            idx_PM = ses.celldata['roi_name']=='PM'
            
            rf_dist_F_Fsmooth = np.sqrt( (ses.celldata['rf_az_F'] - ses.celldata['rf_az_Fsmooth'])**2 + 
                                            (ses.celldata['rf_el_F'] - ses.celldata['rf_el_Fsmooth'])**2 )
            
            idx = (idx_V1 & (rf_dist_F_Fsmooth > rf_thr_V1)) | np.isnan(rf_dist_F_Fsmooth)
            ses.celldata.loc[idx,['rf_az_F','rf_el_F','rf_sx_F','rf_sy_F','rf_r2_F']] = np.NaN
            
            idx = (idx_PM & (rf_dist_F_Fsmooth > rf_thr_PM)) | np.isnan(rf_dist_F_Fsmooth)
            ses.celldata.loc[idx,['rf_az_F','rf_el_F','rf_sx_F','rf_sy_F','rf_r2_F']] = np.NaN

    return sessions

def replace_smooth_with_Fsig(sessions,r2_thr=0.2):
    # Find indices of good fit receptive field neurons 
    # replace Fsmooth receptive fields to their F-based estimates
    for ses in sessions:
        if 'rf_az_F' in ses.celldata and 'rf_az_Fsmooth' in ses.celldata:
            idx     = ses.celldata['rf_r2_F'] > r2_thr
            ses.celldata.loc[idx,'rf_az_Fsmooth'] = ses.celldata.loc[idx,'rf_az_F']
            ses.celldata.loc[idx,'rf_el_Fsmooth'] = ses.celldata.loc[idx,'rf_el_F']
            ses.celldata.loc[idx,'rf_sx_Fsmooth'] = ses.celldata.loc[idx,'rf_sx_F']
            ses.celldata.loc[idx,'rf_sy_Fsmooth'] = ses.celldata.loc[idx,'rf_sy_F']
            ses.celldata.loc[idx,'rf_r2_Fsmooth'] = ses.celldata.loc[idx,'rf_r2_F']

    return sessions
    
    
def plot_delta_rf_across_sessions(sessions,areapairs,rf_type='Fsmooth',r2_thr=0.2):
    clrs_areapairs = get_clr_area_pairs(areapairs)
    binedges    = np.arange(-5,150,5) 
    nbins       = len(binedges)-1
    # binmean     = np.full((len(sessions),len(areapairs),nbins),np.nan)

    fig,axes = plt.subplots(1,len(areapairs),figsize=(len(areapairs)*3,3))
    for iap,areapair in enumerate(areapairs):
        for ses in sessions:
            
            # Define function to filter neuronpairs based on area combination
            areafilter = filter_2d_areapair(ses,areapair)
            nanfilter  = ~np.isnan(ses.distmat_rf)
            r2filter   = np.meshgrid(ses.celldata['rf_r2_' + rf_type]> r2_thr,ses.celldata['rf_r2_'  + rf_type] > r2_thr)
            r2filter   = np.logical_and(r2filter[0],r2filter[1])

            cellfilter = np.all((areafilter,nanfilter,r2filter),axis=0)
            sns.histplot(data=ses.distmat_rf[cellfilter].flatten(),bins=binedges,ax=axes[iap],color=clrs_areapairs[iap],
                         alpha=0.5,fill=False,stat='percent',element='step')
        axes[iap].set_title(areapair)
            # axes[iap].hist(ses.distmat_rf[cellfilter],bins=binedges,color=clrs_areapairs[iap],alpha=0.5)
    sns.despine(fig=fig, top=True, right=True,offset=3)
    plt.tight_layout()
    return fig


def scatter_dXY_dRF(ses,areapairs):
    clrs_areapairs = get_clr_area_pairs(areapairs)
    fig,axes = plt.subplots(1,len(areapairs),figsize=(len(areapairs)*3,3))
    for iap,areapair in enumerate(areapairs):
        ax = axes[iap]

        # Define function to filter neuronpairs based on area combination
        areafilter = filter_2d_areapair(ses,areapair)
        nanfilter  = ~np.isnan(ses.distmat_rf)
        cellfilter = np.all((areafilter,~np.isnan(ses.distmat_rf),~np.isnan(ses.distmat_xy)),axis=0)
        xdata = ses.distmat_xy[cellfilter].flatten()
        ydata = ses.distmat_rf[cellfilter].flatten()

        ax.scatter(xdata,ydata,s=3,alpha=0.01,c='k')	
        sns.set_theme(style="ticks")
        sns.histplot(x=xdata, y=ydata, bins=50, pthresh=.1, cmap="mako",ax=ax)

        # Fit a linear regression model
        slope, intercept = np.polyfit(xdata, ydata, 1)
        regression_line = slope * xdata + intercept
        # Plot the regression line
        ax.plot(xdata, regression_line, color='blue', linewidth=2, label='Fit',linestyle='-')
        # Add text with regression coefficients
        ax.text(0.05, 0.95, f'Slope: {slope:.2f} (deg/um) \nIntercept: {intercept:.2f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(areapair,color=clrs_areapairs[iap])
        ax.set_xlabel(r'dXY ($\mu$m)')
        ax.set_ylabel(u'dRF (\N{DEGREE SIGN})')
        ax.set_xlim([0,650])
        ax.set_ylim([0,140])
        ax.set_aspect('auto')
        ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    return fig

def plot_delta_rf_projections(sessions,areapairs,projpairs,filter_near=False):
    clrs_areapairs  = get_clr_area_pairs(areapairs)
    clrs_projpairs  = get_clr_labelpairs(projpairs)
    binedges        = np.arange(-5,150,5) 
    nbins           = len(binedges)-1
    data            = np.full((len(sessions),len(areapairs),len(projpairs)),np.nan)
    fig,axes = plt.subplots(2,len(areapairs),figsize=(len(areapairs)*3,6))
    # for ises,ses in enumerate([sessions[1]]):
    for iap,areapair in enumerate(areapairs):
        for ipp,projpair in enumerate(projpairs):
            for ises,ses in enumerate(sessions):
                areafilter = filter_2d_areapair(ses,areapair)
                projfilter = filter_2d_projpair(ses,projpair)
                nanfilter  = ~np.isnan(ses.distmat_rf)
                if filter_near:
                    nearfilter = filter_nearlabeled(ses,radius=50)
                    nearfilter = np.outer(nearfilter,nearfilter)
                else: 
                    nearfilter = np.ones(np.shape(areafilter))
                cellfilter = np.all((areafilter,nanfilter,projfilter,nearfilter),axis=0)
                if np.any(cellfilter):
                    sns.histplot(data=ses.distmat_rf[cellfilter].flatten(),bins=binedges,ax=axes[0,iap],color=clrs_projpairs[ipp],
                                alpha=0.5,fill=False,stat='percent',element='step')
                    data[ises,iap,ipp] = np.nanmean(ses.distmat_rf[cellfilter])
            axes[0,iap].set_title(areapair,color=clrs_areapairs[iap])

        # axes[iap].hist(ses.distmat_rf[cellfilter],bins=binedges,color=clrs_areapairs[iap],alpha=0.5)
        sns.stripplot(data=pd.DataFrame(data[:,iap,:],columns=projpairs),ax=axes[1,iap],palette=clrs_projpairs)
        for ipp1,projpair1 in enumerate(projpairs):
            for ipp2,projpair2 in enumerate(projpairs):
                if ipp1 < ipp2:
                    pval = ttest_rel(data[:,iap,ipp1],data[:,iap,ipp2],nan_policy='omit')[1]
                    axes[1,iap].text(ipp1+0.5,np.nanmean(data[:,iap,:])+5,'{:.3f}'.format(pval),ha='center')
        axes[1,iap].plot(np.nanmean(data[:,iap,:],axis=0),color='k',linewidth=2)
        axes[1,iap].set_ylim([0,60])
        axes[1,iap].set_ylabel('RF distance (deg)')
        # for ipp,projpair in enumerate(projpairs):
            # axes[0,iap].plot([ipp-0.2,ipp+0.2],[np.nanmean(data[:,iap,ipp]),np.nanmean(data[:,iap,ipp])],color=clrs_projpairs[ipp],linewidth=2)
    plt.tight_layout()
    return fig


def filter_nearlabeled(ses,radius=50,only_V1_PM=False):

    if not hasattr(ses,'distmat_xyz'):
        [ses] = compute_pairwise_metrics([ses])
    temp = ses.distmat_xyz.copy()
    np.fill_diagonal(temp,0)  #this is to include the labeled neurons themselves
    closemat = temp[ses.celldata['redcell']==1,:] <= radius
    idx = np.any(closemat,axis=0)
    if not only_V1_PM:
        idx = np.logical_or(idx, ~ses.celldata['roi_name'].isin(['V1','PM']))
    return idx

def get_response_triggered_image(ses, natimgdata):
    
    respmean,imageids   = mean_resp_image(ses)

    N                   = np.shape(ses.respmat)[0]

    # Compute response triggered average image:
    ses.RTA             = np.empty((*np.shape(natimgdata)[:2], N))
    weight_sums = np.sum(respmean, axis = 1)
    ses.RTA = np.tensordot(natimgdata[:,:,imageids], respmean, axes=([2], [1])) / weight_sums
        
    return ses

# def get_response_triggered_image(ses, natimgdata):
    
#     respmean,imageids   = mean_resp_image(ses)

#     N                   = np.shape(ses.respmat)[0]

#     # Compute response triggered average image:
#     # N = 100
#     # for iN in range(N):
#     ses.RTA             = np.empty((*np.shape(natimgdata)[:2], N))
#     for iN in tqdm(range(N),desc='Computing average response for neuron'):
#         ses.RTA[:, :, iN] = np.average(natimgdata[:,:,imageids], axis=2, weights=respmean[iN, :])
        
#     return ses


def estimate_rf_IM(ses,show_fig=False): 
    ses.celldata['rf_az_F'] = ses.celldata['rf_el_F'] = ses.celldata['rf_r2_F'] = np.nan
    # natimgdata = load_natural_images(onlyright=True) #Load the natural images:
    natimgdata = load_natural_images(onlyright=False) #Load the natural images:

    if not hasattr(ses,'RTA'):
        ses         = get_response_triggered_image(ses, natimgdata)

    # az_lims     = [45, 135]
    az_lims     = [-135, 135]
    # el_lims     = [50.2, -16.7]
    el_lims     = [-16.7,50.2] #bottom and top of screen displays

    ypix,xpix,N = np.shape(ses.RTA)
    xmap        = np.linspace(*az_lims,xpix)
    ymap        = np.linspace(*el_lims,ypix)
    # N = 100
    zthr        = 3
    rf_data     = pd.DataFrame(data=np.full((N,4),np.nan),columns=['rf_az_F','rf_el_F','rf_sz_F','rf_r2_F'])

    for iN in range(N):
        dev = zscore(ses.RTA[:, :, iN].copy()-128,axis=None)
        dev[np.abs(dev)<zthr]=0
        if np.any(dev):
            (y, x) = np.round(ndimage.center_of_mass(np.abs(dev))).astype(int)
            rf_data.loc[iN,'rf_az_F'] = xmap[x]
            rf_data.loc[iN,'rf_el_F'] = ymap[y]
            rf_data.loc[iN,'rf_sz_F'] = np.sum(dev>zthr)
            print('PROBLEMATIC STILL CONVERT TO R2 FOR IM')
            rf_data.loc[iN,'rf_r2_F'] = 1.015**-(np.sum(np.abs(dev))) #get some significance metric from the total deviation
    
    if show_fig:
        RTA_var = np.var(ses.RTA, axis=(0, 1))

        nExamples = 25

        example_cells = np.argsort(RTA_var)[-nExamples:]

        Rows = int(np.floor(np.sqrt(nExamples)))
        Cols = nExamples // Rows  # Compute Rows required
        if nExamples % Rows != 0:  # If one additional row is necessary -> add one:
            Cols += 1
        Position = range(1, nExamples + 1)  # Create a Position index

        fig = plt.figure(figsize=[18, 9])
        # for iN in range(N):
        for i, iN in enumerate(example_cells):
            # add every single subplot to the figure with a for loop
            ax = fig.add_subplot(Rows, Cols, Position[i])
            ax.imshow(ses.RTA[:, :, iN]-128, cmap='gray',vmin=-15,vmax=15)

            dev = zscore(ses.RTA[:, :, iN].copy()-128,axis=None)
            dev[np.abs(dev)<zthr]=0
            if np.any(dev):
                (y, x) = np.round(ndimage.center_of_mass(np.abs(dev))).astype(int)
                
            ax.plot(x, y, 'r+')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
            ax.set_title("%d" % iN)
        plt.tight_layout(rect=[0, 0, 1, 1])

    return rf_data


def lowrank_RF(Y, IMdata, lam=0.05,nranks=25,nsub=3):
    """
    Compute a linear low-rank approximation of the receptive field (RF) from the natural image responses

    Parameters
    ----------
    Y : array with shape (K,N)
        The neural responses to N natural images
    IMdata : array with shape (H, W, K)
        The natural image data of shape H x W x K 
        where H is the height of the image, W is the width of the image, and K is the number of images
    lam : float, default=0.05
        The regularization parameter for the multiple linear regression
    nranks : int, default=25
        The number of ranks to keep in the low-rank approximation
    nsub : int, default=3
        The downsampling factor for the natural images

    Returns
    -------
    cRF : array with shape (Ly, Lx, N)
        The low-rank approximation of the receptive field
    Y_hat : array with shape (K,N)
        The predicted neural responses
    """
    IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images
    Ly,Lx,K             = np.shape(IMdata)                  #get dimensions
    X                   = np.reshape(IMdata, (Ly*Lx, K)).T  #X is now pixels by images matrix
    X                   = X / np.linalg.norm(X, axis=0)     # normalize the pixels in each image
    assert np.shape(Y)[0] == np.shape(X)[0], 'Number of neuronal responses does not match number of images'

    N                   = np.shape(Y)[1]        # N is the number of neurons

    B_hat               = LM(Y, X, lam=lam)     #fit multiple linear regression (with ridge penalty)

    U, s, V             = svds(B_hat, k=nranks) #truncated singular value decomposition of the coefficients

    B_hat_rrr           = U @ np.diag(s) @ V    #reconstruct the coefficients from low rank

    Y_hat               = X @ B_hat_rrr         #predict the trial to trial response from the low rank coefficients

    cRF                 = np.reshape(B_hat_rrr, (Ly,Lx, N)) #reshape the low rank coefficients to the image space

    return cRF,Y_hat

def lowrank_RF_cv(Y, IMdata, lam=0.05,nranks=25,nsub=3,kfold=2):
    """
    Compute a linear low-rank approximation of the receptive field (RF) from the natural image responses
    Crossvalidated version
    Parameters
    ----------
    Y : array with shape (K,N)
        The neural responses to N natural images
    IMdata : array with shape (H, W, K)
        The natural image data of shape H x W x K 
        where H is the height of the image, W is the width of the image, and K is the number of images
    lam : float, default=0.05
        The regularization parameter for the multiple linear regression
    nranks : int, default=25
        The number of ranks to keep in the low-rank approximation
    nsub : int, default=3
        The downsampling factor for the natural images

    Returns
    -------
    cRF : array with shape (Ly, Lx, N)
        The low-rank approximation of the receptive field
    Y_hat : array with shape (K,N)
        The predicted neural responses
    """

    IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images
    Ly,Lx,K             = np.shape(IMdata)                  #get dimensions
    X                   = np.reshape(IMdata, (Ly*Lx, K)).T  #X is now pixels by images matrix
    X                   = X / np.linalg.norm(X, axis=0)     # normalize the pixels in each image
    assert np.shape(Y)[0] == np.shape(X)[0], 'Number of neuronal responses does not match number of images'

    K,N                 = np.shape(Y)        # K is the number of images, N is the number of neurons
    Y_hat               = np.full((K,N),np.nan)

    kf                  = KFold(n_splits=kfold, shuffle=True)

    B_hat_rrr_folds = np.full((Ly*Lx,N,kfold),np.nan)

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        B_hat               = LM(Y_train, X_train, lam=lam)     #fit multiple linear regression (with ridge penalty)

        U, s, V             = svds(B_hat, k=nranks) #truncated singular value decomposition of the coefficients

        B_hat_rrr           = U @ np.diag(s) @ V    #reconstruct the coefficients from low rank

        Y_hat[test_index,:] = X_test @ B_hat_rrr         #predict the trial to trial response from the low rank coefficients
        
        B_hat_rrr_folds[:,:,i] = B_hat_rrr

    B_hat_rrr   = np.nanmean(B_hat_rrr_folds, axis=2)

    cRF         = np.reshape(B_hat_rrr, (Ly,Lx, N)) #reshape the low rank coefficients to the image space

    return cRF,Y_hat

def linear_RF(Y, IMdata, lam=0.05,nsub=3,kfold=2):
    """
    Compute a linear approximation of the receptive field (RF) from the natural image responses
    Parameters
    ----------
    Y : array with shape (K,N)
        The neural responses to N natural images
    IMdata : array with shape (H, W, K)
        The natural image data of shape H x W x K 
        where H is the height of the image, W is the width of the image, and K is the number of images
    lam : float, default=0.05
        The regularization parameter for the multiple linear regression
    nsub : int, default=3
        The downsampling factor for the natural images

    Returns
    -------
    cRF : array with shape (Ly, Lx, N)
        The linear approximation of the receptive field
    Y_hat : array with shape (K,N)
        The predicted neural responses
    """

    IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images
    Ly,Lx,K             = np.shape(IMdata)                  #get dimensions
    X                   = np.reshape(IMdata, (Ly*Lx, K)).T  #X is now pixels by images matrix
    X                   = X / np.linalg.norm(X, axis=0)     # normalize the pixels in each image
    assert np.shape(Y)[0] == np.shape(X)[0], 'Number of neuronal responses does not match number of images'

    K,N                 = np.shape(Y)        # K is the number of images, N is the number of neurons
    Y_hat               = np.full((K,N),np.nan)

    kf = KFold(n_splits=kfold, shuffle=True)

    B_hat               = LM(Y, X, lam=lam)     #fit multiple linear regression (with ridge penalty)
    
    Y_hat               = X @ B_hat         #predict the trial to trial response from the low rank coefficients


    # B_hat_folds = np.full((Ly*Lx,N,kfold),np.nan)

    # for i, (train_index, test_index) in enumerate(kf.split(X)):

    #     X_train, X_test = X[train_index], X[test_index]
    #     Y_train, Y_test = Y[train_index], Y[test_index]

    #     B_hat               = LM(Y_train, X_train, lam=lam)     #fit multiple linear regression (with ridge penalty)

    #     Y_hat[test_index,:] = X_test @ B_hat         #predict the trial to trial response from the low rank coefficients
        
    #     B_hat_folds[:,:,i] = B_hat

    # B_hat   = np.nanmean(B_hat_folds, axis=2)

    cRF         = np.reshape(B_hat, (Ly,Lx, N)) #reshape the low rank coefficients to the image space

    return cRF,Y_hat

def linear_RF_cv(Y, IMdata, lam=0.05,nsub=3,kfold=2):
    """
    Compute a linear approximation of the receptive field (RF) from the natural image responses
    Crossvalidated version
    Parameters
    ----------
    Y : array with shape (K,N)
        The neural responses to N natural images
    IMdata : array with shape (H, W, K)
        The natural image data of shape H x W x K 
        where H is the height of the image, W is the width of the image, and K is the number of images
    lam : float, default=0.05
        The regularization parameter for the multiple linear regression
    nsub : int, default=3
        The downsampling factor for the natural images

    Returns
    -------
    cRF : array with shape (Ly, Lx, N)
        The linear approximation of the receptive field
    Y_hat : array with shape (K,N)
        The predicted neural responses
    """

    IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images
    Ly,Lx,K             = np.shape(IMdata)                  #get dimensions
    X                   = np.reshape(IMdata, (Ly*Lx, K)).T  #X is now pixels by images matrix
    X                   = X / np.linalg.norm(X, axis=0)     # normalize the pixels in each image
    assert np.shape(Y)[0] == np.shape(X)[0], 'Number of neuronal responses does not match number of images'

    K,N                 = np.shape(Y)        # K is the number of images, N is the number of neurons
    Y_hat               = np.full((K,N),np.nan)

    kf = KFold(n_splits=kfold, shuffle=True)

    B_hat_folds = np.full((Ly*Lx,N,kfold),np.nan)

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        B_hat               = LM(Y_train, X_train, lam=lam)     #fit multiple linear regression (with ridge penalty)

        Y_hat[test_index,:] = X_test @ B_hat         #predict the trial to trial response from the low rank coefficients
        
        B_hat_folds[:,:,i] = B_hat

    B_hat   = np.nanmean(B_hat_folds, axis=2)

    cRF         = np.reshape(B_hat, (Ly,Lx, N)) #reshape the low rank coefficients to the image space

    return cRF,Y_hat

#Fit each cRF with a 2D gaussian:
def fit_2dgauss_cRF(cRF, nsub,celldata):
    N = np.shape(cRF)[2]

    celldata[['rf_az_RRR', 'rf_el_RRR', 'rf_sx_RRR', 'rf_sy_RRR', 'rf_r2_RRR']] = np.nan
    
    for n in tqdm(range(N),total=N,desc='Fitting 2D gauss to RF'):	
        rfdata = np.abs(cRF[:, :, n])
        gaussian_sigma = 1
        rfdata = gaussian_filter(rfdata,sigma=[gaussian_sigma,gaussian_sigma])

        try:
            popt,pcov,r2,z_fit = fit_2d_gaussian(rfdata)

            celldata.loc[n,'rf_az_RRR']   = popt[0]*nsub
            celldata.loc[n,'rf_el_RRR']   = popt[1]*nsub
            celldata.loc[n,'rf_sx_RRR']   = popt[2]*nsub
            celldata.loc[n,'rf_sy_RRR']   = popt[3]*nsub
            celldata.loc[n,'rf_r2_RRR']   = r2
        except:
            pass
    return celldata




    
# def exclude_outlier_rf(sessions,r2_thr=0.2,radius=100,rf_thr=25,mincellsFneu=10):
#     # Filter out neurons with receptive fields that are too far from the local neuropil receptive field:
#     #radius specifies cortical distance of neuropil to include for local rf center
#     #rf_thr specifies cutoff of deviation from local rf center to be excluded
#     # for ses in sessions:
#     for ses in tqdm(sessions,total=len(sessions),desc= 'Setting outlier RFs to NaN: '):
#         if 'rf_az_Fneu' in ses.celldata:
#             rf_az_Fneu_avg = np.full(len(ses.celldata),np.NaN)
#             rf_el_Fneu_avg = np.full(len(ses.celldata),np.NaN)
#             for iN in range(len(ses.celldata)):

#                 # idx_near = ses.distmat_xy[iN,:] < radius
#                 idx_near_Fneu = np.all((ses.distmat_xy[iN,:] < radius,
#                                    ses.celldata['rf_r2_Fneu']>r2_thr,
#                                    ~np.isnan(ses.celldata['rf_az_Fneu'])),axis=0)
#                 if np.sum(idx_near_Fneu)>mincellsFneu:
#                     rf_az_Fneu_avg[iN]         = np.nanmedian(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_az_Fneu'])
#                     rf_el_Fneu_avg[iN]         = np.nanmedian(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_el_Fneu'])

#             rf_dist_F_Fneu = np.sqrt( (ses.celldata['rf_az_F'] - rf_az_Fneu_avg)**2 + (ses.celldata['rf_el_F'] - rf_el_Fneu_avg)**2 )
#             #now set all neurons outside the criterium rf_thr to NaN
#             ses.celldata.loc[rf_dist_F_Fneu > rf_thr,['rf_az_F','rf_el_F','rf_r2_F']] = np.NaN
#     return sessions

# def smooth_rf(sessions,r2_thr=0.2,radius=50,mincellsFneu=10):

#     # for ses in sessions:
#     for ses in tqdm(sessions,total=len(sessions),desc= 'Smoothed interpolation of missing RF: '):
#         if 'rf_az_Fneu' in ses.celldata:
#             ses.celldata['rf_az_Fsmooth']          = ses.celldata['rf_az_F'].copy()
#             ses.celldata['rf_el_Fsmooth']          = ses.celldata['rf_el_F'].copy()
#             ses.celldata['rf_r2_Fsmooth']           = ses.celldata['rf_r2_F'].copy()
            
#             for iN in np.where(~(ses.celldata['rf_r2_Fsmooth'] > r2_thr))[0]:
                
#                 idx_near_Fneu = np.all((ses.distmat_xy[iN,:] < radius,
#                                    ses.celldata['rf_r2_Fneu']>r2_thr,
#                                    ~np.isnan(ses.celldata['rf_az_Fneu'])),axis=0)
#                 if np.sum(idx_near_Fneu)>mincellsFneu:
#                     # idx_near = np.logical_and(ses.distmat_xy[iN,:] < radius,idx_RF)
#                     ses.celldata.loc[iN,'rf_az_Fsmooth']          = np.nanmedian(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_az_Fneu'])
#                     ses.celldata.loc[iN,'rf_el_Fsmooth']          = np.nanmedian(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_el_Fneu'])
#                     ses.celldata.loc[iN,'rf_r2_Fsmooth']           = 0
#                     # ses.celldata.loc[iN,'rf_az_F']          = np.nanmedian(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_az_Fneu'])
#                     # ses.celldata.loc[iN,'rf_el_F']          = np.nanmedian(ses.celldata.loc[ses.celldata[idx_near_Fneu].index,'rf_el_Fneu'])
#                     # ses.celldata.loc[iN,'rf_r2_F']           = 0.0009
#     return sessions
