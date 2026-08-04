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
import seaborn as sns
from scipy.stats import zscore

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import *
from utils.psth import compute_tensor
from utils.plot_lib import * 
from utils.params import load_params

params = load_params()
figdir = os.path.join(params['figdir'],'Behavior')

#%% 
session_list        = np.array([
                                ['LPE12223_2024_06_10'],
                                # ['LPE09830_2023_04_10'], 
                                ['LPE09665_2023_03_14'], 
                                # ['LPE10885_2023_10_23'], 
                                ]) 

sessions,nSessions   = filter_sessions(protocols = ['GN','GR'],only_session_id=session_list)
report_sessions(sessions)

#%% Get all data 
sessions,nSessions   = filter_sessions(protocols = ['GN','GR'])
report_sessions(sessions)

#%% Wrapper function to load the tensor data
params['calciumversion'] = 'deconv'
# params['calciumversion'] = 'dF'
[sessions,t_axis] = load_resid_tensor(sessions,params,regressbehavout=False,
                                      subtract_mean_evoked=False,load_behav=True)

#%% Get mean population rate and behavioral variables:
data = np.full((4,len(t_axis),nSessions),np.nan)

vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
                                ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)
behavfields = np.array(['runspeed','diffrunspeed'])

for ises,ses in enumerate(sessions):
    #normalize deconv by meanF
    data[0,:,ises] = np.nanmean(ses.tensor / np.array(ses.celldata['meanF'])[:,None,None],axis=(0,1)) #mean population rate

    tempdata = ses.tensor_vid[vidfields == 'pupil_area',:,:]
    tempdata -= np.nanmean(tempdata,axis=(0,1),keepdims=True)
    tempdata /= np.nanstd(tempdata,axis=(0,1),keepdims=True)
    data[1,:,ises] = np.nanmean(tempdata,axis=(0,1))

    tempdata = ses.tensor_vid[vidfields == 'videoPC_0',:,:]
    tempdata -= np.nanmean(tempdata,axis=(0,1),keepdims=True)
    tempdata /= np.nanstd(tempdata,axis=(0,1),keepdims=True)
    data[2,:,ises] = np.nanmean(tempdata,axis=(0,1))

    data[3,:,ises] = np.nanmean(ses.tensor_run[behavfields == 'runspeed',:,:],axis=(0,1))

#%% Show mean population rate and behavioral variables:
ylims = [[0.03,0.06],[-.5,.5],[-.5,.5],[0,10]]
figtitles = ['pop. rate','pupil area','video ME','run speed']
ylabels = ['df/f','z-score','z-score','cm/s']
fig,axes = plt.subplots(2,2,figsize=(6*cm,5*cm),sharex=True)
for i in range(4):
    ax = axes.flatten()[i]
    ax.plot(t_axis,np.nanmean(data[i,:,:],axis=1),color='black',linewidth=1)
    ax.fill_between(t_axis,
                    np.nanmean(data[i,:,:],axis=1)-np.nanstd(data[i,:,:],axis=1)/np.sqrt(nSessions),
                    np.nanmean(data[i,:,:],axis=1)+np.nanstd(data[i,:,:],axis=1)/np.sqrt(nSessions),
                    color='grey',alpha=0.3)
    ax.axvline(x=0,color='grey',linestyle='--')
    ax.set_title(figtitles[i])
    ax.set_ylabel(ylabels[i])
    ax.set_ylim(ylims[i])

ax.set_xticks([-1,0,1,2])
ax.set_xlabel('Time (s)')
plt.tight_layout()

sns.despine(fig=fig,top=True,right=True,offset=2)
my_savefig(fig,figdir,'Behav_stimlocked_%dsessions' % (nSessions))

