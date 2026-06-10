# -*- coding: utf-8 -*-
"""
Parameters for the analysis
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""
import os
from loaddata.get_data_folder import get_local_drive

def load_params():
    params = dict(
                figdir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\LabeledSubspace'),
                resultdir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Analysis\\LabeledSubspace'),

                calciumversion = 'dF', #deconv or dF
                t_pre       = -1,         #pre s
                t_post      = 2.17,        #post s
                binsize     = 1/5.35,
                tresp_start = 0, #Time window start for response calculation and analyzing residual activity
                tresp_end   = 1.5,

                # Parameters for session and neuron selection:
                filter_nearby = True,
                radius = 50, # distance in um to look for nearby cells
                maxnoiselevel = 20, #maximum noise level to include cell
                minnneurons = 10, #minimum number of neurons in labeled or unlabeled population to include session

                #Default parameters for RRR:
                lam = 0, #regularization parameter for RRR
                kfold = 5,
                nsubprojection = 20, #number of neurons to sample for each model fit, for each area and label type
                nsubnonlabeled = 100, #number of neurons to sample for each model fit, for each area and label type
                nranks = 25, #number of ranks of RRR to be evaluated
                nmodelfits = 100, #number of times to fit RRR, subsampling different neurons
                nStim = 16, #maximum number of stimulus conditions to evaluate (grating orientations)

                # stilltrialsonly = True, #only use still trials for analysis
                maxvideome = 0.2, #maximum video motion in normalized energy
                maxrunspeed = 0.5, #maximum run speed in cm/s

                # Default parameters for statistical testing:  
                multcomp_method = 'holm', #method for multiple comparisons correction in statistical tests, e.g. 'holm', 'fdr_bh', 'bonferroni'
                )
    return params
