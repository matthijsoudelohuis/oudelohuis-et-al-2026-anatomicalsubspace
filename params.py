# -*- coding: utf-8 -*-
"""
Parameters for the analysis
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""
import os
from loaddata.get_data_folder import get_local_drive

def load_params():
    params = dict(
                savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\LabeledSubspace'),

                calciumversion = 'dF', #deconv or dF
                maxnoiselevel = 20, #maximum noise level to include cell
                minnneurons = 10, #minimum number of neurons in labeled or unlabeled population to include session
                # minrangeresp = 0.04, #minimum range of responses between stimulus conditions to include cell
                
                # stilltrialsonly = True, #only use still trials for analysis
                maxvideome = 0.2, #maximum video motion in normalized energy
                maxrunspeed = 0.5, #maximum run speed in cm/s
                
                radius = 50, # distance in um to look for nearby cells
                )
    return params
