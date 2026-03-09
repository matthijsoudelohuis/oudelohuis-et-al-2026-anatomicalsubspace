# -*- coding: utf-8 -*-
"""
This script analyzes the quality of the recordings and their relation 
to various factors such as depth of recording, being labeled etc.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% Import packages
import os
import numpy as np
import matplotlib.pyplot as plt
from ScanImageTiffReader import ScanImageTiffReader as imread
from pathlib import Path
from matplotlib.patches import Rectangle
from cellpose import models
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

from loaddata.session_info import *
from loaddata.get_data_folder import *
from utils.plot_lib import *
from utils.imagelib import *
# from labeling.tdTom_labeling_cellpose import *
from params import load_params
from labeling.tdTom_labeling_cellpose import *

params = load_params()
figdir = os.path.join(params['figdir'],'Labeling')

#%% 
only_all_areas = np.array(['V1','PM','AL'])
sessions,nSessions   = filter_sessions(protocols = ['GR','GN'],only_all_areas=only_all_areas)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

print('Sessions with recordings in AL: %s' ,sessiondata['session_id'])

#%% Plotting and parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% 
def show_labeling_plane(iplane,plane_folder,saveoverlay=False,showcells=True,overlap_threshold=0.5,gcamp_proj='meanImg'):
    stats       = np.load(os.path.join(plane_folder,'stat.npy'), allow_pickle=True)
    ops         = np.load(os.path.join(plane_folder,'ops.npy'), allow_pickle=True).item()
    iscell      = np.load(os.path.join(plane_folder,'iscell.npy'), allow_pickle=True)
    cm      = 1/2.54  # centimeters in inches

    Nsuite2pcells      = np.shape(stats)[0]

    # From cell masks create outlines:
    masks_suite2p = np.zeros((512,512), np.float32)
    for i,s in enumerate(stats):
        masks_suite2p[s['ypix'],s['xpix']] = i+1

    # load red cell roi information from cellpose gui:
    filenames       = os.listdir(plane_folder)
    cellpose_file   = list(filter(lambda a: 'seg.npy' in a, filenames)) #find the files
    redcell_seg     = np.load(os.path.join(plane_folder,cellpose_file[0]), allow_pickle=True).item()
    masks_cp_red    = redcell_seg['masks']

    #Set red cells at the edge of the screen to zero. This makes sure that a suite2p ROI at the edge that is 
    #is fully red, is labeled as red:
    masks_cp_red[:ops['yrange'][0],:] = 0
    masks_cp_red[ops['yrange'][1]:,:] = 0
    masks_cp_red[:, :ops['xrange'][0]] = 0
    masks_cp_red[:, ops['xrange'][1]:] = 0

    Ncellpose_redcells      = len(np.unique(masks_cp_red))-1

    # Compute fraction of suite2p ROI that is red labeled
    frac_of_ROI_red = np.empty(Nsuite2pcells)
    for i,s in enumerate(stats):
        frac_of_ROI_red[i] = np.sum(masks_cp_red[s['ypix'],s['xpix']] != 0) / len(s['ypix'])

    # Compute fraction of labeled cellpose soma that falls inside suite2p ROI
    # Suite2p ROIs are generally larger and capture calcium activity of membrane and AIS
    # If full tdTomato labeled soma falls inside ROI then labeled. upper measure however could 
    # have lower value if large extra ROI taken as calcium dependent ROI
    frac_red_in_ROI     = np.zeros(Nsuite2pcells)
    npix_redsomas       = np.histogram(masks_cp_red,
                        bins=np.arange(Ncellpose_redcells+1).astype(int))[0]
    npix_redsomas[npix_redsomas==0] = 1000 #artificially set soma's outside cropped FOV to being very large

    if Ncellpose_redcells:
        for i,s in enumerate(stats):
            c = np.histogram(masks_cp_red[s['ypix'],s['xpix']],
                        bins=np.arange(Ncellpose_redcells+1).astype(int))[0]
            c[0] = 0 #set overlap with non-ROI pixels to zero
            frac_red_in_ROI[i] = np.max(c/npix_redsomas) #get maximum overlap (red cell with largest fraction inside suite2p ROI)

    # redcell             = redcell_overlap > overlap_threshold
    redcell             = frac_red_in_ROI > overlap_threshold
    # redcell_cellpose    = np.vstack((redcell,frac_of_ROI_red,frac_red_in_ROI))
    redcell_cellpose    = np.column_stack((redcell,frac_of_ROI_red,frac_red_in_ROI))

    if saveoverlay:
         # Get mean green GCaMP image: 
        if gcamp_proj == 'meanImg': 
            mimg = ops['meanImg']
        elif gcamp_proj == 'max_proj':
            # Get max projection GCaMP image: 
            mimg = np.zeros([512,512])
            mimg[ops['yrange'][0]:ops['yrange'][1],
            ops['xrange'][0]:ops['xrange'][1]]  = ops['max_proj']
        lowprc_green = 10
        uppprc_green = 99.5
        mimg = im_norm8(mimg,min=lowprc_green,max=uppprc_green) #scale between 0 and 255

        lowprc_red = 0.5
        uppprc_red = 99.5
        ## Get red image:
        mimg2 = ops['meanImg_chan2'] #get red channel image from ops
        mimg2 = im_norm(mimg2,min=lowprc_red,max=uppprc_red) #scale between percentiles

        mimg2 = im_sqrt(mimg2) #square root transform to enhance weakly expressing cells

        # mimg2 = im_norm(mimg2,min=0,max=100) #scale between 0 and 255
        mimg2 = im_norm(mimg2,min=5,max=100) #scale between 0 and 255

        #Show labeling results in green and red image:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(14*cm,4*cm))

        rchan = (mimg2 - np.min(mimg2)) / (np.max(mimg2) - np.min(mimg2))
        gchan = (mimg - np.min(mimg)) / (np.max(mimg) - np.min(mimg))
        
        bchan = np.zeros(np.shape(mimg))

        ax1.imshow(gchan,cmap='gray',vmin=0,vmax=1)
        ax2.imshow(rchan,cmap='gray',vmin=0,vmax=1)

        clr_rchan = [1,0,0]
        clr_gchan = [0,1,0]

        # clr_gchan = np.array([3,255,146])/256
        # clr_rchan = np.array([255,3,111])/256
        lw = 0.1

        im3 = rchan[:,:,np.newaxis] * clr_rchan + gchan[:,:,np.newaxis] * clr_gchan

        ax3.imshow(im3)

        # im1 = np.dstack((np.zeros(np.shape(mimg)),mimg,np.zeros(np.shape(mimg)))).astype(np.uint8)
        # ax1.imshow(im1,vmin=0,vmax=255)

        # im2 = np.dstack((mimg2,np.zeros(np.shape(mimg2)),np.zeros(np.shape(mimg2)))).astype(np.uint8)
        # ax2.imshow(im2,vmin=0,vmax=255)

        # im3 = np.dstack((mimg2,mimg,np.zeros(np.shape(mimg2)))).astype(np.uint8)
        # ax3.imshow(im3,vmin=0,vmax=255)

        if showcells:
            outl_green = get_outlines(masks_suite2p)
            #Filter good cells for visualization laters: 
            # outl_green = np.array(outl_green)[iscell[:,0]==1]
            
            # red_filtered = redcell_cellpose[1,iscell[:,0]==1]
            red_filtered = redcell_cellpose[iscell[:,0]==1,0]

            outl_red = get_outlines(masks_cp_red)

            for i,o in enumerate(outl_green):
                if iscell[i,0]: #show only good cells
                    # ax1.plot(o[:,0], o[:,1], color='w',linewidth=lw)
                    # ax1.plot(o[:,0], o[:,1], color='g',linewidth=lw)
                    ax1.plot(o[:,0], o[:,1], color=clr_gchan,linewidth=lw)
                    # if red_filtered[i]:
                    if redcell_cellpose[i,0]:
                        # ax3.plot(o[:,0], o[:,1], color='r',linewidth=lw)
                        ax3.plot(o[:,0], o[:,1], color='w',linewidth=lw)
                        # ax3.text(np.mean(o[:,0]),np.mean(o[:,1]),'1',color='w',fontsize=12,fontweight='bold')
                        ax3.quiver(np.mean(o[:,0])+2,np.mean(o[:,1])-2,-4,-4,color='w',width=0.007,headlength=4,headwidth=2,pivot='tip')

                    # else: 
                    #     ax3.plot(o[:,0], o[:,1], color='w',linewidth=lw)
                    #     h = 2

            for o in outl_red:
                ax2.plot(o[:,0], o[:,1], color=clr_rchan,linewidth=lw)
                # ax2.plot(o[:,0], o[:,1], color='r',linewidth=lw)
                # ax2.plot(o[:,0], o[:,1], color='w',linewidth=lw)
        
        ax1.set_axis_off()
        ax1.set_aspect('auto')
        # ax1.set_title('GCaMP6', fontsize=8, color='black', fontweight='bold',loc='center')
        ax1.text(0.05,0.05,'GCaMP6',color=clr_gchan,transform=ax1.transAxes,fontsize=6,fontweight='bold')
        ax2.set_axis_off()
        ax2.set_aspect('auto')
        # ax2.set_title('tdTomato', fontsize=8, color='black', fontweight='bold',loc='center')
        ax2.text(0.05,0.05,'tdTomato',color=clr_rchan,transform=ax2.transAxes,fontsize=6,fontweight='bold')
        ax3.set_axis_off()
        ax3.set_aspect('auto')
        # ax3.set_title('Merge', fontsize=8, color='black', fontweight='bold',loc='center')
        ax3.text(0.05,0.05,'Merge',color='w',transform=ax3.transAxes,fontsize=6,fontweight='bold')

        # fig.savefig(os.path.join(plane_folder,'labeling_Plane%d.jpg' % iplane),dpi=600)
    return fig

#%% 
animal_id           = 'LPE11622' #If empty than all animals in folder will be processed
animal_id           = 'LPE11086' #If empty than all animals in folder will be processed
sessiondate         = '2024_03_27'
rawdatadir          = 'I:\\RawData\\'
iplane              = 1

date_filter         = []
# date_filter         = ['2023_04_12']
# date_filter         = ['2024_05_07']

animal_ids          = ['LPE12223']
date_filter        = ['2024_06_08']

animal_ids          = ['LPE11086']
date_filter        = ['2024_01_05']

# LPE11998_2024_05_10
# 2    LPE12223_2024_06_10
# 3    LPE11622_2024_03_25
# 4    LPE11622_2024_03_28
# 5    LPE11998_2024_05_02
# 6    LPE12013_2024_05_07
# 7    LPE12223_2024_06_08
# 8    LPE12385_2024_06_13

animal_ids          = ['LPE12223']
date_filter        = ['2024_06_10']

# animal_ids          = ['LPE09665', 'LPE11495', 'LPE11998', 'LPE12013'] #If empty than all animals in folder will be processed
# animal_ids          = ['LPE09665', 'LPE11495', 'LPE11998', 'LPE12013'] #If empty than all animals in folder will be processed

for animal_id in animal_ids: #for each animal

    rawdatadir = get_rawdata_drive([animal_id],protocols=['GR','GN'])

    sessiondates = os.listdir(os.path.join(rawdatadir,animal_id)) 

    if any(date_filter): #If dates specified, then process only those:
        sessiondates = [x for x in sessiondates if x in date_filter]

    for sessiondate in sessiondates: #for each of the sessions for this animal

        sesfolder           = os.path.join(rawdatadir,animal_id,sessiondate)
        suite2p_folder      = os.path.join(sesfolder,"suite2p")

        try:
            plane_folders       = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
        
            for iplane,plane_folder in enumerate(plane_folders):
                plane_folder        = plane_folders[iplane]

                fig     = show_labeling_plane(iplane,plane_folder,saveoverlay=True,showcells=True,overlap_threshold=0.5,gcamp_proj='meanImg')
        except:
            print('suite2p folders not found for %s %s' % (animal_id,sessiondate))

#%% Show example plane V1 for one animal
animal_id           = 'LPE10883' #If empty than all animals in folder will be processed
sessiondate         = '2024_01_26'
iplane              = 0

rawdatadir          = get_rawdata_drive([animal_id],protocols=['GR'])
sesfolder           = os.path.join(rawdatadir,animal_id,sessiondate)
suite2p_folder      = os.path.join(sesfolder,"suite2p")
plane_folders       = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
plane_folder        = plane_folders[iplane]

fig                 = show_labeling_plane(iplane,plane_folder,saveoverlay=True,
                                          showcells=True,overlap_threshold=0.5,gcamp_proj='max_proj')

my_savefig(fig,figdir,'ExamplePlane_%s_%s_Plane%d' % (animal_id,sessiondate,iplane))


#%% Show example plane V1 for one animal
animal_id           = 'LPE11622' #If empty than all animals in folder will be processed
sessiondate         = '2024_03_27'
iplane              = 7

rawdatadir          = get_rawdata_drive([animal_id],protocols=['GR'])
sesfolder           = os.path.join(rawdatadir,animal_id,sessiondate)
suite2p_folder      = os.path.join(sesfolder,"suite2p")
plane_folders       = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
plane_folder        = plane_folders[iplane]

fig                 = show_labeling_plane(iplane,plane_folder,saveoverlay=True,
                                          showcells=True,overlap_threshold=0.5,gcamp_proj='max_proj')
my_savefig(fig,figdir,'ExamplePlane_%s_%s_Plane%d' % (animal_id,sessiondate,iplane))


#%% Show cropped inset of example V1:
# animal_id           = 'LPE10883' #If empty than all animals in folder will be processed
# sessiondate         = '2023_10_31'
# iplane              = 0
# croplims            = [[20,140],[20,140]]

animal_id           = 'LPE12223' #If empty than all animals in folder will be processed
sessiondate         = '2024_06_10'
iplane              = 2
# cornercoords        = [0,0]
# span                = 512

cornercoords        = [220,240]
span                = 150

# gcamp_proj          = 'max_proj'
gcamp_proj          = 'meanImg'

croplims            = [[cornercoords[0],cornercoords[0]+span],[cornercoords[1],cornercoords[1]+span]]

rawdatadir          = get_rawdata_drive([animal_id],protocols=['GR'])
sesfolder           = os.path.join(rawdatadir,animal_id,sessiondate)
suite2p_folder      = os.path.join(sesfolder,"suite2p")
plane_folders       = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
plane_folder        = plane_folders[iplane]

fig                 = show_labeling_plane(iplane,plane_folder,saveoverlay=True,
                                          showcells=False,overlap_threshold=0.5,gcamp_proj=gcamp_proj)

axes = fig.axes
axes[2].set_xlim(croplims[0])
axes[2].set_ylim(croplims[1])
barlength = 10 * 600/512
axes[2].plot([cornercoords[0]+span-barlength-10,cornercoords[0]+span-10],
             [cornercoords[1]+10,cornercoords[1]+10],color='w',linewidth=2)
axes[2].text(cornercoords[0]+span-barlength-30,cornercoords[1]+20,s='10 um',color='w',
             fontweight='bold',fontsize=6)
my_savefig(fig,figdir,'Crop_%s_%s_Plane%d_%s' % (animal_id,sessiondate,iplane,gcamp_proj))

#%% 
# Show cropped inset of example PM: (same animal)
# animal_id           = 'LPE12223' #If empty than all animals in folder will be processed
# sessiondate         = '2024_06_11'
iplane              = 5
# cornercoords        = [350,350]
# span                = 150

# cornercoords        = [0,0]
# span                = 512

cornercoords        = [200,290]
span                = 150

croplims            = [[cornercoords[0],cornercoords[0]+span],[cornercoords[1],cornercoords[1]+span]]

rawdatadir          = get_rawdata_drive([animal_id],protocols=['GR'])
sesfolder           = os.path.join(rawdatadir,animal_id,sessiondate)
suite2p_folder      = os.path.join(sesfolder,"suite2p")
plane_folders       = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
plane_folder        = plane_folders[iplane]

fig                 = show_labeling_plane(iplane,plane_folder,saveoverlay=True,
                                          showcells=False,overlap_threshold=0.5,gcamp_proj=gcamp_proj)

axes = fig.axes
axes[2].set_xlim(croplims[0])
axes[2].set_ylim(croplims[1])

barlength = 10 * 600/512
axes[2].plot([cornercoords[0]+span-barlength-10,cornercoords[0]+span-10],
             [cornercoords[1]+10,cornercoords[1]+10],color='w',linewidth=2)
axes[2].text(cornercoords[0]+span-barlength-30,cornercoords[1]+20,s='10 um',color='w',
             fontweight='bold',fontsize=6)
my_savefig(fig,figdir,'Crop_%s_%s_Plane%d_%s' % (animal_id,sessiondate,iplane,gcamp_proj))

#%% 
# Show cropped inset of example AL: (same animal)
animal_id           = 'LPE12223' #If empty than all animals in folder will be processed
sessiondate         = '2024_06_10'
iplane              = 6
cornercoords        = [150,300]
span                = 150
cornercoords        = [0,0]
span                = 150

croplims            = [[cornercoords[0],cornercoords[0]+span],[cornercoords[1],cornercoords[1]+span]]

rawdatadir          = get_rawdata_drive([animal_id],protocols=['GR'])
sesfolder           = os.path.join(rawdatadir,animal_id,sessiondate)
suite2p_folder      = os.path.join(sesfolder,"suite2p")
plane_folders       = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
plane_folder        = plane_folders[iplane]

fig                 = show_labeling_plane(iplane,plane_folder,saveoverlay=True,
                                          showcells=False,overlap_threshold=0.5,gcamp_proj=gcamp_proj)

axes = fig.axes
axes[2].set_xlim(croplims[0])
axes[2].set_ylim(croplims[1])

barlength = 10 * 600/512
axes[2].plot([cornercoords[0]+span-barlength-10,cornercoords[0]+span-10],
             [cornercoords[1]+10,cornercoords[1]+10],color='w',linewidth=2)
axes[2].text(cornercoords[0]+span-barlength-30,cornercoords[1]+20,s='10 um',color='w',
             fontweight='bold',fontsize=6)
my_savefig(fig,figdir,'Crop_%s_%s_Plane%d_%s' % (animal_id,sessiondate,iplane,gcamp_proj))

#%% 






##### Show mean planes for each session and save in original dir:
protocols           = ['GR','IM','GN']
animal_ids          = [] #If empty than all animals in folder will be processed
date_filter         = []
# animal_ids          = ['LPE11086'] #If empty than all animals in folder will be processed
# date_filter         = ['2023_12_16']

# clr_rchan = np.array(ImageColor.getcolor('#ff0040', "RGB")) / 255
# clr_gchan = np.array(ImageColor.getcolor('#00ffbf', "RGB")) / 255
clr_rchan = [1,0,0]
clr_gchan = [0,1,0]

lowprc = 0.5 #scaling minimum percentile
uppprc = 99 #scaling maximum percentile

## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids =  get_animals_protocol(protocols)
    # [f.name for f in os.scandir(rawdatadir) if f.is_dir() and f.name.startswith(('LPE','NSH'))]

for animal_id in tqdm(animal_ids, desc="Processing animal", total=len(animal_ids)): #for each animal

    rawdatadir = get_rawdata_drive(animal_id,protocols)

    sessiondates = os.listdir(os.path.join(rawdatadir,animal_id))
    sessiondates = [x for x in sessiondates if os.path.isdir(os.path.join(rawdatadir,animal_id,x))]

    if any(date_filter): #If dates specified, then process only those:
        sessiondates = [x for x in sessiondates if x in date_filter]

    for sessiondate in sessiondates: #for each of the sessions for this animal
        suite2pfolder       = os.path.join(rawdatadir,animal_id,sessiondate,'suite2p')
        if os.path.exists(suite2pfolder):
            plane_folders = natsorted([f.path for f in os.scandir(suite2pfolder) if f.is_dir() and f.name[:5]=='plane'])

            fig, axes = plt.subplots(8, 3,figsize=(3*2,8*2))
            for iplane, plane_folder in enumerate(plane_folders):

                # load ops of plane0:
                ops                = np.load(os.path.join(plane_folder, 'ops.npy'), allow_pickle=True).item()

                #standard mean image:
                # mimg = ops['meanImg'] 
                #max projection:
                mimg = np.zeros([512,512])
                mimg[ops['yrange'][0]:ops['yrange'][1],
                    ops['xrange'][0]:ops['xrange'][1]]  = ops['max_proj']

                mimg = im_norm8(mimg,min=lowprc,max=uppprc) #scale between 0 and 255

                ## Get red image:
                mimg2 = ops['meanImg_chan2'] #get red channel image from ops

                mimg2 = im_norm(mimg2,min=lowprc,max=uppprc) #scale between 0 and 255
                mimg2 = im_sqrt(mimg2) #square root transform to enhance weakly expressing cells
                mimg2 = im_norm(mimg2,min=5,max=100) #scale between 0 and 255

                ######

                rchan = (mimg2 - np.min(mimg2)) / (np.max(mimg2) - np.min(mimg2))
                gchan = (mimg - np.min(mimg)) / (np.max(mimg) - np.min(mimg))
                
                bchan = np.zeros(np.shape(mimg))

                axes[iplane,0].imshow(gchan,cmap='gray',vmin=0,vmax=1)
                axes[iplane,1].imshow(rchan,cmap='gray',vmin=0,vmax=1)
        
                im3 = rchan[:,:,np.newaxis] * clr_rchan + gchan[:,:,np.newaxis] * clr_gchan
        
                axes[iplane,2].imshow(im3)

        for ax in axes.flatten():
            ax.set_axis_off()
            ax.set_aspect('auto')
        fig.suptitle(f'{animal_id} - {sessiondate}')
        plt.tight_layout(rect=[0, 0, 1, 1])
        # fig.savefig(os.path.join(rawdatadir,animal_id,sessiondate,'PlaneOverview_%s_%s.png' % (animal_id,sessiondate)), 
        #             dpi=300,format = 'png')       
        # fig.savefig(os.path.join(rawdatadir,animal_id,sessiondate,'PlaneOverview_%s_%s.pdf' % (animal_id,sessiondate)), 
        #             dpi=300,format = 'pdf')

print(f'\n\nPreprocessing Completed')
