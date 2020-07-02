import strax
import straxen
import csv
import json
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import warnings

def pmt_plot(target, PMTs, active_channels,
             label='', figsize=(11, 4), s=230, alpha=1,
             pmt_label_size=7, pmt_label_color='white',
             log_scale=False, extend='neither', vmin=None, vmax=None,
             **kwargs):
    
    """Plots the PMT arrays, using c as a color scale"""
    
    if vmin is None: vmin = target.min() 
    if vmax is None: vmax = target.max() 
    
    r = 79.79 * 1.1 #slightly enlarged tpc radius
    
    f, axes = plt.subplots(3, 1, figsize=figsize)
    for array_i, array_name in enumerate(['top', 'bottom']):
        ax = axes[array_i]
        ax.tick_params(labelsize = 16)
        plt.sca(ax)
        ax.set_aspect('equal')
            
        plt.xlim(-r, r)
        plt.ylim(-r, r)

        #active channels
        pos_x = []
        pos_y = []
        pmt_num = []
        mask_list = []

        #inactive channels
        pos_x_missing = []
        pos_y_missing = []
        pmt_num_missing = []
        mask_list_missing = []

        plt.title(f'Channels:{array_name.capitalize()}',fontsize=20)
        for i in range(len(PMTs)):
            if not PMTs['pmt'][i] in active_channels:
                mask = PMTs['array'][i] == array_name
                mask_list_missing.append(mask)
                if mask == True:
                    pos_x_missing.append(PMTs['x'][i])
                    pos_y_missing.append(PMTs['y'][i])
                    pmt_num_missing.append(PMTs['pmt'][i])
            else:
                mask = PMTs['array'][i] == array_name
                mask_list.append(mask)
                if mask == True:
                    pos_x.append(PMTs['x'][i])
                    pos_y.append(PMTs['y'][i])
                    pmt_num.append(PMTs['pmt'][i])

        plt.scatter(pos_x, pos_y, s=s, c=target[mask_list], alpha=alpha, vmin=vmin, vmax=vmax,
                    norm=matplotlib.colors.LogNorm() if log_scale else None, **kwargs)

        plt.xlabel('distance (cm)',fontsize=16)
        plt.ylabel('distance (cm)',fontsize=16)

        #label active channels
        if pmt_label_size:
            for p in range(len(pos_x)):
                plt.text(pos_x[p], pos_y[p], str(pmt_num[p]),
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=pmt_label_size, color=pmt_label_color)

        #label missing channels
        if pmt_label_size:
            for p in range(len(pos_x_missing)):
                plt.text(pos_x_missing[p], pos_y_missing[p], 'X',
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=pmt_label_size*2, color='red')
    
    cbar = plt.colorbar(ax=axes, extend=extend, label=label)
    cbar.set_label(label, size=16)
    cbar.ax.tick_params(labelsize=16)
    
"""    
def run_channels(run_id,rr):
    #Checks for signal channels with and without data and returns numpy arrays
    #sort rr by channel number
    rr = np.sort(rr, order='channel') #sort data by channel number
    
    #create arrays of channels from TPC(0-493 DM, 500-752 0vbb)
    channels = np.arange(0,494,1)
    
    #determine which channels are present in rr
    active_channels = np.unique(rr[rr['channel'] <= 493]['channel'])
    
    #print which channels have signal, and which channels are missing (i.e. did not get signal)
    print(f"Got signals from {len(active_channels)} channels\n"
    
    return active_channels
"""