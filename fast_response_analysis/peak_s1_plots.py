import sys
import os.path as osp

from copy import deepcopy
import datetime
import os
import pickle
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multihist import Hist1d, Histdd
from tqdm.notebook import tqdm
import pandas as pd
from scipy import stats

import strax
import straxen

def plot_area_risetime(peak_basics):
    ph = Histdd(peak_basics['area'], peak_basics['rise_time'],
                    bins=(np.logspace(1, 5, 200), np.logspace(1, 3, 200)))
    plt.figure(figsize=(12,4))
    ph.plot(log_scale=True, cblabel='events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("rise time (ns)", ha='right', y=1)
    plt.axvline(x=100, ymin=0.39, ymax=0.59, linestyle="-", color = 'r', label='S1/S2 boundary')
    plt.axhline(y=150, xmin=0.25, xmax=1, linestyle="-", color = 'r')
    plt.axhline(y=60, xmin=0, xmax=0.25, linestyle="-", color = 'r')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    
def plot_area_width(peak_basics):
    ph50 = Histdd(peak_basics['area'], peak_basics['range_50p_area'],
                    bins=(np.logspace(1, 5, 200), np.logspace(1, 3, 200)))
    plt.figure(figsize=(12,4))
    ph50.plot(log_scale=True, cblabel='events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("peak width 50% (ns)", ha='right', y=1)
    plt.xscale('log')
    plt.yscale('log')

def plot_area_maxpmt(peak_basics):
    #phmax = Histdd(peak_basics['area'], peak_basics['max_pmt_area'],
    #                bins=(np.logspace(0, 4, 200), np.logspace(0, 3, 200)))
    phmax = Histdd(peak_basics['area'], peak_basics['max_pmt_area']/peak_basics['area']*100,
                    bins=(np.logspace(0, 4, 200), np.logspace(0, 2, 200)))
    plt.figure(figsize=(12,4))
    phmax.plot(log_scale=True, cblabel='events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    #plt.ylabel("max PMT area (PE)", ha='right', y=1)
    plt.ylabel("max PMT area fraction (%)", ha='right', y=1)
    plt.xscale('log')
    plt.yscale('log')
    
    