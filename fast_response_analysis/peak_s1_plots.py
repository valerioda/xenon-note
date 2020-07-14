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

def plot_area(peak_basics,low=0,high=5):
    p_area = Hist1d(peak_basics['area'], bins=(np.logspace(low, high, 200)))
    p_area.plot()
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("events", ha='right', y=1)
    plt.xscale('log')
    plt.yscale('log')
    
def plot_risetime(peak_basics,low=1,high=5):
    rise_time = Hist1d(peak_basics['rise_time'], bins=(np.logspace(low, high, 200)))
    rise_time.plot()
    plt.xlabel("rise time (ns)", ha='right', x=1)
    plt.ylabel("events", ha='right', y=1)
    plt.xscale('log')
    
def plot_area_risetime(peak_basics,low=0,high=5,low2=1,high2=4):
    ph = Histdd(peak_basics['area'], peak_basics['rise_time'],
                    bins=(np.logspace(low, high, 200), np.logspace(low2, high2, 200)))
    plt.figure(figsize=(12,6))
    ph.plot(log_scale=True, cblabel='events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("rise time (ns)", ha='right', y=1)
    plt.xscale('log')
    plt.yscale('log')
    
def plot_area_width(peak_basics,low=0,high=5,low2=1,high2=4):
    ph50 = Histdd(peak_basics['area'], peak_basics['range_50p_area'],
                    bins=(np.logspace(low, high, 200), np.logspace(low2, high2, 200)))
    plt.figure(figsize=(12,6))
    ph50.plot(log_scale=True, cblabel='events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("peak width 50% (ns)", ha='right', y=1)
    plt.xscale('log')
    plt.yscale('log')

def plot_area_maxpmt(peak_basics,low=0,high=5,low2=0,high2=2):
    #phmax = Histdd(peak_basics['area'], peak_basics['max_pmt_area'],
    #                bins=(np.logspace(0, 4, 200), np.logspace(0, 3, 200)))
    phmax = Histdd(peak_basics['area'], peak_basics['max_pmt_area']/peak_basics['area']*100,
                    bins=(np.logspace(low, high, 200), np.logspace(low2, high2, 200)))
    plt.figure(figsize=(12,6))
    phmax.plot(log_scale=True, cblabel='events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    #plt.ylabel("max PMT area (PE)", ha='right', y=1)
    plt.ylabel("max PMT area fraction (%)", ha='right', y=1)
    plt.xscale('log')
    plt.yscale('log')

def plot_area_top(peak_basics,low=0,high=5,low2=-2,high2=0):
    phmax = Histdd(peak_basics['area'], peak_basics['area_fraction_top'],
                    bins=(np.logspace(low, high, 200), np.logspace(low2, high2, 200)))
    plt.figure(figsize=(12,6))
    phmax.plot(log_scale=True, cblabel='events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    #plt.ylabel("max PMT area (PE)", ha='right', y=1)
    plt.ylabel("area fraction top", ha='right', y=1)
    plt.xscale('log')
    plt.yscale('log')
    
def rectangle(bounds1,bounds2,color):
    plt.gca().add_patch(matplotlib.patches.Rectangle((bounds1[0],bounds2[0]),
                                                     bounds1[1]-bounds1[0],
                                                     bounds2[1]-bounds2[0],
                                                     edgecolor=color,facecolor='none'))
    