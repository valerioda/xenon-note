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
from straxen import units
from scipy.optimize import curve_fit

def plot_area_width_aft(st, run_id, low = 0, high = 6, low2 = 0, high2 = 1, binning = 500):
    events = st.get_array(run_id,'event_info')
    ph_s1 = Histdd(events['s1_area'], events['s1_range_50p_area'],
                    bins=(np.logspace(low, high, binning), np.logspace(1, 6, binning)))
    ph_s2 = Histdd(events['s2_area'], events['s2_range_50p_area'],
                    bins=(np.logspace(low, high, binning), np.logspace(1, 6, binning)))
    phcs1 = Histdd(events['s1_area'], events['s1_area_fraction_top'],
                    bins=(np.logspace(low, high, binning), np.linspace(0, 1, binning)))
    phcs2 = Histdd(events['s2_area'], events['s2_area_fraction_top'],
                    bins=(np.logspace(low, high, binning), np.linspace(0, 1, binning)))
    plt.figure(figsize=(12,6))
    ph_s1.plot(log_scale=True, cblabel='S1 events',cmap='plasma')
    ph_s2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("peak width 50% (ns)", ha='right', y=1)
    plt.title(f'run {run_id}')
    plt.xscale('log')
    plt.yscale('log')
    plt.figure(figsize=(12,6))
    phcs1.plot(log_scale=True, cblabel='S1 events',cmap='plasma')
    phcs2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("area fraction top", ha='right', y=1)
    plt.title(f'run {run_id}')
    plt.xscale('log')
    #plt.yscale('log')
    

def plots2_area_width(st, run_id, low = 0, high = 6, low2 = 0, high2 = 6, binning = 500):
    events = st.get_array(run_id,'event_info')
    ph_s2 = Histdd(events['s2_area'], events['s2_range_50p_area'],
                    bins=(np.logspace(low, high, binning), np.logspace(low2, high2, binning)))
    plt.figure(figsize=(12,6))
    ph_s2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("peak width 50% (ns)", ha='right', y=1)
    plt.title(f'run {run_id}')
    plt.xscale('log')
    plt.yscale('log')


def plots2_area_aft(st, run_id, low = 0, high = 6, low3 = 0, high3 = 1, binning = 500):
    events = st.get_array(run_id,'event_info')
    phcs2 = Histdd(events['s2_area'], events['s2_area_fraction_top'],
                    bins=(np.logspace(low, high, binning), np.linspace(low3, high3, binning)))
    plt.figure(figsize=(12,6))
    phcs2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("area fraction top", ha='right', y=1)
    plt.title(f'run {run_id}')
    plt.xscale('log')
    #plt.yscale('log')

    
def drift_velocity(st, run_id, area_bounds, aft_bounds, width_bounds, low = 0, high = 6000, binning = 500,bplot=True,perc=2):
    events = st.get_array(run_id,'event_info',
                      config = dict(left_event_extension = int(3e6),right_event_extension = int(3e6)),
                      max_workers = 10)
    events0 = events
    events = events[(events['s2_area']>area_bounds[0])&(events['s2_area']<area_bounds[1])&
                      (events['s2_range_50p_area']>width_bounds[0]) & 
                      (events['s2_range_50p_area']<width_bounds[1]) &
                      (events['s2_area_fraction_top']>aft_bounds[0]) & 
                      (events['s2_area_fraction_top']<aft_bounds[1])]
    
    plt.figure(figsize=(12,6))
    dt = np.linspace(low, high, binning)
    hdtime0 = Hist1d(events0['drift_time']/1e3, bins=dt)
    hdtime = Hist1d(events['drift_time']/1e3, bins=dt)
    if (bplot): hdtime.plot(color='b',label=f'background events')
    hdtime0.plot(color='black',label='all events')
    plt.ylabel("events", ha='right', y=1)
    plt.xlabel("drift time ($\mu$s)", ha='right', x=1)
    plt.yscale('log')
    dropoff = dt[np.where(np.array(hdtime)>10)[0][-1]]
    plt.axvline(x=dropoff,linewidth=1,linestyle='--', color='r',label=f'cathode drop-off = {dropoff:.1f} us')
    plt.legend(fontsize=14)
    
    plt.figure(figsize=(12,6))
    area_ratio = np.divide(events['cs2'],events['cs1'])
    mh = Histdd(events['drift_time']/1e3, area_ratio,
            bins=(np.linspace(low, high, binning), np.logspace(0, 5, 200)))
    mh.plot(log_scale=True, cblabel='events')
    plt.xlabel("drift time ($\mu$s)", ha='right', x=1,fontsize=12)
    plt.ylabel("cS2/cS1", ha='right', y=1,fontsize=12)
    plt.title(f'run {run_id}',fontsize=14)
    plt.yscale('log')
    plt.axvline(x=dropoff,linewidth=1,linestyle='-', color='r',label=f'cathode drop-off = {dropoff:.1f} us')
    plt.axvline(x=dropoff+dropoff*perc/100,linewidth=1,linestyle='--', color='r')
    plt.axvline(x=dropoff-dropoff*perc/100,linewidth=1,linestyle='--', color='r')
    
    plt.figure(figsize=(12,6))
    dts = np.linspace(0, 20, 200)
    mh_low = Histdd(events['drift_time']/1e3, area_ratio,
            bins=(dts, np.logspace(0, 5, 200)),axis_names=['drift_time', 'area_ratio'])
    mh_low.plot(log_scale=True, cblabel='events')
    median = mh_low.percentile(50, axis='area_ratio')
    plt.xlabel("drift time ($\mu$s)", ha='right', x=1,fontsize=12)
    plt.ylabel("cS2/cS1", ha='right', y=1,fontsize=12)
    plt.title(f'run {run_id}',fontsize=14)
    median.plot(color='red',label='median')
    plt.yscale('log')
    gatedt = dts[np.where(np.array(median[:int(len(median)/2)])>70)[0][-1]]
    plt.axvline(x=gatedt,linewidth=1,linestyle='--', color='b',label=f'gate drift time = {gatedt:.1f} $\mu$s')
    plt.legend(fontsize=14)
    vd = 1485/(dropoff-gatedt)
    print(f'Drift velocity = {vd:.3f} mm/$\mu$s')
    return vd


def diffusion_model(t, D, vd, w0):
    sigma_to_r50p = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)        
    return np.sqrt(2 * sigma_to_r50p**2 * D * t / vd**2 + w0**2)


def diffusion_constant(st, run_id, area_bounds, aft_bounds, width_bounds, fit_range, vd = 600):
    events = st.get_array(run_id,'event_info',
                      config = dict(left_event_extension = int(3e6),right_event_extension = int(3e6)),
                      max_workers = 10)

    data_cut = events[(events['s2_area']>area_bounds[0])&(events['s2_area']<area_bounds[1])&
                      (events['s2_range_50p_area']>width_bounds[0]) & 
                      (events['s2_range_50p_area']<width_bounds[1]) &
                      (events['s2_area_fraction_top']>aft_bounds[0]) & 
                      (events['s2_area_fraction_top']<aft_bounds[1])]
    # s2_width_50 vs drift_time
    t = np.linspace(0, 2000, 200)
    ph = Histdd(data_cut['drift_time']/1e3, data_cut['s2_range_50p_area'],
                bins=(t, np.linspace(100, 15e3, 200)))
    plt.figure(figsize=(12,6))
    ph.plot(log_scale=True, cblabel='events')
    plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
    plt.ylabel("S2 width 50% (ns)", ha='right', y=1,fontsize=12)
    plt.title(f'run {run_id}',fontsize=14)
    
    #mean = np.array(ph.average(axis=1))
    #plt.plot(t[:len(mean)], mean, color='r',linestyle='-', label='mean per drift time slice')
    perc50 = np.array(ph.percentile(percentile=50, axis=1))
    plt.plot(t[:len(perc50)], perc50, color='b',linestyle='--', label='50% percentile')
    
    D_guess = 45e3 * units.cm**2 / units.s
    w0_guess = 500 * units.ns
    print(f'Drift velocity = {vd:.3f} mm/$\mu$s ')
    vd = vd * units.mm / units.us
    guess = np.array([D_guess, vd, w0_guess])
    ys_m = diffusion_model(t, *guess)
    #plt.plot(t, ys_m, c='yellow',linestyle='--',label='initial guess')
    ll = np.where(t>fit_range[0])[0][0]
    hh = np.where(t>fit_range[1])[0][0]
    diffusion = lambda x, D, w0: diffusion_model(x, D, vd, w0)
    #popt, pcov = curve_fit(diffusion, t[ll:hh], mean[ll:hh], p0=(D_guess, w0_guess))
    popt, pcov = curve_fit(diffusion, t[ll:hh], perc50[ll:hh], p0=(D_guess, w0_guess))
    perr = np.sqrt(np.diag(pcov))
    plt.axvspan(*fit_range, alpha=0.1, color='blue', label='fit region')
    ys_u = diffusion(t, *popt) + 1000
    ys_m = diffusion(t, *popt)
    ys_d = diffusion(t, *popt) - 1000

    plt.plot(t, ys_m, label=f'$D = {popt[0]/1e3/(units.cm**2 / units.s):.2f}$ cm$^2$/s',color='r')
    #plt.plot(t, ys_u,color='r')
    #plt.plot(t, ys_d,color='r')
    plt.legend(fontsize=14)
    print(f'Diffusion constant = {popt[0]/1e3/(units.cm**2 / units.s):.2f} +/- {perr[0]/1e3/(units.cm**2 / units.s):.2f} cm$^2$/s ')
    #print(f'Diffusion constant = {popt[0]/1e3/(units.cm**2 / units.s):.2f} cm^2/s ')
    print(f'w0 = {popt[1]/(units.ns):.2f} +/- {perr[1]/(units.ns):.2f} ns ')

    
def expo(t, a, tau):
    return a*np.exp(-t/tau)

def electron_lifetime(st, run_id, area_bounds, aft_bounds, width_bounds, fit_range=(0,1000),
                      low = 0, high = 2000, low2 = 1, high2 = 1e4, binning = 500):
    events = st.get_array(run_id,'event_info')

    events = events[(events['s2_area']>area_bounds[0])&(events['s2_area']<area_bounds[1])&
                      (events['s2_range_50p_area']>width_bounds[0]) & 
                      (events['s2_range_50p_area']<width_bounds[1]) &
                      (events['s2_area_fraction_top']>aft_bounds[0]) & 
                      (events['s2_area_fraction_top']<aft_bounds[1])]
    
    t = np.linspace(low, high, binning)
    h = Histdd(events['drift_time']/1e3, events['s2_area'], 
                    bins=(t, np.linspace(low2, high2, binning) ))
    plt.figure(figsize=(12,6))
    h.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("drift time ($\mu$s)", ha='right', x=1,fontsize=12)
    plt.ylabel("S2 area (PE)", ha='right', y=1,fontsize=12)
    plt.title(f'run {run_id}')
    #plt.yscale('log')
    perc = h.percentile(percentile=50, axis=1)
    perc.plot(color='b',label='50% percentile')
    ll = np.where(t>fit_range[0])[0][0]
    hh = np.where(t>fit_range[1])[0][0]
    popt, pcov = curve_fit(expo, t[ll:hh], perc[ll:hh], p0=(2000, 100))
    perr = np.sqrt(np.diag(pcov))
    ys = expo(t, *popt)
    plt.plot(t, ys, label=f'$EL = {popt[0]:.1f}\pm{perr[0]:.1f} ~\mu$s',color='r')
    plt.legend(fontsize=14)