import sys
import os.path as osp

from copy import deepcopy
import time
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
from scipy.ndimage import gaussian_filter1d
import peaks_utility as psu


def plot_area_width_aft_kr_old(events, run_id, low = 0, high = 6, low2 = 0, high2 = 1, binning = 500):
    print('total events',len(events))
    ph_s1 = Histdd(events['s1_a_area'], events['s1_a_range_50p_area'],
                    bins=(np.logspace(low, high, binning), np.logspace(1, 6, binning)))
    ph_s2 = Histdd(events['s2_a_area'], events['s2_a_range_50p_area'],
                    bins=(np.logspace(low, high, binning), np.logspace(1, 6, binning)))
    phcs1 = Histdd(events['s1_a_area'], events['s1_a_area_fraction_top'],
                    bins=(np.logspace(low, high, binning), np.linspace(0, 1, binning)))
    phcs2 = Histdd(events['s2_a_area'], events['s2_a_area_fraction_top'],
                    bins=(np.logspace(low, high, binning), np.linspace(0, 1, binning)))
    plt.figure(figsize=(12,6))
    ph_s1.plot(log_scale=True, cblabel='S1 events',cmap='plasma')
    ph_s2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("peak width 50% (ns)", ha='right', y=1)
    plt.title(f'{run_id}')
    plt.xscale('log')
    plt.yscale('log')
    plt.figure(figsize=(12,6))
    phcs1.plot(log_scale=True, cblabel='S1 events',cmap='plasma')
    phcs2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("area fraction top", ha='right', y=1)
    plt.title(f'{run_id}')
    plt.xscale('log')
    plt.figure(figsize=(12,6))
    ph_s1s2 = Histdd(events['s1_a_area'], events['s2_a_area'],
                    bins=(np.logspace(0, 5, binning), np.logspace(1, 7, binning)))
    ph_s1s2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("S1 peak area (PE)", ha='right', x=1)
    plt.ylabel("S2 peak area (PE)", ha='right', y=1)
    plt.title(f'{run_id}')
    plt.xscale('log')
    plt.yscale('log')
    
def plot_area_width_aft(events, run_id, low = 0, high = 7, low2 = 0, high2 = 1, binning = 500):
    print('total events',len(events))
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
    plt.title(f'{run_id}')
    plt.xscale('log')
    plt.yscale('log')
    plt.figure(figsize=(12,6))
    phcs1.plot(log_scale=True, cblabel='S1 events',cmap='plasma')
    phcs2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("area fraction top", ha='right', y=1)
    plt.title(f'{run_id}')
    plt.xscale('log')

    
def plots2_area_width(st, run_id, low = 0, high = 6, low2 = 0, high2 = 6, binning = 500):
    events = st.get_array(run_id,'event_info')
    print('total events',len(events))
    ph_s2 = Histdd(events['s2_area'], events['s2_range_50p_area'],
                    bins=(np.logspace(low, high, binning), np.logspace(low2, high2, binning)))
    plt.figure(figsize=(12,6))
    ph_s2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("peak width 50% (ns)", ha='right', y=1)
    plt.title(f'run {run_id}')
    plt.xscale('log')
    plt.yscale('log')

def plotS2_area_aft(events, run_id, low = 0, high = 7, low2 = 0, high2 = 1, binning = 500):
    print('total events',len(events))
    aspace = np.logspace(low, high, binning)
    phcs2 = Histdd(events['s2_area'], events['s2_area_fraction_top'],
                    bins=(aspace, np.linspace(0, 1, binning)))
    plt.figure(figsize=(12,6))
    phcs2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("area fraction top", ha='right', y=1)
    plt.title(f'run {run_id}')
    plt.xscale('log')
    a = np.logspace(2, 6.5, 201)
    _a = np.clip(a, 0, 1e3)
    mu = 0.725
    b = beta.isf(0.0001, ((_a * mu) / 1.2).astype(int),((_a - _a * mu) / 1.2).astype(int))
    c = beta.isf(0.9999, ((_a * mu) / 1.2).astype(int),((_a - _a * mu) / 1.2).astype(int))
    aft_ul = interp1d(a, b, bounds_error=False, fill_value='extrapolate')
    aft_ll = interp1d(a, c, bounds_error=False, fill_value='extrapolate')
    plt.plot(a,aft_ul(a),'r--')
    plt.plot(a,aft_ll(a),'r--')

def mask_FV(ev):
    mask = ev['z'] > -134.238
    mask &= ev['r'] < 60.73
    mask &= ev['z'] < -13.6132
    return mask
    
def mask_KrSingleS1_old(df):
    def line(x):
        return 0.55 * x + 15
    mask = (df['ds_s1_dt'] <= 0)
    mask &= (df['s1_a_n_channels'] >= 90) & (df['s1_a_n_channels'] < 225)
    mask &= (line(df['s1_a_area']) > df['s1_a_n_channels'])
    mask &= (df['s1_a_range_50p_area'] >= 60) & (df['s1_a_range_50p_area'] < 1000)
    mask &= (df['s1_a_area_fraction_top'] < 0.68)
    mask &= (df['s2_a_area'] > 2000) & (df['s2_a_area'] < 15000)
    return mask

def mask_KrSingleS1(events,sr0=False):
    mask = (events['s1_a_n_channels'] < 250) & (events['s1_a_n_channels'] > 110)
    mask &= ((events['s1_a_max_pmt_area'] < 30) & (events['z'] > -130)) | ((events['s1_a_max_pmt_area'] < 100) & (events['z'] < -130))
    mask &= (events['s2_a_max_pmt_area']>=180) & (events['s2_a_max_pmt_area']<1000)
    mask &= (events['s2_a_area'] > 2500) & (events['s2_a_area'] < 20000)
    if sr0: mask &= (events['s2_b_area'] < 60) | (np.isnan(events['s2_b_area']))
    mask &= (np.isnan(events['s1_b_area']))
    return mask


def mask_KrDouble(df,doubleS2=False):
    def line(x,a,b):
        return a*x+b
 
    w_SE= 599.70428e-3
    w_t0= 400.29572e-3
    t_0= 1.0029191e-3
    def diffusion_model(t,w_SE, w_t0, t_0):
        return np.sqrt(w_SE**2 + ((w_t0 - w_SE)**2 /t_0) * t)
 
    mask = (df['s1_a_n_channels'] >= 80) & (df['s1_a_n_channels'] < 225)
    mask &= (df['s1_b_n_channels'] >= 25) & (df['s1_b_n_channels'] < 125)
    mask &= (df['s1_b_distinct_channels'] >= 0) & (df['s1_b_distinct_channels'] < 60)
    mask &= (df['ds_s1_dt'] >= 750) & (df['ds_s1_dt'] < 2000)
    mask &= df['s1_a_area_fraction_top'] < line(df['drift_time'],-2.3e-7,0.70)
    mask &= df['s1_a_area_fraction_top'] > line(df['drift_time'],-2e-7,0.40)
    #mask &= df['s2_a_range_50p_area']/diffusion_model(df['drift_time'], w_SE, w_t0, t_0) > -30/(df['drift_time']*1e-3+10)+0.8
    #mask &= df['s2_a_range_50p_area']/diffusion_model(df['drift_time'], w_SE, w_t0, t_0) < 30/(df['drift_time']*1e-3+10)+1.2
    mask &= df['drift_time']*1e-3 < 2400
    if (doubleS2): mask &= (df['ds_s2_dt'] > 600)&(df['ds_s2_dt'] < 2000)
    return mask


def mask_s2_area_width_aft_kr(events, run_id, area_cut, width_cut,aft_cut, plot = False,
                              low = 1, high = 6, low2 = 2, high2 = 4.5, low3 = 0.4, high3 = 0.9, binning=500):
    print('total events',len(events))
    ph_s2 = Histdd(events['s2_a_area'], events['s2_a_range_50p_area'],
                    bins=(np.logspace(low, high, binning), np.logspace(low2, high2, binning)))
    phcs2 = Histdd(events['s2_a_area'], events['s2_a_area_fraction_top'],
                    bins=(np.logspace(low, high, binning), np.linspace(low3, high3, binning)))
    if plot:
        plt.figure(figsize=(12,6))
        ph_s2.plot(log_scale=True, cblabel='S2 events')
        plt.xlabel("peak area (PE)", ha='right', x=1)
        plt.ylabel("peak width 50% (ns)", ha='right', y=1)
        plt.title(f'{run_id}')
        plt.xscale('log')
        plt.yscale('log')
        psu.rectangle(area_cut, width_cut, 'r')
        
        plt.figure(figsize=(12,6))
        phcs2.plot(log_scale=True, cblabel='S2 events')
        plt.xlabel("peak area (PE)", ha='right', x=1)
        plt.ylabel("area fraction top", ha='right', y=1)
        plt.title(f'{run_id}')
        plt.xscale('log')
        psu.rectangle(area_cut, aft_cut, 'r')
    mask = (events['s2_a_area'] > area_cut[0]) & (events['s2_a_area'] < area_cut[1])
    mask &= (events['s2_a_range_50p_area'] > width_cut[0]) & (events['s2_a_range_50p_area'] < width_cut[1])
    mask &= (events['s2_a_area_fraction_top'] > aft_cut[0]) & (events['s2_a_area_fraction_top'] < aft_cut[1])
    return mask


def mask_s2_area_width_aft(events, run_id, area_cut, width_cut,aft_cut, plot = False,
                               low = 1, high = 6, low2 = 2, high2 = 4.5, low3 = 0.4, high3 = 0.9, binning=500):
    print('total events',len(events))
    ph_s2 = Histdd(events['s2_area'], events['s2_range_50p_area'],
                    bins=(np.logspace(low, high, binning), np.logspace(low2, high2, binning)))
    phcs2 = Histdd(events['s2_area'], events['s2_area_fraction_top'],
                    bins=(np.logspace(low, high, binning), np.linspace(low3, high3, binning)))
    livetime = (events['time'][len(events)-1]-events['time'][0])/1e9
    mask = (events['s2_area'] > area_cut[0]) & (events['s2_area'] < area_cut[1])
    mask &= (events['s2_range_50p_area'] > width_cut[0]) & (events['s2_range_50p_area'] < width_cut[1])
    mask &= (events['s2_area_fraction_top'] > aft_cut[0]) & (events['s2_area_fraction_top'] < aft_cut[1])
    all_rate = len(events)/livetime
    mask_rate = len(events[mask])/livetime
    start_time = datetime.fromtimestamp(events['time'][0]/1e9)
    print(f'run {run_id}, start {start_time}, livetime {livetime:.2f} s, rate: {all_rate:.2f} Hz, selection rate: {mask_rate:.2f} Hz')
    if plot:
        plt.figure(figsize=(12,6))
        ph_s2.plot(log_scale=True, cblabel='S2 events')
        plt.xlabel("peak area (PE)", ha='right', x=1)
        plt.ylabel("peak width 50% (ns)", ha='right', y=1)
        plt.title(f'run {run_id}')
        plt.xscale('log')
        plt.yscale('log')
        psu.rectangle(area_cut, width_cut, 'r')
        
        plt.figure(figsize=(12,6))
        phcs2.plot(log_scale=True, cblabel='S2 events')
        plt.xlabel("peak area (PE)", ha='right', x=1)
        plt.ylabel("area fraction top", ha='right', y=1)
        plt.title(f'run {run_id}')
        plt.xscale('log')
        psu.rectangle(area_cut, aft_cut, 'r')
    return mask, all_rate, mask_rate, start_time


def plots2_area_aft(st, run_id, low = 0, high = 6, low3 = 0, high3 = 1, binning = 500):
    events = st.get_array(run_id,'event_info')
    print('total events',len(events))
    phcs2 = Histdd(events['s2_area'], events['s2_area_fraction_top'],
                    bins=(np.logspace(low, high, binning), np.linspace(low3, high3, binning)))
    plt.figure(figsize=(12,6))
    phcs2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("peak area (PE)", ha='right', x=1)
    plt.ylabel("area fraction top", ha='right', y=1)
    plt.title(f'run {run_id}')
    plt.xscale('log')
    #plt.yscale('log')


def drift_velocity_kr(events, run_id, low = 10, high = 3000, binning = 500, w=8, plot=False, name = None):
    if 'area_ratio' in events: pass
    #else: events.insert(1, 'area_ratio', np.divide(events['cs2_a'],events['cs1_a']))
    else: events.insert(1, 'area_ratio', np.divide(events['s2_a_area'],events['s1_a_area']))
    events = events[events['area_ratio']<1e3]
    
    # cathode drop-off
    dt = np.linspace(low, high, binning)
    dtc = (dt[:-1] + dt[1:])/2.
    hdtime = Hist1d(events['drift_time']/1e3, bins=dt)
    hfilt = gaussian_filter1d(hdtime,w)
    catdt = dtc[np.where(np.gradient(hfilt)==np.gradient(hfilt).min())[0]]
    cathodedt = np.mean(catdt)
    
    if plot:
        plt.figure(figsize=(8,4.5))
        hdtime.plot(color='b',label='data')
        plt.ylabel("events", ha='right', y=1)
        plt.xlabel("drift time ($\mu$s)", ha='right', x=1)
        plt.title(f'{run_id}',fontsize=14)
        plt.axvline(x=cathodedt,linewidth=1,linestyle='-', color='r',label=f'$cathode = {cathodedt:.1f}~\mu$s')
        plt.plot(dtc,hfilt,label='filtered')
        plt.xlim(2100,2300)
        plt.legend(fontsize=14)
        if name is not None: plt.savefig(name+'_cathode_drift_time.png',dpi=600)
        #plt.figure(figsize=(12,6))
        #plt.plot(dtc,np.gradient(hfilt),label='diff. filtered')
        #plt.axvline(x=cathodedt,linewidth=1,linestyle='-', color='r',label=f'$cathode = {cathodedt:.1f}~\mu$s')
        #plt.legend(fontsize=14)
        #plt.xlim(2100,2300)

    mh = Histdd(events['drift_time']/1e3, events['area_ratio'],
            bins=(np.linspace(low, high, binning), np.logspace(0, 3, 200)))
    
    if plot:
        plt.figure(figsize=(8,4.5))
        mh.plot(log_scale=True, cblabel='events')
        plt.xlabel("drift time ($\mu$s)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2/S1", ha='right', y=1,fontsize=12)
        plt.title(f'{run_id}',fontsize=14)
        plt.yscale('log')
        plt.xlim(1500,2500)
        plt.axvline(x=cathodedt,linewidth=1,linestyle='-', color='r',label=f'$cathode = {cathodedt:.1f}~\mu$s')
        #plt.legend(fontsize=14)
    
    # gate drift time
    dts = np.linspace(0, 20, 200)
    dtsc = (dts[:-1] + dts[1:])/2.
    mh_low = Histdd(events['drift_time']/1e3, events['area_ratio'],
            bins=(dts, np.linspace(0, 100, 200)),axis_names=['drift_time', 'area_ratio'])
    median = mh_low.percentile(50, axis='area_ratio')
    mfilt = gaussian_filter1d(median, 4)
    gateidx = np.where(np.gradient(mfilt)==np.gradient(mfilt).min())[0][0]
    gatedt = dts[gateidx] #maximum slope
    s2shiftidx = np.where((mfilt[gateidx:]-mfilt[-100:].mean())<3*mfilt[-100:].std())[0][0]
    s2shift = dts[s2shiftidx+gateidx] # beginning of flat part
    vd = 1485/(cathodedt-gatedt)
    vd_err = vd*(10/cathodedt)
    if plot:
        plt.figure(figsize=(8,4.5))
        mh_low.plot(log_scale=False, cblabel='events')
        plt.xlabel("drift time ($\mu$s)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2/S1", ha='right', y=1,fontsize=12)
        plt.title(f'{run_id}',fontsize=14)
        median.plot(label='median')
        plt.plot(dtsc,mfilt,label='filtered median')
        plt.axvline(x=s2shift,linewidth=1,linestyle='--',color='violet',label=f'$S2~shift = {s2shift:.1f}~\mu$s')
        plt.axvline(x=gatedt,linewidth=1,linestyle='--',color='r',label=f'$gate = {gatedt:.1f}~\mu$s')
        plt.axhline(y=mfilt[50:].mean(),color='b',label='mean')
        plt.legend(fontsize=14)
        if name is not None: plt.savefig(name+'_gate_drift_time.png',dpi=600)
        print(f'Drift velocity = {vd:.3f} +/- {vd_err:.3f} mm/us')
    return vd, vd_err, cathodedt, gatedt, s2shift



def drift_velocity(events, run_id, low = 10, high = 3000, binning = 500, shaping = 4, catlim = 1000, plot = False):
    if 'area_ratio' in events: pass
    else: events.insert(1, 'area_ratio', np.divide(events['cs2'],events['cs1']))
    events = events[events['area_ratio']<1e3]
    
    # cathode drop-off
    dt = np.linspace(low, high, binning)
    hdtime = Hist1d(events['drift_time']/1e3, bins=dt)
    hfilt = gaussian_filter1d(hdtime,shaping)
    idx = np.where(dt>catlim)[0][0]
    hmin = np.argmin(np.gradient(hfilt[idx:]))
    #cathodedt = dt[np.where((np.gradient(hfilt)==np.gradient(hfilt).min)&(dt[1:]>2000))[0][0]]
    cathodedt = dt[hmin+idx]
    if plot:
        plt.figure(figsize=(12,6))
        hdtime.plot(color='b',label='data')
        plt.ylabel("events", ha='right', y=1)
        plt.xlabel("drift time ($\mu$s)", ha='right', x=1)
        plt.title(f'{run_id}',fontsize=14)
        plt.axvline(x=cathodedt,linewidth=1,linestyle='-', color='r',label=f'$cathode = {cathodedt:.1f}~\mu$s')
        plt.legend(fontsize=14)

    mh = Histdd(events['drift_time']/1e3, events['area_ratio'],
            bins=(np.linspace(low, high, binning), np.logspace(0, 5, 200)))
    if plot:
        plt.figure(figsize=(12,6))
        mh.plot(log_scale=True, cblabel='events')
        plt.xlabel("drift time ($\mu$s)", ha='right', x=1,fontsize=12)
        plt.ylabel("cS2/cS1", ha='right', y=1,fontsize=12)
        plt.title(f'{run_id}',fontsize=14)
        plt.yscale('log')
        plt.xlim(1500,high)
        plt.axvline(x=cathodedt,linewidth=1,linestyle='-', color='r',label=f'$cathode = {cathodedt:.1f}~\mu$s')
    
    # gate drift time
    dts = np.linspace(1, 15, 200)
    mh_low = Histdd(events['drift_time']/1e3, events['area_ratio'],
            bins=(dts, np.linspace(0, 200, 200)),axis_names=['drift_time', 'area_ratio'])
    median = mh_low.percentile(50, axis='area_ratio')
    mfilt = gaussian_filter1d(median, shaping)
    gatedt = dts[np.where(np.gradient(mfilt)==np.gradient(mfilt).min())[0][0]] #maximum slope
    s2shift = dts[np.where((mfilt[10:]-mfilt[50:].mean())<2)[0][0]] # beginning of flat part
    vd = 1485/(cathodedt-gatedt)
    vd_err = vd*(10/cathodedt)
    if plot:
        plt.figure(figsize=(12,6))
        mh_low.plot(log_scale=False, cblabel='events')
        plt.xlabel("drift time ($\mu$s)", ha='right', x=1,fontsize=12)
        plt.ylabel("cS2/cS1", ha='right', y=1,fontsize=12)
        plt.title(f'{run_id}',fontsize=14)
        median.plot(label='median')
        plt.plot(dts[1:],mfilt,label='filtered median')
        plt.axvline(x=s2shift,linewidth=1,linestyle='--',color='violet',label=f'$S2~shift = {s2shift:.1f}~\mu$s')
        plt.axvline(x=gatedt,linewidth=1,linestyle='--',color='r',label=f'$gate = {gatedt:.1f}~\mu$s')
        plt.legend(fontsize=14)
        print(f'Drift velocity = {vd:.3f}~mm/$\mu$s')
    return vd, vd_err, cathodedt, gatedt, s2shift


#def diffusion_model(t, D, vd, w0):
#    sigma_to_r50p = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)        
#    return np.sqrt(2 * sigma_to_r50p**2 * D * t / vd**2 + w0**2)

def diffusion_model(t, D, vd, tGate):
    sigma_to_r50p = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
    return sigma_to_r50p * np.sqrt(2 * D * (t-tGate) / vd**2)


def diffusion_constant_kr(events, run_id, fit_range, vd = 600, plot = False, name = None):
    # s2_width_50 vs drift_time
    t = np.linspace(0, 2400, 200)
    dt = events['drift_time']/1e3 # drift time in us
    ph = Histdd(dt, events['s2_a_range_50p_area'],bins=(t, np.linspace(100, 15e3, 200)))
    perc50 = np.array(ph.percentile(percentile=50, axis=1))
    D_guess = 45 * units.cm**2 / units.s * 1e3
    tGate_guess = 5 * units.us / 1e3
    vd = vd * units.mm / units.us
    guess = np.array([D_guess, vd, tGate_guess])
    ys_m = diffusion_model(t, *guess)
    ll = np.where(t>fit_range[0])[0][0]
    hh = np.where(t>fit_range[1])[0][0]
    diffusion = lambda x, D, tGate: diffusion_model(x, D, vd, tGate)
    
    popt, pcov = curve_fit(diffusion, t[ll:hh], perc50[ll:hh], p0=(D_guess, tGate_guess))
    perr = np.sqrt(np.diag(pcov))
    
    ys_u = diffusion(t, *popt) + 1000
    ys_m = diffusion(t, *popt)
    ys_d = diffusion(t, *popt) - 1000
    
    diff_const = popt[0]/(units.cm**2 / units.s)/1e3
    diff_const_err = perr[0]/(units.cm**2 / units.s)/1e3
    
    if plot:
        plt.figure(figsize=(8,4.5))
        ph.plot(log_scale=True, cblabel='events')
        plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{run_id}',fontsize=14)
        plt.plot(t[:len(perc50)], perc50, color='b',linestyle='--', label='50% percentile')
        plt.axvspan(*fit_range, alpha=0.1, color='blue', label='fit region')
        plt.plot(t, ys_m, label=f'$D = {diff_const:.2f} \pm {diff_const_err:.2f}$ cm$^2$/s',color='r')
        #plt.plot(t, diffusion_model(t, *guess), label='guess',color='m')
        plt.legend(fontsize=14)
        if name is not None: plt.savefig(name+'_diffusion_constant.png',dpi=600)
        print(f'Diffusion constant = {diff_const:.2f} +/- {diff_const_err:.2f} cm2/s, tGate = {popt[1]:.2f} us')
    return diff_const, diff_const_err, popt, perr


def diffusion_constant(events, run_id, fit_range, vd = 600, plot = False):
    # s2_width_50 vs drift_time
    t = np.linspace(0, 2400, 200)
    ph = Histdd(events['drift_time']/1e3, events['s2_range_50p_area'],
                bins=(t, np.linspace(100, 15e3, 200)))
    perc50 = np.array(ph.percentile(percentile=50, axis=1))
    
    D_guess = 45e3 * units.cm**2 / units.s
    w0_guess = 500 * units.ns
    vd = vd * units.mm / units.us
    guess = np.array([D_guess, vd, w0_guess])
    ys_m = diffusion_model(t, *guess)
    ll = np.where(t>fit_range[0])[0][0]
    hh = np.where(t>fit_range[1])[0][0]
    diffusion = lambda x, D, w0: diffusion_model(x, D, vd, w0)
    popt, pcov = curve_fit(diffusion, t[ll:hh], perc50[ll:hh], p0=(D_guess, w0_guess))
    perr = np.sqrt(np.diag(pcov))
    
    ys_u = diffusion(t, *popt) + 1000
    ys_m = diffusion(t, *popt)
    ys_d = diffusion(t, *popt) - 1000
    diff_const = popt[0]/1e3/(units.cm**2 / units.s)
    diff_const_err = perr[0]/1e3/(units.cm**2 / units.s)
    
    if plot:
        plt.figure(figsize=(12,6))
        ph.plot(log_scale=True, cblabel='events')
        plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{run_id}',fontsize=14)
        plt.plot(t[:len(perc50)], perc50, color='b',linestyle='--', label='50% percentile')
        plt.axvspan(*fit_range, alpha=0.1, color='blue', label='fit region')
        plt.plot(t, ys_m, label=f'$D = {popt[0]/1e3/(units.cm**2 / units.s):.2f}$ cm$^2$/s',color='r')
        plt.legend(fontsize=14)
        print(f'Diffusion constant = {diff_const:.2f} +/- {diff_const_err:.2f} cm$^2$/s ')
    return diff_const, diff_const_err, popt, perr


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
    plt.title(f'{run_id}')
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
    
    
def diffusion_analysis_kr_old(st,run_kr, area_cut=(5e3,1.1e4),width_cut=(200,1.5e4),aft_cut=(0.65,0.77),
                          radial_cut = None, fit_range=(1,1500), plot = False ):
    run = int(run_kr)
    events = st.get_df(run_kr,'event_info_double',progress_bar=False)
    if(plot): plot_area_width_aft_kr(events, run_kr)
    mask_singleS1 = mask_KrSingleS1(events)
    if(plot): plot_area_width_aft_kr(events[mask_singleS1], run_kr)
    e1 = events[mask_singleS1]
    mask_awt = mask_s2_area_width_aft_kr(e1,run_kr,area_cut=area_cut,width_cut=width_cut,aft_cut=aft_cut,plot=plot)
    e2 = events[mask_singleS1 & mask_awt]
    if radial_cut is not None: e2 = e2[e2['r']>radial_cut]
    vd, vd_err, cathodedt, gatedt, s2shift = drift_velocity_kr(e1, run_kr, plot=plot)
    d, d_err, par, par_err = diffusion_constant_kr(e2,run_kr,fit_range=fit_range,vd = vd,plot=plot)
    return run, vd, vd_err, d, d_err, cathodedt, gatedt, s2shift, par, par_err


def diffusion_analysis(st, run_id, area_cut=(1e4,5e6), fit_range=(1,1500), plot = False ):
    events = st.get_df(run_id,'event_info',progress_bar=False)
    if(plot): plot_area_width_aft(events, run_id)
    mask_awt = mask_s2_area_width_aft(events,run_id,area_cut,width_cut=(200,1.5e4),aft_cut=(0.65,0.77),high=7,plot=plot)
    e1 = events[mask_awt[0]]
    vd, vd_err, cathodedt, gatedt, s2shift = drift_velocity(e1, run_id, low=100,catlim=2000, plot=plot)
    d, d_err, par, par_err = diffusion_constant(e1,run_id,fit_range=(200,1500),vd = vd,plot=plot)
    return int(run_id), vd, vd_err, d, d_err, par, par_err

def merge_runs(st,runs):
    ev0 = st.get_df(runs[0],['event_info'],progress_bar=False)
    print('Reading runs from',runs[-1],'to',runs[0])
    start = time.time()
    for i, run_id in enumerate(runs[1:]):
        if ((i+1)%5) == 0: print(f'n. {i} run {run_id} elapsed time: {time.time()-start:.2f} s')
        ev_temp = st.get_df(run_id,['event_info'],progress_bar=False)
        frames = [ev0,ev_temp]
        ev0 = pd.concat(frames)
    return ev0

def merge_runs_double(st,runs):
    ev0 = st.get_df(runs[0],['event_info_double'],progress_bar=False)
    print('Reading runs from',runs[-1],'to',runs[0])
    start = time.time()
    for i, run_id in enumerate(runs[1:]):
        if ((i+1)%5) == 0: print(f'n. {i} run {run_id} elapsed time: {time.time()-start:.2f} s')
        ev_temp = st.get_df(run_id,['event_info_double'],progress_bar=False)
        frames = [ev0,ev_temp]
        ev0 = pd.concat(frames)
    return ev0

def merge_runs_kr(st,runs):
    ev0 = st.get_df(runs[0],['event_info_double',
                             'cut_s1_max_pmt',
                             'cut_s1_area_fraction_top',
                             'cut_s2_single_scatter',
                             'cut_fiducial_volume',
                             'cut_daq_veto',
                             'cut_Kr_SingleS1S2',
                             'cut_Kr_DoubleS1_SingleS2'],progress_bar=False)
    print('Reading runs from',runs[-1],'to',runs[0])
    start = time.time()
    for i, run_id in enumerate(runs[1:]):
        if ((i+1)%5) == 0: print(f'n. {i} run {run_id} elapsed time: {time.time()-start:.2f} s')
        ev_temp = st.get_df(run_id,['event_info_double',
                             'cut_s1_max_pmt',
                             'cut_s1_area_fraction_top',
                             'cut_s2_single_scatter',
                             'cut_fiducial_volume',
                             'cut_daq_veto',
                             'cut_Kr_SingleS1S2',
                             'cut_Kr_DoubleS1_SingleS2'],progress_bar=False)
        frames = [ev0,ev_temp]
        ev0 = pd.concat(frames)
    return ev0

def plot_area_width_aft_kr(st,run_id, low = 0, high = 7, low2 = 0, high2 = 1, binning = 500,plot=False):
    events0 = st.get_df(run_id,'event_info_double',progress_bar=False)
    mask = mask_KrSingleS1(events0)
    livetime = (events0['time'][len(events0)-1]-events0['time'][0])/1e9
    events = events0[mask]
    ph_s1 = Histdd(events['s1_a_area'], events['s1_a_range_50p_area'],
                    bins=(np.logspace(low, high, binning), np.logspace(1, 6, binning)))
    ph_s2 = Histdd(events['s2_a_area'], events['s2_a_range_50p_area'],
                    bins=(np.logspace(low, high, binning), np.logspace(1, 6, binning)))
    phcs1 = Histdd(events['s1_a_area'], events['s1_a_area_fraction_top'],
                    bins=(np.logspace(low, high, binning), np.linspace(0, 1, binning)))
    phcs2 = Histdd(events['s2_a_area'], events['s2_a_area_fraction_top'],
                    bins=(np.logspace(low, high, binning), np.linspace(0, 1, binning)))
    kr_rate = len(events)/livetime
    all_rate = len(events0)/livetime
    kr_rates, all_rates, slot_times = [], [], []
    for i in range(10):
        idx1, idx2 = int(i*(len(events0)-1)/10), int((i+1)*(len(events0)-1)/10)
        slot_time = (events0['time'][idx2]-events0['time'][idx1])/1e9
        trange = (events0['time']>events0['time'][idx1]) & (events0['time']<events0['time'][idx2])
        kr_rates = np.append(kr_rates,len(events0[mask & trange])/slot_time)
        all_rates = np.append(all_rates,len(events0[trange])/slot_time)
        timestamp = events0['time'][idx1]/1e9
        slot_times = np.append(slot_times,datetime.datetime.fromtimestamp(timestamp))
    if plot:
        plt.figure(figsize=(12,6))
        phcs1.plot(log_scale=True, cblabel='S1 events',cmap='plasma')
        phcs2.plot(log_scale=True, cblabel='S2 events')
        plt.xlabel("peak area (PE)", ha='right', x=1)
        plt.ylabel("area fraction top", ha='right', y=1)
        plt.title(f'{run_id}')
        plt.xscale('log')
        plt.title(f'{run_id} - Kr-rate: {kr_rate:.2f} Hz')
        plt.savefig(f'area_vs_aft_run{run_id}.png')
        
        plt.figure(figsize=(12,6))
        ph_s1.plot(log_scale=True, cblabel='S1 events',cmap='plasma')
        ph_s2.plot(log_scale=True, cblabel='S2 events')
        plt.xlabel("peak area (PE)", ha='right', x=1)
        plt.ylabel("peak width 50% (ns)", ha='right', y=1)
        plt.title(f'{run_id} - Kr-rate: {kr_rate:.2f} Hz')
        plt.xscale('log')
        plt.yscale('log')
               
        plt.figure(figsize=(12,6))
        ph_s1s2 = Histdd(events['s1_area'], events['s2_area'],
                        bins=(np.logspace(0, 5, binning), np.logspace(1, 7, binning)))
        ph_s1s2.plot(log_scale=True, cblabel='S2 events')
        plt.xlabel("S1 peak area (PE)", ha='right', x=1)
        plt.ylabel("S2 peak area (PE)", ha='right', y=1)
        plt.title(f'{run_id} - Kr-rate: {kr_rate:.2f} Hz')
        plt.xscale('log')
        plt.yscale('log')
    print(f'{run_id}, livetime {livetime:.2f} s, tot. events: {len(events0)}, Kr events: {len(events)}, Kr-rate: {kr_rate:.2f} Hz')
    return kr_rate, all_rate, kr_rates, all_rates, slot_times

def plot_kr_rate_vs_time(st,runs, gif=False):
    kr_rates, all_rates, slot_times, tot_kr_rates, tot_all_rates = [], [], [], [], []
    for i, run in enumerate(runs):
        kr_rate, all_rate, krs, alls, slots = plot_area_width_aft_kr(st,run,plot=0)
        slot_times = np.append(slot_times,slots)
        kr_rates = np.append(kr_rates,krs)
        all_rates = np.append(all_rates,alls)
        tot_kr_rates = np.append(tot_kr_rates,kr_rate)
        tot_all_rates = np.append(tot_all_rates,all_rate)
    if gif:
        fp_in = f'area_vs_aft_run*.png'
        fp_out = f'area_vs_aft_run{runs[0]}-{runs[-1]}.gif'
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=500, loop=0)
        for i, run in enumerate(runs):
            os.remove(f'area_vs_aft_run{run}.png')
    
    plt.figure(figsize=(12,6))
    dates = matplotlib.dates.date2num(slot_times + datetime.timedelta(hours=7))
    plt.plot_date(dates, kr_rates,'o',label='Kr-83m rate')
    plt.plot_date(dates, all_rates,'o',label='total rate')
    #plt.axvline(x = dates[22],color='black', linestyle = '--',label=f'valve at 0.2 turns')
    #plt.axvline(x = dates[52],color='m', linestyle = '--',label=f'valve at 0.4 turns')
    #plt.axvline(x = dates[83],color='b', linestyle = '--',label=f'valve at 0.8 turns')
    #plt.axvline(x = dates[130],color='r', linestyle = '--',label=f'valve at 3.2 turns')
    myFmt = matplotlib.dates.DateFormatter('%d%bH%H')
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.title(f'runs {runs[-1]} - {runs[0]}')
    plt.xlabel("LNGS time", ha='right', x=1,fontsize=12)
    plt.ylabel("Kr rate (Hz)", ha='right', y=1,fontsize=12)
    plt.legend(fontsize=14)
    
    
def drift_velocity_temp(events, run_id, low = 10, high = 3000, binning = 500, w=8, plot=False):
    if 'area_ratio' in events: pass
    #else: events.insert(1, 'area_ratio', np.divide(events['cs2_a'],events['cs1_a']))
    else: events.insert(1, 'area_ratio', np.divide(events['s2_area'],events['s1_area']))
    events = events[events['area_ratio']<1e3]
    
    # cathode drop-off
    dt = np.linspace(low, high, binning)
    dtc = (dt[:-1] + dt[1:])/2.
    hdtime = Hist1d(events['drift_time']/1e3, bins=dt)
    hfilt = gaussian_filter1d(hdtime,w)
    catdt = dtc[np.where(np.gradient(hfilt)==np.gradient(hfilt).min())[0]]
    cathodedt = np.mean(catdt)
    
    if plot:
        plt.figure(figsize=(12,6))
        hdtime.plot(color='b',label='data')
        plt.ylabel("events", ha='right', y=1)
        plt.xlabel("drift time ($\mu$s)", ha='right', x=1)
        plt.title(f'run {run_id}',fontsize=14)
        plt.axvline(x=cathodedt,linewidth=1,linestyle='-', color='r',label=f'$cathode = {cathodedt:.1f}~\mu$s')
        plt.plot(dtc,hfilt,label='filtered')
        plt.xlim(2100,2300)
        plt.legend(fontsize=14)
        
        #plt.figure(figsize=(12,6))
        #plt.plot(dtc,np.gradient(hfilt),label='diff. filtered')
        #plt.axvline(x=cathodedt,linewidth=1,linestyle='-', color='r',label=f'$cathode = {cathodedt:.1f}~\mu$s')
        #plt.legend(fontsize=14)
        #plt.xlim(2100,2300)

    mh = Histdd(events['drift_time']/1e3, events['area_ratio'],
            bins=(np.linspace(low, high, binning), np.logspace(0, 3, 200)))
    
    if plot:
        plt.figure(figsize=(12,6))
        mh.plot(log_scale=True, cblabel='events')
        plt.xlabel("drift time ($\mu$s)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2/S1", ha='right', y=1,fontsize=12)
        plt.title(f'run {run_id}',fontsize=14)
        plt.yscale('log')
        plt.xlim(1500,2500)
        plt.axvline(x=cathodedt,linewidth=1,linestyle='-', color='r',label=f'$cathode = {cathodedt:.1f}~\mu$s')
        #plt.legend(fontsize=14)
    
    # gate drift time
    dts = np.linspace(0, 20, 200)
    dtsc = (dts[:-1] + dts[1:])/2.
    mh_low = Histdd(events['drift_time']/1e3, events['area_ratio'],
            bins=(dts, np.linspace(0, 100, 200)),axis_names=['drift_time', 'area_ratio'])
    median = mh_low.percentile(50, axis='area_ratio')
    mfilt = gaussian_filter1d(median, 4)
    gateidx = np.where(np.gradient(mfilt)==np.gradient(mfilt).min())[0][0]
    gatedt = dts[gateidx] #maximum slope
    s2shiftidx = np.where((mfilt[gateidx:]-mfilt[-100:].mean())<3*mfilt[-100:].std())[0][0]
    s2shift = dts[s2shiftidx+gateidx] # beginning of flat part
    vd = 1485/(cathodedt-gatedt)
    vd_err = vd*(10/cathodedt)
    if plot:
        plt.figure(figsize=(12,6))
        mh_low.plot(log_scale=False, cblabel='events')
        plt.xlabel("drift time ($\mu$s)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2/S1", ha='right', y=1,fontsize=12)
        plt.title(f'run {run_id}',fontsize=14)
        median.plot(label='median')
        plt.plot(dtsc,mfilt,label='filtered median')
        plt.axvline(x=s2shift,linewidth=1,linestyle='--',color='violet',label=f'$S2~shift = {s2shift:.1f}~\mu$s')
        plt.axvline(x=gatedt,linewidth=1,linestyle='--',color='r',label=f'$gate = {gatedt:.1f}~\mu$s')
        plt.axhline(y=mfilt[50:].mean(),color='b',label='mean')
        plt.legend(fontsize=14)
        print(f'Drift velocity = {vd:.3f} +/- {vd_err:.3f} mm/us')
    return vd, vd_err, cathodedt, gatedt, s2shift

def diffusion_constant_temp(events, run_id, fit_range, vd = 600, plot = False):
    # s2_width_50 vs drift_time
    t = np.linspace(0, 2400, 200)
    dt = events['drift_time']/1e3 # drift time in us
    ph = Histdd(dt, events['s2_range_50p_area'],bins=(t, np.linspace(100, 15e3, 200)))
    perc50 = np.array(ph.percentile(percentile=50, axis=1))
    D_guess = 45 * units.cm**2 / units.s * 1e3
    tGate_guess = 5 * units.us / 1e3
    vd = vd * units.mm / units.us
    guess = np.array([D_guess, vd, tGate_guess])
    
    ys_m = wcu.s2_width_model(t, *guess)
    ll = np.where(t>fit_range[0])[0][0]
    hh = np.where(t>fit_range[1])[0][0]
    diffusion = lambda x, D, tGate: wcu.s2_width_model(x, D, vd, tGate)
    
    popt, pcov = curve_fit(diffusion, t[ll:hh], perc50[ll:hh], p0=(D_guess, tGate_guess))
    perr = np.sqrt(np.diag(pcov))
    
    ys_u = diffusion(t, *popt) + 1000
    ys_m = diffusion(t, *popt)
    ys_d = diffusion(t, *popt) - 1000
    
    diff_const = popt[0]/(units.cm**2 / units.s)/1e3
    diff_const_err = perr[0]/(units.cm**2 / units.s)/1e3
    
    if plot:
        plt.figure(figsize=(12,6))
        ph.plot(log_scale=True, cblabel='events')
        plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'run {run_id}',fontsize=14)
        plt.plot(t[:len(perc50)], perc50, color='b',linestyle='--', label='50% percentile')
        plt.axvspan(*fit_range, alpha=0.1, color='blue', label='fit region')
        plt.plot(t, ys_m, label=f'$D = {diff_const:.2f} \pm {diff_const_err:.2f}$ cm$^2$/s',color='r')
        #plt.plot(t, diffusion_model(t, *guess), label='guess',color='m')
        plt.legend(fontsize=14)
        print(f'Diffusion constant = {diff_const:.2f} +/- {diff_const_err:.2f} cm2/s, tGate = {popt[1]:.2f} us')
    return diff_const, diff_const_err, popt, perr

def plot_r2_z(ev, title, xylim = (-75,75), r2lim = (0,6300),zlim=(-150,1), bins = 300):
    r2 = ev['r']*ev['r']
    ph_r2z = Histdd(r2, ev['z'],bins=(np.linspace(r2lim[0],r2lim[1],bins), np.linspace(zlim[0],zlim[1],bins)))
    ph_xy = Histdd(ev['x'], ev['y'],bins=(np.linspace(xylim[0],xylim[1],bins), np.linspace(xylim[0],xylim[1],bins)))
    plt.figure(figsize=(12,6))
    ph_r2z.plot(log_scale=True,cblabel='events')
    plt.xlabel(r"r^2 (cm^2)", ha='right', x=1)
    plt.ylabel("z (cm)", ha='right', y=1)
    plt.title(f'{title}',fontsize=14)
    plt.figure(figsize=(10,8))
    ph_xy.plot(log_scale=True,cblabel='events')
    plt.xlabel("x (cm)", ha='right', x=1)
    plt.ylabel("y (cm)", ha='right', y=1)
    plt.title(f'{title}',fontsize=14)
    

def plot_s2width_dt_kr(events, title, width='50p',tlim = (0,2400), wlim = (100,15000),bins = 200,name=None):
    # s2_width_50 vs drift_time
    tspace = np.linspace(tlim[0], tlim[1], bins)
    wspace = np.linspace(wlim[0], wlim[1], bins)
    dt = events['drift_time']/1e3 # drift time in us
    s2width = events['s2_a_range_'+width+'_area']
    ph = Histdd(dt, s2width, bins=(tspace, wspace))
    
    plt.figure(figsize=(8,4.5))
    ph.plot(log_scale=True, cblabel='events')
    plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
    plt.ylabel('S2 width '+width+' (ns)', ha='right', y=1,fontsize=12)
    plt.title(f'{title}',fontsize=14)
    if name is not None:  plt.savefig(name+'_area_width_'+width+'.png',dpi=600)
    #plt.legend(fontsize=14)
