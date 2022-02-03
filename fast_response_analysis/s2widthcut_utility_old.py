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
import time
import strax
import straxen
from straxen import units
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import peaks_utility as psu
from datetime import datetime, timedelta
from matplotlib.colors import LogNorm
from scipy.stats import beta, chi2
from scipy.interpolate import interp1d
import math
import bokeh.plotting as bklt
from IPython.core.display import display, HTML

import drift_diffusion_utility as ddu

def plot_rate_vs_time(runs,mask_rates,all_rates,times):
    plt.figure(figsize=(12,6))
    dates = matplotlib.dates.date2num(times + timedelta(hours=7))
    plt.plot_date(dates, mask_rates,'o',label='selection rate')
    plt.plot_date(dates, all_rates,'o',label='total rate')
    myFmt = matplotlib.dates.DateFormatter('%d%bH%H')
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.title(f'runs {runs[-1]} - {runs[0]}')
    plt.xlabel("LNGS time", ha='right', x=1,fontsize=12)
    plt.ylabel("rate (Hz)", ha='right', y=1,fontsize=12)
    plt.legend(fontsize=14)
    
    
def merge_runs(st,runs):
    ev0 = st.get_df(runs[0],'event_info',progress_bar=False)
    print('Reading runs from',runs[-1],'to',runs[0])
    start = time.time()
    for i, run_id in enumerate(runs[1:]):
        if ((i+1)%5) == 0: print(f'n. {i} run {run_id} elapsed time: {time.time()-start:.2f} s')
        ev_temp = st.get_df(run_id,'event_info',progress_bar=False)
        frames = [ev0,ev_temp]
        ev0 = pd.concat(frames)
    return ev0

def merge_runs_cutax(st,runs):
    ev0 = st.get_df(runs[0],['event_info',
                             'cut_s1_max_pmt',
                             'cut_s1_area_fraction_top',
                             'cut_s2_single_scatter',
                             'cut_s2_width_naive',
                             'cut_fiducial_volume',
                             'cut_daq_veto'],progress_bar=False)
    print('Reading runs from',runs[-1],'to',runs[0])
    start = time.time()
    for i, run_id in enumerate(runs[1:]):
        if ((i+1)%5) == 0: print(f'n. {i} run {run_id} elapsed time: {time.time()-start:.2f} s')
        ev_temp = st.get_df(run_id,['event_info',
                             'cut_s1_max_pmt',
                             'cut_s1_area_fraction_top',
                             'cut_s2_single_scatter',
                             'cut_s2_width_naive',
                             'cut_fiducial_volume',
                             'cut_daq_veto'],progress_bar=False)
        frames = [ev0,ev_temp]
        ev0 = pd.concat(frames)
    return ev0

def basic_cuts(data, aft_cut = False):
    cut = np.ones(len(data), dtype=bool)
    #cut &= data['drift_time'] > 2.0e3
    cut &= data['s2_area'] > 200
    cut &= data['s1_n_channels'] >= 3
    cut &= data['s2_area_fraction_top'] > 0.7
    cut &= data['s2_area_fraction_top'] < 0.77
    return cut

def basic_cuts_old(data, aft_cut = False):
    cut = np.ones(len(data), dtype=bool)
    cut &= data['drift_time'] > 2.0e3
    cut &= data['s2_area'] > 200
    cut &= data['s1_n_channels'] >= 3
    cut &= data['s2_area_fraction_top'] > 0.6
    cut &= data['s2_area_fraction_top'] < 0.9
    #cut &= data['z'] > -144
    #cut &= data['r'] < 70
    #cut_basics = np.ones(len(data), dtype=bool)
    #cut_basics[:] = cut.copy()
    #cut &= data['z'] < -2
    #cut &= data['s1_area_fraction_top'] < 0.6 # conservative
    #cut &= data['alt_s2_area'] < np.clip(data['s2_area']*0.005, 80, np.inf)
    # cut on S2 AFT
    mu = 0.725
    a = np.logspace(2, 6.5, 201)
    _a = np.clip(a, 0, 1e3)
    b = beta.isf(0.0001, ((_a * mu) / 1.2).astype(int),((_a - _a * mu) / 1.2).astype(int))
    c = beta.isf(0.9999, ((_a * mu) / 1.2).astype(int),((_a - _a * mu) / 1.2).astype(int))
    aft_ul = interp1d(a, b, bounds_error=False, fill_value='extrapolate')
    aft_ll = interp1d(a, c, bounds_error=False, fill_value='extrapolate')
    if aft_cut:
        cut &= data['s2_area_fraction_top'] < aft_ul(data['s2_area'])
        cut &= data['s2_area_fraction_top'] > aft_ll(data['s2_area'])
    # cut on S2 width
    b = chi2.isf(0.1, _a / 40) / (_a / 40)
    c = chi2.isf(0.9, _a / 40) / (_a / 40)
    s2w_ul = interp1d(a, b, bounds_error=False, fill_value='extrapolate')
    s2w_ll = interp1d(a, c, bounds_error=False, fill_value='extrapolate')
    #if width_cut:
    #    cut &= data['s2_range_50p_area'] < s2w_ul(data['s2_area'])
    #    cut &= data['s2_range_50p_area'] > s2w_ll(data['s2_area'])
    return cut

def FV_volume_cut(data, z_lim = (-144,-2), r2_lim = 5000):
    cut = np.ones(len(data), dtype=bool)
    cut &= data['z'] > z_lim[0]
    cut &= data['z'] < z_lim[1]
    cut &= (data['r']*data['r']) < r2_lim
    return cut

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

    
def plot_rate_vs_time(title,mask_rates,all_rates,times):
    plt.figure(figsize=(12,6))
    dates = matplotlib.dates.date2num(times + timedelta(hours=7))
    plt.plot_date(dates, mask_rates,'o',label='selection rate')
    plt.plot_date(dates, all_rates,'o',label='total rate')
    myFmt = matplotlib.dates.DateFormatter('%d%bH%H')
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.title(f'{title}',fontsize=14)
    plt.xlabel("LNGS time", ha='right', x=1,fontsize=12)
    plt.ylabel("rate (Hz)", ha='right', y=1,fontsize=12)
    plt.legend(fontsize=14)
    

def S2WidthCut_basic(ev0, title, mod_par = (45, 0.675, 350), wrange = (100,30000), fit_range = (200,1500),
                     perc = (1,99), FVcut = (-144,-2, 5000), S1AFTcut = None, plot = False ):
    
    events = ev0[basic_cuts(ev0)]
    if FVcut is not None: events = events[FV_volume_cut(events,(FVcut[0],FVcut[1]),FVcut[2])]
    if S1AFTcut is not None: events = events[S1AFT_cut(events,prc=S1AFTcut)]
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']/1e3
    s2area = events['s2_area']
    
    print('total events',len(ev0),'selected events',len(events))
    
    t, t0 = np.linspace(0, 2200, 200),np.linspace(0, 2200, 20)
    wspace = np.linspace(wrange[0], wrange[1], 200)
    ph, ph0 = Histdd(drift, s2width, bins=(t, wspace)), Histdd(drift, s2width, bins=(t0, wspace))
    
    perc50 = np.array(ph.percentile(percentile=50, axis=1))
    D_mod = mod_par[0]*1e3 * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    w0_mod = mod_par[2] * units.ns
    guess = np.array([D_mod, vd_mod, w0_mod])
    ll = np.where(t>fit_range[0])[0][0]
    hh = np.where(t>fit_range[1])[0][0]
    diffusion = lambda x, D, w0: ddu.diffusion_model(x, D, vd_mod, w0)
    par, pcov = curve_fit(diffusion, t[ll:hh], perc50[ll:hh], p0=(D_mod, w0_mod))
    perr = np.sqrt(np.diag(pcov))
    ys = diffusion(t, *par)
    ys_mod = ddu.diffusion_model(t, D_mod, vd_mod, w0_mod)
    #cut definition
    perc_l = np.array(ph0.percentile(percentile=perc[0], axis=1))
    perc_h = np.array(ph0.percentile(percentile=perc[1], axis=1))
    tc = (t0[1:]+t0[:-1])/2
    lcut = interp1d(tc, perc_l, bounds_error=False, fill_value='extrapolate',kind='cubic')
    hcut = interp1d(tc, perc_h, bounds_error=False, fill_value='extrapolate',kind='cubic')
    cut = np.ones(len(ev0), dtype=bool)
    cut &= ev0['s2_range_50p_area'] < hcut(ev0['drift_time']/1e3)
    cut &= ev0['s2_range_50p_area'] > lcut(ev0['drift_time']/1e3)
    if plot:
        plt.figure(figsize=(12,6))
        ph.plot(log_scale=True, cblabel='events')
        #plt.axvspan(*fit_range, alpha=0.1, color='blue', label='fit region')
        #plt.plot(tc, lcut(tc), color='b',linestyle='--', label=f'{perc[0]}% percentile')
        #plt.plot(tc, hcut(tc), color='b',linestyle='--', label=f'{perc[1]}% percentile')
        #plt.plot(t, ys, 'r--',label=f'fit: $D = {par[0]/1e3/(units.cm**2 / units.s):.2f}$ cm$^2$/s')
        plt.plot(t, ys_mod, 'black',label=f'model from Kr-83m: $D = {mod_par[0]:.2f}$ cm$^2$/s')
        plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        plt.legend(fontsize=14)
    return cut


def S2WidthCut_percentile(ev0, title, mod_par = (45, 0.675, 350), wrange = (500,15000), fit_range = (200,1500),
                          perc = (1,99), FVcut = (-144,-2, 5000), S1AFTcut = None, delta = 0.25, dlarge = 1.5, plot = False ):
    
    events = ev0[basic_cuts(ev0)]
    if FVcut is not None: events = events[FV_volume_cut(events,(FVcut[0],FVcut[1]),FVcut[2])]
    if S1AFTcut is not None: events = events[S1AFT_cut(events,prc=S1AFTcut)]
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']/1e3
    s2area = events['s2_area']
    
    print('total events',len(ev0),'selected events',len(events))
    
    t = np.linspace(0, 2200, 200)
    wspace = np.linspace(wrange[0], wrange[1], 200)
    ph = Histdd(drift, s2width, bins=(t, wspace))
    
    # fit 50 percentile
    perc50 = np.array(ph.percentile(percentile=50, axis=1))
    D_mod = mod_par[0]*1e3 * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    w0_mod = mod_par[2] * units.ns
    guess = np.array([D_mod, vd_mod, w0_mod])
    ll = np.where(t>fit_range[0])[0][0]
    hh = np.where(t>fit_range[1])[0][0]
    diffusion = lambda x, D, w0: ddu.diffusion_model(x, D, vd_mod, w0)
    par, pcov = curve_fit(diffusion, t[ll:hh], perc50[ll:hh], p0=(D_mod, w0_mod))
    perr = np.sqrt(np.diag(pcov))
    ys = diffusion(t, *par)
    
    ys_mod = ddu.diffusion_model(t, D_mod, vd_mod, w0_mod)
    if plot:
        plt.figure(figsize=(12,6))
        ph.plot(log_scale=True, cblabel='events')
        #plt.axvspan(*fit_range, alpha=0.1, color='blue', label='fit region')
        #plt.plot((t[1:]+t[:-1])/2, perc50, color='b',linestyle='--', label='50% percentile')
        plt.plot(t, ys, 'r--',label=f'fit: $D = {par[0]/1e3/(units.cm**2 / units.s):.2f}$ cm$^2$/s')
        plt.plot(t, ys_mod, 'black',label=f'model: $D = {mod_par[0]:.2f}$ cm$^2$/s')
        plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        plt.legend(fontsize=14)
    
    #normalized width using D from Kr-83m
    #norm50b = events['s2_range_50p_area']/diffusion(events['drift_time']/1e3, *par) # from fit
    #norm50a = events['s2_range_50p_area']/ddu.diffusion_model(events['drift_time']/1e3, D_mod, vd_mod, w0_mod)
    scw = 258.41  # s2_secondary_sc_width median
    SigmaToR50 = 1.349
    normWidth = (np.square(events['s2_range_50p_area'] / SigmaToR50) -
              np.square(scw)) / np.square(s2_width_model(events['drift_time']/1e3, D_mod, vd_mod, w0_mod) )
    aspace, aspace0 = np.logspace(2, 6.5, 300), np.logspace(2, 6.5, 30)
    nspace = np.linspace(0, 10, 300)
    pha = Histdd(events['s2_area'], normWidth, bins=(aspace, nspace))
    pha0 = Histdd(events['s2_area'], normWidth, bins=(aspace0, nspace))
    perc90a = np.array(pha0.percentile(percentile=perc[1], axis=1))
    perc10a = np.array(pha0.percentile(percentile=perc[0], axis=1))
    # lower limit
    aspace1, aspace2 = np.logspace(2, 3.5, 100), np.logspace(3.5,6.5, 200)
    #aspace3 = np.logspace(3.5, 4.5, 50)
    #aspace4, aspace5 = np.logspace(4.5, 5, 50), np.logspace(5, 6.5, 100) # np.logspace(6, 6.5, 50)
    low_cut = np.concatenate((line(np.log10(aspace1),(1-delta)/1.5,(2*delta-2)/1.5),1-delta + 0*np.log10(aspace2)))
    # upper limit
    high_cut = np.concatenate((line(np.log10(aspace1),(delta-1)/1.5,(5-2*delta)/1.5),
                               1+delta + 0*np.log10(aspace2)))
                               #line(np.log10(aspace4),2*(dlarge-delta),1-9*dlarge+10*delta),
                               #1+dlarge +0*np.log10(aspace5)))#,
                               ##line(np.log10(aspace6),-2*(dlarge-2*delta),1+13*dlarge-24*delta)))
    s2width_lcut = interp1d(aspace, low_cut, bounds_error=False, fill_value='extrapolate',kind='cubic')
    s2width_hcut = interp1d(aspace, high_cut, bounds_error=False, fill_value='extrapolate',kind='cubic')
    if plot:
        plt.figure(figsize=(12,6))
        pha.plot(log_scale=True, cblabel='events')
        #plt.plot((aspace0[1:]+aspace0[:-1])/2, perc90a, 'b--', label=f'{perc[1]}% percentile')
        #plt.plot((aspace0[1:]+aspace0[:-1])/2, perc10a, 'b--', label=f'{perc[0]}% percentile')
        plt.axhline(y=1,c='black',ls='-',label='model from Kr-83m')
        plt.xlabel("S2 area (PE)", ha='right', x=1,fontsize=12)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        plt.xscale('log')
        plt.plot(aspace, s2width_lcut(aspace),'r--',label='cut limits')
        plt.plot(aspace, s2width_hcut(aspace),'r--')
        plt.legend(fontsize=14)
    cut = np.ones(len(ev0), dtype=bool)
    norm50 = ev0['s2_range_50p_area']/ddu.diffusion_model(ev0['drift_time']/1e3, D_mod, vd_mod, w0_mod)
    cut &= norm50 < s2width_hcut(ev0['s2_area'])
    cut &= norm50 > s2width_lcut(ev0['s2_area'])
    return cut
    
def line(x,a,b):
    return a * x + b
    
def S1AFT_cut(data, coeff=(-2.2e-4, 0.54), low = 0, high = 0.6, binning = 500, delta = 0.06, prc = None,
              basic_cut = False, plot = False):
    if basic_cut: data = data[basic_cuts(data)]
    
    t = np.linspace(0, 2400, 200)
    t_c = (t[:-1] + t[1:])/2
    dt = data['drift_time']/1e3
    ph = Histdd(dt, data['s1_area_fraction_top'],bins=(t, np.linspace(low, high, binning)))
    
    cut = np.ones(len(data), dtype=bool)
    if prc is not None:
        b = np.array(ph.percentile(percentile=100-prc, axis=1))
        c = np.array(ph.percentile(percentile=prc, axis=1))
        aft_ul = interp1d(t_c, b, bounds_error=False, fill_value='extrapolate')
        aft_ll = interp1d(t_c, c, bounds_error=False, fill_value='extrapolate')
        cut &= data['s1_area_fraction_top'] < aft_ul(dt)
        cut &= data['s1_area_fraction_top'] > aft_ll(dt)
    else:
        cut &= data['s1_area_fraction_top'] < line(dt,*coeff) + delta
        cut &= data['s1_area_fraction_top'] > line(dt,*coeff) - delta
    if plot:
        plt.figure(figsize=(12,6))
        ph.plot(log_scale=True, cblabel='events')
        plt.plot(t, line(t,*coeff)+delta, color='b',linestyle='--',label='basic limits')
        plt.plot(t, line(t,*coeff)-delta, color='b',linestyle='--')
        if prc is not None: plt.plot(t, aft_ul(t), color='r',linestyle='--',label='percentile limits')
        if prc is not None: plt.plot(t, aft_ll(t), color='r',linestyle='--')
        plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
        plt.ylabel("S1 area fraction top", ha='right', y=1,fontsize=12)
        plt.legend(fontsize=14)
    return cut


def display_events(st,run_id, nev = 1,area_cut=(6e4,5e6),width_cut=(1e4,1e6)):
    #display(HTML("<style>.container { width:70% !important; }</style>"))
    st.is_stored(run_id, ('events','event_basics'))
    ev0 = st.get_array(run_id, 'event_info')
    ev1 = ev0[basic_cuts(ev0)]
    mask_awt = ddu.mask_s2_area_width_aft(ev1,run_id,area_cut,width_cut,aft_cut=(0.65,0.77))
    events = ev1[mask_awt[0]]
    print('events before selection',len(ev0),'events after selection',len(events))
    for i in range(nev):
        idx = int(np.random.rand()*len(events))
        fig = st.event_display_interactive(run_id,
                                           time_range=(events[idx]['time'],
                                                       events[idx]['endtime']),
                                           bottom_pmt_array=False)
        bklt.show(fig)

def display_events_S2WidthCut(st,run_id, nev = 1, s2_width_cut = True,area_cut=(6e4,5e6),width_cut=(1e4,1e6)):
    st.is_stored(run_id, ('events','event_basics'))
    ev0 = st.get_array(run_id, 'event_info',progress_bar=False)
    mask_awt = ddu.mask_s2_area_width_aft(ev0,run_id,area_cut=area_cut,width_cut=width_cut,aft_cut=(0.65,0.77))
    width_cut = S2WidthCut_percentile(ev0,run_id,mod_par=(44.47,0.676,375), perc=(0.3,99.7),
                                      FVcut=(-144,-2,5000), S1AFTcut = 2, delta=0.2, dlarge=1.5)
    no_width = ~width_cut
    ev1 = ev0[basic_cuts(ev0) & mask_awt[0]]
    ev_cut = ev0[basic_cuts(ev0) & mask_awt[0] & width_cut]
    ev_noc = ev0[basic_cuts(ev0) & no_width & mask_awt[0]]
    print('events before width cut',len(ev1),'events after selection',len(ev_cut),'no width cut',len(ev_noc))
    if s2_width_cut: events = ev_cut
    else: events = ev_noc
    for i in range(nev):
        idx = int(np.random.rand()*len(events))
        fig = st.event_display_interactive(run_id,
                                           time_range=(events[idx]['time'],
                                                       events[idx]['endtime']),
                                           bottom_pmt_array=False)
        bklt.show(fig)
        


##### new code Sep 2021 #####

def get_events_near_wire(df):
    per_pos1=[-44, -41.5]
    per_pos2=[13.5, 58.2]
    per_pos3=[-20,-57]
    per_pos4=[39.5, 45.8]
    x_pos=np.linspace(-70,70,140)
    y_per1 = ((per_pos2[1]-per_pos1[1])/(per_pos2[0]-per_pos1[0]))*(x_pos-per_pos1[0]) + per_pos1[1]+4
    y_per2 = y_per1-21
    y_per3 = ((per_pos4[1]-per_pos3[1])/(per_pos4[0]-per_pos3[0]))*(x_pos-per_pos3[0]) + per_pos3[1]+4
    y_per4 = y_per3-21
    df1 = df[df['y'] < ((per_pos2[1]-per_pos1[1])/(per_pos2[0]-per_pos1[0]))*(df['x']-per_pos1[0]) + per_pos1[1] + 4 ]
    df1 = df1[df1['y'] > ((per_pos2[1]-per_pos1[1])/(per_pos2[0]-per_pos1[0]))*(df1['x']-per_pos1[0]) + per_pos1[1] - 17 ]
    df2 = df[df['y'] < ((per_pos4[1]-per_pos3[1])/(per_pos4[0]-per_pos3[0]))*(df['x']-per_pos3[0]) + per_pos3[1] + 4 ]
    df2 = df2[df2['y'] > ((per_pos4[1]-per_pos3[1])/(per_pos4[0]-per_pos3[0]))*(df2['x']-per_pos3[0]) + per_pos3[1] - 17 ]
    return pd.concat([df1,df2])

def mask_events_near_wire(df):
    per_pos1=[-44, -41.5]
    per_pos2=[13.5, 58.2]
    per_pos3=[-20,-57]
    per_pos4=[39.5, 45.8]
    x_pos=np.linspace(-70,70,140)
    y_per1 = ((per_pos2[1]-per_pos1[1])/(per_pos2[0]-per_pos1[0]))*(x_pos-per_pos1[0]) + per_pos1[1]+4
    y_per2 = y_per1-21
    y_per3 = ((per_pos4[1]-per_pos3[1])/(per_pos4[0]-per_pos3[0]))*(x_pos-per_pos3[0]) + per_pos3[1]+4
    y_per4 = y_per3-21
    cut1, cut2 = np.ones(len(df), dtype=bool), np.ones(len(df), dtype=bool)
    cut1 &= df['y'] < ((per_pos2[1]-per_pos1[1])/(per_pos2[0]-per_pos1[0]))*(df['x']-per_pos1[0]) + per_pos1[1] + 4 
    cut1 &= df['y'] > ((per_pos2[1]-per_pos1[1])/(per_pos2[0]-per_pos1[0]))*(df['x']-per_pos1[0]) + per_pos1[1] - 17
    cut2 &= df['y'] < ((per_pos4[1]-per_pos3[1])/(per_pos4[0]-per_pos3[0]))*(df['x']-per_pos3[0]) + per_pos3[1] + 4
    cut2 &= df['y'] > ((per_pos4[1]-per_pos3[1])/(per_pos4[0]-per_pos3[0]))*(df['x']-per_pos3[0]) + per_pos3[1] - 17
    cut = cut1 | cut2
    return cut, np.logical_not(cut)

def S2WidthBasic(events, title, mod_par = (44.04,0.673,3.2), bins = 300, wrange = (500,15000), trange = (0,2200),
                 fit_range = (200,1500), plot=False ):
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']/1e3
    s2area = events['s2_area']
    
    print('selected events',len(events))
    
    t = np.linspace(trange[0], trange[1], bins)
    wspace = np.linspace(wrange[0], wrange[1], bins)
    ph = Histdd(drift, s2width, bins=(t, wspace))
    
    # fit 50 percentile
    perc50 = np.array(ph.percentile(percentile=50, axis=1))
    D_mod = mod_par[0]*1e3 * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    tG_mod = mod_par[2] * units.ns
    
    ll = np.where(t>fit_range[0])[0][0]
    hh = np.where(t>fit_range[1])[0][0]
    diffusion = lambda x, D: s2_width_model(x, D, vd_mod, tG_mod)
    par, pcov = curve_fit(diffusion, t[ll:hh], perc50[ll:hh], p0=(D_mod))
    perr = np.sqrt(np.diag(pcov))
    ys = diffusion(t, *par)
    
    ys_mod = s2_width_model(t, D_mod, vd_mod, tG_mod)
    if plot:
        plt.figure(figsize=(12,6))
        ph.plot(log_scale=True, cblabel='events')
        #plt.axvspan(*fit_range, alpha=0.1, color='blue', label='fit region')
        #plt.plot((t[1:]+t[:-1])/2, perc50, color='b',linestyle='--', label='50% percentile')
        #plt.plot(t, ys, 'r--',label=f'fit: $D = {par[0]/1e3/(units.cm**2 / units.s):.2f}$ cm$^2$/s')
        plt.plot(t, ys_mod, 'black',label=f'model: $D = {mod_par[0]:.2f}$ cm$^2$/s')
        plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        plt.legend(fontsize=14)

def s2_width_model(t, D, vd, tGate):
    sigma_to_r50p = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
    return sigma_to_r50p * np.sqrt(2 * D * (t-tGate) / vd**2)

def s2_width_norm(width,s2_width_model):
    scw = 258.41  # s2_secondary_sc_width median
    normWidth = (np.square(width) - np.square(scw)) / np.square(s2_width_model)
    return normWidth

def S2WidthCut(events, title, mod_par = (44.04,0.673,3.2), wrange = (0,10),
                      chi2_cut_parameter = -14, switch = 4.1, pll=(0.8, 1), phh=(1.4, 1.6), plot=False ):
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']
    s2area = events['s2_area']
    
    D_mod = mod_par[0] * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    tG_mod = mod_par[2] * units.us
    normWidth = s2_width_norm(s2width,s2_width_model(drift, D_mod, vd_mod, tG_mod))
    alow, ahigh, bins = 2, 6.5, 300
    bins1 = int((switch-alow)/(ahigh-alow)*bins)
    bins2 = bins - bins1
    aspace, aspace1, aspace2 = np.logspace(alow, ahigh, bins), np.logspace(alow, switch, bins1), np.logspace(switch,ahigh,bins2)
    nspace = np.linspace(wrange[0], wrange[1], bins)
    pha = Histdd(s2area, normWidth, bins=(aspace, nspace))
    print(f'Total events {len(events)}')
    
    # S2WidthCut
    
    scg = 23.0 # pax_config.get('s2_secondary_sc_gain')
    chi2_low = np.sqrt(chi2.ppf((10**chi2_cut_parameter),aspace1/scg)/(aspace1/scg-1))
    chi2_high = np.sqrt(chi2.ppf((1-10**chi2_cut_parameter),aspace1/scg)/(aspace1/scg-1))
    A = ((switch**2,switch,1),(25,5,1),(36,6,1))
    bl, bh = (chi2_low[-1], pll[0],pll[1]), (chi2_high[-1], phh[0],phh[1])
    xl, xh = np.linalg.solve(A,bl), np.linalg.solve(A,bh)
    par_low = xl[0]*np.log10(aspace2)**2 + xl[1]*np.log10(aspace2) + xl[2]
    par_high = xh[0]*np.log10(aspace2)**2 + xh[1]*np.log10(aspace2) + xh[2]
    bound_low = np.concatenate((chi2_low, par_low))
    bound_high = np.concatenate((chi2_high, par_high))
    cut_low = interp1d(aspace, bound_low, bounds_error=False, fill_value='extrapolate',kind='cubic')
    cut_high = interp1d(aspace, bound_high, bounds_error=False, fill_value='extrapolate',kind='cubic')
    mask = np.ones(len(events), dtype=bool)
    mask &= normWidth < cut_high(s2area)
    mask &= normWidth > cut_low(s2area)
    perc = len(events[mask])/len(events)*100
    print(f'Cut with new cut: survived events {len(events[mask])} -> {perc:.2f}%')
    
    if plot:
        plt.figure(figsize=(12,6))
        pha.plot(log_scale=True, cblabel='events')
        plt.axhline(y=1,c='black',ls='--')
        plt.xlabel("S2 area (PE)", ha='right', x=1,fontsize=12)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        plt.xscale('log')
        plt.plot(aspace, cut_low(aspace),'r-',label=f'S2WidthCut')
        plt.plot(aspace, cut_high(aspace),'r-')
        plt.legend(fontsize=14)
    return mask, cut_low, cut_high

def S2WidthCut_plot(events, title, cut = None, mod_par = (44.04,0.673,3.2), wrange = (0,10),
                           bins=100):
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']
    s2area = events['s2_area']
    
    D_mod = mod_par[0] * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    tG_mod = mod_par[2] * units.us
    normWidth = s2_width_norm(s2width,s2_width_model(drift, D_mod, vd_mod, tG_mod))
    aspace, nspace = np.logspace(2, 6.5, 300), np.linspace(wrange[0], wrange[1], 300)
    pha = Histdd(s2area, normWidth, bins=(aspace, nspace))
    print(f'Total events {len(events)}')
    if cut is not None:
        #nlcut, nhcut, llcut, lhcut, plcut, phcut = cut[1:]
        nlcut, nhcut = cut[1:]
        ncut = (normWidth < nhcut(events['s2_area'])) & (normWidth > nlcut(events['s2_area']))
        #lcut = (normWidth < lhcut(events['s2_area'])) & (normWidth > llcut(events['s2_area']))
        #pcut = (normWidth < phcut(events['s2_area'])) & (normWidth > plcut(events['s2_area']))
        perc = len(events[ncut])/len(events)*100
        #lperc = len(events[lcut])/len(events)*100
        #pperc = len(events[pcut])/len(events)*100
        print(f'Cut with new cut: survived events {len(events[ncut])} -> {perc:.2f}%')
        #print(f'Cut with line cut: survived events {len(events[lcut])} -> {lperc:.2f}%')
        #print(f'Cut with perc cut: survived events {len(events[pcut])} -> {pperc:.2f}%')
    
    plt.figure(figsize=(12,6))
    pha.plot(log_scale=True, cblabel='events')
    plt.axhline(y=1,c='black',ls='--')
    plt.xlabel("S2 area (PE)", ha='right', x=1,fontsize=12)
    plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
    plt.title(f'{title}',fontsize=14)
    plt.xscale('log')
    if cut is not None: plt.plot(aspace, nlcut(aspace),'r--',label=f'S2WidthCut: {perc:.2f}%')
    if cut is not None: plt.plot(aspace, nhcut(aspace),'r--')
    plt.legend(fontsize=14)
    if cut is not None:
        aspace0 = np.logspace(2, 6.5, bins)
        aa0 = (aspace0[1:]+aspace0[:-1])/2
        #lc, ee = np.histogram(events[lcut]['s2_area'], bins=aspace0)
        #pc, ee = np.histogram(events[pcut]['s2_area'], bins=aspace0)
        nc, ee = np.histogram(events[ncut]['s2_area'], bins=aspace0)
        cc, ee = np.histogram(events['s2_area'], bins=aspace0)
        plt.figure(figsize=(12,6))
        #plt.plot(aa0,lc/cc,c='black',ds='steps',label='lines cut')
        #plt.plot(aa0,pc/cc,c='b',ds='steps',label='percentile cut')
        plt.plot(aa0,nc/cc,c='r',ds='steps',label='S2WidthCut')
        plt.xlabel("S2 area (PE)", ha='right', x=1,fontsize=12)
        plt.ylabel("acceptance", ha='right', y=1,fontsize=12)
        plt.xscale('log')
        plt.ylim(0,1)
        plt.title(f'{title}',fontsize=14)
        plt.legend(fontsize = 14)

def S2WidthNormLine(events, title, mod_par = (44.04,0.673,3.2), wrange = (0,10), ll = 0, hh = 2, plot=False ):
    
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']/1e3
    s2area = events['s2_area']
    
    D_mod = mod_par[0]*1e3 * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    tG_mod = mod_par[2] * units.ns
    normWidth = s2_width_norm(events['s2_range_50p_area'],s2_width_model(events['drift_time']/1e3, D_mod, vd_mod, tG_mod))
    aspace, aspace0 = np.logspace(2, 6.5, 300), np.logspace(2, 6.5, 30)
    nspace = np.linspace(wrange[0], wrange[1], 300)
    pha = Histdd(events['s2_area'], normWidth, bins=(aspace, nspace))
    pha0 = Histdd(events['s2_area'], normWidth, bins=(aspace0, nspace))
    
    low_cut = line(np.log10(aspace),0,ll)
    high_cut = line(np.log10(aspace),0,hh)
    s2width_lcut = interp1d(aspace, low_cut, bounds_error=False, fill_value='extrapolate',kind='cubic')
    s2width_hcut = interp1d(aspace, high_cut, bounds_error=False, fill_value='extrapolate',kind='cubic')
    if plot:
        plt.figure(figsize=(12,6))
        pha.plot(log_scale=True, cblabel='events')
        plt.axhline(y=1,c='black',ls='-',label='model from Kr-83m')
        plt.xlabel("S2 area (PE)", ha='right', x=1,fontsize=12)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        plt.xscale('log')
        plt.plot(aspace, s2width_lcut(aspace),'r--',label='cut limits')
        plt.plot(aspace, s2width_hcut(aspace),'r--')
        plt.legend(fontsize=14)
    cut = np.ones(len(events), dtype=bool)
    cut &= normWidth < s2width_hcut(events['s2_area'])
    cut &= normWidth > s2width_lcut(events['s2_area'])
    return cut

def plot_position_distribution_cnn(events1, events2, title1="", title2="", rebin=1):
    
    plt.rcParams['figure.figsize'] = (50, 35)
    fig = plt.figure(facecolor = 'w')
    plt.rc('font', size='30') 
    
    plt.subplot(2, 2, 1)
    plt.hist2d(events1['x'], events1['y'],
               bins = ([np.linspace(-75, 75, int(150/rebin)), np.linspace(-75, 75, int(150/rebin))]), 
               norm=LogNorm(), cmap = 'viridis')
    plt.colorbar()
    #utils.plt_config(ylabel = 'y  [cm]', xlabel = 'x [cm]', 
    #           colorbar = True,
    #           title=title1)
    plt.xlabel("x (cm)", ha='right', x=1)
    plt.ylabel("y (cm)", ha='right', y=1)
    plt.title(f'{title1}',fontsize=30)
    
    plt.subplot(2, 2, 2)
    plt.hist2d(events2['x'], events2['y'],
               bins = ([np.linspace(-75, 75, int(150/rebin)), np.linspace(-75, 75, int(150/rebin))]), 
               norm=LogNorm(), cmap = 'viridis')
    plt.colorbar()
    #utils.plt_config(ylabel = 'y [cm]', xlabel = 'x [cm]', 
    #           colorbar = True,
    #           title=title2)
    plt.xlabel("x (cm)", ha='right', x=1)
    plt.ylabel("y (cm)", ha='right', y=1)
    plt.title(f'{title2}',fontsize=30)
    
    plt.subplot(2, 2, 3)
    plt.hist2d(events1['r']**2,  
               events1['z'],
               bins = ([np.linspace(0, 75**2, int(150/rebin)), 
                        np.linspace(-160, 0, int(160/rebin))]), 
               norm=LogNorm(), cmap = 'viridis')
    plt.colorbar()
    #utils.plt_config(ylabel = 'Z [cm]', xlabel = 'R$^2$ [cm$^2$]', 
    #           colorbar = True, title=title1)
    plt.xlabel(r"r^2 (cm^2)", ha='right', x=1)
    plt.ylabel("z (cm)", ha='right', y=1)
    plt.title(f'{title1}',fontsize=30)
    
    plt.subplot(2, 2, 4)
    plt.hist2d(events2['r']**2,  
               events2['z'],
               bins = ([np.linspace(0, 75**2, int(150/rebin)), 
                        np.linspace(-160, 0, int(160/rebin))]), 
               norm=LogNorm(), cmap = 'viridis')
    plt.colorbar()
    #utils.plt_config(ylabel = 'Z [cm]', xlabel = 'R$^2$ [cm$^2$]', 
    #           colorbar = True, title=title2)
    plt.xlabel(r"r^2 (cm^2)", ha='right', x=1)
    plt.ylabel("z (cm)", ha='right', y=1)
    plt.title(f'{title2}',fontsize=30)
    plt.show()
    
def S2WidthNormalized_drift(events, title, mod_par = (44.04,0.673,3.2), wrange = (0,10), trange = (0,2200), bins=300,plot=False ):
    
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']/1e3
    s2area = events['s2_area']
    
    D_mod = mod_par[0]*1e3 * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    tG_mod = mod_par[2] * units.ns
    normWidth = s2_width_norm(events['s2_range_50p_area'],s2_width_model(events['drift_time']/1e3, D_mod, vd_mod, tG_mod))
    aspace, aspace0 = np.logspace(2, 6.5, 300), np.logspace(2, 6.5, 30)
    
    low_cut1 = line(np.log10(aspace),0,0.25)
    s2width_lcut1 = interp1d(aspace, low_cut1, bounds_error=False, fill_value='extrapolate',kind='cubic')
    cut1 = np.ones(len(events), dtype=bool)
    cut1 &= normWidth > s2width_lcut1(events['s2_area'])
    events =  events[cut1]
    
    tspace = np.linspace(trange[0], trange[1], bins)
    nspace = np.linspace(wrange[0], wrange[1], bins)
    normWidth = s2_width_norm(events['s2_range_50p_area'],s2_width_model(events['drift_time']/1e3, D_mod, vd_mod, tG_mod))
    pha = Histdd(events['drift_time']/1e3, normWidth, bins=(tspace, nspace))
    
    if plot:
        plt.figure(figsize=(12,6))
        pha.plot(log_scale=True, cblabel='events')
        #plt.axhline(y=1,c='black',ls='-',label='model from Kr-83m')
        plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        #plt.xscale('log')
        #plt.legend(fontsize=14)
        
def cutevents_wires(ev, FV = False):
    cut=(ev['cut_s1_max_pmt']) & (ev['cut_daq_veto']) & (ev['cut_s1_area_fraction_top']) & (ev['cut_s2_single_scatter'])
    mask = basic_cuts(ev) & cut
    if FV: mask = mask & ev['cut_fiducial_volume']
    #maskNW, maskFW = mask_events_near_wire(ev)
    maskFW, maskNW = mask_S2Width_vs_pos(ev)
    return ev[mask], ev[mask&maskNW], ev[mask&maskFW]

def S2Width_vs_drift(events, title,mod_par=(44.04,0.673,3.2),wrange=(0,20),nrange=(0,10),trange=(0,2200),bins=300):    
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']/1e3
    s2area = events['s2_area']
    
    D_mod = mod_par[0]*1e3 * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    tG_mod = mod_par[2] * units.ns
    normWidth = s2_width_norm(s2width,s2_width_model(drift, D_mod, vd_mod, tG_mod))
    tspace = np.linspace(trange[0], trange[1], bins)
    wspace = np.linspace(wrange[0], wrange[1], bins)
    nspace = np.linspace(nrange[0], nrange[1], bins)
    pha = Histdd(drift, s2width/1e3, bins=(tspace, wspace))
    phn = Histdd(drift, normWidth, bins=(tspace, nspace))
    
    plt.figure(figsize=(12,6))
    pha.plot(log_scale=True, cblabel='events')
    #plt.axhline(y=1,c='black',ls='-',label='model from Kr-83m')
    plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
    plt.ylabel("S2 width 50% (us)", ha='right', y=1,fontsize=12)
    plt.title(f'{title}',fontsize=14)
    
    plt.figure(figsize=(12,6))
    phn.plot(log_scale=True, cblabel='events')
    #plt.axhline(y=1,c='black',ls='-',label='model from Kr-83m')
    plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
    plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
    plt.title(f'{title}',fontsize=14)


def mask_S2Width_vs_pos(events, title = 'Kr83m', mod_par = (44.04,0.673,3.2), wrange = (0,15), nrange = (0,10),angledeg = 30, xrange = (-60,60), xcut = (10,17.5), bins = 300, plot = False):  
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']/1e3
    s2area = events['s2_area']
    
    D_mod = mod_par[0]*1e3 * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    tG_mod = mod_par[2] * units.ns
    normWidth = s2_width_norm(s2width,s2_width_model(drift, D_mod, vd_mod, tG_mod))
    xspace = np.linspace(xrange[0], xrange[1], bins)
    wspace = np.linspace(wrange[0], wrange[1], bins)
    nspace = np.linspace(nrange[0], nrange[1], bins)

    angle = math.radians(angledeg)
    nx = math.cos(angle) * events['s2_x_mlp'] - math.sin(angle) * events['s2_y_mlp']
    ny = math.sin(angle) * events['s2_x_mlp'] + math.cos(angle) * events['s2_y_mlp']
    phxy = Histdd(nx, ny, bins=(xspace, xspace))
    phw = Histdd(nx, s2width/1e3, bins=(xspace, wspace))
    phx = Histdd(nx, normWidth, bins=(xspace, nspace))
    phy = Histdd(ny, normWidth, bins=(xspace, nspace))
    
    cut1, cut2 = np.ones(len(events), dtype=bool), np.ones(len(events), dtype=bool)
    cut1 = nx > xcut[0]
    cut1 &= nx < xcut[1]
    cut2 = nx < -xcut[0]
    cut2 &= nx > -xcut[1]
    cutnw = cut1 | cut2
    cutfw = np.logical_not(cutnw)
    
    if plot:
        plt.figure(figsize=(10,10))
        phxy.plot(log_scale=True, cblabel='events')
        plt.xlabel("rotate s2_x_mlp (cm)", ha='right', x=1,fontsize=12)
        plt.ylabel("rotate s2_y_mlp (cm)", ha='right', y=1,fontsize=12)
        plt.axvline(x=xcut[0],c='red',ls='--')
        plt.axvline(x=xcut[1],c='red',ls='--')
        plt.axvline(x=-xcut[0],c='red',ls='--')
        plt.axvline(x=-xcut[1],c='red',ls='--')
        plt.title(f'{title}',fontsize=14)
        
        plt.figure(figsize=(12,6))
        phw.plot(log_scale=True, cblabel='events')
        plt.xlabel("rotate s2_x_mlp (cm)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2 width 50% (us)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
    
        plt.figure(figsize=(12,6))
        phx.plot(log_scale=True, cblabel='events')
        plt.axvline(x=xcut[0],c='red',ls='--',label='position cut')
        plt.axvline(x=xcut[1],c='red',ls='--')
        plt.axvline(x=-xcut[0],c='red',ls='--')
        plt.axvline(x=-xcut[1],c='red',ls='--')
        plt.xlabel("rotate s2_x_mlp (cm)", ha='right', x=1,fontsize=12)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        
        plt.figure(figsize=(12,6))
        phy.plot(log_scale=True, cblabel='events')
        plt.xlabel("rotate s2_y_mlp (cm)", ha='right', x=1,fontsize=12)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
    return cutfw, cutnw

def S2WidthNormalized_old(events, title, mod_par = (44.04,0.673,3.2), wrange = (0,10), perc = (1,99), delta=0.25,
                      cut_chi2 = -14, pll=(0.8, 1), phh=(1.4, 1.6), plot=False ):
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']/1e3
    s2area = events['s2_area']
    
    D_mod = mod_par[0]*1e3 * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    tG_mod = mod_par[2] * units.ns
    normWidth = s2_width_norm(events['s2_range_50p_area'],s2_width_model(events['drift_time']/1e3, D_mod, vd_mod, tG_mod))
    aspace, aspace0 = np.logspace(2, 6.5, 300), np.logspace(2, 6.5, 50)
    aa, aa0 = (aspace[1:]+aspace[:-1])/2, (aspace0[1:]+aspace0[:-1])/2
    #low_cut1 = line(np.log10(aspace),0,0.25)
    #s2width_lcut1 = interp1d(aspace, low_cut1, bounds_error=False, fill_value='extrapolate',kind='cubic')
    #cut1 = np.ones(len(events), dtype=bool)
    #cut1 &= normWidth > s2width_lcut1(events['s2_area'])
    #events =  events[cut1]
    
    nspace = np.linspace(wrange[0], wrange[1], 300)
    normWidth = s2_width_norm(events['s2_range_50p_area'],s2_width_model(events['drift_time']/1e3, D_mod, vd_mod, tG_mod))
    pha = Histdd(events['s2_area'], normWidth, bins=(aspace, nspace))
    s2area = events['s2_area']
    print(f'Total events {len(events)}')
    
    # cut with lines
    aspace1, aspace2 = np.logspace(2, 3.5, 100), np.logspace(3.5,6.5, 200)
    low_cut =  np.concatenate((line(np.log10(aspace1),(1-delta)/1.5,(2*delta-2)/1.5),
                               1.0 - delta + 0*np.log10(aspace2)))
    high_cut = np.concatenate((line(np.log10(aspace1),(delta-3)/1.5,(12-2*delta)/1.5),
                               1.0 + delta + 0*np.log10(aspace2)))
    s2width_lcut = interp1d(aspace, low_cut, bounds_error=False, fill_value='extrapolate',kind='cubic')
    s2width_hcut = interp1d(aspace, high_cut, bounds_error=False, fill_value='extrapolate',kind='cubic')
    cut = np.ones(len(events), dtype=bool)
    cut &= normWidth < s2width_hcut(events['s2_area'])
    cut &= normWidth > s2width_lcut(events['s2_area'])
    perc0 = len(events[cut])/len(events)*100
    print(f'Cut with lines: survived events {len(events[cut])} -> {perc0:.2f}%')
    
    #cut with percentile
    pha0 = Histdd(events['s2_area'], normWidth, bins=(aspace0, nspace))
    perc90a = np.array(pha0.percentile(percentile=perc[1], axis=1))
    perc10a = np.array(pha0.percentile(percentile=perc[0], axis=1))
    s2width_lperc = interp1d(aa0, perc10a, bounds_error=False, fill_value='extrapolate',kind='cubic')
    s2width_hperc = interp1d(aa0, perc90a, bounds_error=False, fill_value='extrapolate',kind='cubic')
    cutp = np.ones(len(events), dtype=bool)
    cutp &= normWidth < s2width_hperc(events['s2_area'])
    cutp &= normWidth > s2width_lperc(events['s2_area'])
    perc1 = len(events[cutp])/len(events)*100
    print(f'Cut with percentile: survived events {len(events[cutp])} -> {perc1:.2f}%')
    
    # chi2 cut
    scg = 23.0 # pax_config.get('s2_secondary_sc_gain')
    cutc = chi2.logpdf(normWidth * (s2area/scg -1), s2area/scg) > cut_chi2
    chi2_ll = np.sqrt(chi2.ppf(0.01,s2area/scg)/(s2area/scg-1))
    chi2_hh = np.sqrt(chi2.ppf(0.99,s2area/scg)/(s2area/scg-1))
    #cutc = np.ones(len(events), dtype=bool)
    #cutc &= normWidth < chi2_hh
    #cutc &= normWidth > chi2_ll
    nWc = s2_width_norm(events[cutc]['s2_range_50p_area'],s2_width_model(events[cutc]['drift_time']/1e3, D_mod, vd_mod, tG_mod))
    ph = Histdd(events[cutc]['s2_area'], nWc, bins=(aspace0, nspace))
    p00 = np.array(ph.percentile(percentile=1e-5, axis=1))
    p99 = np.array(ph.percentile(percentile=100-1e-5, axis=1))
    c_ll = interp1d(aa0, p00, bounds_error=False, fill_value='extrapolate',kind='quadratic')
    c_hh = interp1d(aa0, p99, bounds_error=False, fill_value='extrapolate',kind='quadratic')
    perc2 = len(events[cutc])/len(events)*100
    print(f'Cut with chi2: survived events {len(events[cutc])} -> {perc2:.2f}%')
    
    # new cut
    na1, na2 = np.logspace(2, 4.1, 20), np.logspace(4.1,6.5, 25)
    naa = np.logspace(2, 6.5, 45)
    na01 = (na1[1:]+na1[:-1])/2
    ph = Histdd(events[cutc]['s2_area'], nWc, bins=(na1, nspace))
    p00 = np.array(ph.percentile(percentile=1e-5, axis=1))
    p99 = np.array(ph.percentile(percentile=100-1e-5, axis=1))
    nc_ll = interp1d(na01, p00, bounds_error=False, fill_value='extrapolate',kind='quadratic')
    nc_hh = interp1d(na01, p99, bounds_error=False, fill_value='extrapolate',kind='quadratic')
    A = ((16,4,1),(25,5,1),(36,6,1))
    bl, bh = (nc_ll(na1[-1]), pll[0],pll[1]), (nc_hh(na1[-1]), phh[0],phh[1])
    xl, xh = np.linalg.solve(A,bl), np.linalg.solve(A,bh)
    nll_cut = np.concatenate((nc_ll(na1), xl[0]*np.log10(na2)**2 + xl[1]*np.log10(na2) + xl[2]))
    nhh_cut = np.concatenate((nc_hh(na1), xh[0]*np.log10(na2)**2 + xh[1]*np.log10(na2) + xh[2]))
    nlcut = interp1d(naa, nll_cut, bounds_error=False, fill_value='extrapolate',kind='cubic')
    nhcut = interp1d(naa, nhh_cut, bounds_error=False, fill_value='extrapolate',kind='cubic')
    ncut = np.ones(len(events), dtype=bool)
    ncut &= normWidth < nhcut(events['s2_area'])
    ncut &= normWidth > nlcut(events['s2_area'])
    perc3 = len(events[ncut])/len(events)*100
    print(f'Cut with necut: survived events {len(events[ncut])} -> {perc3:.2f}%')
    
    if plot:
        plt.figure(figsize=(12,6))
        pha.plot(log_scale=True, cblabel='events')
        plt.axhline(y=1,c='black',ls='--')
        plt.xlabel("S2 area (PE)", ha='right', x=1,fontsize=12)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        plt.xscale('log')
        plt.plot(aspace, s2width_lcut(aspace),c='black',ls='--',label=f'lines cut')
        plt.plot(aspace, s2width_hcut(aspace),c='black',ls='--')
        plt.plot(aa0, perc10a, 'b--', ds='steps',label=f'{perc[0]}-{perc[1]}% percentile')
        plt.plot(aa0, perc90a, 'b--', ds='steps')
        #plt.plot(aa0, c_ll(aa0),'r--',label=f'chi2 cut: {perc2:.2f}%')
        #plt.plot(aa0, c_hh(aa0),'r--')
        plt.plot(aspace, nlcut(aspace),'r--',label=f'S2WidthCut')
        plt.plot(aspace, nhcut(aspace),'r--')
        plt.legend(fontsize=14)
    return ncut, nlcut, nhcut, s2width_lcut, s2width_hcut, s2width_lperc, s2width_hperc

def plot_ERband(events,evcut, title, low = 0, high = 5000, bins = 500,guess=(300,2225,100,20,1)):
    print('total events',len(events[events['cs1']<high]))
    print('total events',len(evcut[evcut['cs1']<high]))
    cs1, cs2 = events['cs1'], events['cs2']
    cs1c, cs2c = evcut['cs1'], evcut['cs2']
    area_ratio = np.log10(np.divide(events['cs2'],events['cs1']))
    area_ratioc = np.log10(np.divide(evcut['cs2'],evcut['cs1']))
    ph_s2 = Histdd(cs1,area_ratio,bins=(np.linspace(low, high, bins), np.linspace(0, 5, bins)))
    ph_s2c = Histdd(cs1c,area_ratioc,bins=(np.linspace(low, high, bins), np.linspace(0, 5, bins)))
    plt.figure(figsize=(12,6))
    ph_s2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("cs1 (PE)", ha='right', x=1,size=14)
    plt.ylabel("log10(cs2/cs1)", ha='right', y=1,size=14)
    plt.title(title)
    plt.figure(figsize=(12,6))
    ph_s2c.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("cs1 (PE)", ha='right', x=1,size=14)
    plt.ylabel("log10(cs2/cs1)", ha='right', y=1,size=14)
    plt.title(title+' after S2WidthCut')
    
    
def gauss(x,a,mu,sigma,c,d):
    return a*np.exp(-(x-mu)**2 / (2.*sigma**2))+c+d*x

def plot_cs1_cs2(events,evcut, title, low = 0, high = 5000, low2 = 0, high2 = 1e6, bins = 500,guess=(300,2225,100,20,1)):
    print('total events',len(events))
    print('total events',len(evcut))
    cs1, cs2 = events['cs1'], events['cs2']
    cs1c, cs2c = evcut['cs1'], evcut['cs2']
    ph_s2 = Histdd(cs1,cs2,bins=(np.logspace(3, 5, 500), np.logspace(4, 6.5, 500)))
    plt.figure(figsize=(12,6))
    ph_s2.plot(log_scale=True, cblabel='S2 events')
    plt.xlabel("cS1 (PE)", ha='right', x=1,size=14)
    plt.ylabel("cS2 (PE)", ha='right', y=1,size=14)
    plt.title(title)
    plt.xscale('log')
    plt.yscale('log')
    # calibration
    W = 0.0137 #keV
    g1 = 0.1674
    g2 = 15.0369
    ene = W * (cs1/g1 + cs2/g2)
    enec = W * (cs1c/g1 + cs2c/g2)
    espace = np.linspace(low, high, bins)
    cs2space = np.linspace(low2, high2, bins)
    s0 = (cs2space[1:]+cs2space[:-1])/2
    e0 = (espace[1:]+espace[:-1])/2
    hene = Hist1d(ene, bins = espace)
    henec = Hist1d(enec, bins = espace)
    hene2 = Hist1d(cs2, bins = cs2space)
    hene2c = Hist1d(cs2c, bins = cs2space)
    popt, pcov = curve_fit(gauss, e0, hene,p0=guess)
    perr = np.sqrt(np.diag(pcov))
    poptc, pcovc = curve_fit(gauss, e0, henec,p0=guess)
    plt.figure(figsize=(12,6))
    plt.plot(e0,hene,label='before S2WidthCut')
    plt.plot(e0,gauss(e0,*popt),label=f'mu={popt[1]:.1f} keV, FWHM={popt[2]*2.35:.1f} keV')
    plt.plot(e0,henec,label='after S2WidthCut')
    plt.plot(e0,gauss(e0,*poptc),label=f'mu={poptc[1]:.1f} keV, FWHM={poptc[2]*2.35:.1f} keV')
    #plt.axvline(x=2225)
    plt.xlabel("energy (keV)", ha='right', x=1,size=14)
    plt.ylabel("counts", ha='right', y=1,size=14)
    plt.title(title,fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=14)
    plt.figure(figsize=(12,6))
    plt.plot(s0,hene2,label='before S2WidthCut')
    plt.plot(s0,hene2c,label='after S2WidthCut')
    #plt.axvline(x=2225)
    plt.xlabel("cS2 (PE)", ha='right', x=1,size=14)
    plt.ylabel("counts", ha='right', y=1,size=14)
    plt.title(title,fontsize=14)
    #plt.yscale('log')
    plt.legend(fontsize=14)