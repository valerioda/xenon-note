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

def S2WidthBasic(events, title, mod_par = (45, 0.675, 3), bins = 300, wrange = (500,15000), trange = (0,2200),
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
    SigmaToR50 = 1.349
    normWidth = (np.square(width) - np.square(scw)) / np.square(s2_width_model)
    return normWidth



def S2WidthNormLine(events, title, mod_par = (45, 0.675, 3), wrange = (0,10), ll = 0, hh = 2, plot=False ):
    
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
    
def S2WidthNormalized_drift(events, title, mod_par = (45, 0.675, 3), wrange = (0,10), trange = (0,2200), bins=300,plot=False ):
    
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