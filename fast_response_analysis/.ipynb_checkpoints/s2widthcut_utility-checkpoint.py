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
                             'cut_fiducial_volume',
                             'cut_daq_veto'],progress_bar=False)
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


def basic_cuts(data, aft_cut = False):
    cut = np.ones(len(data), dtype=bool)
    #cut &= data['drift_time'] > 2.0e3
    cut &= data['s2_area'] > 200
    cut &= data['s1_n_channels'] >= 3
    cut &= data['s2_area_fraction_top'] > 0.7
    cut &= data['s2_area_fraction_top'] < 0.77
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

    
def line(x,a,b):
    return a * x + b


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
        nlcut, nhcut = cut[1:]
        ncut = (normWidth < nhcut(events['s2_area'])) & (normWidth > nlcut(events['s2_area']))
        perc = len(events[ncut])/len(events)*100
        print(f'Cut with new cut: survived events {len(events[ncut])} -> {perc:.2f}%')
    
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
        nc, ee = np.histogram(events[ncut]['s2_area'], bins=aspace0)
        cc, ee = np.histogram(events['s2_area'], bins=aspace0)
        plt.figure(figsize=(12,6))
        plt.plot(aa0,nc/cc,c='r',ds='steps',label='S2WidthCut')
        plt.xlabel("S2 area (PE)", ha='right', x=1,fontsize=12)
        plt.ylabel("acceptance", ha='right', y=1,fontsize=12)
        plt.xscale('log')
        plt.ylim(0,1)
        plt.title(f'{title}',fontsize=14)
        plt.legend(fontsize = 14)

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

def mask_S2Width_vs_pos_kr(events, title = 'Kr83m', mod_par = (44.04,0.673,3.2),
                        wrange = (0,15), nrange = (0,10),angledeg = 30, xrange = (-60,60),
                        xcut = (10,17.5), bins = 300, plot = False):  
    s2width = events['s2_a_range_50p_area']
    drift = events['drift_time']/1e3
    s2area = events['s2_a_area']
    
    D_mod = mod_par[0]*1e3 * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    tG_mod = mod_par[2] * units.ns
    normWidth = s2_width_norm(s2width,s2_width_model(drift, D_mod, vd_mod, tG_mod))
    xspace = np.linspace(xrange[0], xrange[1], bins)
    wspace = np.linspace(wrange[0], wrange[1], bins)
    nspace = np.linspace(nrange[0], nrange[1], bins)

    angle = math.radians(angledeg)
    nx = math.cos(angle) * events['s2_a_x_mlp'] - math.sin(angle) * events['s2_a_y_mlp']
    ny = math.sin(angle) * events['s2_a_x_mlp'] + math.cos(angle) * events['s2_a_y_mlp']
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
    
    return cutfw, cutnw

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