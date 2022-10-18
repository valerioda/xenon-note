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
from cutax import cut_efficiency
import matplotlib.colors as mcolors
mcolors.colors = dict(mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS)
cnames = []
for cname, value in mcolors.colors.items(): cnames.append(cname)
from decimal import Decimal
from scipy import special
from straxen.get_corrections import get_correction_from_cmt
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

# old definition
#def s2_width_model(t, D, vd, tGate):
#    scw = 375
#    sigma_to_r50p = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
#    return sigma_to_r50p * np.sqrt(2 * D * (t-tGate) / vd**2 + scw**2)
#def s2_width_norm(width,s2_width_model):
#    normWidth  = width / s2_width_model
#    return normWidth

def s2_width_model(t, D, vd, tGate):
    sigma_to_r50p = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
    return sigma_to_r50p * np.sqrt(2 * D * (t-tGate) / vd**2)

def s2_width_norm(width,s2_width_model):
    scw = 375
    normWidth = (np.square(width) - np.square(scw)) / np.square(s2_width_model)
    return normWidth

def S2WidthCut(events, title, mod_par = (44.04,0.673,3.2), wrange = (0,10), alow=2, ahigh=6.5, bins=300,
                      chi2_cut_parameter = -14, switch = 4.1, pll=(0.8, 1), phh=(1.4, 1.6), plot=False ):
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']
    s2area = events['s2_area']
    print(f'Total events {len(events)}')
    D_mod = mod_par[0] * units.cm**2 / units.s
    vd_mod = mod_par[1] * units.mm / units.us
    tG_mod = mod_par[2] * units.us
    normWidth = s2_width_norm(s2width,s2_width_model(drift, D_mod, vd_mod, tG_mod))
    
    bins1 = int((switch-alow)/(ahigh-alow)*bins)
    bins2 = bins - bins1
    aspace, aspace1, aspace2 = np.logspace(alow, ahigh, bins), np.logspace(alow, switch, bins1), np.logspace(switch,ahigh,bins2)
    nspace = np.linspace(wrange[0], wrange[1], bins)
    pha = Histdd(s2area, normWidth, bins=(aspace, nspace))
    print(f'Total events {len(events)}')
    
    # S2WidthCut
    
    scg = 23.0 # pax_config.get('s2_secondary_sc_gain')
    #cutc = chi2.logpdf(normWidth * (s2area/scg -1), s2area/scg) > cut_chi2
    chi2_low = np.sqrt(chi2.ppf((10**chi2_cut_parameter),aspace1/scg)/(aspace1/scg-1))
    chi2_high = np.sqrt(chi2.ppf( (Decimal(1)-Decimal(10**chi2_cut_parameter) ),aspace1/scg)/(aspace1/scg-1))
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
    print('1111111')
    if plot:
        print('ddsadas')
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
    
    """
    Update May 2022
    """

def s2_width_model(t, D, vd, tGate,width_type='50p'):
    if width_type == '50p': sigma_to_ = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
    if width_type == '90p': sigma_to_ = stats.norm.ppf(0.95) - stats.norm.ppf(0.05)
    return sigma_to_ * np.sqrt(2 * D * (t-tGate) / vd**2)

def s2_width_norm(width,s2_width_model):
    scw = 375
    normWidth = (np.square(width) - np.square(scw)) / np.square(s2_width_model)
    return normWidth

def S2WidthCut(events, title, mod_par = (4.566e-08,6.77e-05,2700.0), near_wires = False, bins = 200, bins0 = 2000,
               width_type='50p', perc = (1,99), arange = (2, 4), wrange = (0,20), switch = 4.1, afit = (2.2,6),
               pll=(0.8, 0.95), phh=(1.25, 1.35), chi2_cut = -14, fit_par = None, ev_leak = None, perc_plot = True,
               plot = False, name = None ):
    print('Deprecated Version')
    if width_type == '50p': s2width = events['s2_range_50p_area']
    if width_type == '90p': s2width = events['s2_range_90p_area']
    drift = events['drift_time']
    s2area = events['s2_area']
    s2arealog = np.log10(s2area)
    
    normWidth = s2_width_norm(s2width,s2_width_model(drift, *mod_par, width_type=width_type))
    #aspace = np.logspace(arange[0], arange[1], bins)
    aspace = np.linspace(arange[0], arange[1], bins)
    nspace, n0 = np.linspace(wrange[0], wrange[1], bins), np.linspace(0, 20, bins0)
    pha = Histdd(s2arealog, normWidth, bins=(aspace, nspace))
    if ev_leak is not None:
        if width_type == '50p': s2width_leak = ev_leak['s2_range_50p_area']
        if width_type == '90p': s2width_leak = ev_leak['s2_range_90p_area']
        normWidth_leak = s2_width_norm(s2width_leak,s2_width_model(ev_leak['drift_time'], *mod_par,width_type=width_type))
        ph_leak = Histdd(np.log10(ev_leak['s2_area']),normWidth_leak, bins=(aspace, nspace))
        r2_leak = ev_leak['r']**2
    
    if switch < arange[1]:
        bins1 = int((switch-arange[0])/(arange[1]-arange[0])*bins)
        bins2 = bins - bins1
        aspace1, aspace2 = np.linspace(arange[0], switch, bins1), np.linspace(switch,arange[1],bins2)
    else:
        bins1 = bins
        aspace1 = aspace
    
    ################### S2WidthCut as defined in cutax ###################
    
    scg = 23.0 #pax_config.get('s2_secondary_sc_gain')
    
    # chi2 distribution for low area events
    
    chi2_low = np.sqrt(chi2.ppf((10**chi2_cut),(10**aspace1)/scg-1)/((10**aspace1)/scg-1))
    chi2_high = np.sqrt(chi2.ppf((1-10**chi2_cut),(10**aspace1)/scg-1)/((10**aspace1)/scg-1))
    
    # parabolas for high area events
    if switch < arange[1]:
        A = ((switch**2,switch,1),(25,5,1),(36,6,1))
        bl, bh = (chi2_low[-1], pll[0],pll[1]), (chi2_high[-1], phh[0],phh[1])
        xl, xh = np.linalg.solve(A,bl), np.linalg.solve(A,bh)
        #par_low = xl[0]*np.log10(aspace2)**2 + xl[1]*np.log10(aspace2) + xl[2]
        #par_high = xh[0]*np.log10(aspace2)**2 + xh[1]*np.log10(aspace2) + xh[2]
        par_low = xl[0]*aspace2**2 + xl[1]*aspace2 + xl[2]
        par_high = xh[0]*aspace2**2 + xh[1]*aspace2 + xh[2]
        bound_low = np.concatenate((chi2_low, par_low))
        bound_high = np.concatenate((chi2_high, par_high))
        cut_low = interp1d(aspace, bound_low, bounds_error=False,
                           fill_value='extrapolate',kind='cubic')
        cut_high = interp1d(aspace, bound_high, bounds_error=False, 
                            fill_value='extrapolate',kind='cubic')
    else:
        cut_low = interp1d(aspace, chi2_low, bounds_error=False, 
                           fill_value='extrapolate',kind='cubic')
        cut_high = interp1d(aspace, chi2_high, bounds_error=False, 
                            fill_value='extrapolate',kind='cubic')
    
    mask, maskn = np.ones(len(events), dtype=bool), np.ones(len(events), dtype=bool)
    mask &= normWidth > cut_low(s2arealog)
    mask &= normWidth < cut_high(s2arealog)
    surv = len(events[mask])/len(events)*100
    #print(f'Chi2-low + Chi2-log-high cut: total {len(events)}, survived {len(events[mask])} -> {surv:.2f}%')
    
    # log pdf
    mask_log = np.ones(len(events), dtype=bool)
    mask_log &= chi2.logpdf(normWidth * (s2area/scg -1), s2area/scg-1) > chi2_cut
    phc = Histdd(s2arealog[mask_log], normWidth[mask_log], bins=(aspace1, n0))
    #surv_log = len(events[mask_log])/len(events)*100
    #print(f'Chi2 log pdf: total {len(events)}, survived {len(events[mask_log])} -> {surv_log:.2f}%')
    # pdf
    #mask_pdf = np.ones(len(events), dtype=bool)
    #mask_pdf &= chi2.pdf(normWidth * (s2area/scg -1), s2area/scg-1) > np.exp(chi2_cut)
    #php = Histdd(s2arealog[mask_pdf], normWidth[mask_pdf], bins=(aspace, nspace))
    #surv_pdf = len(events[mask_pdf])/len(events)*100
    #print(f'Chi2 pdf: total {len(events)}, survived {len(events[mask_pdf])} -> {surv_pdf:.2f}%')
    if ev_leak is not None:
        mask_l, maskn_l = np.ones(len(ev_leak), dtype=bool), np.ones(len(ev_leak), dtype=bool)
        mask_l &= normWidth_leak > cut_low(np.log10(ev_leak['s2_area']))
        surv_l = len(ev_leak[mask_l])/len(ev_leak)*100
        #print(f'ER leakage: total {len(ev_leak)}, survived {len(ev_leak[mask_l])} -> {surv_l:.2f}%')
    
    ph0 = Histdd(s2arealog, normWidth, bins=(aspace, n0))
    perc_l = np.array(ph0.percentile(percentile=perc[0], axis=1))
    perc_h = np.array(ph0.percentile(percentile=perc[1], axis=1))
    ac, ac1 = (aspace[1:]+aspace[:-1])/2, (aspace1[1:]+aspace1[:-1])/2
    lcut = interp1d(ac, perc_l, bounds_error=False, fill_value='extrapolate',kind='cubic')
    hcut = interp1d(ac, perc_h, bounds_error=False, fill_value='extrapolate',kind='cubic')
    """
    if upper_cut is not None:
        print('Upper boundary not calculated')
    else:
        #ppl = np.array(phc.percentile(percentile=1, axis=1))
        pph = np.array(phc.percentile(percentile=99.9, axis=1))
        #ccl = interp1d(ac, ppl, bounds_error=False, fill_value='extrapolate',kind='cubic')
        upper_cut = interp1d(ac1, pph, bounds_error=False, fill_value='extrapolate',kind='cubic')
    # parabolas for high area events
    if switch < arange[1]:
        bh = upper_cut(ac1)[-1], phh[0],phh[1]
        xh = np.linalg.solve(A,bh)
        par_high = xh[0]*aspace2**2 + xh[1]*aspace2 + xh[2]
        bound_high = np.concatenate((upper_cut(aspace1), par_high))
        upper_cut = interp1d(aspace, bound_high, bounds_error=False,fill_value='extrapolate',kind='cubic')
    """
    if fit_par is not None:
        if plot: print('Lower and higher boundary not calculated (fit not performed)')
        lower_cut = f_lower_boundary(s2arealog,*fit_par[0])
        upper_cut = f_upper_boundary(s2arealog,*fit_par[1])
        if switch < arange[1]:
            bl = f_lower_boundary(ac1,*fit_par[0])[-1], pll[0], pll[1]
            bu = f_upper_boundary(ac1,*fit_par[1])[-1], phh[0], phh[1]
            xl = np.linalg.solve(A,bl)
            xu = np.linalg.solve(A,bu)
            par_low = xl[0]*aspace2**2 + xl[1]*aspace2 + xl[2]
            par_upp = xu[0]*aspace2**2 + xu[1]*aspace2 + xu[2]
            bound_low = np.concatenate((f_lower_boundary(aspace1,*fit_par[0]), par_low))
            bound_upp = np.concatenate((f_upper_boundary(aspace1,*fit_par[1]), par_upp))
            low_bound = interp1d(aspace, bound_low, bounds_error=False,fill_value='extrapolate',kind='cubic')
            upp_bound = interp1d(aspace, bound_upp, bounds_error=False,fill_value='extrapolate',kind='cubic')
            lower_cut = low_bound(s2arealog)
            upper_cut = upp_bound(s2arealog)
        maskn &= normWidth > lower_cut
        maskn &= normWidth < upper_cut
        survn = len(events[maskn])/len(events)*100
        if plot: print(f'Fit cut: total {len(events)}, survived {len(events[maskn])} -> {survn:.2f}%')
        # ER leakage
        maskn_l &= normWidth_leak > f_lower_boundary(np.log10(ev_leak['s2_area']),*fit_par[0])
        maskn_l &= normWidth_leak < f_upper_boundary(np.log10(ev_leak['s2_area']),*fit_par[1])
        survn_l = len(ev_leak[maskn_l])/len(ev_leak)*100
        if plot: print(f'ER leakage: total {len(ev_leak)}, survived {len(ev_leak[maskn_l])} -> {survn_l:.2f}%')
    elif (afit[1] <= arange[1]):
        ll, hh = np.where(ac>=afit[0])[0][0], np.where(ac>=afit[1])[0][0]
        guess_l = (perc_l[hh]-np.mean(perc_l[ll:ll+20]), (ac[ll]+ac[hh])*0.5, (ac[hh]-ac[ll])*0.5 )
        guess_u, b_l, b_u = (2.5, 1, ac[ll], perc_h[hh]), (1, 0.2, 1.8, 0.8), (5, 10, 2.2, 2)
        fit_par_l, pcov_l = curve_fit( f_lower_boundary, ac[ll:hh], perc_l[ll:hh], p0 = guess_l )
        perr_l = np.sqrt(np.diag(pcov_l))
        if width_type == '90p': ll += 40
        fit_par_u, pcov_u = curve_fit( f_upper_boundary, ac[ll:hh], perc_h[ll:hh], p0=guess_u,bounds=(b_l,b_u))
        perr_u = np.sqrt(np.diag(pcov_u))
        fit_par = (fit_par_l,fit_par_u)
        lower_cut = f_lower_boundary(s2arealog,*fit_par[0])
        upper_cut = f_upper_boundary(s2arealog,*fit_par[1])
        print('Fit lower boundary:',fit_par[0])
        print('Fit upper boundary:',fit_par[1])
        if switch < arange[1]:
            bl = f_lower_boundary(ac1,*fit_par[0])[-1], pll[0], pll[1]
            bu = f_upper_boundary(ac1,*fit_par[1])[-1], phh[0], phh[1]
            xl = np.linalg.solve(A,bl)
            xu = np.linalg.solve(A,bu)
            par_low = xl[0]*aspace2**2 + xl[1]*aspace2 + xl[2]
            par_upp = xu[0]*aspace2**2 + xu[1]*aspace2 + xu[2]
            bound_low = np.concatenate((f_lower_boundary(aspace1,*fit_par[0]), par_low))
            bound_upp = np.concatenate((f_upper_boundary(aspace1,*fit_par[1]), par_upp))
            low_bound = interp1d(aspace, bound_low, bounds_error=False,fill_value='extrapolate',kind='cubic')
            upp_bound = interp1d(aspace, bound_upp, bounds_error=False,fill_value='extrapolate',kind='cubic')
            lower_cut = low_bound(s2arealog)
            upper_cut = upp_bound(s2arealog)
        maskn &= normWidth > lower_cut
        maskn &= normWidth < upper_cut
        #survn = len(events[maskn])/len(events)*100
        #print(f'Fit-low cut: total {len(events)}, survived {len(events[maskn])} -> {survn:.2f}%')
        #maskn &= normWidth < upper_cut(s2arealog)
        survn = len(events[maskn])/len(events)*100
        if plot: print(f'Fit cut: total {len(events)}, survived {len(events[maskn])} -> {survn:.2f}%')
        # ER leakage
        maskn_l &= normWidth_leak > f_lower_boundary(np.log10(ev_leak['s2_area']),*fit_par[0])
        maskn_l &= normWidth_leak < f_upper_boundary(np.log10(ev_leak['s2_area']),*fit_par[1])
        survn_l = len(ev_leak[maskn_l])/len(ev_leak)*100
        if plot: print(f'ER leakage: total {len(ev_leak)}, survived {len(ev_leak[maskn_l])} -> {survn_l:.2f}%')
    
    if plot:
        plt.figure(figsize=(10,5.625))
        pha.plot(log_scale=True, cblabel='events',colorbar=False)#,alpha=0.2)
        #phc.plot(log_scale=True, cblabel='events',colorbar=False)
        if perc_plot: plt.plot(ac, lcut(ac), 'rx', label=f'{perc[0]}-{perc[1]}% percentile')
        if perc_plot: plt.plot(ac, hcut(ac), 'rx')
        #plt.plot(ac, ccl(ac), 'g-')
        #plt.plot(ac, gaussian_filter1d(upper_cut(ac),4), 'g-', label=f'upper boundary\nchi2_par={chi2_cut}')
        #plt.plot(aspace, cut_low(aspace),'b-',label=f'cutax cut\nchi2_par={chi2_cut}')
        #plt.plot(aspace, cut_high(aspace),'b-')
        if fit_par is not None:
            if switch < arange[1]:
                plt.plot(aspace, low_bound(aspace),'b-',label='lower boundary')
                plt.plot(aspace, upp_bound(aspace),'g-',label='higher boundary')
            else:
                plt.plot(aspace, f_lower_boundary(aspace,*fit_par[0]),'b-',label='lower boundary')
                plt.plot(aspace, f_upper_boundary(aspace,*fit_par[1]),'g-',label='higher boundary')
        #plt.plot(aspace, f_lower_boundary(aspace,*par_50p_width),'--',label='guess')
        if ev_leak is not None:
            sc=plt.scatter(np.log10(ev_leak['s2_area']),normWidth_leak,c=r2_leak,s=30,cmap='Wistia',label='ER leakage')
            plt.colorbar(sc,label='R$^2$ (cm$^2$)')
        #plt.axhline(y=1,c='black',ls='--')
        plt.xlabel('log10(S2 area (PE))', ha='right', x=1,fontsize=12)
        if width_type == '50p': plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        if width_type == '90p': plt.ylabel("Normalized S2 width 90% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        #plt.xscale('log')
        plt.ylim(*wrange)
        plt.legend(fontsize=14,loc='upper right')
        if name is not None: plt.savefig(name+'_s2widthcut.png',dpi=600)
        if (fit_par is not None) & (afit[1] <= arange[1]):
            #plt.figure(figsize=(10,5.625))
            fig, ax = plt.subplots(2, 1,figsize=(10,5.625))
            sel_err = perr_l[0]
            l_data = (perc_l[ll:hh] - f_lower_boundary(ac,*fit_par[0])[ll:hh])/sel_err
            ax[0].errorbar(ac[ll:hh], l_data, color='b', fmt='s', ms=2, capsize=3,label='low fit residuals')
            par_ll = (fit_par[0][0], fit_par[0][1], fit_par[0][2]-perr_l[2])
            par_lu = (fit_par[0][0], fit_par[0][1], fit_par[0][2]+perr_l[2])
            par_ll2 = (fit_par[0][0], fit_par[0][1], fit_par[0][2]-2*perr_l[2])
            par_lu2 = (fit_par[0][0], fit_par[0][1], fit_par[0][2]+2*perr_l[2])
            l_min = (f_lower_boundary(ac,*fit_par[0])[ll:hh]-f_lower_boundary(ac,*par_ll)[ll:hh])/sel_err
            l_max = (f_lower_boundary(ac,*fit_par[0])[ll:hh]-f_lower_boundary(ac,*par_lu)[ll:hh])/sel_err
            l_min2 = (f_lower_boundary(ac,*fit_par[0])[ll:hh]-f_lower_boundary(ac,*par_ll2)[ll:hh])/sel_err
            l_max2 = (f_lower_boundary(ac,*fit_par[0])[ll:hh]-f_lower_boundary(ac,*par_lu2)[ll:hh])/sel_err
            ax[0].fill_between(ac[ll:hh],-1, 1, color='r',alpha=0.2)
            ax[0].fill_between(ac[ll:hh],1, 2, color='b',alpha=0.2)
            ax[0].fill_between(ac[ll:hh],-2,-1, color='b',alpha=0.2)
            #ax[0].set_xlabel('log10(S2 area (PE))', ha='right', x=1,fontsize=12)
            ax[0].set_ylabel('residuals [$\sigma_{ampl}$]', ha='right', y=1,fontsize=12)
            ax[0].legend(fontsize=14,loc='upper right')
            ax[0].set_ylim(-5,5)
            ax[0].set_xlim(ac[ll],ac[hh])
            if name is not None: plt.savefig(name+'_low_fit_residuals.png',dpi=600)
            sel_err = perr_l[0]
            u_data = (perc_h[ll:hh] - f_upper_boundary(ac,*fit_par[1])[ll:hh])/sel_err
            ax[1].errorbar(ac[ll:hh], u_data, color='g', fmt='s', ms=2, capsize=5,label='high fit residuals')
            ax[1].set_xlabel('log10(S2 area (PE))', ha='right', x=1,fontsize=12)
            ax[1].fill_between(ac[ll:hh],-1, 1, color='r',alpha=0.2)
            ax[1].fill_between(ac[ll:hh],1, 2, color='b',alpha=0.2)
            ax[1].fill_between(ac[ll:hh],-2, -1, color='b',alpha=0.2)
            ax[1].set_ylabel('residuals [$\sigma_{ampl}$]', ha='right', y=1,fontsize=12)
            ax[1].legend(fontsize=14,loc='upper right')
            ax[1].set_ylim(-10,10)
            ax[1].set_xlim(ac[ll],ac[hh])
            if name is not None: plt.savefig(name+'_high_fit_residuals.png',dpi=600)
    return mask, maskn, fit_par, len(events), len(events[maskn]), len(ev_leak), len(ev_leak[maskn_l])


def S2WidthCut(events, title, mod_par = (4.566e-08,6.77e-05,2700.0), bins = 200, bins0 = 2000,
               width_type='50p', perc = (1,99), arange = (2, 4), wrange = (0,20), afit = (2, 3.8),
               pll=(0.8, 0.95), phh=(1.25, 1.35), ext_par = None, ev_leak = None, perc_plot = True,
               plot = False, name = None, real_data = True, near_wires = False ):
    
    if width_type == '50p': s2width = events['s2_range_50p_area']
    if width_type == '90p': s2width = events['s2_range_90p_area']
    drift = events['drift_time']
    s2area = events['s2_area']
    s2arealog = np.log10(s2area)
    
    normWidth = s2_width_norm(s2width,s2_width_model(drift, *mod_par, width_type=width_type))
    area_space = (2, 7)
    aspace = np.linspace(area_space[0], area_space[1], bins)
    ac = (aspace[1:]+aspace[:-1])/2
    nspace, n0 = np.linspace(wrange[0], wrange[1], bins), np.linspace(0, 20, bins0)
    pha = Histdd(s2arealog, normWidth, bins=(aspace, nspace))
    if ev_leak is not None:
        if width_type == '50p': s2width_leak = ev_leak['s2_range_50p_area']
        if width_type == '90p': s2width_leak = ev_leak['s2_range_90p_area']
        normWidth_leak = s2_width_norm(s2width_leak,s2_width_model(ev_leak['drift_time'], *mod_par,width_type=width_type))
        ph_leak = Histdd(np.log10(ev_leak['s2_area']),normWidth_leak, bins=(aspace, nspace))
        r2_leak = ev_leak['r']**2
    
    # definition of fit limits
    ll, hh = np.where(ac>=afit[0])[0][0], int((afit[1]-area_space[0])/(area_space[1]-area_space[0])*bins)-1
    
    # S2WidthCut as defined in cutax
    chi2_cut = -14
    scg = 23.0
    chi2_low = np.sqrt(chi2.ppf((10**chi2_cut),(10**aspace[:hh])/scg-1)/((10**aspace[:hh])/scg-1))
    chi2_high = np.sqrt(chi2.ppf((1-10**chi2_cut),(10**aspace[:hh])/scg-1)/((10**aspace[:hh])/scg-1))
    
    A = ((afit[1]**2,afit[1],1),(25,5,1),(36,6,1))
    bl, bh = (chi2_low[-1], pll[0],pll[1]), (chi2_high[-1], phh[0],phh[1])
    xl, xu = np.linalg.solve(A,bl), np.linalg.solve(A,bh)
    par_low = xl[0]*ac[hh:]**2 + xl[1]*ac[hh:] + xl[2]
    par_upp = xu[0]*ac[hh:]**2 + xu[1]*ac[hh:] + xu[2]
    bound_low = np.concatenate((chi2_low, par_low))
    bound_upp = np.concatenate((chi2_high, par_upp))
    cut_low = interp1d(ac, bound_low, bounds_error=False, fill_value='extrapolate',kind='cubic')
    cut_upp = interp1d(ac, bound_upp, bounds_error=False, 
                        fill_value='extrapolate',kind='cubic')
    
    mask, maskn = np.ones(len(events), dtype=bool), np.ones(len(events), dtype=bool)
    mask &= normWidth > cut_low(s2arealog)
    mask &= normWidth < cut_upp(s2arealog)
    
    # percentile definition
    ph0 = Histdd(s2arealog, normWidth, bins=(aspace, n0))
    perc_l = np.array(ph0.percentile(percentile=perc[0], axis=1))
    perc_h = np.array(ph0.percentile(percentile=perc[1], axis=1))
    lcut = interp1d(ac, perc_l, bounds_error=False, fill_value='extrapolate',kind='cubic')
    hcut = interp1d(ac, perc_h, bounds_error=False, fill_value='extrapolate',kind='cubic')
    
    # fit parameters provided externally
    if ext_par is not None:
        if plot: print('Lower and higher boundary provided externally')
        fit_par = ext_par
    # perform the fits
    else:
        guess_l = (perc_l[hh]-np.mean(perc_l[ll:ll+20]), (ac[ll]+ac[hh])*0.5,(ac[hh]-ac[ll])*0.5)
        perc_hf = gaussian_filter1d(hcut(ac),8)
        dec_u = perc_hf[np.where(perc_hf<perc_hf[0]*0.5)[0][0]]*np.log(2)
        guess_u, b_l, b_u = ( perc_hf[0], dec_u, ac[ll], perc_hf[hh]), (perc_hf[0]*0.5, dec_u*0.8, ac[ll]*0.9, perc_hf[hh]*0.8), (perc_hf[0]*2, dec_u*1.5, ac[ll]*1.1, perc_hf[hh]*1.5)
        fit_par_l, pcov_l = curve_fit( f_lower_boundary, ac[ll:hh], perc_l[ll:hh], p0 = guess_l)
        perr_l = np.sqrt(np.diag(pcov_l))
        if width_type == '90p': ll += 40
        fit_par_u, pcov_u = curve_fit( f_upper_boundary, ac[ll:hh], perc_h[ll:hh], 
                                      p0=guess_u,bounds=(b_l,b_u))
        perr_u = np.sqrt(np.diag(pcov_u))
        fit_par = (fit_par_l,fit_par_u)
        print('Fit lower boundary:',fit_par[0])
        print('Fit upper boundary:',fit_par[1])
    
    # add parabola sides
    A = ((afit[1]**2,afit[1],1),(25,5,1),(36,6,1))
    bl = f_lower_boundary(ac,*fit_par[0])[hh], pll[0], pll[1]
    bu = f_upper_boundary(ac,*fit_par[1])[hh], phh[0], phh[1]
    xl = np.linalg.solve(A,bl)
    xu = np.linalg.solve(A,bu)
    print('param_parabola_low:',xl)
    print('param_parabola_high:',xu)
    par_low = xl[0]*ac[hh:]**2 + xl[1]*ac[hh:] + xl[2]
    par_upp = xu[0]*ac[hh:]**2 + xu[1]*ac[hh:] + xu[2]
    bound_low = np.concatenate((f_lower_boundary(ac[:hh],*fit_par[0]), par_low))
    bound_upp = np.concatenate((f_upper_boundary(ac[:hh],*fit_par[1]), par_upp))
    low_bound = interp1d(ac, bound_low, bounds_error=False,fill_value='extrapolate',kind='cubic')
    upp_bound = interp1d(ac, bound_upp, bounds_error=False,fill_value='extrapolate',kind='cubic')
    lower_cut = low_bound(s2arealog)
    upper_cut = upp_bound(s2arealog)
    
    # definition of cut mask
    maskn = np.ones(len(events), dtype=bool)
    maskn &= normWidth > lower_cut
    if not near_wires: maskn &= normWidth < upper_cut
    survn = len(events[maskn])/len(events)*100
    if plot: print(f'Fit cut: total {len(events)}, survived {len(events[maskn])} -> {survn:.2f}%')
    
    # ER leakage
    if ev_leak is not None:
        maskn_l = np.ones(len(ev_leak), dtype=bool)
        maskn_l &= normWidth_leak > f_lower_boundary(np.log10(ev_leak['s2_area']),*fit_par[0])
        maskn_l &= normWidth_leak < f_upper_boundary(np.log10(ev_leak['s2_area']),*fit_par[1])
        survn_l = len(ev_leak[maskn_l])/len(ev_leak)*100
        if plot: print(f'ER leakage: total {len(ev_leak)}, survived {len(ev_leak[maskn_l])} -> {survn_l:.2f}%')
    
    if plot:
        ev_c = events[maskn]
        ev_r = events[~maskn]
        plt.figure(figsize=(10,5.625))
        pha.plot(log_scale=True, cblabel='events',colorbar=False)
        plt.scatter(np.log10(ev_r['s2_area']),get_norm_width(ev_r),
                    c='r',s=5,cmap='Wistia',label='event rejected')
        if perc_plot: 
            plt.plot(ac, lcut(ac), 'rx', label=f'{perc[0]}-{perc[1]}% percentile')
            plt.plot(ac, hcut(ac), 'rx')
        plt.plot(ac, low_bound(ac),'b-',label='lower boundary')
        if not near_wires: plt.plot(ac, upp_bound(ac),'g-',label='higher boundary')
        plt.xlim(arange[0],arange[1])
        if ev_leak is not None:
            sc=plt.scatter(np.log10(ev_leak['s2_area']),normWidth_leak,
                           c=r2_leak,s=30,cmap='Wistia',label='ER leakage')
            plt.colorbar(sc,label='R$^2$ (cm$^2$)')
        plt.xlabel('log10(S2 area (PE))', ha='right', x=1,fontsize=12)
        if width_type == '50p': plt.ylabel("Normalized S2 width 50% (ns)",
                                           ha='right', y=1,fontsize=12)
        if width_type == '90p': plt.ylabel("Normalized S2 width 90% (ns)",
                                           ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        #plt.xscale('log')
        plt.ylim(*wrange)
        plt.legend(fontsize=14,loc='upper right')
        if name is not None: plt.savefig(name+'_s2widthcut.png',dpi=600)
        
        # plots to be removed
        tspace = np.linspace(0, 2300, bins)
        wspace = np.linspace(0, 20, bins)
        phw = Histdd(ev_c['drift_time']/1e3, ev_c['s2_range_50p_area']/1e3, bins=(tspace, wspace))
        plt.figure(figsize=(10,5.625))
        phw.plot(log_scale=True, cblabel='events')
        plt.scatter(ev_r['drift_time']/1e3,ev_r['s2_range_50p_area']/1e3,
                    c='r',s=5,cmap='Wistia',label='event rejected')
        plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2 width 50% (us)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        
        zspace = np.linspace(-155, 0, bins)
        phz = Histdd(ev_c['z'], ev_c['s2_range_50p_area']/1e3, bins=(zspace, wspace))
        plt.figure(figsize=(10,5.625))
        phz.plot(log_scale=True, cblabel='events')
        plt.scatter(ev_r['z'],ev_r['s2_range_50p_area']/1e3,
                    c='r',s=5,cmap='Wistia',label='event rejected')
        plt.xlabel("z (cm)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2 width 50% (us)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        
        plt.figure(figsize=(10,5.625))
        phzn = Histdd(ev_c['z'], get_norm_width(ev_c), bins=(zspace, nspace))
        phzn.plot(log_scale=True, cblabel='events')
        plt.scatter(ev_r['z'],get_norm_width(ev_r),
                    c='r',s=5,cmap='Wistia',label='event rejected')
        plt.xlabel("z (cm)", ha='right', x=1,fontsize=12)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        
        plt.figure(figsize=(10,5.625))
        xspace = np.linspace(-70, 70, bins)
        phxy = Histdd(ev_c['x'], ev_c['y'], bins=(xspace, xspace),alpha=0.2)
        phxy.plot(log_scale=True, cblabel='events')
        plt.scatter(ev_r['x'], ev_r['y'], c='r',s=5,cmap='Wistia',label='event rejected')
        plt.xlabel("x (cm)", ha='right', x=1,fontsize=12)
        plt.ylabel("y (cm)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        
        plt.figure(figsize=(10,5.625))
        rspace = np.linspace(0, 4500, bins)
        phrz = Histdd(ev_c['r']*ev_c['r'], ev_c['z'], bins=(rspace, zspace),alpha=0.2)
        phrz.plot(log_scale=True, cblabel='events')
        plt.scatter(ev_c['r']*ev_c['r'], ev_c['z'], c='r',s=5,cmap='Wistia',label='event rejected')
        plt.xlabel("r^2 (cm^2)", ha='right', x=1,fontsize=12)
        plt.ylabel("z (cm)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        
        ##############
        if 'perr_l' in locals():
            #plt.figure(figsize=(10,5.625))
            fig, ax = plt.subplots(2, 1,figsize=(10,5.625))
            sel_err = perr_l[0]
            l_data = (perc_l[ll:hh] - f_lower_boundary(ac,*fit_par[0])[ll:hh])/sel_err
            ax[0].errorbar(ac[ll:hh], l_data, color='b', fmt='s', ms=2, capsize=3,label='low fit residuals')
            par_ll = (fit_par[0][0], fit_par[0][1], fit_par[0][2]-perr_l[2])
            par_lu = (fit_par[0][0], fit_par[0][1], fit_par[0][2]+perr_l[2])
            par_ll2 = (fit_par[0][0], fit_par[0][1], fit_par[0][2]-2*perr_l[2])
            par_lu2 = (fit_par[0][0], fit_par[0][1], fit_par[0][2]+2*perr_l[2])
            l_min = (f_lower_boundary(ac,*fit_par[0])[ll:hh]-f_lower_boundary(ac,*par_ll)[ll:hh])/sel_err
            l_max = (f_lower_boundary(ac,*fit_par[0])[ll:hh]-f_lower_boundary(ac,*par_lu)[ll:hh])/sel_err
            l_min2 = (f_lower_boundary(ac,*fit_par[0])[ll:hh]-f_lower_boundary(ac,*par_ll2)[ll:hh])/sel_err
            l_max2 = (f_lower_boundary(ac,*fit_par[0])[ll:hh]-f_lower_boundary(ac,*par_lu2)[ll:hh])/sel_err
            ax[0].fill_between(ac[ll:hh],-1, 1, color='r',alpha=0.2)
            ax[0].fill_between(ac[ll:hh],1, 2, color='b',alpha=0.2)
            ax[0].fill_between(ac[ll:hh],-2,-1, color='b',alpha=0.2)
            #ax[0].set_xlabel('log10(S2 area (PE))', ha='right', x=1,fontsize=12)
            ax[0].set_ylabel('residuals [$\sigma_{ampl}$]', ha='right', y=1,fontsize=12)
            ax[0].legend(fontsize=14,loc='upper right')
            ax[0].set_ylim(-5,5)
            ax[0].set_xlim(ac[ll],ac[hh])
            if name is not None: plt.savefig(name+'_low_fit_residuals.png',dpi=600)
            sel_err = perr_l[0]
            u_data = (perc_h[ll:hh] - f_upper_boundary(ac,*fit_par[1])[ll:hh])/sel_err
            ax[1].errorbar(ac[ll:hh], u_data, color='g', fmt='s', ms=2, capsize=5,label='high fit residuals')
            ax[1].set_xlabel('log10(S2 area (PE))', ha='right', x=1,fontsize=12)
            ax[1].fill_between(ac[ll:hh],-1, 1, color='r',alpha=0.2)
            ax[1].fill_between(ac[ll:hh],1, 2, color='b',alpha=0.2)
            ax[1].fill_between(ac[ll:hh],-2, -1, color='b',alpha=0.2)
            ax[1].set_ylabel('residuals [$\sigma_{ampl}$]', ha='right', y=1,fontsize=12)
            ax[1].legend(fontsize=14,loc='upper right')
            ax[1].set_ylim(-10,10)
            ax[1].set_xlim(ac[ll],ac[hh])
            if name is not None: plt.savefig(name+'_high_fit_residuals.png',dpi=600)
    return mask, maskn, fit_par, len(events), len(events[maskn])

#def f_lower_boundary(x, p0, p1, p2, p3, p4):
#    return p0/(1.+p1*np.exp(-p2*x)) + p3*x + p4

def f_lower_boundary(x, p0, p1, p2):
    return p0*0.5 * special.erf((np.sqrt(2) * (x - p1)) / p2) + p0*0.5

#def f_upper_boundary(x, p0, p1, p2, p3):
#    return p0*np.exp(-p1*x) + p2*x + p3

def f_upper_boundary(x, p0, p1, p2, p3):
    return p0*np.exp(-p1*(x-p2)) + p3
    
def all_cuts(data, cs1_cut = None, near_wires = False,far_wires = False, AmBe = False, AC = False):
    cut_list=['cut_interaction_exists',
              'cut_main_is_valid_triggering_peak', 
              'cut_daq_veto',
              'cut_run_boundaries',
              #'cut_fiducial_volume',
              'cut_s1_tightcoin_3fold',
              'cut_s1_area_fraction_top',
              'cut_cs2_area_fraction_top',
              'cut_s1_max_pmt',
              'cut_s1_single_scatter',
              'cut_s2_single_scatter',
              'cut_s1_pattern_top',
              'cut_s1_pattern_bottom',
              'cut_s2_pattern',
              'cut_s1_width',
              #'cut_pres2_junk',
              'cut_s2_recon_pos_diff',
              'cut_ambience',
              'cut_time_veto', 
              'cut_time_shadow', 
              'cut_position_shadow',
              'cut_s1_naive_bayes',
              'cut_s2_naive_bayes']
    cut_list_ambe=[
          'cut_interaction_exists',
          'cut_main_is_valid_triggering_peak',
          'cut_daq_veto',
          'cut_run_boundaries',
          'cut_fiducial_volume_ambe',
          'cut_s1_tightcoin_3fold',
          'cut_s1_area_fraction_top',
          'cut_cs2_area_fraction_top',
          'cut_s1_max_pmt',
          'cut_s1_single_scatter',
          'cut_s2_single_scatter',
          'cut_s1_pattern_top',
          'cut_s1_pattern_bottom',
          'cut_s2_pattern',
          'cut_s1_width',
          'cut_s2_recon_pos_diff',
          'cut_nv_tpc_coincidence_ambe' ]
    cut_list_AC=[
        'cut_interaction_exists',
        'cut_main_is_valid_triggering_peak',
        'cut_s1_max_pmt',
        'cut_s1_single_scatter',
        'cut_s1_width', 'cut_s1_naive_bayes',
        'cut_cs2_area_fraction_top',
        'cut_s2_recon_pos_diff',
        'cut_s2_pattern',
        'cut_s2_single_scatter',
        'cut_s2_naive_bayes',
        'cut_s1_area_fraction_top',
        'cut_s1_pattern_top',
        'cut_s1_pattern_bottom',
        'cut_ambience',
        'cut_time_veto',
        'cut_time_shadow',
        'cut_fiducial_volume']
        
    cut = np.ones(len(data), dtype=bool)
    if AmBe: cut_list = cut_list_ambe
    if AC: cut_list = cut_list_AC
    for cut_ in cut_list:
        cut &= data[cut_]
    if cs1_cut is not None: cut &= data['cs1'] < cs1_cut
    cut &= data['s2_area'] > 200
    if near_wires: cut &= data['cut_near_wires']
    if far_wires: cut &= data['cut_near_wires']==False
    return cut
    
    
def get_lower_upper_bound(h_n, h_n_1, method="cp"):
    h_lower_bound = h_n.similar_blank_histogram()
    h_upper_bound = h_n.similar_blank_histogram()
    upper_bound_s = []
    lower_bound_s = []
    for k, n in zip(h_n.histogram, h_n_1.histogram):
        if method == "cp":
            lower, upper = cut_efficiency.compute_acceptance_uncertainty_CP(k, n)
        elif method == "bayesian":
            lower, upper = cut_efficiency.compute_acceptance_uncertainty_bayesian(k, n)
        else:
            raise ValueError(f"{method} can only be 'cp' or 'beyesian'!")
            
        avg = k / n
        lower_bound_s.append(avg - lower)
        upper_bound_s.append(upper - avg)
        
    h_lower_bound.histogram = np.array(lower_bound_s)
    h_upper_bound.histogram = np.array(upper_bound_s)
    return h_lower_bound, h_upper_bound
    
def ces(cs1, cs2):
    #https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:analysis:ntsciencerun0:g1g2_update
    g1 = 0.1532
    g2 = 16.2655
    w = 0.0137
    return (cs1/g1 + cs2/g2) * w
    
def rotated_x(x, y):
    """get rotated x, i.e. perpendicular to the transverse wires"""
    angle = np.pi / 6
    rotated_x = x*np.cos(angle) - y*np.sin(angle)
    return rotated_x
    
def get_acceptance(df, cut_mask, title, bins = 40, alim=(0.9,1), unc_method="cp", name = None):
    #df['ces'] = ces(df["cs1"], df["cs2"])
    #df['r2'] = df["r"]**2
    #df['s2_x_rotated'] = rotated_x(df["s2_x"], df["s2_y"])
    cut_s2 = df['s2_area']<10000
    #param_list = ('ces','s1_area','s2_area','r2','s2_x_rotated')
    #name_list = ('CES (keV)','S1 area (PE)','S2 area (PE)','R$^2$ (cm$^2$)','S2 x rotated (cm)')
    #col_list = ('b','g','r','c','m')
    #boundary_list = ((0,30), (0,200), (0,10000), (0,3600), (-57,57))
    param_list = ('s1_area','s2_area','z')
    name_list = ('S1 area (PE)','S2 area (PE)','z (cm)')
    data = pd.DataFrame(columns=('s1_area','s1_area_acc','s2_area','s2_area_acc','z','z_acc'))
    col_list = ('g','r','m')
    boundary_list = ((0,185), (0,10000), (-145,-5))
    for i,param_name in enumerate(param_list):
        xspace = np.linspace(*boundary_list[i],bins)
        h_bef = Hist1d(df[param_name], bins=xspace)
        h_aft = Hist1d(df[cut_mask][param_name], bins=xspace)
        h_acc_l, h_acc_u = get_lower_upper_bound(h_aft, h_bef, unc_method)
        h_acc = h_aft / h_bef
        acc_mean = np.mean(h_acc.histogram[np.isfinite(h_acc.histogram)])
        plt.figure(figsize=(12,6))
        plt.errorbar(h_acc.bin_centers, h_acc.histogram,yerr=(h_acc_l.histogram, h_acc_u.histogram),
                        color=col_list[i],fmt="s", ms=2, capsize=3,label='acceptance vs '+param_name)
        plt.xlabel(name_list[i], ha='right', x=1,fontsize=12)
        plt.ylabel("S2WidthCut Acceptance", ha='right', y=1,fontsize=12)
        plt.ylim(alim[0],alim[1])
        plt.title(f'{title} Total Acceptance = {len(df[cut_mask&cut_s2])/len(df[cut_s2])*100:.2f}%',fontsize=16)
        #plt.title(f'{title}',fontsize=16)
        plt.legend(fontsize = 14)
        data[param_name] = h_acc.bin_centers
        data[param_name+'_acc'] = h_acc.histogram
        data[param_name+'_acc_err_l'] = h_acc_l.histogram
        data[param_name+'_acc_err_h'] = h_acc_u.histogram
        if name is not None: plt.savefig(name+param_list[i]+'.png',dpi=600)
    return data
        
def S2WidthCutER_var(events, ev_leak, title, mod_par = (4.566e-08,6.77e-05,2700.0),
                     wrange = (0,10), arange = (2, 4), bins = 200,chi2_cut_hh = -14,
                     chi2_cuts = (-14,-10), scg = 23.0, plot=False, name = None ):
    
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']
    s2area = events['s2_area']
    print(f'Total events {len(events)}, ER leakage events {len(ev_leak)}')
    
    normWidth = s2_width_norm(s2width,s2_width_model(drift, *mod_par))
    
    aspace = np.logspace(arange[0], arange[1], bins)
    nspace = np.linspace(wrange[0], wrange[1], bins)
    pha = Histdd(s2area, normWidth, bins=(aspace, nspace))
    
    # ER leakage
    normWidth_leak = s2_width_norm(ev_leak['s2_range_50p_area'],s2_width_model(ev_leak['drift_time'], *mod_par))
    ph_leak = Histdd(ev_leak['s2_area'],normWidth_leak, bins=(aspace, nspace))
    r2_leak = ev_leak['r']**2
    
    plt.figure(figsize=(12,6))
    pha.plot(log_scale=True, cblabel='events')
    sc=plt.scatter(ev_leak['s2_area'],normWidth_leak,c=r2_leak,s=30,cmap='Wistia',label='ER leakage')
    plt.colorbar(sc,label='R$^2$ (cm$^2$)')
    plt.axhline(y=1,c='black',ls='--')
    plt.xlabel("log10(S2 area(PE))", ha='right', x=1,fontsize=12)
    plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
    plt.title(f'{title}',fontsize=14)
    plt.xscale('log')
    chi2_high = np.sqrt(chi2.ppf((1-10**chi2_cut_hh),aspace/scg)/(aspace/scg-1))
    cut_high = interp1d(aspace, chi2_high, bounds_error=False, fill_value='extrapolate',kind='cubic')
    plt.plot(aspace, cut_high(aspace),'r--')
    for i, chi2_cut in enumerate(chi2_cuts):
        chi2_low = np.sqrt(chi2.ppf((10**chi2_cut),aspace/scg)/(aspace/scg-1))
        cut_low = interp1d(aspace, chi2_low, bounds_error=False, fill_value='extrapolate',kind='cubic')
        mask = np.ones(len(events), dtype=bool)
        mask &= normWidth < cut_high(s2area)
        mask &= normWidth > cut_low(s2area)
        perc = len(events[mask])/len(events)*100
        maskl = np.ones(len(ev_leak), dtype=bool)
        maskl &= normWidth_leak < cut_high(ev_leak['s2_area'])
        maskl &= normWidth_leak > cut_low(ev_leak['s2_area'])
        percl = len(ev_leak[maskl])/len(ev_leak)*100
        print(f'Survived events: total {len(events[mask])} -> {perc:.2f}%, ER leakage {len(ev_leak[maskl])} -> {percl:.2f}%')
        plt.plot(aspace, cut_low(aspace),c=cnames[i],ls='--',label=f'low cut parameter = {chi2_cut}')
    plt.legend(fontsize=14)
    #if name is not None: plt.savefig(name+'_s2widthcut.png',dpi=600)

    
def merge_runs_cutax(st,runs):
    ev0 = st.get_df(runs[0],['event_info',
                             'cut_interaction_exists',
                             'cut_main_is_valid_triggering_peak', 
                             'cut_daq_veto',
                             'cut_run_boundaries',
                             #'cut_fiducial_volume_cylinder',
                             'cut_s1_tightcoin_3fold',
                             'cut_s1_area_fraction_top',
                             'cut_cs2_area_fraction_top',
                             'cut_s1_max_pmt',
                             'cut_s1_single_scatter',
                             'cut_s2_single_scatter',
                             'cut_s1_pattern_top',
                             'cut_s1_pattern_bottom',
                             'cut_s2_pattern',
                             'cut_s1_width',
                             #'cut_pres2_junk',
                             'cut_s2_recon_pos_diff',
                             'cut_ambience',
                             'cut_time_veto', 
                             'cut_time_shadow', 
                             'cut_position_shadow',
                             'cut_s1_naive_bayes',
                             'cut_s2_naive_bayes',
                             'cut_near_wires',
                             #'cut_s2_width',
                             'cut_s2_width_wimps',
                             'cut_s2_width_low_er',
                             'cut_s2_width_chi2',
                             'cut_s2_width_wire_modeled_wimps',
                             'cut_s2_width_wire_modeled_low_er'],
                    progress_bar=False)
    print('Reading runs from',runs[-1],'to',runs[0])
    start = time.time()
    for i, run_id in enumerate(runs[1:]):
        if ((i+1)%5) == 0: print(f'n. {i} run {run_id} elapsed time: {time.time()-start:.2f} s')
        ev_temp = st.get_df(run_id,['event_info',
                             'cut_interaction_exists',
                             'cut_main_is_valid_triggering_peak', 
                             'cut_daq_veto',
                             'cut_run_boundaries',
                             #'cut_fiducial_volume_cylinder',
                             'cut_s1_tightcoin_3fold',
                             'cut_s1_area_fraction_top',
                             'cut_cs2_area_fraction_top',
                             'cut_s1_max_pmt',
                             'cut_s1_single_scatter',
                             'cut_s2_single_scatter',
                             'cut_s1_pattern_top',
                             'cut_s1_pattern_bottom',
                             'cut_s2_pattern',
                             'cut_s1_width',
                             #'cut_pres2_junk',
                             'cut_s2_recon_pos_diff',
                             'cut_ambience',
                             'cut_time_veto', 
                             'cut_time_shadow', 
                             'cut_position_shadow',
                             'cut_s1_naive_bayes',
                             'cut_s2_naive_bayes',
                             'cut_near_wires',
                             #'cut_s2_width',
                             'cut_s2_width_wimps',
                             'cut_s2_width_low_er',
                             'cut_s2_width_chi2',
                             'cut_s2_width_wire_modeled_wimps',
                             'cut_s2_width_wire_modeled_low_er'],
                            #config=dict(s2_secondary_sc_width=375),
                    progress_bar=False)
        frames = [ev0,ev_temp]
        ev0 = pd.concat(frames)
    return ev0

def position_resolution(s2_area_top):      
    flat, amp, power, cutoff = 0.1, 4e2, 0.528, 3e4
    return flat + (amp / np.clip(s2_area_top, 0, cutoff)) ** power

def uppper_lim_with_wire(model_pred, nwidth_lim, s2_area_top, x_to_wire):
    ry = model_pred * (nwidth_lim - 1)                                              
    rx = position_resolution(s2_area_top) * stats.norm.isf(0.02)                      
    x_tmp = x_to_wire / rx     
    s2_width_wire_model_slope = 5e3
    k0 = s2_width_wire_model_slope * rx / ry  # slope in rescaled space    
    y0 = 1 / np.cos(np.arctan(k0))  # interception of the upper limit         
    x0 = np.sin(np.arctan(k0))  # switch poit from linear to circle
    s2_width_wire_model_intercep = 9e3
    yc = s2_width_wire_model_intercep / ry  # interception of the center 
    y_linear = y0 + yc - np.abs(x_tmp) * k0        
    y_circle = yc + np.sqrt(1 - np.clip(np.abs(x_tmp)**2, 0, 1))     
    y_lim = np.select([np.abs(x_tmp) > x0, np.abs(x_tmp) <= x0],
                      [y_linear, y_circle], 1)
    m = y_lim > yc               
    y_lim[m] = (y_lim[m] - yc[m]) * 0.5 + yc[m]  # half the peak
    return np.clip(y_lim, 1, np.inf) * ry

def cut_s2_width_chi2(events, chi2_cut = -14, mod_par = (4.566e-08, 6.77e-05, 2700.0), 
                      pll=(0.8, 0.95), phh=(1.25, 1.35), switch = 4.1, bins = 200):
    drift = events['drift_time']
    s2area = events['s2_area']
    s2arealog = np.log10(s2area)
    s2width = events['s2_range_50p_area']
    normWidth = s2_width_norm(s2width,s2_width_model(drift, *mod_par))
    area_space = (2, 7)
    aspace = np.linspace(area_space[0], area_space[1], bins)
    bins1 = int((switch-area_space[0])/(area_space[1]-area_space[0])*bins)
    bins2 = bins - bins1
    aspace1, aspace2 = np.linspace(area_space[0], switch, bins1), np.linspace(switch,area_space[1],bins2)
    scg = 23.0
    chi2_low = np.sqrt(chi2.ppf((10**chi2_cut),(10**aspace1)/scg-1)/((10**aspace1)/scg-1))
    chi2_high = np.sqrt(chi2.ppf((1-10**chi2_cut),(10**aspace1)/scg-1)/((10**aspace1)/scg-1))
    
    A = ((switch**2,switch,1),(25,5,1),(36,6,1))
    bl, bh = (chi2_low[-1], pll[0],pll[1]), (chi2_high[-1], phh[0],phh[1])
    xl, xu = np.linalg.solve(A,bl), np.linalg.solve(A,bh)
    par_low = xl[0]*aspace2**2 + xl[1]*aspace2 + xl[2]
    par_upp = xu[0]*aspace2**2 + xu[1]*aspace2 + xu[2]
    bound_low = np.concatenate((chi2_low, par_low))
    bound_upp = np.concatenate((chi2_high, par_upp))
    cut_low = interp1d(aspace, bound_low, bounds_error=False, fill_value='extrapolate',kind='cubic')
    cut_upp = interp1d(aspace, bound_upp, bounds_error=False, 
                        fill_value='extrapolate',kind='cubic')
    
    mask, maskn = np.ones(len(events), dtype=bool), np.ones(len(events), dtype=bool)
    mask &= normWidth > cut_low(s2arealog)
    mask &= normWidth < cut_upp(s2arealog)
    return mask, cut_low, cut_upp

def cut_parabola(area, parabola_par2, parabola_par1, parabola_par0):
    return parabola_par2 * np.power(np.log10(area), 2) + parabola_par1 * np.log10(area) + parabola_par0 

def get_x_to_wire(ev):
    wire_median_displacement = 13.06
    return  np.abs(ev['s2_x_mlp'] * np.cos(-np.pi/6) + ev['s2_y_mlp'] * np.sin(-np.pi/6)) - wire_median_displacement

def get_model_pred(ev):
    vd = get_correction_from_cmt('024075',('electron_drift_velocity', 'ONLINE', True))
    gd = get_correction_from_cmt('024075',('electron_drift_time_gate', 'ONLINE', True))
    dc = get_correction_from_cmt('024075',('electron_diffusion_cte', 'ONLINE', True))
    par = (dc,vd,gd)
    drift = ev['drift_time']
    model_pred = s2_width_model(drift, *par)
    return model_pred

def get_diff_width(ev):
    s2width = ev['s2_range_50p_area']
    drift = ev['drift_time']
    model_pred = get_model_pred(ev)
    return s2width - model_pred

def get_norm_width(ev):
    vd = get_correction_from_cmt('024075',('electron_drift_velocity', 'ONLINE', True))
    gd = get_correction_from_cmt('024075',('electron_drift_time_gate', 'ONLINE', True))
    dc = get_correction_from_cmt('024075',('electron_diffusion_cte', 'ONLINE', True))
    par = (dc,vd,gd)
    s2width = ev['s2_range_50p_area']
    drift = ev['drift_time']
    model_pred = s2_width_model(ev['drift_time'], *par)
    return s2_width_norm(s2width, s2_width_model(drift, *par))

def plot_cut_s2_wdith(events, fit_par, par_par, title, arange = (2,4.5), wrange = (0,2),
                      bins=400, wires = 'far',cut='cut_s2_wdith',wire_model=False,
                      xrange = (-5,5), drange = (-5000,15000), name = None ):
    print('Total events',len(events))
    if wires == 'far': events = events[events['cut_near_wires']==0]
    elif wires == 'near': events = events[events['cut_near_wires']]
    print('Events',wires,len(events))
    area_space = (2, 7)
    aspace = np.linspace(area_space[0], area_space[1], bins)
    ac = (aspace[1:]+aspace[:-1])/2
    nspace = np.linspace(wrange[0], wrange[1], bins)
    
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']
    s2area = events['s2_area']
    s2arealog = np.log10(s2area)
    normWidth = get_norm_width(events)
    pha = Histdd(s2arealog, normWidth, bins=(aspace, nspace))
    
    # defining new cut boundaries
    switch_from_quantile = 3.8
    hh = int((switch_from_quantile-area_space[0])/(area_space[1]-area_space[0])*bins)-1
    par_low = par_par[0][0]*ac[hh:]**2 + par_par[0][1]*ac[hh:] + par_par[0][2]
    par_upp = par_par[1][0]*ac[hh:]**2 + par_par[1][1]*ac[hh:] + par_par[1][2]
    bound_low = np.concatenate((f_lower_boundary(ac[:hh],*fit_par[0]), par_low))
    bound_upp = np.concatenate((f_upper_boundary(ac[:hh],*fit_par[1]), par_upp))
    low_bound = interp1d(ac, bound_low, bounds_error=False,fill_value='extrapolate',kind='cubic')
    upp_bound = interp1d(ac, bound_upp, bounds_error=False,fill_value='extrapolate',kind='cubic')
    mask_cut = events[cut]
    ev_c = events[mask_cut]
    ev_r = events[~mask_cut]
    print('Events after',cut,'from cutax',len(ev_c))
    phc = Histdd(np.log10(ev_c['s2_area']), get_norm_width(ev_c), bins=(aspace, nspace))
    
    plt.figure(figsize=(10,5.625))
    plt.plot(0,0,c='white',label=cut)
    pha.plot(log_scale=True, cblabel='events',colorbar=False,alpha=0.2)
    phc.plot(log_scale=True, cblabel='events',colorbar=True)
    plt.scatter(np.log10(ev_r['s2_area']),get_norm_width(ev_r),
                c='r',s=5,cmap='Wistia',label='event rejected')
    plt.plot(ac, low_bound(ac),'b-',label='lower boundary')
    if not wires == 'near': plt.plot(ac, upp_bound(ac),'g-',label='higher boundary')
    #plt.axvline(x=switch_from_quantile,color='k',ls=':',label='switch_from_quantile')
    plt.xlim(arange[0],arange[1])
    plt.legend(fontsize=14,loc='upper right')
    plt.xlabel('log10(S2 area (PE))', ha='right', x=1,fontsize=12)
    plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
    plt.title(f'{title}',fontsize=14)
    if name is not None: plt.savefig(name+cut+'.png',dpi=600)
    
    if wire_model:
        bins = 200
        xspace = np.linspace(xrange[0], xrange[1], bins)
        dspace = np.linspace(drange[0], drange[1], bins)
        phx_c = Histdd(get_x_to_wire(ev_c),get_diff_width(ev_c),
                         bins=(xspace, dspace))
        plt.figure(figsize=(10,5.625))
        plt.plot(0,0,c='white',label=cut)
        phx_c.plot(log_scale=True, cblabel='events after cut',colorbar=True)
        plt.scatter(get_x_to_wire(ev_r),get_diff_width(ev_r),c='r',s=5,cmap='Wistia',
                    label='event rejected')
        x2w = np.linspace(-10, 10, 1001)
        x1 = np.ones_like(x2w)
        aa1 ,aa2, aa3 = 3, 4, 5
        vd = get_correction_from_cmt('024075',
                                     ('electron_drift_velocity', 'ONLINE', True))
        gd = get_correction_from_cmt('024075',
                                     ('electron_drift_time_gate', 'ONLINE', True))
        dc = get_correction_from_cmt('024075',
                                     ('electron_diffusion_cte', 'ONLINE', True))
        par = (dc,vd,gd)
        plt.plot(x2w, uppper_lim_with_wire(x1 * s2_width_model(100e3,*par),
                                           x1 * f_upper_boundary(aa1,*fit_par[1]),
                                           x1 * 10**aa1 * 0.75, x2w),
                 color='r',label=f'limit with S2=10$^{aa1}$ PE, dt=100us')
        plt.plot(x2w, uppper_lim_with_wire(x1 * s2_width_model(100e3,*par),
                                           x1 * f_upper_boundary(aa2,*fit_par[1]), 
                                           x1 * 10**aa2 * 0.75, x2w),
                 color='b',label=f'limit with S2=10$^{aa2}$ PE, dt=100us')
        plt.ylabel('S2 width - model ($\mu$s)', ha='right', y=1,fontsize=12)
        plt.xlabel("distance from wire (cm)", ha='right', x=1,fontsize=12)
        plt.legend(fontsize=14,loc='upper right')
        plt.title(f'{title}',fontsize=14)
        if name is not None: plt.savefig(name+cut+'_diff_width.png',dpi=600)
        
        ############## test Tianyu model ##############
        
        phx = Histdd(get_x_to_wire(events), get_diff_width(events), bins=(xspace, dspace))
        area_top = s2area * events['s2_area_fraction_top']
        mask_chi2, chi2_low, chi2_upp = cut_s2_width_chi2(events)
        
        high_cut_mod_qua =  uppper_lim_with_wire(get_model_pred(events),
                                                 f_upper_boundary(s2arealog,*fit_par[1]),
                                                 area_top, get_x_to_wire(events))
        high_cut_mod_par =  uppper_lim_with_wire(get_model_pred(events),
                                                 cut_parabola(s2area,*par_par[1]),
                                                 area_top, get_x_to_wire(events))
        mask_high_modeled = ( (s2arealog >= switch_from_quantile) |
                             (get_diff_width(events) < high_cut_mod_qua) )
        mask_high_modeled &= ( (s2arealog < switch_from_quantile) |
                              (get_diff_width(events) < high_cut_mod_par) )
        mask_low_modeled = ( (s2arealog >= switch_from_quantile) |
                            (normWidth > f_lower_boundary(s2arealog,*fit_par[0])) )
        mask_low_modeled &= ( (s2arealog < switch_from_quantile) |
                             (normWidth > cut_parabola(s2area,*par_par[0])) )
        mask_mod = mask_high_modeled & mask_low_modeled
        ev_cut = events[mask_mod] # accepted events
        ev_rej = events[~mask_mod] # rejected events
        print('Events after S2WidthCut wire modeled (local)',len(ev_cut))
        phx_cut = Histdd(get_x_to_wire(ev_cut), get_diff_width(ev_cut),
                         bins=(xspace, dspace))
        ph_cut = Histdd(np.log10(ev_cut['s2_area']), get_norm_width(ev_cut),
                        bins=(aspace, nspace))
        
        plt.figure(figsize=(10,5.625))
        #pha.plot(log_scale=True, cblabel='events',colorbar=False,alpha=0.2)
        ph_cut.plot(log_scale=True, cblabel='events',colorbar=True)
        plt.scatter(np.log10(ev_rej['s2_area']), get_norm_width(ev_rej),c='r',
                    s=5,cmap='Wistia',label='event rejected')
        plt.plot(ac, low_bound(ac),'b-',label='lower boundary')
        plt.legend(fontsize=14,loc='upper right')
        plt.xlabel('log10(S2 area (PE))', ha='right', x=1,fontsize=12)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        
        plt.figure(figsize=(10,5.625))
        #phx.plot(log_scale=True, cblabel='events',colorbar=False,alpha=0.2)
        phx_cut.plot(log_scale=True, cblabel='events after cut',colorbar=True)
        plt.scatter(get_x_to_wire(ev_rej),get_diff_width(ev_rej),c='r',s=5,cmap='Wistia',
                    label='event rejected')
        plt.plot(x2w, uppper_lim_with_wire(x1 * s2_width_model(100e3,*par),
                                           x1 * f_upper_boundary(aa1,*fit_par[1]),
                                           x1 * 10**aa1 * 0.75, x2w),
                 color='r',label=f'limit with S2=10$^{aa1}$ PE, dt=100us')
        plt.plot(x2w, uppper_lim_with_wire(x1 * s2_width_model(100e3,*par),
                                           x1 * f_upper_boundary(aa2,*fit_par[1]), 
                                           x1 * 10**aa2 * 0.75, x2w),
                 color='b',label=f'limit with S2=10$^{aa2}$ PE, dt=100us')
        plt.ylabel('S2 width - model ($\mu$s)', ha='right', y=1,fontsize=12)
        plt.xlabel("distance from wire (cm)", ha='right', x=1,fontsize=12)
        plt.legend(fontsize=14,loc='upper right')
        plt.title(f'{title}',fontsize=14)
        

def cut_s2_wdith_sim(events, fit_par, par_par, title, arange = (2,4.5),
                     wrange = (0,2), bins=400, wires = 'far', wire_model = False,
                     xrange = (-5,5), drange = (-5000,15000), name = None ):
    print('Total events',len(events))
    mask_cut = np.ones(len(events), dtype=bool)
    area_space = (2, 7)
    aspace = np.linspace(area_space[0], area_space[1], bins)
    ac = (aspace[1:]+aspace[:-1])/2
    nspace = np.linspace(wrange[0], wrange[1], bins)
    
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']
    s2area = events['s2_area']
    s2arealog = np.log10(s2area)
    normWidth = get_norm_width(events)
    pha = Histdd(s2arealog, normWidth, bins=(aspace, nspace))
    
    # defining new cut boundaries
    switch_from_quantile = 3.8
    hh = int((switch_from_quantile-area_space[0])/(area_space[1]-area_space[0])*bins)-1
    par_low = par_par[0][0]*ac[hh:]**2 + par_par[0][1]*ac[hh:] + par_par[0][2]
    par_upp = par_par[1][0]*ac[hh:]**2 + par_par[1][1]*ac[hh:] + par_par[1][2]
    bound_low = np.concatenate((f_lower_boundary(ac[:hh],*fit_par[0]), par_low))
    bound_upp = np.concatenate((f_upper_boundary(ac[:hh],*fit_par[1]), par_upp))
    low_bound = interp1d(ac, bound_low, bounds_error=False,fill_value='extrapolate',kind='cubic')
    upp_bound = interp1d(ac, bound_upp, bounds_error=False,fill_value='extrapolate',kind='cubic')

    mask_cut &= ( (s2arealog >= switch_from_quantile) |
                            (normWidth > f_lower_boundary(s2arealog,*fit_par[0])) )
    mask_cut &= ( (s2arealog < switch_from_quantile) |
                             (normWidth > cut_parabola(s2area,*par_par[0])) )
    if not wires == 'near':
        mask_cut &= ( (s2arealog >= switch_from_quantile) |
                     (normWidth < f_upper_boundary(s2arealog,*fit_par[1])) )
        mask_cut &= ( (s2arealog < switch_from_quantile) |
                     (normWidth < cut_parabola(s2area,*par_par[1])) )
    ev_c = events[mask_cut]
    ev_r = events[~mask_cut]
    print('Events after cut',len(ev_c))
    phc = Histdd(np.log10(ev_c['s2_area']), get_norm_width(ev_c), bins=(aspace, nspace))
    
    plt.figure(figsize=(10,5.625))
    #pha.plot(log_scale=True, cblabel='events',colorbar=False,alpha=0.2)
    phc.plot(log_scale=True, cblabel='events',colorbar=True)
    plt.scatter(np.log10(ev_r['s2_area']),get_norm_width(ev_r),
                c='r',s=5,cmap='Wistia',label='event rejected')
    plt.plot(ac, low_bound(ac),'b-',label='lower boundary')
    if not wires == 'near': plt.plot(ac, upp_bound(ac),'g-',label='higher boundary')
    #plt.axvline(x=switch_from_quantile,color='k',ls=':',label='switch_from_quantile')
    plt.xlim(arange[0],arange[1])
    plt.legend(fontsize=14,loc='upper right')
    plt.xlabel('log10(S2 area (PE))', ha='right', x=1,fontsize=12)
    plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
    plt.title(f'{title}',fontsize=14)
    #if name is not None: plt.savefig(name+cut+'.png',dpi=600)
    
    if wire_model:
        bins = 200
        xspace = np.linspace(xrange[0], xrange[1], bins)
        dspace = np.linspace(drange[0], drange[1], bins)
        phx_c = Histdd(get_x_to_wire(ev_c),get_diff_width(ev_c),
                         bins=(xspace, dspace))
        plt.figure(figsize=(10,5.625))
        phx_c.plot(log_scale=True, cblabel='events after cut',colorbar=True)
        plt.scatter(get_x_to_wire(ev_r),get_diff_width(ev_r),c='r',s=5,cmap='Wistia',
                    label='event rejected')
        x2w = np.linspace(-10, 10, 1001)
        x1 = np.ones_like(x2w)
        aa1 ,aa2, aa3 = 3, 4, 5
        vd = get_correction_from_cmt('024075',
                                     ('electron_drift_velocity', 'ONLINE', True))
        gd = get_correction_from_cmt('024075',
                                     ('electron_drift_time_gate', 'ONLINE', True))
        dc = get_correction_from_cmt('024075',
                                     ('electron_diffusion_cte', 'ONLINE', True))
        par = (dc,vd,gd)
        plt.plot(x2w, uppper_lim_with_wire(x1 * s2_width_model(100e3,*par),
                                           x1 * f_upper_boundary(aa1,*fit_par[1]),
                                           x1 * 10**aa1 * 0.75, x2w),
                 color='r',label=f'limit with S2=10$^{aa1}$ PE, dt=100us')
        plt.plot(x2w, uppper_lim_with_wire(x1 * s2_width_model(100e3,*par),
                                           x1 * f_upper_boundary(aa2,*fit_par[1]), 
                                           x1 * 10**aa2 * 0.75, x2w),
                 color='b',label=f'limit with S2=10$^{aa2}$ PE, dt=100us')
        plt.ylabel('S2 width - model ($\mu$s)', ha='right', y=1,fontsize=12)
        plt.xlabel("distance from wire (cm)", ha='right', x=1,fontsize=12)
        plt.legend(fontsize=14,loc='upper right')
        plt.title(f'{title}',fontsize=14)
        if name is not None: plt.savefig(name+cut+'_diff_width.png',dpi=600)
        
        ############## test Tianyu model ##############
        
        phx = Histdd(get_x_to_wire(events), get_diff_width(events), bins=(xspace, dspace))
        area_top = s2area * events['s2_area_fraction_top']
        mask_chi2, chi2_low, chi2_upp = cut_s2_width_chi2(events)
        
        high_cut_mod_qua =  uppper_lim_with_wire(get_model_pred(events),
                                                 f_upper_boundary(s2arealog,*fit_par[1]),
                                                 area_top, get_x_to_wire(events))
        high_cut_mod_par =  uppper_lim_with_wire(get_model_pred(events),
                                                 cut_parabola(s2area,*par_par[1]),
                                                 area_top, get_x_to_wire(events))
        mask_high_modeled = ( (s2arealog >= switch_from_quantile) |
                             (get_diff_width(events) < high_cut_mod_qua) )
        mask_high_modeled &= ( (s2arealog < switch_from_quantile) |
                              (get_diff_width(events) < high_cut_mod_par) )
        mask_low_modeled = ( (s2arealog >= switch_from_quantile) |
                            (normWidth > f_lower_boundary(s2arealog,*fit_par[0])) )
        mask_low_modeled &= ( (s2arealog < switch_from_quantile) |
                             (normWidth > cut_parabola(s2area,*par_par[0])) )
        mask_mod = mask_high_modeled & mask_low_modeled
        ev_cut = events[mask_mod] # accepted events
        ev_rej = events[~mask_mod] # rejected events
        print('Events after S2WidthCut wire modeled (local)',len(ev_cut))
        phx_cut = Histdd(get_x_to_wire(ev_cut), get_diff_width(ev_cut),
                         bins=(xspace, dspace))
        ph_cut = Histdd(np.log10(ev_cut['s2_area']), get_norm_width(ev_cut),
                        bins=(aspace, nspace))
        mask_cut = mask_mod
    return mask_cut