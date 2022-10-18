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


def line(x,a,b):
    return a * x + b

def gauss(x,a,mu,sigma,c,d):
    return a*np.exp(-(x-mu)**2 / (2.*sigma**2))+c+d*x


def plot_r2z_xy(ev, title, xylim = (-75,75), r2lim = (0,6300),zlim=(-150,1), bins = 300):
    r2 = ev['r']*ev['r']
    ph_r2z = Histdd(r2, ev['z'],bins=(np.linspace(r2lim[0],r2lim[1],bins), np.linspace(zlim[0],zlim[1],bins)))
    ph_xy = Histdd(ev['x'], ev['y'],bins=(np.linspace(xylim[0],xylim[1],bins), np.linspace(xylim[0],xylim[1],bins)))
    plt.figure(figsize=(8,4.5))
    ph_r2z.plot(log_scale=True,cblabel='events')
    plt.xlabel(r"r^2 (cm^2)", ha='right', x=1)
    plt.ylabel("z (cm)", ha='right', y=1)
    plt.title(f'{title}',fontsize=14)
    plt.figure(figsize=(10,8))
    ph_xy.plot(log_scale=True,cblabel='events')
    plt.xlabel("x (cm)", ha='right', x=1)
    plt.ylabel("y (cm)", ha='right', y=1)
    plt.title(f'{title}',fontsize=14)


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
        
        plt.figure(figsize=(8,4.5))
        phy.plot(log_scale=True, cblabel='events')
        plt.xlabel("rotate s2_y_mlp (cm)", ha='right', x=1,fontsize=12)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}')
    return cutfw, cutnw

def mask_S2Width_vs_pos(events, title = 'Kr83m', mod_par = (44.04,0.673,3.2),
                        wrange = (0,15), nrange = (0,10),angledeg = 30, xrange = (-60,60),
                        xcut = (10,17.5), bins = 300, plot = False):  
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
    
    return cutfw, cutnw 

def s2_width_model(t, D, vd, tGate,width_type='50p'):
    if width_type == '50p': sigma_to_ = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
    if width_type == '90p': sigma_to_ = stats.norm.ppf(0.95) - stats.norm.ppf(0.05)
    return sigma_to_ * np.sqrt(2 * D * (t-tGate) / vd**2)

def s2_width_norm(width,s2_width_model):
    scw = 375
    normWidth = (np.square(width) - np.square(scw)) / np.square(s2_width_model)
    return normWidth

def S2WidthCut(events, title, mod_par = (4.566e-08,6.77e-05,2700.0), bins = 200, bins0 = 2000,
               width_type='50p', perc = (1,99), arange = (2, 4), wrange = (0,20), afit = (2, 3.8),
               pll=(0.8, 0.95), phh=(1.25, 1.35), ext_par = None, ev_leak = None, perc_plot = True,
               plot = False, name = None, real_data = True, near_wires = False, kr = False,
               wire_model = False, xrange = (-5,5), drange = (-5000,15000) ):
    if kr:
        s2area = events['s2_a_area']
        s2width = events['s2_a_range_50p_area']
        area_top = s2area * events['s2_a_area_fraction_top']
    else:
        s2area = events['s2_area']
        area_top = s2area * events['s2_area_fraction_top']
        if width_type == '50p': s2width = events['s2_range_50p_area']
        if width_type == '90p': s2width = events['s2_range_90p_area']
    drift = events['drift_time']
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
    if plot:
        print('param_parabola_low:',xl)
        print('param_parabola_high:',xu)
        print(f'Cut: total {len(events)}, survived {len(events[maskn])} -> {survn:.2f}%')
    
    # ER leakage
    if ev_leak is not None:
        maskn_l = np.ones(len(ev_leak), dtype=bool)
        maskn_l &= normWidth_leak > f_lower_boundary(np.log10(ev_leak['s2_area']),*fit_par[0])
        maskn_l &= normWidth_leak < f_upper_boundary(np.log10(ev_leak['s2_area']),*fit_par[1])
        survn_l = len(ev_leak[maskn_l])/len(ev_leak)*100
        if plot: print(f'ER leakage: total {len(ev_leak)}, survived {len(ev_leak[maskn_l])} -> {survn_l:.2f}%')
    
    
    if wire_model:
        xspace = np.linspace(xrange[0], xrange[1], bins)
        dspace = np.linspace(drange[0], drange[1], bins)
        
        phx = Histdd(get_x_to_wire(events,kr), get_diff_width(events,kr), bins=(xspace, dspace))
        
        #mask_chi2, chi2_low, chi2_upp = cut_s2_width_chi2(events)
        
        high_cut_mod_qua =  uppper_lim_with_wire(get_model_pred(events),
                                                 f_upper_boundary(s2arealog,*fit_par[1]),
                                                 area_top, get_x_to_wire(events,kr))
        high_cut_mod_par =  uppper_lim_with_wire(get_model_pred(events),
                                                 cut_parabola(s2area,*xu),
                                                 area_top, get_x_to_wire(events,kr))
        mask_high_modeled = ( (s2arealog >= afit[1]) |
                             (get_diff_width(events,kr) < high_cut_mod_qua) )
        mask_high_modeled &= ( (s2arealog < afit[1]) |
                              (get_diff_width(events,kr) < high_cut_mod_par) )
        mask_low_modeled = ( (s2arealog >= afit[1]) |
                            (normWidth > f_lower_boundary(s2arealog,*fit_par[0])) )
        mask_low_modeled &= ( (s2arealog < afit[1]) |
                             (normWidth > cut_parabola(s2area,*xl)) )
        mask_mod = mask_high_modeled & mask_low_modeled
        ev_cut = events[mask_mod] # accepted events
        ev_rej = events[~mask_mod] # rejected events
        maskn = mask_mod
        survn = len(events[maskn])/len(events)*100
        if plot: print(f'Cut wire modeled: total {len(events)}, survived {len(events[maskn])} -> {survn:.2f}%')
    
    if plot:
        ev_c = events[maskn]
        ev_r = events[~maskn]
        plt.figure(figsize=(10,5.625))
        pha.plot(log_scale=True, cblabel='events',colorbar=True)
        plt.scatter(np.log10(ev_r['s2_area']),get_norm_width(ev_r),
                    c='r',s=1,cmap='Wistia',label='event rejected')
        if perc_plot: 
            plt.plot(ac, lcut(ac), 'rx', label=f'{perc[0]}-{perc[1]}% percentile')
            plt.plot(ac, hcut(ac), 'rx')
        plt.plot(ac, low_bound(ac),'b-',label='SR0 S2WidthCut')
        if not near_wires: plt.plot(ac, upp_bound(ac),'b-')
        #plt.plot(ac, cut_low(ac),'m-',label=f'cutax v8')
        #plt.plot(ac, cut_upp(ac),'m-')
        plt.xlim(arange[0],arange[1])
        if ev_leak is not None:
            sc=plt.scatter(np.log10(ev_leak['s2_area']),normWidth_leak,
                           c=r2_leak,s=30,cmap='Wistia',label='ER leakage')
            plt.colorbar(sc,label='R$^2$ (cm$^2$)')
        plt.xlabel('log10(S2 area (PE))', ha='right', x=1)
        if width_type == '50p': plt.ylabel("Normalized S2 width 50% (ns)",
                                           ha='right', y=1)
        if width_type == '90p': plt.ylabel("Normalized S2 width 90% (ns)",
                                           ha='right', y=1)
        plt.title(f'{title}')
        #plt.xscale('log')
        plt.ylim(*wrange)
        plt.legend(loc='upper right')
        if name is not None: plt.savefig(name+'_s2widthcut.png',dpi=600)
        
        plt.figure(figsize=(10,5.625))
        zspace = np.linspace(-155, 0, bins)
        phzn = Histdd(ev_c['z'], get_norm_width(ev_c,kr), bins=(zspace, nspace))
        phzn.plot(log_scale=True, cblabel='events')
        #plt.scatter(ev_r['z'],get_norm_width(ev_r),
        #            c='r',s=1,cmap='Wistia',label='event rejected')
        plt.xlabel("z (cm)", ha='right', x=1)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1)
        plt.title(f'{title}')
        #plt.legend(loc='upper right')
        #if name is not None: plt.savefig(name+'_s2widthcut_z.png',dpi=600)
        
        plt.figure(figsize=(10,5.625))
        harea = Hist1d(s2arealog, bins=aspace)
        if kr: harea_cut = Hist1d(np.log10(ev_c['s2_a_area']), bins=aspace)
        else: harea_cut = Hist1d(np.log10(ev_c['s2_area']), bins=aspace)
        harea.plot(label='total events')
        harea_cut.plot(label='after cut')
        plt.yscale('log')
        plt.xlabel("S2 area (PE)", ha='right', x=1)
        plt.ylabel("counts", ha='right', y=1)
        plt.title(f'{title}')
        plt.legend(loc='upper right')
        plt.xlim(arange[0],arange[1])
        
        """
        plt.figure(figsize=(10,5.625))
        xspace = np.linspace(-70, 70, bins)
        phxy = Histdd(ev_c['x'], ev_c['y'], bins=(xspace, xspace))
        phxy.plot(log_scale=True, cblabel='events')
        plt.scatter(ev_r['x'], ev_r['y'], c='r',s=1,
                    cmap='Wistia',label='event rejected')
        plt.xlabel("x (cm)", ha='right', x=1,fontsize=12)
        plt.ylabel("y (cm)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        plt.legend(fontsize=14,loc='upper right')
        if name is not None: plt.savefig(name+'_s2widthcut_xy.png',dpi=600)
        
        plt.figure(figsize=(10,5.625))
        rspace = np.linspace(0, 4500, bins)
        phrz = Histdd(ev_c['r']*ev_c['r'], ev_c['z'], bins=(rspace, zspace))
        phrz.plot(log_scale=True, cblabel='events')
        plt.scatter(ev_r['r']*ev_r['r'], ev_r['z'], c='r',s=1,
                    cmap='Wistia',label='event rejected')
        plt.xlabel("r^2 (cm^2)", ha='right', x=1,fontsize=12)
        plt.ylabel("z (cm)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        plt.legend(fontsize=14,loc='upper right')
        if name is not None: plt.savefig(name+'_s2widthcut_r2z.png',dpi=600)
        
        tspace = np.linspace(0, 2300, bins)
        wspace = np.linspace(0, 20, bins)
        phw = Histdd(ev_c['drift_time']/1e3, ev_c['s2_range_50p_area']/1e3, bins=(tspace, wspace))
        plt.figure(figsize=(10,5.625))
        phw.plot(log_scale=True, cblabel='events')
        plt.scatter(ev_r['drift_time']/1e3,ev_r['s2_range_50p_area']/1e3,
                    c='r',s=1,cmap='Wistia',label='event rejected')
        plt.xlabel("drift time (us)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2 width 50% (us)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        
        phz = Histdd(ev_c['z'], ev_c['s2_range_50p_area']/1e3, bins=(zspace, wspace))
        plt.figure(figsize=(10,5.625))
        phz.plot(log_scale=True, cblabel='events')
        plt.scatter(ev_r['z'],ev_r['s2_range_50p_area']/1e3,
                    c='r',s=1,cmap='Wistia',label='event rejected')
        plt.xlabel("z (cm)", ha='right', x=1,fontsize=12)
        plt.ylabel("S2 width 50% (us)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        """
        
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
            #ax[0].set_xlabel('log10(S2 area (PE))', ha='right', x=1)
            ax[0].set_ylabel('residuals [$\sigma_{ampl}$]', ha='right', y=1)
            ax[0].legend(loc='upper right')
            ax[0].set_ylim(-5,5)
            ax[0].set_xlim(ac[ll],ac[hh])
            sel_err = perr_l[0]
            u_data = (perc_h[ll:hh] - f_upper_boundary(ac,*fit_par[1])[ll:hh])/sel_err
            ax[1].errorbar(ac[ll:hh], u_data, color='g', fmt='s', ms=2, capsize=5,label='high fit residuals')
            ax[1].set_xlabel('log10(S2 area (PE))', ha='right', x=1)
            ax[1].fill_between(ac[ll:hh],-1, 1, color='r',alpha=0.2)
            ax[1].fill_between(ac[ll:hh],1, 2, color='b',alpha=0.2)
            ax[1].fill_between(ac[ll:hh],-2, -1, color='b',alpha=0.2)
            ax[1].set_ylabel('residuals [$\sigma_{ampl}$]', ha='right', y=1)
            ax[1].legend(loc='upper right')
            ax[1].set_ylim(-10,10)
            ax[1].set_xlim(ac[ll],ac[hh])
            if name is not None: plt.savefig(name+'_fit_residuals.png',dpi=600)
        if wire_model:
            if kr: s2r, s2c = ev_rej['s2_a_area'], ev_cut['s2_a_area']
            else: ev_rej['s2_area'], ev_cut['s2_area']
            phx_cut = Histdd(get_x_to_wire(ev_cut,kr), get_diff_width(ev_cut,kr),
                         bins=(xspace, dspace))
            ph_cut = Histdd(np.log10(s2c), get_norm_width(ev_cut,kr),
                        bins=(aspace, nspace))
        
            plt.figure(figsize=(8,4.5))
            #pha.plot(log_scale=True, cblabel='events',colorbar=False,alpha=0.2)
            ph_cut.plot(log_scale=True, cblabel='events',colorbar=True)
            plt.scatter(np.log10(s2r), get_norm_width(ev_rej,kr),c='r',
                        s=5,cmap='Wistia',label='event rejected')
            plt.plot(ac, low_bound(ac),'b-',label='lower boundary')
            plt.legend(loc='upper right')
            plt.xlabel('log10(S2 area (PE))', ha='right', x=1)
            plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1)
            plt.title(f'{title}')
            plt.xlim(arange[0],arange[1])
            if name is not None: plt.savefig(name+'_s2widthcut.png',dpi=600)
            
            plt.figure(figsize=(8,4.5))
            #phx.plot(log_scale=True, cblabel='events',colorbar=False,alpha=0.2)
            phx_cut.plot(log_scale=True, cblabel='events after cut',colorbar=True)
            plt.scatter(get_x_to_wire(ev_rej,kr),get_diff_width(ev_rej,kr),c='r',s=5,cmap='Wistia',
                    label='event rejected')
            x2w = np.linspace(-10, 10, 1001)
            x1 = np.ones_like(x2w)
            aa1 ,aa2, aa3 = 3, 4, 5
            plt.plot(x2w, uppper_lim_with_wire(x1 * s2_width_model(100e3,*mod_par),
                                               x1 * f_upper_boundary(aa1,*fit_par[1]),
                                               x1 * 10**aa1 * 0.75, x2w),
                     color='r',label=f'limit with S2=10$^{aa1}$ PE, dt=100us')
            plt.plot(x2w, uppper_lim_with_wire(x1 * s2_width_model(100e3,*mod_par),
                                               x1 * f_upper_boundary(aa2,*fit_par[1]), 
                                               x1 * 10**aa2 * 0.75, x2w),
                     color='b',label=f'limit with S2=10$^{aa2}$ PE, dt=100us')
            plt.ylabel('S2 width - model ($\mu$s)', ha='right', y=1)
            plt.xlabel("distance from wire (cm)", ha='right', x=1)
            plt.legend(loc='upper right')
            plt.title(f'{title}')
            if name is not None: plt.savefig(name+'_s2widthcut_wire_model.png',dpi=600)
    return mask, maskn, fit_par, len(events), len(events[maskn])

def S2WidthCut_SR0_plot(events, fit_par, par_par, title='', mod_par = (4.566e-08,6.77e-05,2700.0), bins = 200, bins0 = 2000,
                        arange = (2, 4), wrange = (0,20), ext_par = None, name = None, real_data = True, near_wires = False,
                        xrange = (-4.5,4.5), drange = (-3000,25000),label='events' ):
    
    s2area = events['s2_area']
    area_top = s2area * events['s2_area_fraction_top']
    s2width = events['s2_range_50p_area']
    drift = events['drift_time']
    s2arealog = np.log10(s2area)
    
    normWidth = s2_width_norm(s2width,s2_width_model(drift, *mod_par))
    nspace, n0 = np.linspace(wrange[0], wrange[1], bins), np.linspace(wrange[0], wrange[1], bins)
    pha = Histdd(s2arealog, normWidth, bins=(np.linspace(arange[0], arange[1], bins), nspace))
    
    def f_lower_boundary(x, p0, p1, p2):
        return p0*0.5 * special.erf((np.sqrt(2) * (x - p1)) / p2) + p0*0.5
    def f_upper_boundary(x, p0, p1, p2, p3):
        return p0*np.exp(-p1*(x-p2)) + p3
    def parabola(x, p0, p1, p2):
        return p0*x**2 + p1*x + p2
    
    area_space=(2,7)
    aspace = np.linspace(2, 7, bins)
    hh = int(2/5*bins)-1
    ac = (aspace[1:]+aspace[:-1])/2
    bound_low = np.concatenate((f_lower_boundary(ac[:hh],*fit_par[0]), parabola(ac[hh:],*par_par[0]) ))
    bound_upp = np.concatenate((f_upper_boundary(ac[:hh],*fit_par[1]), parabola(ac[hh:],*par_par[1]) ))
    low_bound = interp1d(ac, bound_low, bounds_error=False,fill_value='extrapolate',kind='cubic')
    upp_bound = interp1d(ac, bound_upp, bounds_error=False,fill_value='extrapolate',kind='cubic')
    lower_cut = low_bound(s2arealog)
    upper_cut = upp_bound(s2arealog)
    
    # cut mask
    maskn = events['cut_s2_width_wire_modeled_low_er']
    
    if near_wires:
        xspace = np.linspace(xrange[0], xrange[1], bins)
        dspace = np.linspace(drange[0], drange[1], bins)
        phx = Histdd(get_x_to_wire(events), get_diff_width(events), bins=(xspace, dspace))
        high_cut_mod_qua =  uppper_lim_with_wire(get_model_pred(events),
                                                 f_upper_boundary(s2arealog,*fit_par[1]),
                                                 area_top, get_x_to_wire(events))
        high_cut_mod_par =  uppper_lim_with_wire(get_model_pred(events),
                                                 cut_parabola(s2area,*par_par[1]),
                                                 area_top, get_x_to_wire(events))
    #plot
    ev_c = events[maskn]
    ev_r = events[~maskn]
    plt.figure(figsize=(8,4.5), facecolor='white')
    pha.plot(log_scale=True, cblabel=label,colorbar=True)
    plt.scatter(np.log10(ev_r['s2_area']),get_norm_width(ev_r), c='r',s=1,cmap='Wistia',label='event rejected')
    plt.plot(ac, low_bound(ac),'b',ls=':',label='cut boundary')
    if not near_wires: plt.plot(ac, upp_bound(ac),'b',ls=':')
    plt.xlim(arange[0],arange[1])
    plt.xlabel('log10(S2 area (PE))', ha='right', x=1)
    plt.ylabel("Normalized S2 width 50%", ha='right', y=1)
    plt.title(f'{title}')
    plt.ylim(*wrange)
    plt.legend(loc='upper right')
    if name is not None: plt.savefig(name+'_s2widthcut.png',dpi=600)
        
    if near_wires:
        phx = Histdd(get_x_to_wire(events), get_diff_width(events)/1e3, bins=(xspace, dspace/1e3))
        #phx_cut = Histdd(get_x_to_wire(ev_c), get_diff_width(ev_c), bins=(xspace, dspace))
        plt.figure(figsize=(8,4.5), facecolor='white')
        phx.plot(log_scale=True, cblabel='$^{220}$Rn data',colorbar=True)
        #phx_cut.plot(log_scale=True, cblabel='$^{220}$Rn data',colorbar=True)
        plt.scatter(get_x_to_wire(ev_r),get_diff_width(ev_r)/1e3,c='r',s=1,cmap='Wistia', label='event rejected')
        x2w = np.linspace(-10, 10, 1001)
        x1 = np.ones_like(x2w)
        aa1 ,aa2, aa3 = 3, 4, 5
        plt.plot(x2w, uppper_lim_with_wire(x1 * s2_width_model(100e3,*mod_par),
                                               x1 * f_upper_boundary(aa1,*fit_par[1]),
                                               x1 * 10**aa1 * 0.75, x2w)/1e3,
                     color='g',ls=':',label=f'cut boundary with S2=10$^{aa1}$ PE, dt=100 $\mu$s')
        plt.plot(x2w, uppper_lim_with_wire(x1 * s2_width_model(100e3,*mod_par),
                                               x1 * f_upper_boundary(aa2,*fit_par[1]), 
                                               x1 * 10**aa2 * 0.75, x2w)/1e3,
                     color='b',ls=':',label=f'cut boundary with S2=10$^{aa2}$ PE, dt=100 $\mu$s')
        plt.ylabel('S2 width - model ($\mu$s)', ha='right', y=1)
        plt.xlabel("distance from wire (cm)", ha='right', x=1)
        plt.legend(loc='upper right')
        plt.title(f'{title}')
        if name is not None: plt.savefig(name+'_s2widthcut_wire_model.png',dpi=600)

def f_lower_boundary(x, p0, p1, p2):
    return p0*0.5 * special.erf((np.sqrt(2) * (x - p1)) / p2) + p0*0.5

def f_upper_boundary(x, p0, p1, p2, p3):
    return p0*np.exp(-p1*(x-p2)) + p3

def parabola(x, p0, p1, p2):
    return p0*x**2 + p1*x + p2

def all_cuts(data, s2_cut = 200, cs1_cut = None, FV = True, near_wires = False, far_wires = False,
             AmBe = False, AC = False, analysis = False):
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
              'cut_position_shadow']
              #'cut_s1_naive_bayes',
              #'cut_s2_naive_bayes']
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
          'cut_s2_recon_pos_diff']
          #'cut_nv_tpc_coincidence_ambe' ]
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
    cut_list_analysis=['cut_interaction_exists',
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
          'cut_s2_recon_pos_diff',
          'cut_ambience', 
          'cut_time_veto', 
          'cut_time_shadow', 
          'cut_position_shadow',
          'cut_s1_naive_bayes', 
          'cut_s2_naive_bayes',
         ]
    cut = np.ones(len(data), dtype=bool)
    if FV: cut &= data['cut_fiducial_volume']
    if AmBe: cut_list = cut_list_ambe
    if AC: cut_list = cut_list_AC
    if analysis: cut_list = cut_list_analysis
    for cut_ in cut_list:
        cut &= data[cut_]
    if cs1_cut is not None: cut &= data['cs1'] < cs1_cut
    cut &= data['s2_area'] > s2_cut
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
    
def get_ces(cs1, cs2):
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
    
def get_acceptance(df, cut_mask, title, bins = 50, alim=(0.9,1), unc_method="cp", Ar37 = False, sim = False, name = None):
    if sim: df['e_ces'] = get_ces(df['cs1'], df['cs2'])
    #ces = get_ces(df['cs1'], df['cs2'])
    #df['r2'] = df["r"]**2
    #df['s2_x_rotated'] = rotated_x(df["s2_x"], df["s2_y"])
    cut_s2 = df['s2_area'] < 1e4
    #param_list = ('ces','s1_area','s2_area','r2','s2_x_rotated')
    #name_list = ('CES (keV)','S1 area (PE)','S2 area (PE)','R$^2$ (cm$^2$)','S2 x rotated (cm)')
    #col_list = ('b','g','r','c','m')
    #boundary_list = ((0,30), (0,200), (0,10000), (0,3600), (-57,57))
    #data = pd.DataFrame(columns=('s1_area','s1_area_acc','s2_area','s2_area_acc','z','z_acc'))
    
    #param_list = ('s1_area','s2_area','z')
    #name_list = ('S1 area (PE)','S2 area (PE)','z (cm)')
    #col_list = ('g','r','m')
    #boundary_list = ((0,185), (200,10000), (-142,-5))
    param_list = ('e_ces','s2_area','cs2','z')
    name_list = ('CES (keV)','S2 area (PE)','cS2 (PE)','z (cm)')
    data = pd.DataFrame(columns=('e_ces','e_ces_acc','s2_area','s2_area_acc','cs2','cs2_acc','z','z_acc'))
    col_list = ('b','r','r','m')
    #s2 = np.linspace(625.0, 24875.0, 98)
    #boundary_list = ((0,140), (200, 15000), (200, 15000), (-142,-5))
    spaces = ((np.linspace(1, 140, bins)),(np.linspace(200, 10000, bins)),(np.linspace(200, 10000, bins)),(np.linspace(-142, -5, bins)))
    if Ar37: spaces = ((np.linspace(2.7, 2.9, bins)),(np.linspace(200, 10000, bins)),(np.linspace(200, 10000, bins)),(np.linspace(-142, -5, bins)))
    for i, param_name in enumerate(param_list):
        #xspace = np.linspace(*boundary_list[i],bins[i])
        #if param_name == 'ces':
        #    h_bef = Hist1d(ces[cut_s2], bins=spaces[i])
        #    h_aft = Hist1d(ces[cut_mask & cut_s2], bins=spaces[i])
        #else:
        h_bef = Hist1d(df[cut_s2][param_name], bins=spaces[i])
        h_aft = Hist1d(df[cut_mask & cut_s2][param_name], bins=spaces[i])
        h_acc_l, h_acc_u = get_lower_upper_bound(h_aft, h_bef, unc_method)
        h_acc = h_aft / h_bef
        #acc_mean = np.mean(h_acc.histogram[np.isfinite(h_acc.histogram)])
        acc_mean = len(df[cut_mask&cut_s2])/len(df[cut_s2])
        plt.figure(figsize=(8,4.5))
        plt.errorbar(h_acc.bin_centers, h_acc.histogram,yerr=(h_acc_l.histogram, h_acc_u.histogram),
                        color=col_list[i],fmt="s", ms=2, capsize=3,label='acceptance vs '+param_name)
        #plt.axhline(y=acc_mean, color='k', linestyle='--',label=f'Tot. Acc. = {acc_mean*100:.2f}%')
        #if param_name == 'ces':
        #    plt.axvline(x=1, color='r', linestyle='--',label='threshold 1 keV')
        plt.xlabel(name_list[i], ha='right', x=1)
        plt.ylabel("S2WidthCut Acceptance", ha='right', y=1)
        plt.ylim(alim[0],alim[1])
        #plt.title(f'{title} Total Acceptance = {acc_mean:.2f}%',fontsize=16)
        plt.title(f'{title}',fontsize=16)
        plt.legend(fontsize = 14)
        data[param_name] = h_acc.bin_centers
        data[param_name+'_acc'] = h_acc.histogram
        data[param_name+'_acc_err_l'] = h_acc_l.histogram
        data[param_name+'_acc_err_h'] = h_acc_u.histogram
        if name is not None: plt.savefig(name+param_list[i]+'.png',dpi=600)
    return data

def get_acceptance_2d(df, cut_mask, title, bins = 50, xlim = 75, zlim = (-150,0), rlim = (0,70), vmin=0.7, name = None):
    cut_s2 = df['s2_area'] < 3e4
    xspace = np.linspace(xlim*-1, xlim, bins)
    zspace = np.linspace(zlim[0], zlim[1], bins)
    rspace = np.linspace(rlim[0], rlim[1], bins)
    
    # XY    
    xy_bef = Histdd(df[cut_s2]['x'],df[cut_s2]['y'], bins=(xspace,xspace) )
    xy_aft = Histdd(df[cut_s2 & cut_mask]['x'],df[cut_s2 & cut_mask]['y'], bins=(xspace,xspace) )
    xy_acc = xy_aft / xy_bef
    plt.figure(figsize=(8,6))
    xy_acc.plot(cblabel='acceptance',vmin=vmin)
    plt.xlabel("x (cm)", ha='right', x=1)
    plt.ylabel("y (cm)", ha='right', y=1)
    plt.title(f'{title}',fontsize=16)
    if name is not None: plt.savefig('acc_2d_xy_'+name+'.png',dpi=600)
    # RZ
    rz_bef = Histdd(df[cut_s2]['r'], df[cut_s2]['z'], bins=(rspace,zspace) )
    rz_aft = Histdd(df[cut_s2 & cut_mask]['r'], df[cut_s2 & cut_mask]['z'], bins=(rspace,zspace) )
    rz_acc = rz_aft / rz_bef
    plt.figure(figsize=(8,6))
    rz_acc.plot(cblabel='acceptance',vmin=vmin)
    plt.xlabel("r (cm)", ha='right', x=1)
    plt.ylabel("z (cm)", ha='right', y=1)
    plt.title(f'{title}',fontsize=16)
    if name is not None: plt.savefig('acc_2d_rz_'+name+'.png',dpi=600)
    return xy_acc

def get_acceptance_kr(df, cut_mask, title, bins = 70, alim=(0.9,1), unc_method="cp", plot= False, name = None):
    cut_s2 = df['s2_a_area'] < 1e4
    param_list = ('e_ces','s2_a_area','z')
    name_list = ('CES (keV)','S2 area (PE)','z (cm)')
    data = pd.DataFrame(columns=('e_ces','e_ces_acc','s2_a_area','s2_a_area_acc','z','z_acc'))
    col_list = ('b','r','m')
    spaces = (np.linspace(37,47, bins),(np.linspace(200, 10000, bins)),(np.linspace(-142, -5, bins)))
    for i, param_name in enumerate(param_list):
        h_bef = Hist1d(df[cut_s2][param_name], bins=spaces[i])
        h_aft = Hist1d(df[cut_mask & cut_s2][param_name], bins=spaces[i])
        h_acc_l, h_acc_u = get_lower_upper_bound(h_aft, h_bef, unc_method)
        h_acc = h_aft / h_bef
        #acc_mean = np.mean(h_acc.histogram[np.isfinite(h_acc.histogram)])
        acc_mean = len(df[cut_mask&cut_s2])/len(df[cut_s2])
        if plot:
            plt.figure(figsize=(8,4.5))
            plt.errorbar(h_acc.bin_centers, h_acc.histogram,yerr=(h_acc_l.histogram, h_acc_u.histogram),
                        color=col_list[i],fmt="s", ms=2, capsize=3)
            plt.axhline(y=acc_mean, color='k', linestyle='--',label=f'Tot. Acc. = {acc_mean*100:.2f}%')
            plt.xlabel(name_list[i], ha='right', x=1)
            plt.ylabel("survival fraction", ha='right', y=1)
            plt.ylim(alim[0],alim[1])
            plt.title(f'{title}')
            plt.legend()
        data[param_name] = h_acc.bin_centers
        data[param_name+'_acc'] = h_acc.histogram
        data[param_name+'_acc_err_l'] = h_acc_l.histogram
        data[param_name+'_acc_err_h'] = h_acc_u.histogram
        if name is not None: plt.savefig(name+'_'+param_list[i]+'.png',dpi=600)
    return data

def merge_runs_cutax(st,runs):
    ev0 = st.get_df(runs[0],['event_info',
                             'cut_interaction_exists',
                             'cut_main_is_valid_triggering_peak', 
                             'cut_daq_veto',
                             'cut_run_boundaries',
                             'cut_fiducial_volume',
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
                             'cut_s2_width_wire_modeled_low_er',
                            ],
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
                             'cut_fiducial_volume',
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

def get_x_to_wire(ev,kr=False):
    wire_median_displacement = 13.06
    if kr: s2_x_mlp, s2_y_mlp = ev['s2_a_x_mlp'], ev['s2_a_y_mlp']
    else: s2_x_mlp, s2_y_mlp = ev['s2_x_mlp'], ev['s2_y_mlp']
    return np.abs(s2_x_mlp * np.cos(-np.pi/6) + s2_y_mlp * np.sin(-np.pi/6)) - wire_median_displacement

def get_model_pred(ev):
    vd = get_correction_from_cmt('024075',('electron_drift_velocity', 'ONLINE', True))
    gd = get_correction_from_cmt('024075',('electron_drift_time_gate', 'ONLINE', True))
    dc = get_correction_from_cmt('024075',('electron_diffusion_cte', 'ONLINE', True))
    par = (dc,vd,gd)
    drift = ev['drift_time']
    model_pred = s2_width_model(drift, *par)
    return model_pred

def get_diff_width(ev,kr=False):
    if kr: s2width = ev['s2_a_range_50p_area']
    else: s2width = ev['s2_range_50p_area']
    drift = ev['drift_time']
    model_pred = get_model_pred(ev)
    return s2width - model_pred

def get_norm_width(ev,kr=False):
    vd = get_correction_from_cmt('024075',('electron_drift_velocity', 'ONLINE', True))
    gd = get_correction_from_cmt('024075',('electron_drift_time_gate', 'ONLINE', True))
    dc = get_correction_from_cmt('024075',('electron_diffusion_cte', 'ONLINE', True))
    par = (dc,vd,gd)
    if kr: s2width = ev['s2_a_range_50p_area']
    else: s2width = ev['s2_range_50p_area']
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
    
    plt.figure(figsize=(8,4.5))
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
        plt.figure(figsize=(8,4.5))
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
        
        plt.figure(figsize=(8,4.5))
        #pha.plot(log_scale=True, cblabel='events',colorbar=False,alpha=0.2)
        ph_cut.plot(log_scale=True, cblabel='events',colorbar=True)
        plt.scatter(np.log10(ev_rej['s2_area']), get_norm_width(ev_rej),c='r',
                    s=5,cmap='Wistia',label='event rejected')
        plt.plot(ac, low_bound(ac),'b-',label='lower boundary')
        plt.legend(fontsize=14,loc='upper right')
        plt.xlabel('log10(S2 area (PE))', ha='right', x=1,fontsize=12)
        plt.ylabel("Normalized S2 width 50% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'{title}',fontsize=14)
        
        plt.figure(figsize=(8,4.5))
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
    
    plt.figure(figsize=(8,4.5))
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
        plt.figure(figsize=(8,4.5))
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


def set_cut_on_data(ev, par, width_type = '50p', wrange = (0,2), plot = False, wire_model=False, near_wires = False, far_wires = False,
                     perc =(1,99), afit=(2,3.6), acc_calc = False, alim=(0.8,1), info = None):
    #mask_near_wires = NearWires.cut_by(ev_sim)
    if near_wires:
        print('Select events close to perpendicular wires')
        ev = ev[ev['cut_near_wires']]
    if far_wires:
        print('Select events close to perpendicular wires')
        ev = ev[ev['cut_near_wires']==False]
    if info is not None: name = f'cut_SR0_sim_{nome}_{width_type}_p{perc[0]}_p{perc[1]}_'
    else: name = None
    cut_sim = S2WidthCut(ev, title=info, mod_par=par, perc=perc, bins=400, perc_plot=0,near_wires = near_wires,
                           wrange = wrange, real_data = 0, #ev_leak=ev_leak
                           plot = plot, arange = (2, 4.5), afit=afit,wire_model=wire_model,
                           width_type = width_type, name=name)
    if acc_calc: data_acc = get_acceptance(ev, cut_sim[1], title=info, bins = 40, alim=alim, name = name)
    return cut_sim[2], data_acc


def apply_cut_on_data(ev, par, cut_par, width_type = '50p', near_wires = False, far_wires = False, wrange = (0,2),
                      acc_calc = False, alim=(0.8,1), info = None, plot = False, wire_model=False):
    if near_wires: ev = ev[ev['cut_near_wires']]
    if far_wires: ev = ev[ev['cut_near_wires']==False]
    if info is not None: name = f'cut_SR0_{info}_{width_type}_'
    else: name = None
    cut_data = S2WidthCut(ev, title=info, mod_par=par, perc_plot=0, bins=400,
                              wrange=wrange, near_wires = near_wires, #ev_leak=ev_leak
                              plot = plot, arange = (2.2,6.5),wire_model=wire_model,
                              width_type=width_type, name=name,
                              ext_par=cut_par)
    if acc_calc: data_acc = get_acceptance(ev, cut_data[1], title=info, bins = 40, alim=alim, name = name)
    return data_acc, cut_data[1]


def plot_acceptance(acc_data, acc_sim = None, acc_ar = None, acc_kr = None, alim=(0.8,1),title='',info='data',save=False):
    param_list = ('e_ces','s2_area','cs2','z')
    col_list = ('b','r','m','g')
    name_list = ('CES (keV)','S2 area (PE)','cS2 (PE)','z (cm)')
    for i, param_name in enumerate(param_list):
        plt.figure(figsize=(8,4.5))
        plt.errorbar(acc_data[param_name], acc_data[param_name+'_acc'],c='r',
                     yerr=(acc_data[param_name+'_acc_err_l'], acc_data[param_name+'_acc_err_h']),
                     fmt="s", ms=2, capsize=3,label=info)
        if (param_name != 'e_ces') & (acc_sim is not None):
            plt.errorbar(acc_sim[param_name], acc_sim[param_name+'_acc'],c='g',
                         yerr=(acc_sim[param_name+'_acc_err_l'], acc_sim[param_name+'_acc_err_h']),
                         fmt="s", ms=2, capsize=3,label='WFsim')
        if (param_name == 'e_ces') & (acc_ar is not None):
            plt.errorbar(acc_ar[param_name], acc_ar[param_name+'_acc'],c='b',
                         yerr=(acc_ar[param_name+'_acc_err_l'], acc_ar[param_name+'_acc_err_h']),xerr=([1.2], [1.2]),
                         fmt="s", ms=2, capsize=3,label='Ar37')
        if (param_name == 'e_ces') & (acc_kr is not None):
            plt.errorbar(acc_kr[param_name], acc_kr[param_name+'_acc'],c='m',
                         yerr=(acc_kr[param_name+'_acc_err_l'], acc_kr[param_name+'_acc_err_h']),xerr=([1.2], [1.2]),
                         fmt="s", ms=2, capsize=3,label='Kr83m')
            plt.xscale('log')
        acc_mean = np.mean(acc_data[param_name+'_acc'][np.isfinite(acc_data[param_name+'_acc'])])
        #plt.axhline(y=acc_mean, color='k', linestyle='--',label=f'Tot. Acc. = {acc_mean*100:.2f}%')
        if param_name == 'ces':
            plt.axvline(x=1, color='r', linestyle='--',label='threshold 1 keV')
        plt.xlabel(name_list[i], ha='right', x=1,fontsize=12)
        plt.ylabel("Acceptance", ha='right', y=1,fontsize=12)
        plt.ylim(alim[0],alim[1])
        plt.legend(fontsize = 14,loc='lower right')
        plt.title(title,fontsize = 14)
        if save: plt.savefig('acc_'+param_list[i]+'_'+title+'.png',dpi=600)
        #acc_diff = np.abs(acc_data[param_name+'_acc']-acc_sim[param_name+'_acc'])
        """plt.figure(figsize=(12,6))
        plt.errorbar(acc_data[param_name], acc_diff,
                     yerr=(acc_sim[param_name+'_acc_err_l'], acc_sim[param_name+'_acc_err_h']),
                     color=col_list[i],fmt="s", ms=2, capsize=3,label='acceptance difference')
        acc_mean = np.mean(acc_data[param_name+'_acc'][np.isfinite(acc_data[param_name+'_acc'])])
        #plt.axhline(y=acc_mean, color='k', linestyle='--',label=f'Tot. Acc. = {acc_mean*100:.2f}%')
        plt.xlabel(param_list[i], ha='right', x=1,fontsize=12)
        plt.ylabel("Acceptance Difference", ha='right', y=1,fontsize=12)
        plt.legend(fontsize = 14)"""
        
        """
        plt.figure(figsize=(12,6))
        err_hh = acc_sim[param_name+'_acc_err_h']+acc_diff
        for j, a in enumerate(np.array(acc_sim[param_name+'_acc']+err_hh)):
            if a > 1: err_hh[j] = 1 - acc_sim[param_name+'_acc'][j]
        plt.errorbar(acc_sim[param_name], acc_sim[param_name+'_acc'],
                     yerr=(acc_sim[param_name+'_acc_err_l']+acc_diff, err_hh),
                     color=col_list[i],fmt="s", ms=2, capsize=3,label='acceptance')
        acc_mean = np.mean(acc_data[param_name+'_acc'][np.isfinite(acc_data[param_name+'_acc'])])
        #plt.axhline(y=acc_mean, color='k', linestyle='--',label=f'Tot. Acc. = {acc_mean*100:.2f}%')
        plt.xlabel(param_list[i], ha='right', x=1,fontsize=12)
        plt.ylabel("Acceptance", ha='right', y=1,fontsize=12)
        plt.ylim(alim[0],alim[1])
        plt.legend(fontsize = 14,loc='lower right')
        plt.title(title,fontsize = 14)"""