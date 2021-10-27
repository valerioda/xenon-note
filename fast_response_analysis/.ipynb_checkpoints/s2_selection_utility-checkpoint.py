import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.integrate import quad
import scipy.special as sc
from scipy.integrate import odeint
import scipy.integrate as integ
from scipy.optimize import curve_fit
from scipy.special import erf, erfc, gammaln
import time

import socket 
import strax
import straxen
from multihist import Hist1d, Histdd

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']

def select_peaks_times(st, run_id, peaks, events, ndiv = 10, div = 0, s2_start = 0):
    area = []
    width = []
    areachn = []
    aft = []
    s1_times = []
    s1_endtimes = []
    s2_times = []
    s2_endtimes = []
    drift_times = []
    r_data = pd.DataFrame(columns=['area','range_50p_area','area_per_channel','area_fraction_top',
                                  's1_time','s1_endtime','s2_time','s2_endtime','drift_time'])
    
    t_start = time.time()
    tsec = (peaks['endtime'][-1]-peaks['time'][0])/1e9
    csec = tsec/ndiv
    print('run',run_id,'Total events',len(events),'total peaks',len(peaks),'duration',tsec)
    k = s2_start
    plt.figure(k,figsize=(12,6))
    #for i in range(nd):
    pp = st.get_array(run_id,['peaks','peak_basics'],
                  seconds_range=(csec*div,csec*(div+1)),progress_bar=False)
    nn = len(pp)
    dt = pp['dt'][0]
    dts = np.arange(0,pp['data'].shape[1]*dt,dt)
    ft = pp['endtime'][-1]
    for j in range(nn):
        t = pp['time'][j]
        if t >= ft or k == len(events): break
        elif events['s2_time'][k]==0:
            area.append(0)
            width.append(0)
            areachn.append(0)
            aft.append(0)
            s1_times.append(events['s1_time'][k])
            s1_endtimes.append(events['s1_endtime'][k])
            s2_times.append(events['s2_time'][k])
            s2_endtimes.append(events['s2_endtime'][k])
            drift_times.append(events['drift_time'][k])
            k += 1
        elif t == events['s2_time'][k] and k < len(events)-1:
            area.append(pp['area'][j])
            width.append(pp['range_50p_area'][j])
            areachn.append(pp['area_per_channel'][j])
            aft.append(pp['area_fraction_top'][j])
            s1_times.append(events['s1_time'][k])
            s1_endtimes.append(events['s1_endtime'][k])
            s2_times.append(events['s2_time'][k])
            s2_endtimes.append(events['s2_endtime'][k])
            drift_times.append(events['drift_time'][k])
            if (k<s2_start+10):
                plt.plot( dts,pp['data'][j],drawstyle='steps',label=f'{k} t = {t}')
                plt.xlabel("time (ns)", ha='right', x=1,fontsize=12)
                plt.ylabel(f"PE", ha='right', y=1,fontsize=12)
                plt.legend(fontsize=12)
            k += 1
    diff = time.time() - t_start
    print(f'division n. {div}, tot. events {nn}, selected {len(area)}, tot.sel. {k} time to process {diff:.2f} s')
    del pp
    r_data['area'] = area
    r_data['range_50p_area'] = width
    r_data['area_per_channel'] = areachn
    r_data['area_fraction_top'] = aft
    r_data['s1_time'] = s1_times
    r_data['s1_endtime'] = s1_endtimes
    r_data['s2_time'] = s2_times
    r_data['s2_endtime'] = s2_endtimes
    r_data['drift_time'] = drift_times
    r_data.to_hdf(f'data/select_peaks_run{run_id}_div{div}.h5', key='df', mode='w')
    return r_data, k


def select_records_times(st, run_id, peaks, events, pdata, PMTs, ndiv = 10, div = 0, s2_start = 0):
    area = []
    width = []
    area_per_channel = []
    aft = []
    channel = []
    tempo = []
    et = []
    ev = []
    wf = []
    dt = []
    s1_times = []
    s1_endtimes = []
    s2_times = []
    s2_endtimes = []
    drift_times = []
    r_data = pd.DataFrame(columns=['event_number','event_time','time','channel',
                                   'area','range_50p_area','area_per_channel','area_fraction_top','data','dt',
                                   's1_time','s1_endtime','s2_time','s2_endtime','drift_time'])
    
    areas = np.array(pdata['area'])
    widths = np.array(pdata['range_50p_area'])
    afts = np.array(pdata['area_fraction_top'])
    area_chn = np.array(pdata['area_per_channel'])
    
    s1_t = np.array(pdata['s1_time'])
    s1_et = np.array(pdata['s1_endtime'])
    s2_t = np.array(pdata['s2_time'])
    s2_et = np.array(pdata['s2_endtime'])
    d_t = np.array(pdata['drift_time'])
    
    t_start = time.time()
    tsec = (peaks['endtime'][-1]-peaks['time'][0])/1e9
    csec = tsec/ndiv
    print('run',run_id,'Total events',len(events),'total peaks',len(peaks),'duration',tsec)
    k, kk = s2_start, 0
    #for ii in range(1):
    rr = st.get_array(run_id,'records',
                        seconds_range=(csec*div,csec*(div+1)),progress_bar=False)
    nr = len(rr)
    dts = np.arange(0,rr['data'].shape[1]*rr['dt'][0],rr['dt'][0])
    ft = rr['time'][-1] + rr['dt'][0]*rr['length'][-1]
    for j in range(nr):
        t = rr['time'][j]
        if t >= ft or k == len(events): break
        elif events['s2_time'][k] == 0: k += 1
        elif t >= events['s2_time'][k] and kk < len(pdata):
            jj = j
            while rr['time'][jj] < events['s2_endtime'][k]:
                for i, PMT in enumerate(PMTs):
                    chn = rr['channel'][jj]
                    areai = rr['area'][jj]
                    if chn == PMT and rr['record_i'][jj] > 0:
                        ev.append(k)
                        et.append(events['s2_time'][k])
                        channel.append(chn)
                        tempo.append(rr['time'][jj])
                        area.append(areas[kk])
                        width.append(widths[kk])
                        aft.append(afts[kk])
                        try: area_channel = area_chn[kk][chn]
                        except: area_channel = 0
                        area_per_channel.append(area_channel)
                        wf.append(rr['data'][jj])    
                        dt.append(rr['dt'][jj])
                        
                        s1_times.append(s1_t[kk])
                        s1_endtimes.append(s1_et[kk])
                        s2_times.append(s2_t[kk])
                        s2_endtimes.append(s2_et[kk])
                        drift_times.append(d_t[kk])
                jj += 1
            k += 1
            kk += 1
    diff = time.time() - t_start
    print(f'division n. {div}, tot. events {nr}, selected events: {len(ev)} {k}, time to process: {diff:.2f} s')
    del rr
    r_data['event_number'] = ev
    r_data['event_time'] = et
    r_data['time'] = tempo
    r_data['channel'] = channel
    r_data['data'] = wf
    r_data['dt'] = dt
    
    r_data['area'] = area
    r_data['range_50p_area'] = width
    r_data['area_fraction_top'] = aft
    r_data['area_per_channel'] = area_per_channel
    
    r_data['s1_time'] = s1_times
    r_data['s1_endtime'] = s1_endtimes
    r_data['s2_time'] = s2_times
    r_data['s2_endtime'] = s2_endtimes
    r_data['drift_time'] = drift_times
    try: os.mkdir('./data')
    except: pass
    r_data.to_hdf(f'data/select_records_run{run_id}_div{div}.h5', key='df', mode='w')
    return r_data


def merge_records(run_id, PMTs, ndiv = 10, path='./data'):
    ev = []
    et = []
    tempo = []
    channel = []
    area = []
    area_chn = []
    width = []
    width50 = []
    width90 = []
    aft = []
    wf = []
    dt = []
    position_x = []
    position_y = []
    s1_times = []
    s1_endtimes = []
    s2_times = []
    s2_endtimes = []
    drift_times = []
    r_data = pd.DataFrame(columns=['event_number','event_time','time','channel','area','area_per_channel','area_fraction_top',
                                   'range_50p_area','width50','width90','data','dt','position_x','position_y',
                                   's1_time','s1_endtime','s2_time','s2_endtime','drift_time'])
    positions = straxen.pmt_positions()
    t_start = time.time()
    for div in range(ndiv):
        filename = f'{path}/select_records_run{run_id}_div{div}.h5'
        rdata = pd.read_hdf(filename)
        lev = rdata['event_number'][rdata.last_valid_index()]
        fev = rdata['event_number'][rdata.first_valid_index()]
        print(f'Reading: {filename}, initial records: {len(rdata)}, events: {lev-fev}')
        for k in range(fev,lev):
            if k % 1000 == 0 or k == fev: print(f'{k}, time to process: {time.time()-t_start:.2f}')
            area_sum, rec_x, rec_y = 0, 0, 0
            for i, PMT in enumerate(PMTs):
                rr = rdata[(rdata['event_number']==k) & (rdata['channel']==PMT)]
                idx = rr.first_valid_index()
                if idx is None: break
                tt = rr['time'][idx] - rr['event_time'][idx]
                wftot = []
                for ww in rr['data']:
                    wftot.append(ww)
                wftot = np.concatenate(wftot)
                ev.append(rr['event_number'][idx])
                et.append(rr['event_time'][idx])
                channel.append(rr['channel'][idx])
                tempo.append(rr['time'][idx])
                area_channel = rr['area_per_channel'][idx]
                area_chn.append(area_channel)
                width.append(rr['range_50p_area'][idx])
                area.append(rr['area'][idx])
                aft.append(rr['area_fraction_top'][idx])
                wf.append(wftot)
                dt.append(rr['dt'][idx])
                s1_times.append(rr['s1_time'][idx])
                s1_endtimes.append(rr['s1_endtime'][idx])
                s2_times.append(rr['s2_time'][idx])
                s2_endtimes.append(rr['s2_endtime'][idx])
                drift_times.append(rr['drift_time'][idx])
                # width calculation
                ii, areafrac = 1, 0.1
                ilo50, ihi50 = int(wftot.argmax()-ii), int(wftot.argmax()+ii)
                areatot = np.sum(wftot[:])
                while areafrac < 0.5 and areafrac > 0 and areatot > 0:
                    #ilo, ihi = int(len(wftot)/2-ii), int(len(wftot)/2+ii)
                    ilo50, ihi50 = int(wftot.argmax()-ii), int(wftot.argmax()+ii)
                    if ilo50 < 0: ilo50 = 0
                    if ihi50 > len(wftot): ihi50 = len(wftot)-1
                    if ilo50 == 0 and ihi50 == len(wftot)-1: break
                    areafrac = np.sum(wftot[ilo50:ihi50])/areatot
                    ii += 1
                wid50 = (ihi50 - ilo50)*rr['dt'][idx]
                width50.append(wid50)
                
                ii, areafrac = 1, 0.1
                ilo90, ihi90 = int(wftot.argmax()-ii), int(wftot.argmax()+ii)
                while areafrac < 0.9 and areafrac > 0 and areatot > 0:
                    ilo90, ihi90 = int(wftot.argmax()-ii), int(wftot.argmax()+ii)
                    if ilo90 < 0: ilo90 = 0
                    if ihi90 > len(wftot): ihi90 = len(wftot)-1
                    if ilo90 == 0 and ihi90 == len(wftot)-1: break
                    areafrac = np.sum(wftot[ilo90:ihi90])/areatot
                    ii += 1
                wid90 = (ihi90 - ilo90)*rr['dt'][idx]
                width90.append(wid90)
                
                # position reconstruction
                pos_x = np.float(positions['x'][positions['i']==PMT])
                pos_y = np.float(positions['y'][positions['i']==PMT])
                rec_x += area_channel * pos_x
                rec_y += area_channel * pos_y
                area_sum += area_channel
            try:
                rec_x /= area_sum
                rec_y /= area_sum
            except:
                rec_x = 0
                rec_y = 0
            
            for i, PMT in enumerate(PMTs):
                rr = rdata[(rdata['event_number']==k) & (rdata['channel']==PMT)]
                idx = rr.first_valid_index()
                if idx is None: break
                position_x.append(rec_x)
                position_y.append(rec_y)
                
        print('Merged records',len(ev))
    r_data['event_number'] = ev
    r_data['event_time'] = et
    r_data['time'] = tempo
    r_data['channel'] = channel
    r_data['area'] = area
    r_data['area_per_channel'] = area_chn
    r_data['area_fraction_top'] = aft
    r_data['range_50p_area'] = width
    r_data['width50'] = width50
    r_data['width90'] = width90
    r_data['data'] = wf
    r_data['dt'] = dt
    r_data['position_x'] = position_x
    r_data['position_y'] = position_y
    r_data['s1_time'] = s1_times
    r_data['s1_endtime'] = s1_endtimes
    r_data['s2_time'] = s2_times
    r_data['s2_endtime'] = s2_endtimes
    r_data['drift_time'] = drift_times
    r_data.to_hdf(f'{path}/merged_records_run{run_id}.h5', key='df', mode='w')
    return r_data


def position_reconstruction(run_id, mdata, PMTs, path='./data'):
    positions = straxen.pmt_positions()
    nev = mdata['event_number'][mdata.last_valid_index()]
    rec_x, rec_y, area_sum  = np.zeros(nev), np.zeros(nev), np.zeros(nev)
    position_x = []
    position_y = []
    pos_data = pd.DataFrame(columns=['position_x','position_y'])
    t_start = time.time()
    for k in range(nev):
        area_sum = 0
        for i, PMT in enumerate(PMTs):
            rr = mdata[(mdata['event_number']==k) & (mdata['channel']==PMT)]
            try: area_chn = np.float(rr['area_per_channel'])
            except: break
            pos_x = np.float(positions['x'][positions['i']==PMT])
            pos_y = np.float(positions['y'][positions['i']==PMT])
            #print(k,'PMT n.',PMT,'area',area_chn,' PE position',pos_x,pos_y,'cm')
            rec_x[k] += area_chn * pos_x
            rec_y[k] += area_chn * pos_y
            area_sum += area_chn
        rec_x[k] /= area_sum
        rec_y[k] /= area_sum
        position_x.append(rec_x[k])
        position_y.append(rec_y[k])
        if k % 1000 == 0: print(f'{k}, pos {rec_x[k]} {rec_y[k]} cm, time: {time.time()-t_start:.2f} s')
    pos_data['position_x'] = position_x
    pos_data['position_y'] = position_y
    pos_data.to_hdf(f'{path}/position_reconstruction_run{run_id}.h5', key='df', mode='w')
    return pos_data, rec_x, rec_y


def width_distribution_slice(run_id, PMTs, area_lim = 1e4, width_lim = 1e4, nslices = 10, plot = False):
    mdata = pd.read_hdf(f'data/merged_records_run{run_id}.h5')
    media = np.zeros((len(PMTs),nslices))
    stand = np.zeros((len(PMTs),nslices))
    median = np.zeros((len(PMTs),nslices))
    arear = np.zeros((len(PMTs),nslices))
    for i, PMT in enumerate(PMTs):
        mdata1 = mdata[(mdata['channel']==PMT)]
        print('PMT',PMT,'events',len(mdata1))
        binn= int(width_lim/20)
        if plot: plt.figure(figsize=(12,6))
        for ii in range(nslices):
            alo, ahi = area_lim/nslices*ii, area_lim/nslices*(ii+1)
            mdata2 = mdata1[(mdata1['area_per_channel']>alo)&(mdata1['area_per_channel']<ahi)]
            media[i][ii] = mdata2['width90'].mean()
            stand[i][ii] = mdata2['width90'].std()/np.sqrt(len(mdata2))
            median[i][ii] = mdata2['width90'].median()
            arear[i][ii] = ahi
            #print(PMT,ii,width,median,)
            if plot:
                ph = Hist1d(mdata2['width90'],bins=(np.linspace(0, width_lim, binn)))
                ph.plot(label=f'area<{ahi} PE, mean ${media[i][ii]:.0f} \pm {stand[i][ii]:.0f}$ ns')
                #plt.axvline(x=width,color=colors[ii],linestyle='-')
                #plt.axvline(x=median,color=colors[ii],linestyle='--')
                plt.ylabel("events", ha='right', y=1,fontsize=12)
                plt.xlabel("peak width 90% (ns)", ha='right', x=1,fontsize=12)
                plt.title(f'run {run_id} PMT {PMT}',fontsize=14)
                plt.legend(fontsize=12)
    plt.figure(figsize=(12,6))
    for i, PMT in enumerate(PMTs):
        plt.errorbar(arear[i],media[i],yerr=stand[i],label=f'PMT {PMT}')
        plt.xlabel("PMT area (PE)", ha='right', x=1,fontsize=12)
        plt.ylabel("PMT width 90% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'run {run_id} PMT {PMT}',fontsize=14)
        plt.legend(fontsize=14)
    return media, stand

def plot_positions_area(run_id,PMTs):
    mdata = pd.read_hdf(f'data/merged_records_run{run_id}.h5')
    positions = straxen.pmt_positions()
    for i, PMT in enumerate(PMTs):
        mdata1 = mdata[mdata['channel']==PMT]
        pos_x0 = np.float(positions['x'][positions['i']==PMTs[0]])
        pos_y0 = np.float(positions['y'][positions['i']==PMTs[0]])
        pos_x1 = np.float(positions['x'][positions['i']==PMTs[1]])
        pos_y1 = np.float(positions['y'][positions['i']==PMTs[1]])
        pos_x2 = np.float(positions['x'][positions['i']==PMTs[2]])
        pos_y2 = np.float(positions['y'][positions['i']==PMTs[2]])
        plt.figure(figsize=(12,6))
        ph50x = Histdd(mdata1['width90'], mdata1['position_x'],
                      bins=(np.linspace(0, 10000, 500), np.linspace(-10, 35, 200)))
        ph50x.plot(log_scale=True, cblabel='events')
        plt.axhline(pos_x0,color='r',label=f'x position of PMT {PMTs[0]}')
        plt.axhline(pos_x1,color='b',label=f'x position of PMT {PMTs[1]}')
        plt.axhline(pos_x2,color='y',label=f'x position of PMT {PMTs[2]}')
        plt.xlabel("peak width 90% (ns)", ha='right', x=1,fontsize=12)
        plt.ylabel("position x (cm)", ha='right', y=1,fontsize=12)
        plt.title(f'PMT {PMT}',fontsize=14)
        plt.legend(fontsize=14,loc='upper right')
    
        ph50y = Histdd(mdata1['width90'], mdata1['position_y'],
                      bins=(np.linspace(0, 10000, 500), np.linspace(-15, 35, 200)))
        plt.figure(figsize=(12,6))
        ph50y.plot(log_scale=True, cblabel='events')
        plt.axhline(pos_y0,color='r',label=f'y position of PMT {PMTs[0]}')
        plt.axhline(pos_y1,color='b',label=f'y position of PMT {PMTs[1]}')
        plt.axhline(pos_y2,color='y',label=f'y position of PMT {PMTs[2]}')
        plt.xlabel("peak width 90% (ns)", ha='right', x=1,fontsize=12)
        plt.ylabel("position y (cm)", ha='right', y=1,fontsize=12)
        plt.title(f'run {run_id} PMT {PMT}',fontsize=14)
        plt.legend(fontsize=14,loc='upper right')
        
        
        
def plot_records(run_id,PMTs,n):
    mdata = pd.read_hdf(f'data/merged_records_run{run_id}.h5')
    for k in range(n):
        plt.figure(k,figsize=(12,8))
        for i, PMT in enumerate(PMTs):
            rr = mdata[(mdata['event_number']==k) & (mdata['channel']==PMT)]
            idx = rr.first_valid_index()
            if idx is None: break
            tt = rr['time'][idx] - rr['event_time'][idx]
            wf = mdata['data'][idx]
            dt = mdata['dt'][idx]
            area = mdata['area_per_channel'][idx]
            width = mdata['width90'][idx]
            dts = np.arange(tt,len(wf)*dt+tt,dt)
            plt.plot( dts, wf, drawstyle='steps', color=colors[i],
                     label=f'{k} PMT {PMT} area = {area:.2f} PE, width(90%) = {width} ns')
            plt.xlabel("time since S2 (ns)", ha='right', x=1,fontsize=12)
            plt.ylabel(f"ADC", ha='right', y=1,fontsize=12)
            plt.legend(fontsize=12)


def plot_positions (run_id, PMTs):
    mdata = pd.read_hdf(f'data/merged_records_run{run_id}.h5')
    conteggi = np.zeros(494)
    for PMT in PMTs:
        conteggi[PMT] = 1
    plt.figure(figsize=(10,10))
    straxen.plot_on_single_pmt_array(conteggi,array_name='top',pmt_label_size=8,show_tpc=1,r=70)
    ph = Histdd(mdata['position_x'], mdata['position_y'], bins=(np.linspace(-70, 70, 500), np.linspace(-70, 70, 500)))
    ph.plot(log_scale=True, cblabel='events')
    plt.xlabel("distance x (cm)", ha='right', x=1,fontsize=12)
    plt.ylabel("distance y (cm)", ha='right', y=1,fontsize=12)


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2. * sigma**2))


def gauss_tail(x, a, mu, sigma, htail, tau, components=False):
    tail = a * htail
    tail *= erfc(-(x - mu) / (sigma * np.sqrt(2)) + sigma / (tau * np.sqrt(2)))
    tail *= np.exp(-(x - mu) / tau)
    #tail /= (2 * tau * np.exp(-(sigma / (np.sqrt(2) * tau))**2))

    if not components:
        # add up all the peak shape components
        return (1 - htail)* gaussian(x, a, mu, sigma) + tail
    else:
        # return individually to make a pretty plot
        return (1 - htail), gaussian(x, a, mu, sigma), tail
    
def plot_area_width90_position(run_id, PMTs, area_lim=1e4, width_lim=1e4,position=False,distance=8, plot=False):
    mdata = pd.read_hdf(f'data/merged_records_run{run_id}.h5')
    positions = straxen.pmt_positions()

    for i, PMT in enumerate(PMTs):
        pos_x = np.float(positions['x'][positions['i']==PMT])
        pos_y = np.float(positions['y'][positions['i']==PMT])
        mdata1 = mdata[(mdata['channel']==PMT)]
        if position:
            mdata1 = mdata1[(mdata1['position_x']<pos_x+distance) & (mdata1['position_x']>pos_x-distance) &
                            (mdata1['position_y']<pos_y+distance) & (mdata1['position_y']>pos_y-distance)]
        print('PMT n.',PMT,'events',len(mdata1))
        
        # width
        plt.figure(0,figsize=(12,6))
        bins = np.linspace(0, width_lim, 100)
        ph = Hist1d(mdata1['width90'],bins=bins)
        media_tot = mdata1['width90'].mean()
        stand_tot = mdata1['width90'].std()/np.sqrt(len(mdata1))
        median_tot = mdata1['width90'].median()
        pha = np.array(ph)
        popt, pcov = curve_fit(gauss_tail, bins[:len(ph)], pha, p0 = np.array([2e3, 2e3, 1e3,0.5,2000]))
        perr = np.sqrt(np.diag(pcov))
        mu, mu_err = popt[1], perr[1]
        #ph.plot(label=f'PMT n. {PMT}, mean {media_tot:.0f}$\pm${stand_tot:.0f} {mu:.0f}$\pm${mu_err:.0f} ns')
        ph.plot(label=f'PMT n. {PMT}, mean {mu:.0f}$\pm${mu_err:.0f} ns')
        if plot:
            #plt.plot(bins,gauss_tail(bins, *popt), c='r',label=f'fit {popt[1]} +/- {perr[1]}')
            h, gaus, tail = gauss_tail(bins, *popt, components=True)
            gaus = np.array(gaus)
            tail = np.array(tail)
            plt.plot(bins, gaus, ls="--", lw=2, c='g')
            plt.plot(bins, tail, ls='--', lw=2, c='m')
        plt.ylabel("events", ha='right', y=1,fontsize=12)
        plt.xlabel("peak width 90% (ns)", ha='right', x=1,fontsize=12)
        plt.title(f'run {run_id}',fontsize=14)
        plt.legend(fontsize=14)
        
        # area vs width
        ph90 = Histdd(mdata1['area_per_channel'], mdata1['width90'],
                      bins=(np.linspace(0, area_lim, 500), np.linspace(100, width_lim, 500)))
        plt.figure(i+1,figsize=(12,6))
        ph90.plot(log_scale=True, cblabel='events')
        plt.axhline(media_tot,color='r',linestyle='--',label='mean value total')
        plt.axhline(mu,color='b',linestyle='--',label='mean from fit')
        plt.xlabel("peak area (PE)", ha='right', x=1,fontsize=12)
        plt.ylabel("peak width 90% (ns)", ha='right', y=1,fontsize=12)
        plt.title(f'run {run_id} - PMT n. {PMT}',fontsize=14)
        plt.legend(fontsize=14)

