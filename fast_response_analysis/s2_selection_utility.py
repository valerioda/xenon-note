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
import time

import socket 
import strax
import straxen


def select_peaks_times(st, run_id, peaks, times, ndiv = 10, div = 0, s2_start = 0):
    area = []
    width = []
    areachn = []
    r_data = pd.DataFrame(columns=['area','range_50p_area','area_per_channel'])
    t_start = time.time()
    tsec = (peaks['endtime'][-1]-peaks['time'][0])/1e9
    csec = tsec/ndiv
    print('run',run_id,'Total events',len(times),'total peaks',len(peaks),'duration',tsec)
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
        if t >= ft or k == len(times): break
        elif times[k]==0:
            area.append(0)
            width.append(0)
            areachn.append(0)
            k += 1
        elif t == times[k] and k < len(times)-1:
            area.append(pp['area'][j])
            width.append(pp['range_50p_area'][j])
            areachn.append(pp['area_per_channel'][j])
            if (k<s2_start+10):
                plt.plot( dts,pp['data'][j],drawstyle='steps',label=f'{k} t = {t}')
                plt.xlabel("time (ns)", ha='right', x=1,fontsize=12)
                plt.ylabel(f"ADC", ha='right', y=1,fontsize=12)
                plt.legend(fontsize=12)
            k += 1
    diff = time.time() - t_start
    print(f'division n. {div}, tot. events {nn}, selected {len(area)}, tot.sel. {k} time to process {diff:.2f} s')
    del pp
    r_data['area'] = area
    r_data['range_50p_area'] = width
    r_data['area_per_channel'] = areachn
    r_data.to_hdf(f'data/select_peaks_run{run_id}_div{div}.h5', key='df', mode='w')
    return r_data, k


def select_records_times(st, run_id, peaks, times, endtimes, areas, area_chn, widths, PMTs, ndiv = 10, div = 0, s2_start = 0):
    area = []
    width = []
    area_per_channel = []
    channel = []
    tempo = []
    et = []
    ev = []
    wf = []
    dt = []
    r_data = pd.DataFrame(columns=['event_number','event_time','time','channel',
                                   'area','range_50p_area','area_per_channel','data','dt'])
    
    t_start = time.time()
    tsec = (peaks['endtime'][-1]-peaks['time'][0])/1e9
    csec = tsec/ndiv
    print('run',run_id,'Total events',len(times),'total peaks',len(peaks),'duration',tsec)
    k, kk = s2_start, 0
    #for ii in range(1):
    rr = st.get_array(run_id,'records',
                        seconds_range=(csec*div,csec*(div+1)),progress_bar=False)
    nr = len(rr)
    dts = np.arange(0,rr['data'].shape[1]*rr['dt'][0],rr['dt'][0])
    ft = rr['time'][-1] + rr['dt'][0]*rr['length'][-1]
    for j in range(nr):
        t = rr['time'][j]
        if t >= ft or k == len(times): break
        elif times[k] == 0: k += 1
        elif t >= times[k] and kk < len(areas):
            jj = j
            while rr['time'][jj] < endtimes[k]:
                for i, PMT in enumerate(PMTs):
                    chn = rr['channel'][jj]
                    areai = rr['area'][jj]
                    if chn == PMT and rr['record_i'][jj] > 0:
                        ev.append(k)
                        et.append(times[k])
                        channel.append(chn)
                        tempo.append(rr['time'][jj])
                        area.append(areas[kk])
                        width.append(widths[kk])
                        try: area_channel = area_chn[kk][chn]
                        except: area_channel = 0
                        area_per_channel.append(area_channel)
                        wf.append(rr['data'][jj])    
                        dt.append(rr['dt'][jj])
                jj += 1
            k += 1
            kk += 1
    diff = time.time() - t_start
    print(f'division n. {div}, tot. events {nr}, selected events: {len(area)} {k}, time to process: {diff:.2f} s')
    del rr
    r_data['event_number'] = ev
    r_data['event_time'] = et
    r_data['time'] = tempo
    r_data['channel'] = channel
    r_data['area'] = area
    r_data['area_per_channel'] = area_per_channel
    r_data['range_50p_area'] = width
    r_data['data'] = wf
    r_data['dt'] = dt
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
    wf = []
    dt = []
    position_x = []
    position_y = []
    r_data = pd.DataFrame(columns=['event_number','event_time','time','channel','area','area_per_channel',
                                   'range_50p_area','width50','width90','data','dt','position_x','position_y'])
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
                wf.append(wftot)
                dt.append(rr['dt'][idx])
                
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
    r_data['range_50p_area'] = width
    r_data['width50'] = width50
    r_data['width90'] = width90
    r_data['data'] = wf
    r_data['dt'] = dt
    r_data['position_x'] = position_x
    r_data['position_y'] = position_y
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