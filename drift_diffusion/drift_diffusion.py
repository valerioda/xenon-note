import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import straxen
from multihist import Histdd, Hist1d

import sys
import os
import os.path as osp
from straxen import units
import pandas as pd
import math
from tqdm import tqdm
import emcee
import corner

def select_data_from_runs( runs, nrun, firstrun = 0 ):
    s2 = []
    cs1 = []
    cs2 = []
    s2_50 = []
    s2_aft = []
    drift_times = []
    data = pd.DataFrame(columns=['cs1','cs2','s2','s2_a50','s2_aft','drift_times'])
    print(data)
    for i in range(nrun):
        run_info = runs.iloc[i+firstrun]
        events = st.get_array(run_info['name'],'event_basics')
        c_areas = st.get_array(run_info['name'],'corrected_areas')
        for k in range(len(events)):
            s2.append(events['s2_area'][k]) 
            s2_50.append(events['s2_range_50p_area'][k])
            s2_aft.append(events['s2_area_fraction_top'][k])
            drift_times.append(events['drift_time'][k] / units.us)
            cs1.append(c_areas['cs1'][k])
            cs2.append(c_areas['cs2'][k])
    data['cs1'] = cs1
    data['cs2'] = cs2
    data['s2'] = s2
    data['s2_50'] = s2_50
    data['s2_aft'] = s2_aft
    data['drift_times'] = drift_times
    frun = runs.iloc[firstrun]['name']
    lrun = runs.iloc[nrun+firstrun]['name']
    print('First run:',frun,'Last run:',lrun)
    print('Total number of events:',len(s2))
    fname = f'data_{frun}-{lrun}.h5'
    data.to_hdf(fname, key='data', mode='w')
    print('Data saved in file',fname)
    return data

def drift_velocity( data, plot = False ):
    #cathode drift time
    drift_hist, drift_bin = np.histogram(data['drift_times'],bins=800, range=(400,800))
    drift_max = drift_bin[int(drift_hist.argmax())]
    if plot:
        plt.figure(1)
        plt.plot(drift_bin[1:],drift_hist)
        plt.xlabel('drift time ($\mu$s)', ha='right', x=1)
        plt.ylabel('events', ha='right', y=1)
        plt.axvline(drift_max, c='red', linestyle=":",label=f'cathode drop-off: {drift_max:.1f} $\mu s$')
        plt.legend()
        plt.savefig("drift-time.png")
    
    #area ratio
    area_ratio = np.zeros(len(data))
    area_ratio10 = np.zeros(len(data))
    #bool = (data['cs1']>0) & (data['cs2']>0) & (np.invert(np.isnan(data['cs1'])))
    area_ratio = np.divide(data['cs2'],data['cs1'])
    area_ratio10 = np.log10(area_ratio)
    d_vs_a = Histdd(data['drift_times'], area_ratio10,
                    bins=(np.linspace(400, 800, 200), np.linspace(-2, 4, 100)))
    if plot:
        plt.figure(2)
        d_vs_a.plot(log_scale=True, cblabel='events')
        plt.xlabel('drift time ($\mu$s)', ha='right', x=1)
        plt.ylabel('log10(cS2/cS1)', ha='right', y=1)
        plt.axvline(drift_max, c='red', linestyle=":",label=f'cathode drop-off {drift_max:.1f} $\mu s$')
        plt.legend()
        plt.savefig("dt-area_high.png")
    
    mh = Histdd(data['drift_times'], area_ratio10,
                bins=(np.linspace(0, 7, 70), np.linspace(1, 3.5, 100)),
                axis_names=['drift_time', 'area_ratio'])
    median = mh.percentile(50, axis='area_ratio')

    if plot:
        plt.figure(3)
        mh.plot(log_scale=True, cblabel='Events / bin')
        plt.xlabel("drift time ($\mu$s)", ha='right', x=1)
        plt.ylabel("log10(cS2/cS1)", ha='right', y=1)
        median.plot(color='red',label='median')
        plt.legend()
        plt.savefig("dt-area_low.png")
    
    #drift velocity
    cathode_dt = drift_max * units.us
    dt_offset = 1.5 * units.us
    tpc_length = 97 * units.cm #lunghezza approssimata della TPC di XENON1T
    drift_velocity = tpc_length / (cathode_dt - dt_offset)
    print(f'Drift velocity determined at {drift_velocity/(units.km/units.s):.3f} km/s')
    return drift_velocity

def diffusion_model(t, D, vd, w0):
    sigma_to_r50p = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
    return np.sqrt(2 * sigma_to_r50p**2 * D * t / vd**2 + w0**2)

def fit_diffusion(drift_times, s2_50, vd, outlier_cap=-15):
    def loglikelihood(params):
        D, w0, wsigma = params
        
        # Range constraint / uniform prior. Probably not necessary anymore.
        if not (5 < D < 50 and (100  < w0/units.ns < 500) and (10 < wsigma/units.ns < 100)):
            return -float('inf')
        
        D *= (units.cm**2 / units.s)
        
        y = s2_50
        model = diffusion_model(drift_times, D, vd, w0)
        
        # Gaussian loglikelihood, with outlier contribution capped
        inv_sigma2 = 1.0/wsigma**2
        result = -0.5 * ((y-model)**2*inv_sigma2 - np.log(inv_sigma2))
        result = np.clip(result, outlier_cap, 1)
        return np.sum(result)
    D_guess = 30 * units.cm**2 / units.s
    w0_guess = 300 * units.ns
    wsigma_guess = 30 * units.ns
    guess = np.array([D_guess/(units.cm**2 / units.s), w0_guess, wsigma_guess])
    n_walkers = 50
    n_steps = 250
    n_dim = len(guess)
    
    # Hack to show a progress bar during the computation
    def lnprob(x):
        lnprob.t.update(1)
        return loglikelihood(x)
    lnprob.t = tqdm(desc='Computing likelihoods', total=n_walkers * n_steps)
    
    # Run the MCMC sampler
    p0 = np.array([np.random.uniform(0.9, 1.1, size=n_dim) for i in range(n_walkers)]) * guess
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob)
    samples = sampler.run_mcmc(p0, n_steps)
    
    # Remove first 50 samples for each walker (burn-in)
    samples = sampler.chain.ravel()
    samples = sampler.chain[:, 50:, :].reshape((-1, n_dim))
    return samples

def diffusion_constant( data, s2_bounds, aft_bounds, fit_range, vd, plot = False ):
    # S2_area vs S2_area_fraction_top
    s2_top2 = Histdd(data['s2'], data['s2_aft'],bins=(np.logspace(2, 6, 50), np.linspace(0.4, 0.8, 50)))
    if plot:
        plt.figure(1)
        s2_top2.plot(log_scale=True)
        plt.gca().add_patch(matplotlib.patches.Rectangle(
            (s2_bounds[0], aft_bounds[0]), s2_bounds[1] - s2_bounds[0],
            aft_bounds[1] - aft_bounds[0],
            edgecolor='red',facecolor='none'))
        plt.xscale('log')
        plt.xlabel('S2 (PE)', ha='right', x=1)
        plt.ylabel('S2 area fraction top', ha='right', y=1)
        plt.tight_layout()
    
    # drift_time vs S2_width
    width_hist = Histdd(data['drift_times'], data['s2_50'],
                        bins=(np.linspace(0, 800, 800), np.linspace(0, 3e3, 200)));
    if plot:
        plt.figure(2)
        width_hist.plot(log_scale=True, cblabel='events')
        plt.xlabel('drift time ($\mu$s)')
        plt.ylabel('S2 width range 50% area (ns)')
    
    # cut on S2_area and S2_area_fraction_top
    data_cut = data[(data['s2']>s2_bounds[0]) & (data['s2']<s2_bounds[1]) &
                    (data['s2_aft']>aft_bounds[0]) & (data['s2_aft']<aft_bounds[1])]
    width_hist_cut = Histdd(data_cut['drift_times'], data_cut['s2_50'],
                            bins=(np.linspace(0, 800, 800), np.linspace(0, 3e3, 200)));
    data_fit = data_cut[(data_cut['drift_times']>fit_range[0]) &
                        (data_cut['drift_times']<fit_range[1])]
    # calculate the diffusion constant
    samples = fit_diffusion(data_fit['drift_times'], data_fit['s2_50'], vd)
    fit_result = np.median(samples, axis=0)
    l, r = np.percentile(samples, 100 * stats.norm.cdf([-1, 1]), axis=0)
    sigma = (r - l)/2
    q = np.round(fit_result, 2), np.round(sigma, 2)
    D = fit_result[0] * (units.cm**2 / units.s)
    w0 = fit_result[1]
    diffusion_const = fit_result[0]
    diffusion_const_err = sigma[0]
    print(f'Diffusion constant = {fit_result[0]:.2f} +/- {sigma[0]:.2f} cm^2/s ')
    if plot:
        plt.figure(3)
        width_hist_cut.plot(log_scale=True, cblabel='events')
        plt.xlabel('drift time ($\mu$s)')
        plt.ylabel('S2 width range 50% area (ns)')
        ts = np.linspace(0, 800, 100) * units.us
        plt.plot(ts / units.us, diffusion_model(ts, D, vd, w0),
                 linestyle=':', linewidth=2, c='r',label='model')
        plt.axvspan(*fit_range, alpha=0.2, color='blue', label='fit region')
        plt.legend()
        plt.savefig('diffusion_constant.png')
        #corner.corner(samples, show_titles=True, labels=['D', 'w0', 'wsig'])
    return diffusion_const, diffusion_const_err
