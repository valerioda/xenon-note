import os
import matplotlib
import numpy as np
from iminuit import Minuit
from IPython.display import display
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colorbar import Colorbar
from multihist import Histdd
import h5py

from numba import jit

from iminuit import Minuit, describe
from iminuit.util import make_func_code

class Fit2dHisto(object):

    def __init__(self):
        pass
    
    def from_events(self, events, bin_edges, axis_names, plot=True):
        
        self.drift_time = events['drift_time']
        self.s2_width = events['s2_a_range_50p_area']
        
        self.bin_edges = bin_edges
        self.axis_names = axis_names
        self.mh = Histdd(self.drift_time, self.s2_width, bins = self.bin_edges, axis_names = self.axis_names)
        
        print('Histogrammed %d events' % self.mh.n)
        self.prepare_fits()
        if plot:
            self.plot_roi()
    
    def prepare_fits(self):
        # The Histogram has to be transposed!
        self.z = self.mh.histogram.T    
        
        # Get the edges and the centers 
        # edges for plotting
        # centers for fits
        self.x_edges = self.mh.bin_edges[0]
        self.y_edges = self.mh.bin_edges[1]
        self.x = self.mh.bin_centers()[0]
        self.y = self.mh.bin_centers()[1]
        
        # Create meshgrid
        self.x_mesh, self.y_mesh = np.meshgrid(self.x, self.y)
        self.x_mesh_ravel = self.x_mesh.ravel()
        self.y_mesh_ravel = self.y_mesh.ravel()
        self.z_ravel = self.z.ravel()
        
        # Assume symetric uncertainties. (only needed for chi2 fit)
        self.z_err = yerror(self.z_ravel)

    def plot_roi(self,**kwdarg):
        '''
        Plot the Histogram!
        To adjust the binning and find the region of interest
        just use the plot function of multihist
        '''
        fig = plt.figure()
        self.mh.plot(log_scale=True, cblabel='Events/bin', **kwdarg)
            
        # take the labels from the dimensions
        plt.xlabel(self.mh.axis_names[0])
        plt.ylabel(self.mh.axis_names[1])
        plt.show()
        
    def fit(self, fitfunction,start_values,error_values, fixed_values={}, limit_values={}, fit_type='n2log'):
        
        self.fitfunction = fitfunction
        self.fit_type = fit_type
        
        # Chi2 fit with sqrt(n) uncertainty
        if self.fit_type == 'Chi2':
            self.m = Minuit(ChiSquare(
                self.fitfunction, self.x_mesh_ravel,
                self.y_mesh_ravel, self.z_ravel, self.z_err),
                **start_values)
            
            for key in error_values.keys():
                self.m.errors[key]=error_values[key]
            for key in fixed_values.keys():
                self.m.fixed[key]=fixed_values[key]
            for key in limit_values.keys():
                self.m.limits[key]=limit_values[key]

            status = self.m.migrad()
            display(status)

            self.chi_sq         = self.m.fval
            self.ndof        = (len(self.z_ravel)
                                - (len(self.m.values[:])-sum(self.m.fixed[:])))
            self.chi_sq_red     = self.chi_sq / self.ndof
            self.chi_sq_red_err = np.sqrt(2 / self.ndof)
        
        # Log-likelihood fit!
        # Better use this one due to empty bins
        elif self.fit_type == 'n2log':
            self.m = Minuit(n2logL(
                self.fitfunction, self.x_mesh_ravel,
                self.y_mesh_ravel, self.z_ravel),
                **start_values)
            
            for key in error_values.keys():
                self.m.errors[key]=error_values[key]
            for key in fixed_values.keys():
                self.m.fixed[key]=fixed_values[key]
            for key in limit_values.keys():
                self.m.limits[key]=limit_values[key]

            status = self.m.migrad()
            display(status)
            
            # Goodnes of Fit X2p
            self.x2p         = X2P(self.fitfunction, self.x_mesh_ravel,
                                   self.y_mesh_ravel, self.z_ravel, self.m.values[:])
            self.ndof        = (len(self.z_ravel)
                                - (len(self.m.values[:])-sum(self.m.fixed[:])))
            self.x2p_red     = self.x2p / self.ndof
            self.x2p_red_err = np.sqrt(2 / self.ndof)

        else:
            print('Please enter Chi2 or n2log')
            return
        
        self.fit_results = self.m.values
        self.fit_errors = self.m.errors

        
        
# 2D fit functions
@jit(nopython=True)
def two_d_gaussian(x, y, amp, muX, a, muY, b, theta):
    
    cos = np.cos(np.radians(theta))
    sin = np.sin(np.radians(theta))
    
    A = (((x - muX) * cos - (y - muY) * sin)**2) / (2 * a**2)
    B = (((x - muX) * sin + (y - muY) * cos)**2) / (2 * b**2)
    
    gaussian = amp * np.exp(- A - B)
    
    return gaussian

def yerror(ydata):
    
    ydataerr = []
    
    for y in ydata:
        if y <= 1:
            yerr = 1
        else:
            yerr = np.sqrt(y)
        # if yerr <= :
            # yerr = 1.
        ydataerr.append(yerr)
    return ydataerr


# The negative log-likelihood class for the 2d fits
class n2logL:
    
    def __init__(self, model, x, y, z):
        
        self.errordef = Minuit.LIKELIHOOD

        self.model = model
        
        self.x, self.y, self.z = (x, y, z)
        
        self.func_code = make_func_code(describe(self.model)[2:])

    def __call__(self, *par):
        
        with np.errstate(invalid='ignore'):
            return - np.sum(self.z
                            * np.log(self.model(self.x, self.y, *par))
                            - self.model(self.x, self.y, *par))

# Chi2 class for 2D fits
class ChiSquare:
    
    def __init__(self, model, x, y, z, zerr):
        
        self.errordef = Minuit.LEAST_SQUARES

        self.model = model
        
        self.x, self.y, self.z, self.zerr = (x, y, z, zerr)
        
        self.func_code = make_func_code(describe(self.model)[2:])

    def __call__(self, *par):

        return np.sum(((self.model(self.x, self.y, *par) - self.z)
                       / (self.zerr))**2)

# 1D Chi2 class
class ChiSquare_1d:
    
    def __init__(self, model, x, y, yerr):
        
        self.errordef = Minuit.LEAST_SQUARES

        self.model = model
        
        self.x, self.y, self.yerr = (x, y, yerr)
        
        self.func_code = make_func_code(describe(self.model)[1:])

    def __call__(self, *par):

        return np.sum(((self.model(self.x, *par) - self.y) / (self.yerr))**2)

def X2P(fitfunction, x, y, z, par):
    
    with np.errstate(divide='ignore', invalid='ignore'):
        x2p = sum(np.where(z > 0,
                           fitfunction(x, y, *par) - z
                           + z * np.log(z / fitfunction(x, y, *par)),
                           fitfunction(x, y, *par)))
    
    return 2 * x2p


# Used for the g1g2 likelihood fit 
class Loglikelihood_g1g2:
    
    def __init__(self, model, x, y, yerr):
        
        self.errordef = Minuit.LIKELIHOOD
        self.model = model
        
        self.x = x
        self.y = y
        self.yerr = yerr
        
        self.func_code = make_func_code(describe(self.model)[1:] + ('f',))
        
    def __call__(self, *par):
        
        fpar = par[:-1]  # cut the f..
        f = par[-1]
    
        sn2 = (f * self.model(self.x, *fpar))**2 + self.yerr**2
    
        logli = 0.5 * np.sum(((self.model(self.x, *fpar) - self.y)**2)
                             / sn2 + np.log(2 * np.pi * sn2))
    
        return logli