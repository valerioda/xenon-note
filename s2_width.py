import strax
import numpy as np
import straxen
from scipy.stats import chi2
from scipy.stats import norm
from scipy.interpolate import interp1d
from straxen.get_corrections import get_correction_from_cmt
from scipy import special
import warnings

'''
The S2 width cut has been changing considerably over time and 
therefore this file went from ok to not perfect to a perfect mess.
This is a try at refactoring this file to be easier to read, 
understand and implement any further changes.
The basic structure is:
  * S2WidthBase(strax.CutPlugin)
     This class has all the functions and general parameters 
     needed for all the subsequent cuts.

    - S2WidthNearWires(S2WidthBase)
    - S2WidthFarWires(S2WidthBase)
    - S2WidthWireModeled(S2WidthBase)
  *S2WidthWIMPs(S2WidthNearWires,S2WidthFarWires)
     Collection of near and far wires parameters to apply to WIMP analysis
  *S2WidthLowER(S2WidthNearWires,S2WidthFarWires)
     Collection of near and far wires parameters to apply to WIMP analysis
  
  * S2WidthNaive(stax.CutPlugin)
'''

quantile_parameters = dict(WIMP = dict(far_wires = dict(high = ( 4.43451121, 1.84825809, 2.15736571, 1.2112417 ),
                                                        low = ( 0.63734546, 2.99678884, 1.03246656 ) 
                                                        ),
                                       near_wires = dict(high = ( 4.43451121,  1.84825809, 2.15736571, 1.2112417 ),
                                                         low = ( 0.70507915,  2.80001544, 1.10057148 )
                                                         ),
                                       ),
                           LowER = dict(far_wires = dict(high = ( 4.43451121, 1.84825809, 2.15736571, 1.2112417 ),
                                                        low = ( 0.63734546, 2.99678884, 1.03246656 ) 
                                                         ),
                                       near_wires = dict(high = ( 4.43451121,  1.84825809, 2.15736571, 1.2112417 ),
                                                         low = ( 0.63734546, 2.99678884, 1.03246656 ) 
                                                         ),
                                       ),
                          )   

parabola_parameters = dict(WIMP = dict(far_wires = dict(high = (0.11170658, -1.12877236,  4.10119736),
                                                        low = (-0.00798012,  0.23778128, -0.1894035)
                                                        ),
                                       near_wires = dict(high = ( 0.11170658, -1.12877236,  4.10119736),
                                                         low = ( 0.02292279, -0.10215068,  0.73768366)
                                                         ),
                                       ),
                           LowER = dict(far_wires = dict(high = (0.11170658, -1.12877236,  4.10119736),
                                                        low = (-0.00798012,  0.23778128, -0.1894035)
                                                         ),
                                       near_wires = dict(high = (0.11170658, -1.12877236,  4.10119736),
                                                         low = (-0.00798012,  0.23778128, -0.1894035)
                                                         ),
                                       ),
                          )   

@strax.takes_config(
    strax.Option('min_s2_area_width_cut', default=0,
                 help='Min area [PE] of a S2 for the S2WidthCut'),
    strax.Option('max_s2_area_width_cut', default=1e8,
                 help='Max area [PE] of a S2 for the S2WidthCut'),
    strax.Option('s2_secondary_sc_width', default=375,
                 help='S2 secondary sc width median for the S2WidthCut'),
    strax.Option('switch_boundary', default=10**3.8,
                 help='S2 area value where to switch from low s2 area to a parabolas for high area'),
    #strax.Option('s2_width_cut_only_apply_lb', default=True, type=bool,
    #             help=('If true only applies lower boundary cut near wires. '
    #                   'If false do not apply any cut near wires.')),
    strax.Option('perp_wires_cut_distance', default=4.45,
                 help=('Distance in x to apply exception from the center of the '
                       'gate perpendicular wires [cm]'))
    )


class S2WidthBase(strax.CutPlugin): # is this really a cut or just a collection 
                                    #of functions and parameters?
    """
    Base of S2WidthCut. REDO THIS TEXT

    Cut on S2 width for SR0 optimized on WFsim percentiles.
    The current version divides the cut into two regions: below and
    above 1e3.8 PE on S2 area. The region below is cut by functions reproducing
    the WFsim percentiles and the region above is cut by a parabola due an increase
    of width at high energies.
    To be updated:
      - model the percentile boundaries with better models
      - change the boundaries for the high s2 area region, also based on WFsim
    
    Current developers: rperes@physik.uzh.ch; valerio.dandrea@lngs.infn.it
    
    Notes:
      * v0.1.0: dandrea:s2widthcutsr0update
      * v0.0.0: dandrea:s2widthcutsr0regionalmeeting
    """
    depends_on = ('event_basics',)
    #provides = ('cut_s2_width',) #should not provide anything?
    #cut_name = 'cut_s2_width'
    #cut_description = ('S2WidthCut base class. Use S2WidthWIMP '+
    #                   'or S2WidthLowER for cut')
    __version__ = '0.1.0'

    electron_drift_velocity = straxen.URLConfig(
        default='cmt://electron_drift_velocity?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    electron_drift_time_gate = straxen.URLConfig(
        default='cmt://electron_drift_time_gate?version=ONLINE&run_id=plugin.run_id',
        help='Electron drift time from the gate in ns',
        cache=True)
    
    diffusion_constant = straxen.URLConfig(
        default='cmt://electron_diffusion_cte?version=ONLINE&run_id=plugin.run_id',
        help='Longitudinal diffusion constant value (time dependant from CMT).',
        cache=True) 
    
    def setup(self):
        self.sigma_to_r50p = norm.ppf(0.75) - norm.ppf(0.25)                

    def s2_width_model(self, drift_time):
        return self.sigma_to_r50p * np.sqrt(2 * self.diffusion_constant *
                                            (drift_time - self.electron_drift_time_gate) / self.electron_drift_velocity**2 )
        
    def correct_cut_boundaries(self, area, mask):
        """Forces to accept events tha tare outside the cut defined
        tested boundaries"""
        mask[area > self.config['max_s2_area_width_cut']] = True
        mask[area < self.config['min_s2_area_width_cut']] = True
        return mask 

    def normalize_width(self, events):
        return ( events['s2_range_50p_area']**2 - self.config['s2_secondary_sc_width']**2 ) / self.s2_width_model(events['drift_time'])**2
        
    def cut_low_boundary(self, area, p0, p1, p2):
        '''
        Function to apply to the lower boundary of the cut 
        (based on the erf special function)
        '''
        return p0*0.5 * special.erf((np.sqrt(2) * (np.log10(area) - p1)) / p2) + p0*0.5
    
    def cut_high_boundary(self, area, p0, p1, p2, p3):
        '''
        Function to apply to the upper boundary of the cut
        (based on an exponential)
        '''
        return p0*np.exp(-p1*(np.log10(area)-p2)) + p3
        
    def cut_parabola(self, area, parabola_par2, parabola_par1, parabola_par0):
        """
        Returns the value of the 2nd deg polynomial cut
        function at the given area
        """
        return parabola_par2 * np.power(np.log10(area), 2) + parabola_par1 * np.log10(area) + parabola_par0
    
    def get_cut_mask_parabola(self, events):
        area = events['s2_area']
        norm_width = self.normalize_width(events)
        '''
        Get masks of parabola section of the cut.
        '''
        mask_high_parabola = ((area < self.config['switch_boundary']) |
                              (norm_width < (self.cut_parabola(area, 
                                                               self.parabola_par2, 
                                                               self.parabola_par1,
                                                               self.parabola_par0) 
                                             + self.delta_par0)))
    
        mask_low_parabola  = ((area < self.config['switch_boundary']) |
                              (norm_width > (self.cut_parabola(area, 
                                                               self.parabola_par2, 
                                                               self.parabola_par1, 
                                                               self.parabola_par0) 
                                             - self.delta_par0)))
    
        mask_high_parabola = self.correct_cut_boundaries(area, mask_high_parabola)
        mask_low_parabola = self.correct_cut_boundaries(area, mask_low_parabola)
    
        return mask_high_parabola, mask_low_parabola
    
    def cut_chi2(self, area, band):
        """
        Cut based on the chi2.ppf distribution.
        Params:
          * area: aree [pe] of the peak
          * band: True for upper boundary cut, False for lower boundary cut.
        """
        _n_electron = self.n_electron(area)
        if band:
            return np.sqrt(chi2.ppf((1 - 10 ** self.config['min_norm_s2_width_cut']), _n_electron) / (_n_electron - 1))
        else:
            return np.sqrt(chi2.ppf((10 ** self.config['min_norm_s2_width_cut']), _n_electron) / (_n_electron - 1))

    def get_cut_mask_near_wires(self, events, ev_type):
        area = events['s2_area']
        norm_width = self.normalize_width(events)
        # only lower boundaries for events close to perpendicular wires
        mask_low = ((area >= self.config['switch_boundary']) |
                    (norm_width > self.cut_low_boundary(area, *quantile_parameters[ev_type]['near_wires']['low'] )))

        mask_low_parabola  = ((area < self.config['switch_boundary']) |
                              (norm_width > (self.cut_parabola(area, *parabola_parameters[ev_type]['near_wires']['low'] ))))
        
        mask_low = self.correct_cut_boundaries(area, mask_low)
        mask_low_parabola = self.correct_cut_boundaries(area, mask_low_parabola)

        cut_mask = (mask_low & mask_low_parabola)

        return cut_mask
    
    def get_cut_mask_far_wires(self, events, ev_type):
        area = events['s2_area']
        norm_width = self.normalize_width(events)
        
        mask_high = ((area >= self.config['switch_boundary']) |
                     (norm_width < self.cut_high_boundary(area, *quantile_parameters[ev_type]['far_wires']['high'] )))
        mask_low = ((area >= self.config['switch_boundary']) |
                    (norm_width > self.cut_low_boundary(area, *quantile_parameters[ev_type]['far_wires']['low'] )))
        mask_high_parabola = ((area < self.config['switch_boundary']) |
                              (norm_width < (self.cut_parabola(area, *parabola_parameters[ev_type]['far_wires']['high'] ))))
        mask_low_parabola  = ((area < self.config['switch_boundary']) |
                              (norm_width > (self.cut_parabola(area, *parabola_parameters[ev_type]['far_wires']['low'] ))))
        
        mask_high = self.correct_cut_boundaries(area, mask_high)
        mask_low = self.correct_cut_boundaries(area, mask_low)                           
        mask_high_parabola = self.correct_cut_boundaries(area, mask_high_parabola)
        mask_low_parabola = self.correct_cut_boundaries(area, mask_low_parabola)
        
        cut_mask = (mask_high & mask_low & mask_high_parabola & mask_low_parabola)
        return cut_mask
        
    """def cut_by(self, events):
        mask_high, mask_high_parabola, mask_low, mask_low_parabola = self.get_cut_masks(events) 
        cut_mask = (mask_high &
                    mask_low &
                    mask_high_parabola &
                    mask_low_parabola)
        return cut_mask"""
      
class S2WidthCutWIMP(S2WidthBase):
    '''
    Use this when trying to find WIMPs.
    '''

    __version__ = '0.0.1'
    provides = ('cut_s2_width_wimps',)
    cut_name = 'cut_s2_width_wimps'
    cut_description = 'S2 Width cut for WIMP search analysis.'
    depends_on = ('event_basics', 'cut_near_wires')
    child_plugin = True

    def cut_by(self, events):
        
        mask_near_wires = self.get_cut_mask_near_wires(events, 'WIMP')
        mask_far_wires = self.get_cut_mask_far_wires(events, 'WIMP')
        mask_near_wires[~events['cut_near_wires']] = True
        mask_far_wires[events['cut_near_wires']] = True 
        
        cut_mask = mask_near_wires & mask_far_wires

        return cut_mask


class S2WidthCutLowER(S2WidthBase):
    '''
    Use this when trying to find tritium.
    '''

    __version__ = '0.0.1'
    provides = ('cut_s2_width_low_er')
    cut_name = 'cut_s2_width_low_er'
    cut_description = 'S2 Width cut for Low-energy ER search analysis.'
    depends_on = ('event_basics', 'cut_near_wires')
    child_plugin = True

    def cut_by(self, events):
        mask_near_wires = self.get_cut_mask_near_wires(events, 'LowER')
        mask_far_wires = self.get_cut_mask_far_wires(events, 'LowER')
        mask_near_wires[~events['cut_near_wires']] = True
        mask_far_wires[events['cut_near_wires']] = True 

        cut_mask = mask_near_wires & mask_far_wires

        return cut_mask


@strax.takes_config(
    strax.Option('s2_width_wire_model_slope', default=5e3,
                 help='Slope [ns/cm] of model for S2 width near wire'),
    strax.Option('s2_width_wire_model_intercept', default=9e3,
                 help='Intercept [ns] of model for S2 width near wire'),
    strax.Option('wire_median_displacement', default=13.06,
                 help='Median distance [cm] of wire population from center',),
    strax.Option('position_resolution_params', default=(0.1, 4e2, 0.528, 3e4),
                 help='Parameters to model position resolution')
    )


class S2WidthWireModeled(S2WidthBase):
    """
    Cut on S2 width based on preliminary boundaries for SR0.
    The current version divides the cut into two regions: below and
    above 1e4.1 PE on S2 area. The region below is cut by a chi2.ppf
    function and the region above is cut by a parabola due an increase
    of width at high energies.
    To be updated:
      - Add a half circle cap on width of events near the wire
      - Add dependence on xy due to wires;
      - Return the cut with updated diffusion, SE gain and SE width
      - Go to chi2.pdf instead of np.sqrt(chi2.ppf(...))
    Current developers: rperes@physik.uzh.ch; valerio.dandrea@lngs.infn.it; tz2263@columbia.edu
    First implementation and pretty code: j.angevaare@nikhef.nl
    Notes:
      * v0.2.0: xenon:xenon1t:tzhu:s2wwire
      * v0.1.0: dandrea:s2widthcutsr0regionalmeeting
      * v0.0.0: xenon:xenon1t:sim:notes:tzhu:width_cut_tuning
    """
    depends_on = ('event_basics', 'cut_near_wires')
    provides = 'cut_s2_width_wire_modeled'
    cut_name = 'cut_s2_width_wire_modeled'
    cut_description = 'Cut S2Width triangle model near wire'
    __version__ = '0.2.0'
    
    def position_resolution(self, s2_area_top):
        flat, amp, power, cutoff = self.config['position_resolution_params']
        return flat + (amp / np.clip(s2_area_top, 0, cutoff)) ** power

    def uppper_lim_with_wire(self, model_pred, nwidth_lim, s2_area_top, x_to_wire):
        ry = model_pred * (nwidth_lim - 1)
        rx = self.position_resolution(s2_area_top) * norm.isf(0.02)
        
        x_tmp = x_to_wire / rx
        k0 = self.config['s2_width_wire_model_slope'] * rx / ry  # slope in rescaled space
        y0 = 1 / np.cos(np.arctan(k0))  # interception of the upper limit
        x0 = np.sin(np.arctan(k0))  # switch poit from linear to circle
        yc = self.config['s2_width_wire_model_intercept'] / ry  # interception of the center

        y_linear = y0 + yc - np.abs(x_tmp) * k0
        y_circle = yc + np.sqrt(1 - np.clip(np.abs(x_tmp)**2, 0, 1))

        y_lim = np.select([np.abs(x_tmp) > x0, np.abs(x_tmp) <= x0],
                          [y_linear, y_circle],
                          1)
        m = y_lim > yc
        y_lim[m] = (y_lim[m] - yc[m]) * 0.5 + yc[m]  # half the peak
        return np.clip(y_lim, 1, np.inf) * ry

    def get_cut_masks(self, events,ev_type):
        area = events['s2_area']
        area_top = events['s2_area'] * events['s2_area_fraction_top']
        width = events['s2_range_50p_area']
        norm_width = self.normalize_width(events)
        model_pred = self.s2_width_model(events['drift_time'])
        x_to_wire = np.abs(events['s2_x_mlp'] * np.cos(-np.pi/6) + events['s2_y_mlp'] * np.sin(-np.pi/6)) \
            - self.config['wire_median_displacement']
        
        """mask_high_chi2     = ((area >= self.config['switch_boundary']) |
                              (width - model_pred < self.uppper_lim_with_wire(
                                  model_pred, self.cut_chi2(area, True),
                                  area_top, x_to_wire)))
        mask_high_parabola = ((area < self.config['switch_boundary']) |
                              (width - model_pred < self.uppper_lim_with_wire(
                                  model_pred, self.cut_parabola(area, *self.config['param_parabola_high']),
                                  area_top, x_to_wire)))"""
        mask_high  = ((area >= self.config['switch_boundary']) |
                      (width - model_pred < self.uppper_lim_with_wire(
                          model_pred, self.cut_high_boundary(area, *quantile_parameters[ev_type]['near_wires']['high']), area_top, x_to_wire)))
        
        mask_high_parabola = ((area < self.config['switch_boundary']) |
                              (width - model_pred < self.uppper_lim_with_wire(
                                  model_pred, self.cut_parabola(area, *parabola_parameters[ev_type]['near_wires']['high']), area_top, x_to_wire)))
        
        mask_low = ((area >= self.config['switch_boundary']) |
                    (norm_width > self.cut_low_boundary(area, *quantile_parameters[ev_type]['near_wires']['low'] )))

        mask_low_parabola  = ((area < self.config['switch_boundary']) |
                              (norm_width > (self.cut_parabola(area, *parabola_parameters[ev_type]['near_wires']['low'] ))))

        
        mask_high = self.correct_cut_boundaries(area, mask_high)
        mask_high_parabola = self.correct_cut_boundaries(area, mask_high_parabola)
        mask_low = self.correct_cut_boundaries(area, mask_low)
        mask_low_parabola = self.correct_cut_boundaries(area, mask_low_parabola)
        
        cut_mask = (mask_high & mask_low & mask_high_parabola & mask_low_parabola)
        return cut_mask

    def cut_by(self, events):
        
        mask_near_wires = self.get_cut_masks(events,'WIMP')
        mask_far_wires = self.get_cut_mask_far_wires(events, 'WIMP')
        
        mask_near_wires[~events['cut_near_wires']] = True
        mask_far_wires[events['cut_near_wires']] = True 

        cut_mask = mask_near_wires & mask_far_wires
        
        return cut_mask




@strax.takes_config(
    strax.Option('min_norm_s2_width_cut', default=-14,  # change for the chi2 section
                 help='Minimum Chi2 norm of S2 width cut'),
    strax.Option('switch_from_chi2', default=10**4.1,
                 help='S2 area value where to switch from low s2 area to a parabolas for high area'),
    strax.Option('param_parabola_deg2', default= 0.05175732,
                 help='Fixed parameters of the polynomial function to apply at high energy, deg2 parameter'),
    strax.Option('param_parabola_deg1', default= -0.45385021,
                 help='Fixed parameters of the polynomial function to apply at high energy, deg1 parameter.')
    )


class S2WidthChi2(S2WidthBase):
    """
    Implementation of S2Width with chi2.
    """
    depends_on = ('event_basics',)
    provides = ('cut_s2_width_chi2',)
    cut_name = 'cut_s2_width_chi2'
    cut_description = 'S2Width with chi2 boundaries'
    __version__ = '0.1.2'
    child_plugin = True
    
    s2_secondary_sc_gain = straxen.URLConfig(
        default='cmt://se_gain?version=ONLINE&run_id=plugin.run_id',
        help='Single electron average gain value.',
        cache=True)
    
    def setup(self):
        self.sigma_to_r50p = norm.ppf(0.75) - norm.ppf(0.25)                
        
        self.parabola_par2 = self.config['param_parabola_deg2']
        self.parabola_par1 = self.config['param_parabola_deg1']
        
        #Get the last vertical offset of the parabola
        ref_high = self.cut_chi2(self.config['switch_from_chi2'], band = True)
        ref_low  = self.cut_chi2(self.config['switch_from_chi2'], band = False)
        #ref_high = self.cut_high_boundary(self.config['switch_boundary'], *param_quantile_high )))
        
        par0_high = (ref_high -
                     self.parabola_par2*np.log10(self.config['switch_from_chi2'])**2 -
                     self.parabola_par1*np.log10(self.config['switch_from_chi2']))
        par0_low = (ref_low -
                    self.parabola_par2*np.log10(self.config['switch_from_chi2'])**2 -
                    self.parabola_par1*np.log10(self.config['switch_from_chi2']))
        
        self.parabola_par0 = (par0_high + par0_low)/2
        self.delta_par0 = par0_high - self.parabola_par0

    
    def n_electron(self, area):
        return np.clip(area, self.s2_secondary_sc_gain, np.inf) / self.s2_secondary_sc_gain


    def get_cut_masks_chi2(self, events):
        area = events['s2_area']
        norm_width = self.normalize_width(events)

        mask_high_chi2     = ((area >= self.config['switch_boundary']) |
                              (norm_width < self.cut_chi2(area, True)))

        mask_low_chi2      = ((area >= self.config['switch_boundary']) | 
                              (norm_width > self.cut_chi2(area, False)))

        mask_high_chi2 = self.correct_cut_boundaries(area, mask_high_chi2)
        mask_low_chi2 = self.correct_cut_boundaries(area, mask_low_chi2)

        cut_mask = (mask_high_chi2 & mask_low_chi2)
        return cut_mask

    def cut_by(self, events):

        mask_high_parabola, mask_low_parabola = self.get_cut_mask_parabola(events)
        cut_mask = self.get_cut_masks_chi2(events)

        cut_mask = (cut_mask & mask_high_parabola & mask_low_parabola)

        return cut_mask


@strax.takes_config(
    strax.Option('single_electron_width', default=599.70428e-3,
                 help='50 % area width of a single electron signal (without any diffusion) [ms].'),
    strax.Option('reference_s2_width', default=400.29572e-3,
                 help='Empirically found S2 reference 50 % area width [ms].'),
    strax.Option('reference_s2_drift_time', default=1.0029191e-3,
                 help='Empirically found S2 reference drift time [ms].'),
)

class NaiveS2Width(strax.CutPlugin):
    """
    naive S2 width cut based on S2 width versus drift time. As the S2 width depends on area and
    drift time the more appropriate model is given by the S2Width cut. However, this cut is
    easier to adjust empirically for a first rough selection of data using the 5th and 95th 
    percentile lines.

    References:
        * v0.1.0 reference: xenon:peres:analysis:xenonnt:s2_width_cut_may
        * v0.0.0 reference: xenon:xenonnt:wenz:nveto:comissioning:nveto_straxen:ambecalibration
    """
    depends_on = ('event_info',)
    provides = ('cut_s2_width_naive',)
    cut_name = 'cut_s2_width_naive'
    cut_description = ('Naive S2 width cut which is defined in the S2 width versus drift time ' 
                       'space and ignores S2 area.')
    __version__ = '0.1.0'

    @np.errstate(invalid='ignore')
 
    def _diffusion_model(self, t, a, b):
        return np.sqrt(a*t + b)

    def s2width(self, t, kind):
        """Simplified width function fitted from P5 and P95"""
  
        if kind == "lower":
            a = 24.65469946878112
            b = -651811.2798348855
        elif kind == "upper":
            a = 45.57908198636116
            b = 4310876.7330609495
        else:
            raise ValueError('Kind should be upper or lower')

        ret = self._diffusion_model(t,a,b)

        return ret

    def cut_by(self, events):
        t = events['drift_time']
        width = events['s2_range_50p_area']
        mask_s2_width = (width >= self.s2width(t, 'lower'))
        mask_s2_width &= (width < self.s2width(t, 'upper'))
        return mask_s2_width


class Rn220S2Width(strax.CutPlugin):
    # Preliminary from analyst
    depends_on = ('event_info', )
    provides = ('cut_rn220_s2_width',)
    cut_name = 'cut_rn220_s2_width'
    cut_description = 'S2 width cut'
    __version__ = '0.0.0'
    a = np.logspace(2, 6.5, 201)
    _a = np.clip(np.logspace(2, 6.5, 201), 0, 1e3)
    b = chi2.isf(0.1, _a / 40) / (_a / 40)
    c = chi2.isf(0.9, _a / 40) / (_a / 40)
    s2w_ul = interp1d(a, b, bounds_error=False, fill_value='extrapolate')
    s2w_ll = interp1d(a, c, bounds_error=False, fill_value='extrapolate')

    def cut_by(self, events):
        w0 = 0
        D = 50
        v = 0.677
        mask = events['s2_range_50p_area'] / np.sqrt(
            w0 ** 2 + 2 * D * events['drift_time'] * 0.1 / v ** 2) / 1.349 < self.s2w_ul(events['s2_area'])
        mask &= events['s2_range_50p_area'] / np.sqrt(
            w0 ** 2 + 2 * D * events['drift_time'] * 0.1 / v ** 2) / 1.349 > self.s2w_ll(events['s2_area'])
        return mask


class S2Width():
    '''
    For testing we don't want s1_single_scatter to crash.
    '''
    warnings.warn("WARNING! You are fetching S2Widh, whih is only a placeholder from the past!")
    pass
