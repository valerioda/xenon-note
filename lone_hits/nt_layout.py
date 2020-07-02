#Alvaro A. Loya Villalpando - loya.alvaro@gmail.com
#The functions below are used to generate a pandas dataframe of the PMT and hardware connections + a plot of these

import csv
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

def enum(array, meta, color, top_bot):
    patches = []
    if top_bot == 'PMTs_top':
        start, stop = 0, meta['PMTs_top']
    else:
        start, stop = meta['PMTs_top'], meta['PMTs_top']+meta['PMTs_bottom']   
    for ch in range(start, stop):
        for i, n in enumerate(array):
            if ch == n :
                circle = plt.Circle((meta['PMT_positions'][ch]['x'],meta['PMT_positions'][ch]['y']),\
                                    meta['PMTOuterRingRadius'])
                patches.append(circle)
                plt.annotate(str(ch), xy=(meta['PMT_positions'][ch]['x'],meta['PMT_positions'][ch]['y']),\
                 fontsize=12, ha='center', va='center')
            else:
                continue
    p = matplotlib.collections.PatchCollection(patches, alpha=1.0, edgecolor='black',  facecolor=color)
    ax = plt.gca()
    ax.add_collection(p)

def plot_pmts_hv_layout(PMTs,meta):    
    """ Function to plot the nT PMT arrays, diplaying each PMT's corresponding high voltage connector by 
    groups of similar color, and groups of PMTs connected to a common ADC by the same subcolor within the 
    larger group. 
    
    CHANNEL GROUPING MAP
    * 24 channels to each HV connector (8 PMTs x 3 digitizers max)
    * see https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:giovo:pmtcablingplan
    * see https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:dsg:daq:channel_groups


    XENONnT CABLING MAP SUMMARY
    * crates = 0 - 4 (physical location in DAQ room)
    * slots = 0 - 18 (slot on crate)
    * channels = 0 - 7 (digitizer channel)


    * 21 connectors - i.e. xy.z = connector xy (00-21), pin z (0,1,2)
    """

    #----TOP ARRAY----
    
    top_dict =  { 'ch_split': {'top_array' : {
        
                # VME crate 0
                '00.0' : [0,2,8,16,20,30,46,45],
                '00.1' : [1,6,10,17,19,31,33,43],
                '00.2' : [3,7,9,18,29,32,44,47],

                '01.0' : [4,13,23,26,34,37,49,65],
                '01.1' : [5,12,21,24,35,38,48,51],
                '01.2' : [11,14,22,25,36,39,50,52],

                '02.0' : [53,80,83,98,101,116,131,134],
                '02.1' : [66,68,82,97,100,115,133,149],
                '02.2' : [67,81,84,99,114,117,132,150],

                '03.0' : [148,166,181,195,198,211,237,223],
                '03.1' : [164,167,182,196,209,212,224,235],
                '03.2' : [165,180,183,197,210,246,225,236],

                '04.0' : [252,249,243,240,234,231,219,207],
                '04.1' : [208,218,221,232,241,244,247,250],
                '04.2' : [220,222,230,233,242,245,248,251],

                '06.0' : [200,186,172,169,154,151,136,118],
                '06.1' : [187,184,170,155,152,137,121,119],
                '06.2' : [120,153,135,138,168,171,185,199],

                '07.0' : [28,40,55,69,72,87,102,105],
                '07.1' : [15,41,56,70,73,85,88,103],
                '07.2' : [27,42,54,57,71,86,89,104],

                # VME crate 1
                '08.0' : [58,75,90,93,108,123,126,141],
                '08.1' : [59,76,91,106,109,124,139,142],
                '08.2' : [60,74,92,107,122,125,140,143],

                '09.0' : [61,64,78,96,110,128,144,146],
                '09.1' : [62,79,94,112,127,130,147,162],
                '09.2' : [63,77,95,111,113,129,145,163],

                '10.0' : [156,159,174,177,188,191,194,204],
                '10.1' : [157,160,175,178,189,192,203,205],
                '10.2' : [158,161,173,176,179,190,193,206],

                '05.0' : [201,202,213,215,217,227,229,238],
                '05.1' : [214,216,226,228,239,471]}
                  }
                }

    #---BOTTOM ARRAY---
    bot_dict =  { 'ch_split': {'bot_array' : {
        
                # VME crate 1 (continued)
                '15.0' : [253,258,261,266,269,279,282,294],
                '15.1' : [254,256,259,267,270,278,280,295],
                '15.2' : [255,257,262,260,268,271,281,293],

                '16.0' : [263,274,284,287,296,299,302,315],
                '16.1' : [264,272,275,285,297,300,314,316],
                '16.2' : [265,273,283,286,288,298,301,317],

                '21.0' : [309,312,324,326,329,340,343,357],
                '21.1' : [308,311,313,325,327,339,356,359],
                '21.2' : [310,323,328,341,342,344,355,358],

                '17.0' : [330,333,347,361,364,379,394,397],
        
                # VME crate 2
                '17.1' : [331,345,348,362,377,380,395,412],
                '17.2' : [332,346,360,363,378,381,396,411],

                '18.0' : [410,427,441,454,457,469,479,487],
                '18.1' : [425,428,442,455,467,470,480,488],
                '18.2' : [426,440,443,453,456,468,478,489],

                '19.0' : [372,375,390,393,407,421,424,438],
                '19.1' : [373,376,391,405,408,422,436,439],
                '19.2' : [374,389,392,406,409,420,423,437],

                '11.0' : [451,463,466,473,476,483,486,491],
                '11.1' : [452,461,464,474,477,481,484,492],
                '11.2' : [462,465,472,475,482,485,490,493],

                '13.0' : [365,383,398,415,429,432,445,460],
                '13.1' : [366,384,399,413,416,431,447,458],
                '13.2' : [367,382,400,414,430,444,446,459],

                '20.0' : [338,369,371,387,404,417,434,448],
                '20.1' : [353,368,386,401,403,418,435,449],
                '20.2' : [354,370,385,388,402,419,433,450],

                '14.0' : [276,291,304,307,320,334,337,351],
                '14.1' : [277,290,303,306,319,322,336,350],

                # VME crate 3
                '14.2' : [289,292,305,318,321,335,349,352]}
                  }
                }


    #create color lists for arrays
    color_top = ['royalblue', 'lightsteelblue', 'slategray', 'forestgreen', 'lime', 'springgreen', 'tan', 'darkgoldenrod',
                  'darkorange', 'lightcoral', 'firebrick', 'rosybrown', 'cadetblue', 'deepskyblue', 'steelblue', 'gray',
                 'silver', 'crimson', 'palevioletred', 'deeppink', 'chocolate', 'sienna', 'peachpuff', 'indianred',
                  'brown', 'red', 'aqua', 'teal', 'deepskyblue', 'khaki', 'darkkhaki', 'orange']

    color_bot = [ 'darkcyan', 'c', 'mediumspringgreen', 'g', 'dodgerblue', 'royalblue', 'slateblue', 'indianred', 'pink',
                 'red', 'royalblue', 'c', 'dodgerblue', 'peachpuff', 'peru', 'tan', 'violet', 'fuchsia', 'blueviolet',
                 'orangered', 'lightsalmon', 'chocolate', 'red', 'maroon', 'firebrick', 'goldenrod', 'burlywood', 'bisque',
                 'greenyellow', 'olivedrab', 'lightgreen']

    top_dict['ch_split'].update(bot_dict['ch_split'])
    pmt_map = {}
    pmt_map.update(top_dict)

    with open('ch_split.json', 'w') as fp:
        json.dump(pmt_map, fp)
    fp.close()

    
    #Plot PMT arrays with HV connector colors
    fig = plt.figure(figsize=(20,9))
    
    #top array
    plt.subplot(121)

    keys_top= []
    for key in pmt_map['ch_split']['top_array']:
        keys_top.append(key) #00.0, 00.1,...11.2, ...

    for i, key in enumerate(np.sort(keys_top)):
        enum(pmt_map['ch_split']['top_array'][key],meta, color_top[i],'PMTs_top')

    patches = []
    for ch in range(0, meta['PMTs_top']):
        circle = plt.Circle((meta['PMT_positions'][ch]['x'],meta['PMT_positions'][ch]['y']), meta['PMTOuterRingRadius'])
        patches.append(circle)
        plt.annotate(str(ch), xy=(meta['PMT_positions'][ch]['x'],meta['PMT_positions'][ch]['y']),\
                     fontsize=12, ha='center', va='center')

    p = matplotlib.collections.PatchCollection(patches, cmap='jet', alpha=1.0, edgecolor='black',  facecolor='none')
    ax = plt.gca()
    ax.add_collection(p)


    ax.add_collection(matplotlib.collections.PatchCollection([plt.Circle((0,0),meta['tpc_radius'])],\
                                                             facecolor='none', edgecolor='black', alpha=1.0))
    ax.text(0.05, 0.95, "Top PMTs", transform=ax.transAxes, horizontalalignment='left',fontsize=18,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

    plt.xlabel('x-position [cm]',fontsize=18)
    plt.ylabel('y-position [cm]',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.axis('equal')
    
    #LABEL SECTORS
    ax.set_xlim(-80,80)
    ax.set_ylim(-80,80)
    
    ax.text(-26, 68,"conn00", size=16, rotation=27,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(38, 62, "conn01", size=16, rotation=-27,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(70, 20, "conn02", size=16, rotation=-75,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(60, -40, "conn03", size=16, rotation=60,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(0, -70, "conn04", size=16, rotation=0,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(-50, -55, "conn05", size=16, rotation=-40,
            ha="center", va="center",
            bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(-69, -25, "conn06", size=16, rotation=-60,
            ha="center", va="center",
            bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(-66, 29,"conn07", size=16, rotation=63,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(-12, 10.5,"conn08", size=16, rotation=0,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))

    ax.text(20, 3, "conn09", size=16, rotation=0,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(0, -23.5, "conn10", size=16, rotation=0,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    #bottom array
    plt.subplot(122)

    keys_bot= []
    for key in pmt_map['ch_split']['bot_array']:
        keys_bot.append(key)

    for i, key in enumerate(np.sort(keys_bot)):
        enum(pmt_map['ch_split']['bot_array'][key],meta, color_bot[i], 'bot')

    patches = []
    for ch in range(meta['PMTs_top'], meta['PMTs_top']+meta['PMTs_bottom']):
        circle = plt.Circle((meta['PMT_positions'][ch]['x'],meta['PMT_positions'][ch]['y']), meta['PMTOuterRingRadius'])
        pp = patches.append(circle)
        plt.annotate(str(ch), xy=(meta['PMT_positions'][ch]['x'],meta['PMT_positions'][ch]['y']),\
                     fontsize=12, ha='center', va='center')

    p = matplotlib.collections.PatchCollection(patches, cmap='jet', alpha=1.0, edgecolor='black', facecolor='none')

    ax = plt.gca()
    ax.add_collection(p)
    ax.add_collection(matplotlib.collections.PatchCollection([plt.Circle((0,0),meta['tpc_radius'])],\
                                                             facecolor='none', edgecolor='black', alpha=1.0))
    ax.text(0.15, 0.95, "Bottom PMTs", transform=ax.transAxes, horizontalalignment='center', fontsize=18,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

    ax.get_yaxis().set_visible(False)
    plt.xlabel('x-position [cm]',fontsize=18)
    plt.xticks(fontsize=18)
    plt.axis('equal')
    plt.tight_layout()
    
    #LABEL SECTORS
    ax.text(-24, 66,"conn15", size=16, rotation=25,
     ha="center", va="center",
     bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(45, 55, "conn16", size=16, rotation=-40,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(73, 0, "conn17", size=16, rotation=-90,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(60, -40, "conn18", size=16, rotation=60,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(-10, -70, "conn11", size=16, rotation=-10,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(-69, -25, "conn13", size=16, rotation=-65,
            ha="center", va="center",
            bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(-66, 29,"conn14", size=16, rotation=63,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(3, 18,"conn21", size=16, rotation=0,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))

    ax.text(14, -11.2, "conn19", size=16, rotation=0,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(-22, -13, "conn20", size=16, rotation=-60,
         ha="center", va="center",
         bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.text(-46, -54, "conn05", size=16, rotation=0,
        ha="center", va="center",
        bbox=dict(facecolor='white', edgecolor='black'))

    plt.show()

#FUNCTIONS BELOW ARE USED TO CREATE PANDAS DATAFRAME OF PMT CONNECTION TO ADC/HV MODULES/ETC.
def find_between( s, first, last ):
    """ Function to separate crate.chanel.slot into individual components"""
    
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def find_between_r( s, first, last ):
    """ Function to separate crate.chanel.slot into individual components"""
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

def PMT_info(hv_connector_file = 'hv_connector_20032020.csv', 
             cable_map_file = 'xenonnt_cable_map.csv', plot = False):
    """Provides the following information associated to each signal channel in a 
    pandas dataframe for each signal channel:
    
    note: assinged value of '-1' means not applicable (NA) 
    
    1. pmt (PMT number [0-493])
    2. x (x-coordinate of pmt)
    3. y (y-coordinate of pmt)
    4. array (top or bottom)
    5. pmt_lg (0vbb channel number [500-752] for corresponding pmt)
    6. adc_dm (crate.slot.channel of dark matter adc corresponding to pmt)
    7. adc_0vbb (crate.slot.channel of neutrinoless double beta decay adc corresponding to pmt)
    8. adc_dm_id (dark matter adc id value corresponding to pmt)
    10. adc_0vbb_id (0vbb adc id value where pmt is connected)
    11. amp (crate.slot.channel of amplifier corresponding to pmt)
    12. dm_crate (crate of dark matter adc corresponding to pmt)
    13. dm_slot (slot of adrk matter adc corresponding to pmt)
    14. dm_channel (channel of dark matter adc corresponding to pmt)
    15. 0vbb_crate (crate of neutrinoless double beta decay adc corresponding to pmt)
    16. 0vbb_slot (slot of neutrinoless double beta decay adc corresponding to pmt)
    17. 0vbb_channel (channel of neutrinoless double beta decay adc corresponding to pmt)
    18. amp_crate (crate of amplifier adc corresponding to pmt)
    19. amp_slot (slot of amplifier corresponding to pmt)
    20. amp_channel (channel of amplifier corresponding to pmt)
    21. hv_connector (hv module number corresponding to pmt)
    22. hv_connector_pin (hv module pin correspoinding to pmt)
    23. hv_crate (crate where hv module is located)
    24. hv_slot (slot where hv module is located)
    """
    
    PMTs = []
    
    # hexagonal top pattern
    PMTs_top = 253
    row_nPMT = [0, 6, 9, 12, 13, 14, 15, 16, 17, 16, 17, 16, 17, 16, 15, 14, 13, 12, 9, 6]
    row_nPMT_cumsum = [sum(row_nPMT[0:x + 1]) for x in range(0, len(row_nPMT))]
    y_start = 62.1863

    for pmt in range(0, PMTs_top):
        _row = [i for i,x in enumerate(row_nPMT_cumsum) if x <= pmt][-1]
        _row_position = pmt - row_nPMT_cumsum[_row]

        y = y_start - _row * 6.90959
        if (row_nPMT[_row+1] % 2):
            x = (int(row_nPMT[_row+1] / -2.) + _row_position) * 7.97850
        else:
            x = (int(row_nPMT[_row+1] / -2.) + _row_position) * 7.97850 + 3.98925

        PMTs.append({'pmt': pmt,'x': x, 'y': y, 'r': np.sqrt(x**2 + y**2)})

    # hexagonal bottom pattern
    PMTs_bottom = 241
    row_nPMT = [0, 4, 9, 10, 13, 14, 15, 16, 15, 16, 17, 16, 15, 16, 15, 14, 13, 10, 9, 4]
    row_nPMT_cumsum = [sum(row_nPMT[0:x + 1]) for x in range(0, len(row_nPMT))]
    y_start = 62.1863

    for pmt in range(0, PMTs_bottom):
        _row = [i for i,x in enumerate(row_nPMT_cumsum) if x <= pmt][-1]
        _row_position = pmt - row_nPMT_cumsum[_row]

        y = y_start - _row * 6.90959
        if (row_nPMT[_row+1] % 2):
            x = (int(row_nPMT[_row+1] / -2.) + _row_position) * 7.97850
        else:
            x = (int(row_nPMT[_row+1] / -2.) + _row_position) * 7.97850 + 3.98925

        PMTs.append({'pmt': PMTs_top+pmt,'x': x, 'y': y, 'r': np.sqrt(x**2 + y**2)})

    meta = {'tpc_radius': 66.4,
             'PMTs_top': PMTs_top,
             'PMTs_bottom': PMTs_bottom,
             'PMT_positions': PMTs,
             'PMTOuterRingRadius': 3.875, # cm
            }
    
    #assign top or bottom to each PMT
    for i in range(len(PMTs)):
        if PMTs[i]['pmt'] <= 252:
            PMTs[i].update({'array': 'top'})
            #add the low gain (lg) channels
            PMTs[i].update({'pmt_lg': int(PMTs[i]['pmt'])+500})
        else:
            PMTs[i].update({'array': 'bottom'})
            PMTs[i].update({'pmt_lg': -1}) #-1 means NA/not applicable
            
    #assign adc values
    cable_map_file = cable_map_file

    with open(cable_map_file, 'rt') as f:
        csv_reader = csv.reader(f)

        counter = 0
        for line in csv_reader:
            if line[0] == 'PMT Location':
                continue
            pmt = line[0]
            array = line[1]

            dm_adc = line[2] #crate.slot.channel
            v0bb_adc = line[3] #crate.slot.channel
            dm_adc_id = line[4]
            v0bb_adc_id = line[5]
            amp = line[6] #crate.slot.channel
            PMTs[counter].update({'adc_dm':dm_adc})
            PMTs[counter].update({'adc_0vbb':v0bb_adc})
            PMTs[counter].update({'adc_dm_id':dm_adc_id})
            PMTs[counter].update({'adc_0vbb_id':v0bb_adc_id})
            PMTs[counter].update({'amp':amp})

            #assign crate, slot and channel of ADC individually
            
            #dm
            dm_adc_split = dm_adc.split()
            dm_adc_crate = dm_adc_split[0][0]
            dm_adc_slot = find_between(dm_adc,'.','.')
            dm_adc_channel = find_between_r(dm_adc,'.','')
            PMTs[counter].update({'dm_crate':dm_adc_crate})
            PMTs[counter].update({'dm_slot':dm_adc_slot})
            PMTs[counter].update({'dm_channel':dm_adc_channel})

            #0vbb
            if array == 'bottom':
                PMTs[counter].update({'0vbb_crate':-1})
                PMTs[counter].update({'0vbb_slot':-1})
                PMTs[counter].update({'0vbb_channel':-1})
            else:
                v0bb_adc_split = v0bb_adc.split()
                v0bb_adc_crate = v0bb_adc_split[0][0]
                v0bb_adc_slot = find_between(v0bb_adc,'.','.')
                v0bb_adc_channel = find_between_r(v0bb_adc,'.','')
                PMTs[counter].update({'0vbb_crate':v0bb_adc_crate})
                PMTs[counter].update({'0vbb_slot':v0bb_adc_slot})
                PMTs[counter].update({'0vbb_channel':v0bb_adc_channel})
            
            #amp
            amp_split = amp.split()
            amp_crate = amp_split[0][0]
            amp_slot = find_between(amp,'.','.')
            amp_channel = find_between_r(amp,'.','')
            PMTs[counter].update({'amp_crate':amp_crate})
            PMTs[counter].update({'amp_slot':amp_slot})
            PMTs[counter].update({'amp_channel':amp_channel})
            
            counter += 1
            
            
            
    #assign HV connector module and pin number - board and crate info
    with open(hv_connector_file, 'rt') as f:
        csv_reader = csv.reader(f)

        counter = 0
        for line in csv_reader:
            if line[0] == 'PMT':
                continue
            pmt = line[0]
            hv_connector = line[1]
            hv_connector_pin = line[2]
            
            PMTs[counter].update({'hv_connector':hv_connector})
            PMTs[counter].update({'hv_connector_pin':hv_connector_pin})
            
            #hv crate, board, channel info
            hv = line[3] #crate.slot.channel
            hv_split = hv.split()
            hv_crate = hv_split[0][0]
            hv_slot = find_between(hv,'.','.')
            hv_channel = find_between_r(hv,'.','')
            PMTs[counter].update({'hv_crate':hv_crate})
            PMTs[counter].update({'hv_slot':hv_slot})

            counter += 1
    if plot == True:
        plot_pmts_hv_layout(PMTs,meta)
        
    return (pd.DataFrame(data=PMTs))
    

def missing_channels_info(PMT_info,missing_channels):
    """Prints information about missing channels, sorted by ADC value"""
    dm_d = [] 
    v0bb_d = []
    for i in range(len(PMT_info)):
        if PMT_info['pmt'][i] in missing_channels:
            dm_d.append((PMT_info['pmt'][i],PMT_info['adc_dm_id'][i],PMT_info['dm_channel'][i],
                         PMT_info['dm_crate'][i]))
        if PMT_info['pmt_lg'][i] in missing_channels:
            v0bb_d.append((PMT_info['pmt_lg'][i],PMT_info['adc_0vbb_id'][i],PMT_info['0vbb_channel'][i],
                         PMT_info['0vbb_crate'][i]))
            

    dm_info_missing_channels = pd.DataFrame(dm_d,columns=('signal_channel', 'adc_dm_id','dm_adc_channel', 'dm_crate'))
    
    v0bb_info_missing_channels = pd.DataFrame(v0bb_d,columns=('signal_channel', 'adc_0vbb_id','0vbb_channel', '0vbb_crate'))
    
    pd.set_option('display.max_rows', len(missing_channels))
    
    return (dm_info_missing_channels.sort_values('adc_dm_id'), 
            v0bb_info_missing_channels.sort_values('adc_0vbb_id'))