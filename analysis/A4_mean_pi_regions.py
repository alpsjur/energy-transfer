#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 13:13:27 2022

@author: alsjur
"""
import xarray as xr
import xgcm
import numpy as np
import pickle
import sys
import glob
from LLC2A4 import readROMSfile, LLC2A4

datadir = '/projects/NS9869K/LLC2160/A4_filtered/'
outdir = '/nird/home/annals/data_temp/'

depths = ['mean']

gridData = readROMSfile('/tos-project3/NS9081K/NORSTORE_OSL_DISK/NS9081K/shared/A4/A4_nudging_off/outputs/'+'ocean_avg_1827.nc')
# LLCgrid = xr.open_dataset('/projects/NS9869K/LLC260/LLC2160_grid.nc')

istart = int(sys.argv[1])
istop = int(sys.argv[2])
jstart = int(sys.argv[3])
jstop = int(sys.argv[4])
region = int(sys.argv[5])

# istart, jstart = LLC2A4([istart, jstart], gridData, LLCgrid)
# istop, jstop = LLC2A4([istop, jstop], gridData, LLCgrid)

def spectrum_at_depth(depth, grid):
    dss = []
    #files = sorted(glob.glob(datadir+f'A4_filtered_day*_depth{depth:03n}.nc'))
    files = sorted(glob.glob(datadir+'A4_filtered_day*_depthmean.nc'))
    for file in files:
       ds = xr.open_dataset(file)#.isel(i=slice(istart,istop),j=slice(jstart,jstop),
                                 #   i_g=slice(istart,istop),j_g=slice(jstart,jstop)
                                 #                 )
                        
       ds = ds.isel(i=slice(istart,istop),j=slice(jstart,jstop),
                i_g=slice(istart,istop),j_g=slice(jstart,jstop)
                )                           
       dss.append(ds)
       
    data = xr.concat(dss, dim='time')

    u = grid.interp(data.ubar, axis=['X'], boundary='fill')
    v = grid.interp(data.vbar, axis=['Y'], boundary='fill')
    pi = data.energy_transfer

    ls = data.scale.values

    E = ((u**2+v**2)/2)
    e = -E.differentiate('scale')#*scales**2
    
    # meanpi = grid.average(pi.mean(dim=('time')), ['X','Y'])
    # meane = grid.average(e.mean(dim=('time')), ['X','Y'])*(ls**2)
    
    # not 100% correct, since grid is not regular. 
    meanpi = pi.mean(dim=('time', 'i', 'j'))
    meane = e.mean(dim=('time', 'i', 'j'))*(ls**2)
    
    stdpi = pi.std(dim=('time', 'i', 'j'))
    stde = e.std(dim=('time', 'i', 'j'))*(ls**2)
    
    #meane = np.gradient(meanE.values, scales)
    
    return meanpi.load(), stdpi.load(), meane.load(), stde.load()


gridData = xr.open_dataset('/projects/NS9869K/LLC2160/LLC2160_grid.nc')
bath = gridData.Depth
gridData = gridData.isel(i=slice(istart,istop),j=slice(jstart,jstop),
                 i_g=slice(istart,istop),j_g=slice(jstart,jstop)
                 )



### Grid information ###
coords={'X':{'center':'i', 'left':'i_g'}, 
    'Y':{'center':'j', 'left':'j_g'}, 
    's':{'center':'s_rho', 'outer':'s_w'}}

grid = xgcm.Grid(gridData
                 , coords=coords
                 , periodic=False
                 )

datadict = {
    }
for depth in depths:
    
    pi, dpi, e, de = spectrum_at_depth(depth, grid)
    
    scales = 1/pi.scale.values


    # save results for later
    datadict[f'depth{depth}'] = {
        'pi' : pi,
        'e' : e,
        'dpi' : dpi,
        'de' : de,
        'scales' : scales,
        'region' : region
        }

#with open(outdir+f'A4_region{region}.pickle', 'wb') as f:
with open(outdir+f'A4_depthmean_region{region}.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(datadict, f, pickle.HIGHEST_PROTOCOL)
