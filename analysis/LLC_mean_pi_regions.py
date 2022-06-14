#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:33:22 2022

@author: alsjur
"""
import xarray as xr
import xgcm
import numpy as np
import pickle
import sys

datadir = '/projects/NS9869K/LLC2160/gcm_filtered/'
outdir = '/nird/home/annals/data_temp/'

levels = [16, 21, 28, 34, 39]


start_day = 0
# want exactly two years
stop_day = 365*2#778
step = 5
days = np.arange(start_day,stop_day,step)

idx_start = int(sys.argv[1])
idx_stop = int(sys.argv[2])
idy_start = int(sys.argv[3])
idy_stop = int(sys.argv[4])
region = int(sys.argv[5])


def spectrum_at_level(level, days, grid):
    dss = []
    files = [datadir+f'LLC2160_filtered_day{day:03n}_k{level:02n}.nc' for day in days]
    for file in files:
       ds = xr.open_dataset(file)#.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop),
                                 #   i_g=slice(idx_start,idx_stop),j_g=slice(idy_start,idy_stop)
                                 #                 )
                        
       ds = ds.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop),
                i_g=slice(idx_start,idx_stop),j_g=slice(idy_start,idy_stop)
                )                           
       dss.append(ds)
       
    data = xr.concat(dss, dim='time')

    u = grid.interp(data.ubar, axis=['X'], boundary='fill')
    v = grid.interp(data.vbar, axis=['Y'], boundary='fill')
    pi = data.energy_transfer

    ls = data.scale.values

    E = ((u**2+v**2)/2)
    e = -E.differentiate('scale')#*scales**2
    
    meanpi = grid.average(pi.mean(dim=('time')), ['X','Y'])
    meane = grid.average(e.mean(dim=('time')), ['X','Y'])*(ls**2)
    
    stdpi = pi.std(dim=('time', 'i', 'j'))
    stde = e.std(dim=('time', 'i', 'j'))*(ls**2)
    
    #meane = np.gradient(meanE.values, scales)
    
    return meanpi.load(), stdpi.load(), meane.load(), stde.load()
    
files = [[datadir+f'LLC2160_filtered_day{day:03n}_k{level:02n}.nc' for \
          level in levels] for day in days]


gridData = xr.open_dataset('/projects/NS9869K/LLC2160/LLC2160_grid.nc')
bath = gridData.Depth
gridData = gridData.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop),
                 i_g=slice(idx_start,idx_stop),j_g=slice(idy_start,idy_stop)
                 )



# create grid object 
metrics = {
    ('X',): ['dxF', 'dxC', 'dxG'],    # X distances
    ('Y',): ['dyF', 'dyC', 'dyG'],    # Y distances
    ('X', 'Y'): ['rA', 'rAs', 'rAw']  # Areas
}

grid = xgcm.Grid(gridData
                 , metrics=metrics
                 , periodic=False
                 )


datadict = {
    }
for level in levels:
    
    pi, dpi, e, de = spectrum_at_level(level, days, grid)
    
    scales = 1/pi.scale.values


    # save results for later
    datadict[f'level{level}'] = {
        'pi' : pi,
        'e' : e,
        'dpi' : dpi,
        'dd' : de,
        'scales' : scales,
        'region' : region
        }

with open(outdir+f'LLC_region{region}.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(datadict, f, pickle.HIGHEST_PROTOCOL)
