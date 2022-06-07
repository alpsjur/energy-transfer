#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:26:08 2022

@author: alsjur
"""
import numpy as np
import xarray as xr
import time 
import xgcm
from gcmFilterFunction import filter, calculate_energy_transfer, get_grid_vars
from os.path import exists
import sys
from joblib import Parallel, delayed

# sycle through days in a way than ensures a spread
start_day = 0
stop_day = 778
step = 5
tot_screens = 4
screen = 1#int(sys.argv[1])

all_days = np.arange(start_day,stop_day,step)
days = np.array_split(all_days,tot_screens)[screen]

print(days)
# days = []

# start = 0
# while len(days)<len(linear_days):
#     for d in linear_days[start::scycle]:
#         days.append(d)
#     start += 1
    

def calculate_filtered(day):
    tic = time.perf_counter()
    
    filepath = '/projects/NS9869K/LLC2160/'
    #filepath = '/home/alsjur/PhD/Kode/test_data/'
    outpath = '/projects/NS9869K/LLC2160/LLC_filtered/'

    scales = np.geomspace(2100,150000,num=20,dtype=int)
    #scales = np.geomspace(2100,15000,num=3,dtype=int)
    level = 21
    filetype = 'snapshot'
    
    # number of iterations depend on scale
    iterations = []
    for scale in scales:
        if scale <= 8000:
            iterations.append(1)
        elif scale <= 100000:
            iterations.append(20)
        else:
            iterations.append(30)
        
    ### Grid information ###
    dsGrid = xr.open_dataset(filepath+'LLC2160_grid.nc')#.isel(i=slice(1000,1500),
                                                         #     i_g=slice(1000,1500),
                                                         #     j=slice(0,500),
                                                         #     j_g=slice(0,500)
                                                         #     )#.squeeze()

    # create grid object 
    metrics = {
        ('X',): ['dxF', 'dxC', 'dxG'],    # X distances
        ('Y',): ['dyF', 'dyC', 'dyG'],    # Y distances
        ('X', 'Y'): ['rA', 'rAs', 'rAw']  # Areas
    }

    grid = xgcm.Grid(dsGrid
                     , metrics=metrics
                     , periodic=False
                     )
    
    maskU = xr.open_dataset(filepath+f'LLC2160_U_Arctic_day_{day:04n}_k{level:03n}_{filetype}.nc').squeeze().U.values
    grid_vars_visc, grid_vars_diff, dx_min = get_grid_vars(dsGrid, maskU)
    filename = f'LLC2160_filtered_day{day:03n}_k{level:02n}.nc'

    
    print(f'Level {level} day {day}')
    dsU = xr.open_dataset(filepath+f'LLC2160_U_Arctic_day_{day:04n}_k{level:03n}_{filetype}.nc').squeeze()
    dsV = xr.open_dataset(filepath+f'LLC2160_V_Arctic_day_{day:04n}_k{level:03n}_{filetype}.nc').squeeze()
    # set fill value of 9.9..e36 to nan
    dsU = dsU.where(dsU['U']<1e30)
    dsV = dsV.where(dsV['V']<1e30)

    # dsU = dsU.sel(i_g=slice(i_start,i_stop), j=slice(j_start,j_stop))
    # dsV = dsV.sel(i=slice(i_start,i_stop), j_g=slice(j_start,j_stop))
    
    ds = xr.Dataset()
    ds['u'] = dsU.U
    ds['v'] = dsV.V
    ds['u'] = ds.u.swap_dims({'i_g':'i'})
    ds['v'] = ds.v.swap_dims({'j_g':'j'})
    ds['uu'] = ds.u*ds.u
    ds['vv'] = ds.v*ds.v
    ds['uv'] = grid.interp(dsU.U, axis=['X'], boundary='fill')*grid.interp(dsV.V, axis=['Y'], boundary='fill')


    #for n_iterations in iterations:
    #    print(f'Starting {n_iterations} iteration(s)')
    bars = []
    
    for scale, n_iterations in zip(scales, iterations):
        print(f'Starting scale {scale} with {n_iterations} iteration(s)')
        dsbar = filter(ds, scale, grid_vars_visc, grid_vars_diff, dx_min, n_iterations)
        pi = calculate_energy_transfer(dsbar, grid)
        dsbar['energy_transfer'] = pi.energy_transfer
        bars.append(dsbar)
        
    ds_out = xr.concat(bars, dim='scale').drop_vars(['CS', 'SN', 'Depth', 'dxF',
                                                 'dyF', 'rA', 'rLowC', 'rSurfC',
                                                 'dxC', 'dyG', 'rAw', 'rLowW',
                                                 'rSurfW', 'dxG', 'dyC', 'rAs',
                                                 'rLowS', 'rSurfS', 'rAz', 'dxV',
                                                 'dyU'
                                                 # these variables can be found in the grid file
                                                 #, 'XC', 'YC', 'XG', 'YG'
                                                 ])
    ds_out.coords['scale'] = scales
    toc = time.perf_counter()
    runtime = (toc-tic)/60
    print(f"Runtime day {day} {runtime:0.4f} minutes")
    
    
    ds_out.to_netcdf(outpath+filename)
            
    return None

out = Parallel(n_jobs=4)(delayed(calculate_filtered)(day) for day in days)
            