#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:21:44 2022

@author: alsjur
"""
import numpy as np
import xarray as xr
import time 
import xgcm
from gcmFilterFunction import filter, calculate_energy_transfer, get_grid_vars
# import matplotlib.pyplot as plt
# from os.path import exists
from depth import sdepth, zslice
from joblib import Parallel, delayed

filepath = '/tos-project3/NS9081K/NORSTORE_OSL_DISK/NS9081K/shared/A4/A4_nudging_off/outputs/'
#filepath = '/home/alsjur/PhD/Data/test_data/A4/'
outpath = '/projects/NS9869K/LLC2160/A4_filtered/'
#outpath = '/home/alsjur/PhD/Kode/test_data/'

scales = np.geomspace(7000, 500000, num=30, dtype=int)[:22]
# all_days = np.arange(1,4452,10)
all_days = np.arange(1827,4452,10)
depth = 'mean'
filetype = 'avg'

#all_days = np.arange(1481,2241,10)

tot_screens = 2
screen = 1#int(sys.argv[1])

days = np.array_split(all_days,tot_screens)[screen]

### Grid information ###
coords={'X':{'center':'i', 'left':'i_g'}, 
    'Y':{'center':'j', 'left':'j_g'}, 
    's':{'center':'s_rho', 'outer':'s_w'}}

def readROMSfile(filename):
    '''
    This functions makes sure that the dimensions match between i and i_g.
    This means that we need to remove one row and one column.
    Here, the first row and column is removed, menaing that 
    i_g and j_g is later spesified as left in xgcm coords.
    If the last clumn/row is removed instead, i_g and j_g is right in the 
    coordinates. 
    '''
    # read file with xarray
    ds_temp = xr.open_dataset(filename)
    # remove first column and row 
    ds = ds_temp.isel(eta_rho=slice(1,None),
                      xi_rho=slice(1,None),
                      eta_u=slice(1,None),
                      xi_v=slice(1,None)
                      )
    # rename variables to match MITgcm
    ds = ds.rename({'eta_rho': 'j'
                    , 'xi_rho': 'i'
                    , 'eta_u': 'j'
                    , 'xi_u': 'i_g'
                    , 'eta_v': 'j_g'
                    , 'xi_v' : 'i'
                    , 'lon_rho' : 'XC'
                    , 'lat_rho' : 'YC'
                    })
    
    dx = 1/ds.pm.values
    dy = 1/ds.pn.values
    area = dx*dy

    ds = ds.assign(dxF=(['j','i'],dx), dyF=(['j','i'],dy), rA=(['j','i'],area),
                   dxC=(['j','i_g'],dx), dyC=(['j_g','i'],dy), rAs=(['j_g','i'],area),
                   dxG=(['j_g','i'],dx), dyG=(['j','i_g'],dy), rAw=(['j','i_g'],area),
                   dxV=(['j_g','i_g'],dx), dyU=(['j_g','i_g'],dy)
        )
    return ds


dsGrid = readROMSfile(filepath+'ocean_avg_1827.nc')

grid = xgcm.Grid(dsGrid
                 , coords=coords
                 , periodic=False
                 )

# number of iterations depend on scale
iterations = []
for scale in scales:
    if scale <= 15000:
        iterations.append(1)  
    elif scale <= 30000:
        iterations.append(5)
    elif scale <= 60000:
        iterations.append(15)
    elif scale <= 100000:
        iterations.append(30)
    elif scale <= 200000:
        iterations.append(60)
    else:
        iterations.append(75)

grid_vars_visc, grid_vars_diff, dx_min = get_grid_vars(dsGrid, dsGrid.h.values, depth=depth, ROMS=True)

#%%
def calculate_filtered(day):
    tic = time.perf_counter()
    #filename = f'A4_filtered_day{day:04n}_depth{depth:03n}.nc'
    filename = f'A4_filtered_day{day:04n}_depthmean.nc'
    
    print(f'Depth {depth} day {day}')

    ds_temp = readROMSfile(filepath+f'ocean_{filetype}_{day:04n}.nc').squeeze()
    
    H = ds_temp.h.values
    C = ds_temp.Cs_r.values
    Hc = ds_temp.hc.values
    z_rho = sdepth(H, Hc, C)
    landmask = ds_temp.mask_rho.values
    
    # u = zslice(ds_temp.u, z_rho, -depth, Vtransform=2)
    # v = zslice(ds_temp.v, z_rho, -depth, Vtransform=2)
    
    # u[H<depth] = 0
    # v[H<depth] = 0
    
    # u[landmask==0] = 0
    # v[landmask==0] = 0

    u = ds_temp.ubar.values
    v = ds_temp.vbar.values

    u[landmask==0] = 0
    v[landmask==0] = 0

    ds_temp['u'] = (('j','i_g'),u)
    ds_temp['v'] = (('j_g','i'),v)
    
    ds = xr.Dataset()
    ds['u'] = ds_temp.u#bar
    ds['v'] = ds_temp.v#bar
    utemp = grid.interp(ds.u, axis=['X'], boundary='fill')
    vtemp = grid.interp(ds.v, axis=['Y'], boundary='fill')
    ds['uv'] = utemp*vtemp
    ds['u'] = ds.u.swap_dims({'i_g':'i'})
    ds['v'] = ds.v.swap_dims({'j_g':'j'})
    ds['uu'] = ds.u*ds.u
    ds['vv'] = ds.v*ds.v
    
    bars = []
    
    for scale, n_iterations in zip(scales, iterations):
        #print(f'Starting scale {scale} with {n_iterations} iteration(s)')
        dsbar = filter(ds, scale, grid_vars_visc, grid_vars_diff, dx_min, n_iterations)
        dsbar['pn'] = ds_temp.pn
        dsbar['pm'] = ds_temp.pm
        pi = calculate_energy_transfer(dsbar, grid, ROMS=True)
        dsbar['energy_transfer'] = pi.energy_transfer
        bars.append(dsbar)

        
    ds_out = xr.concat(bars, dim='scale').drop_vars(['pn','pm'])
    ds_out['lon'] = grid.interp(ds.lon_u, axis=['X'], boundary='extend')
    ds_out['lat'] = grid.interp(ds.lat_u, axis=['X'], boundary='extend')
    ds_out.coords['scale'] = scales
    toc = time.perf_counter()
    runtime = (toc-tic)/60
    print(f"Runtime day {day} {runtime:0.4f} minutes")
    
    ds_out.to_netcdf(outpath+filename)
        
        # ax.plot(ds_out.scale,ds_out.energy_transfer.mean(dim=('i','j')))
        # ax.set_ylim(-2e-10,2e-10)
    return None
    #return ds_out

out = Parallel(n_jobs=6)(delayed(calculate_filtered)(day) for day in days)