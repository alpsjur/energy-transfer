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
import gcm_filters
from gcmFilterFunction import filter, calculate_energy_transfer, \
    get_grid_vars,coarsen_grid, coarsen_data, coarsen_depth, readROMSfile
# import matplotlib.pyplot as plt
# from os.path import exists
from depth import sdepth, zslice
from joblib import Parallel, delayed

filepath = '/tos-project3/NS9081K/NORSTORE_OSL_DISK/NS9081K/shared/A4/A4_nudging_off/outputs/'
#filepath = '/home/alsjur/PhD/Data/test_data/A4/'
outpath = '/projects/NS9869K/LLC2160/A4_filtered/'
#outpath = '/home/alsjur/PhD/Kode/test_data/'

scales = np.geomspace(7000, 500000, num=30, dtype=int)[17:]
#all_days = np.arange(1,4452,10)
all_days = np.arange(1827,4452,10)
depth = 'mean'
filetype = 'avg'

coarsen_factor = 3

#all_days = np.arange(1481,2241,10)

tot_screens = 2
screen = 0#int(sys.argv[1])

days = np.array_split(all_days,tot_screens)[screen]
#days = np.arange(981,2231,10)
#days = np.arange(3251,4452,10)

### Grid information ###
coords={'X':{'center':'i', 'left':'i_g'}, 
    'Y':{'center':'j', 'left':'j_g'}, 
    's':{'center':'s_rho', 'outer':'s_w'}}


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
    elif scale <= 100000:
        iterations.append(20)
    elif scale <= 200000:
        iterations.append(30)
    else:
        iterations.append(60)


#%%
def calculate_filtered(day):
    tic = time.perf_counter()
    #filename = f'A4_filtered_day{day:04n}_depth{depth:03n}_coarse{coarsen_factor}.nc'
    filename = f'A4_filtered_day{day:04n}_depthmean_coarse{coarsen_factor}.nc'

    
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
    
    coarsen_filter = gcm_filters.Filter(
        filter_scale=coarsen_factor,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.TAPER,
        grid_type=gcm_filters.GridType.REGULAR,
    )
    #filter to intermediate scale
    Ubar = coarsen_filter.apply(ds.u, dims=['j', 'i'])
    UUbar = coarsen_filter.apply(ds.uu, dims=['j', 'i'])
    Vbar = coarsen_filter.apply(ds.v, dims=['j', 'i'])
    VVbar = coarsen_filter.apply(ds.vv, dims=['j', 'i'])
    UVbar = coarsen_filter.apply(ds.uv, dims=['j', 'i'])

    Ubar = Ubar.swap_dims({'i':'i_g'})
    UUbar = UUbar.swap_dims({'i':'i_g'})
    Vbar = Vbar.swap_dims({'j':'j_g'})
    VVbar = VVbar.swap_dims({'j':'j_g'})

    dsbar = xr.Dataset()
    dsbar['ubar'] = Ubar
    dsbar['vbar'] = Vbar
    dsbar['uubar'] = UUbar
    dsbar['vvbar'] = VVbar
    dsbar['uvbar'] = UVbar


    dsGridc = coarsen_grid(dsGrid, coarsen_factor)
    dsc = coarsen_data(dsbar, coarsen_factor)

    dsdc = coarsen_depth(dsGrid.h, coarsen_factor)
    #dsc = coarsen_data(ds, coarsen_factor)

    dsc['u'] = dsc['u'].swap_dims({'i_g':'i'})
    dsc['uu'] = dsc['uu'].swap_dims({'i_g':'i'})
    dsc['v'] = dsc['v'].swap_dims({'j_g':'j'})
    dsc['vv'] = dsc['vv'].swap_dims({'j_g':'j'})


    gridc = xgcm.Grid(dsGridc
                      , coords=coords
                      , periodic=False
                      )
    # gridc = xgcm.Grid(dsGridc
    #                  , metrics=metrics
    #                  , periodic=False
    #                  )

     
    grid_vars_visc, grid_vars_diff, dx_min = get_grid_vars(dsGridc, dsdc.values, depth=0, ROMS=True)
    
    
    bars = []
    
    for scale, n_iterations in zip(scales, iterations):
        #print(f'Starting scale {scale} with {n_iterations} iteration(s)')
        dsbar = filter(dsc, scale, grid_vars_visc, grid_vars_diff, dx_min, n_iterations)
        dsbar['pn'] = 1/dsGridc.dxF
        dsbar['pm'] = 1/dsGridc.dyF
        pi = calculate_energy_transfer(dsbar, gridc, ROMS=True)
        dsbar['energy_transfer'] = pi.energy_transfer
        bars.append(dsbar)

        
    ds_out = xr.concat(bars, dim='scale').drop_vars(['pn','pm'])
    ds_out['lon'] = grid.interp(dsGridc.lon_u, axis=['X'], boundary='extend')
    ds_out['lat'] = grid.interp(dsGridc.lat_u, axis=['X'], boundary='extend')
    ds_out.coords['scale'] = scales
    toc = time.perf_counter()
    runtime = (toc-tic)/60
    print(f"Runtime day {day} {runtime:0.4f} minutes")
    
    ds_out.to_netcdf(outpath+filename)
        
        # ax.plot(ds_out.scale,ds_out.energy_transfer.mean(dim=('i','j')))
        # ax.set_ylim(-2e-10,2e-10)
    #return None
    return ds_out

#ds = calculate_filtered(1827)
out = Parallel(n_jobs=4)(delayed(calculate_filtered)(day) for day in days)