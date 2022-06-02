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
from gcmFilterFunction import filter, calculate_energy_transfer
import matplotlib.pyplot as plt
from os.path import exists
from depth import sdepth, zslice


filepath = '/tos-project3/NS9081K/NORSTORE_OSL_DISK/NS9081K/shared/A4/A4_nudging_off/outputs/'
#filepath = '/home/alsjur/PhD/Kode/test_data/'
outpath = '/projects/NS9869K/LLC2160/A4_filtered/'
#outpath = '/home/alsjur/PhD/Kode/test_data/'

scales = np.geomspace(7000, 250000, num=20, dtype=int)
days = np.arange(1,4452,10)
levels = [21]
depths = [100, 50, 5]
filetype = 'avg'

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
                    })
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
    elif scale <= 100000:
        iterations.append(20)
    elif scale <= 200000:
        iterations.append(50)
    else:
        iterations.append(75)

# grid info centered at T-points
wet_mask_t = dsGrid.mask_rho
dxT = 1/dsGrid.pm
dyT = 1/dsGrid.pn
area = dxT*dyT
# grid info centered at U-points
dxCu = dxT
dyCu = dyT
area_u = area
# grid info centered at V-points
dxCv = dxT
dyCv = dyT
area_v = area
# # grid info centered at vorticity points
wet_mask_q = wet_mask_t
dxBu = dxT
dyBu = dyT

dx_min = min(dxT.where(wet_mask_t).min(), dyT.where(wet_mask_t).min())
dx_min = dx_min.values

dx_max = max(dxT.max(), dyT.max(), dxCu.max(), dyCu.max(), dxCv.max(), \
              dyCv.max(), dxBu.max(), dyBu.max())
dx_max = dx_max.values

kappa_iso = xr.ones_like(dxT)
kappa_aniso = xr.zeros_like(dyT)

kappa_w = xr.ones_like(dxCu)
kappa_s = xr.ones_like(dxCu)

grid_vars_visc={
    'wet_mask_t': wet_mask_t, 'wet_mask_q': wet_mask_q,
    'dxT': dxT, 'dyT': dyT, 
    'dxCu': dxCu, 'dyCu': dyCu, 'area_u': area_u, 
    'dxCv': dxCv, 'dyCv': dyCv, 'area_v': area_v,
    'dxBu': dxBu, 'dyBu': dyBu,
    'kappa_iso': kappa_iso, 'kappa_aniso': kappa_aniso
}

grid_vars_diff={
    'wet_mask': wet_mask_t, 
    'dxw' : dxCu, 'dyw' : dyCu,
    'dxs' : dxCv, 'dys' : dyCv,
    'area' : area, 'kappa_w' : kappa_w, 'kappa_s' : kappa_s
}
#grid_vars_visc, grid_vars_diff, dx_min = get_grid_vars(dsGrid)

fig, ax = plt.subplots()

for depth in depths:
    for day in days:
        tic = time.perf_counter()
        #filename = f'A4_filtered_day{day:03n}_depth{depth:03n}.nc'
        filename = f'A4_filtered_day{day:03n}_depth{depth:03n}.nc'
        
        if exists(outpath+filename):
            print('hei')
            continue
        
        print(f'Depth {depth} day {day}')

        ds_temp = readROMSfile(filepath+f'ocean_{filetype}_{day:04n}.nc').squeeze()
        
        H = ds_temp.h.values
        C = ds_temp.Cs_r.values
        Hc = ds_temp.hc.values
        z_rho = sdepth(H, Hc, C)
        
        ds_temp['u'] = (('j','i_g'),zslice(ds_temp.u, z_rho, -depth))
        ds_temp['v'] = (('j_g','i'),zslice(ds_temp.v, z_rho, -depth))
        
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
            if exists(outpath+filename):
                continue
            print(f'Starting scale {scale} with {n_iterations} iteration(s)')
            dsbar = filter(ds, scale, grid_vars_visc, grid_vars_diff, dx_min, n_iterations)
            dsbar['pn'] = ds_temp.pn
            dsbar['pm'] = ds_temp.pm
            pi = calculate_energy_transfer(dsbar, grid, ROMS=True)
            dsbar['energy_transfer'] = pi.energy_transfer
            bars.append(dsbar)
            
        if exists(outpath+filename):
            continue
            
        ds_out = xr.concat(bars, dim='scale').drop_vars(['pn','pm'])
        ds_out['lon'] = grid.interp(ds.lon_u, axis=['X'], boundary='extend')
        ds_out['lat'] = grid.interp(ds.lat_u, axis=['X'], boundary='extend')
        ds_out.coords['scale'] = scales
        toc = time.perf_counter()
        runtime = (toc-tic)/60
        print(f"Runtime day {day} {(toc - tic)/60:0.4f} minutes")
        
        ds_out.to_netcdf(outpath+filename)
        
        # ax.plot(ds_out.scale,ds_out.energy_transfer.mean(dim=('i','j')))
        # ax.set_ylim(-2e-10,2e-10)