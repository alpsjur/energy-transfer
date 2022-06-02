#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 13:54:25 2022

@author: alsjur
"""
import xarray as xr
import xgcm
import xesmf as xe
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj
import gcm_filters

import sys
sys.path.insert(0, '/home/alsjur/nird/energy-transfer/analysis')
from gcmFilterFunction import get_grid_vars, filter, calculate_energy_transfer, coarsen_grid, coarsen_data


#%%

datapath = '/home/alsjur/PhD/Data/test_data/LLC2160/'
figpath = '/home/alsjur/PhD/Figurer/EnergyTransfer/method/'

scales = np.geomspace(80000,300000,num=10,dtype=int)
iterations = [15]
day = 0


# load data
dsU = xr.open_dataset(datapath+f'LLC2160_U_Arctic_day_{day:04n}_k021_snapshot.nc').squeeze()
dsV = xr.open_dataset(datapath+f'LLC2160_V_Arctic_day_{day:04n}_k021_snapshot.nc').squeeze()
dsGrid = xr.open_dataset(datapath+'LLC2160_grid.nc')#.squeeze()

# select region
i_start = 1080-200
i_stop = 1080+200
j_start = 1080-200
j_stop = 1080+200

dsU = dsU.sel(i_g=slice(i_start,i_stop), j=slice(j_start,j_stop))
dsV = dsV.sel(i=slice(i_start,i_stop), j_g=slice(j_start,j_stop))
dsGrid = dsGrid.sel(i=slice(i_start,i_stop),i_g=slice(i_start,i_stop),
                     j=slice(j_start,j_stop),j_g=slice(j_start,j_stop)
                     )

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

# make dataset with variables to filter. All must have i and j as dims
ds = xr.Dataset()
ds['u'] = dsU.U
ds['v'] = dsV.V
ds['u'] = ds.u.swap_dims({'i_g':'i'})
ds['v'] = ds.v.swap_dims({'j_g':'j'})
ds['uu'] = ds.u*ds.u
ds['vv'] = ds.v*ds.v
ds['uv'] = grid.interp(dsU.U, axis=['X'], boundary='fill')*grid.interp(dsV.V, axis=['Y'], boundary='fill')

grid_vars_visc, grid_vars_diff, dx_min = get_grid_vars(dsGrid)

#%%
def gcm_filter_many_iterations(ds, grid, scales, grid_vars_visc, grid_vars_diff, dx_min, iterations, kernel):
    results = []
    suptic = time.perf_counter()
    for n_iterations in iterations:
        print(f'Starting on {n_iterations} iterations')
        pis = []
        tic = time.perf_counter()
        for scale in scales:
            #print('Starting scale',scale)
            dsbar = filter(ds, scale, grid_vars_visc, grid_vars_diff, dx_min, n_iterations, kernel)
            pi = calculate_energy_transfer(dsbar, grid, calculate_energy=True)
            pis.append(pi)
            
        dspi = xr.concat(pis, dim='L')
        dspi.coords['L'] = scales
        toc = time.perf_counter()
        runtime = (toc-tic)/60
        print(f"Runtime {n_iterations} iteration(s) {runtime:0.4f} minutes")
        results.append(dspi)
    suptac = time.perf_counter()
    supruntime = (suptac-suptic)/60
    print(f"Runtime {kernel} full dataset {supruntime} minutes")

    ds_gcm = xr.concat(results, dim='n_iterations')
    ds_gcm.coords['n_iterations'] = iterations
    
    return ds_gcm
#%%
# filter using gaussian kernel
ds_fine = gcm_filter_many_iterations(ds, grid, scales, grid_vars_visc, grid_vars_diff, dx_min, iterations, 'gauss')


#%%
# coarsen data 

# first, filter data to intermediate scale. Use scipy to coarsen in index space?

coarsen_factor = 3

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

#%%
dsGridc = coarsen_grid(dsGrid, coarsen_factor)
dsc = coarsen_data(dsbar, coarsen_factor)
#%%

dsc['u'] = dsc['u'].swap_dims({'i_g':'i'})
dsc['uu'] = dsc['uu'].swap_dims({'i_g':'i'})
dsc['v'] = dsc['v'].swap_dims({'j_g':'j'})
dsc['vv'] = dsc['vv'].swap_dims({'j_g':'j'})


gridc = xgcm.Grid(dsGridc
                  , metrics=metrics
                  , periodic=False
                  )
 
grid_vars_viscc, grid_vars_diffc, dx_minc = get_grid_vars(dsGridc)

dsGridr = dsGrid.rename({"XC": "lon", "YC": "lat"})
dsGridcr = dsGridc.rename({"XC": "lon", "YC": "lat"})

regridder = xe.Regridder(dsGridcr, dsGridr, "bilinear")

#%%
ds_coarse = gcm_filter_many_iterations(dsc, gridc, scales, grid_vars_viscc, grid_vars_diffc, dx_minc, iterations, 'gauss')

#%%
# plot results

# calculate mean for plotting
meanpi_fine = grid.average(ds_fine.energy_transfer, ['X','Y'])
meanpi_coarse = gridc.average(ds_coarse.energy_transfer, ['X','Y'])


meane_fine = grid.average(ds_fine.energy, ['X','Y'])
meane_coarse = gridc.average(ds_coarse.energy, ['X','Y'])



sns.set_theme()
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,10))
colors = sns.color_palette("viridis", len(iterations)*2)

for i, n in enumerate(iterations):
    
    pif = meanpi_fine.sel(n_iterations=n)
    ef = meane_fine.sel(n_iterations=n)
    ax1.plot(scales/1e3,pif, marker='x', label='fine'
             #, color=colors[i]
             , color = 'red'
             )
    ax2.plot(scales/1e3,ef, marker='x', label='fine'
             #, color=colors[i]
             , color = 'red'
             )
    
    pic = meanpi_coarse.sel(n_iterations=n)
    ec = meane_coarse.sel(n_iterations=n)
    ax1.plot(scales/1e3,pic, marker='x', label='coarse'
             #, color=colors[i]
             , color = 'blue'
             )
    ax2.plot(scales/1e3,ec, marker='x', label='coarse'
             #, color=colors[i]
             , color = 'blue'
             )
    

#ax1.plot(scales/1e3,meanpi_taper, marker='x', label='taper', color = 'red')
#ax2.plot(scales/1e3,meane_taper, marker='x', color = 'red')

#axes[0].set_yscale('log')
#ax.set_xscale('log')
#ax.set_ylim(-1.5e-10,0)
#ax1.set_xlim(0,300)
ax1.set_ylim(-2.5e-10,1e-10)
ax2.set_ylim(0,1.5e-3)


ax1.legend(ncol = 4)