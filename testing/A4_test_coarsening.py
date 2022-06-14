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
#sys.path.insert(0, '/home/alsjur/PhD/Kode/energy-transfer/analysis')
from gcmFilterFunction import get_grid_vars, filter, calculate_energy_transfer, coarsen_grid, coarsen_data, coarsen_depth, readROMSfile
from depth import sdepth, zslice

#%%

filepath = '/home/alsjur/PhD/Data/test_data/A4/'
figpath = '/home/alsjur/PhD/Figurer/EnergyTransfer/method/'

scales = np.geomspace(60000,150000,num=10,dtype=int)
iterations = [1, 2, 5, 10, 25, 40]
coarsen_factors = [2,3]
day = 1827
depth = 100
filetype = 'avg'


# load data
dsGrid = readROMSfile(filepath+'ocean_avg_1827.nc').squeeze()

# select region
i_start = 1080-300
i_stop = 1080+300
j_start = 1080-300
j_stop = 1080+300

# i_start = 0
# i_stop = 2159
# j_start = 0
# j_stop = 2159


dsGrid = dsGrid.sel(i=slice(i_start,i_stop),i_g=slice(i_start,i_stop),
                     j=slice(j_start,j_stop),j_g=slice(j_start,j_stop)
                     )

### Grid information ###
coords={'X':{'center':'i', 'left':'i_g'}, 
    'Y':{'center':'j', 'left':'j_g'}, 
    's':{'center':'s_rho', 'outer':'s_w'}}


grid = xgcm.Grid(dsGrid
                  , coords=coords
                  , periodic=False
                  )
# create grid object 
# metrics = {
#     ('X',): ['dxF', 'dxC', 'dxG'],    # X distances
#     ('Y',): ['dyF', 'dyC', 'dyG'],    # Y distances
#     ('X', 'Y'): ['rA', 'rAs', 'rAw']  # Areas
# }

# grid = xgcm.Grid(dsGrid
#                  , metrics=metrics
#                  , periodic=False
#                  )

ds_temp = dsGrid

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
ds['pm'] = ds_temp.pm
ds['pn'] = ds_temp.pn

grid_vars_visc, grid_vars_diff, dx_min = get_grid_vars(dsGrid, dsGrid.h.values, depth=depth, ROMS=True)

#%%
def gcm_filter_many_iterations(ds, grid, dsGrid, scales, grid_vars_visc, grid_vars_diff, dx_min, iterations, kernel):
    results = []
    suptic = time.perf_counter()
    for n_iterations in iterations:
        #print(f'Starting on {n_iterations} iterations')
        pis = []
        tic = time.perf_counter()
        for scale in scales:
            #print('Starting scale',scale)
            dsbar = filter(ds, scale, grid_vars_visc, grid_vars_diff, dx_min, n_iterations, kernel)
            #set pn and pm from the coarsened varibles, sice pn and pm in the dataset is not coarsened 
            dsbar['pn'] = 1/dsGrid.dxF
            dsbar['pm'] = 1/dsGrid.dyF
            pi = calculate_energy_transfer(dsbar, grid, calculate_energy=True, ROMS=True)
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

def gcm_filter_many_iterations_coarsening_factors(ds, dsGrid, iterations, coarsen_factors, kernel):
    results = []
    cs = []
    for coarsen_factor in coarsen_factors:
        print(coarsen_factor)
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

         
        grid_vars_viscc, grid_vars_diffc, dx_minc = get_grid_vars(dsGridc, dsdc.values, depth=depth, ROMS=True)

        dsGridr = dsGrid.rename({"XC": "lon", "YC": "lat"})
        dsGridcr = dsGridc.rename({"XC": "lon", "YC": "lat"})

        regridder = xe.Regridder(dsGridcr, dsGridr, "bilinear")
        
        #try:
        ds_temp = gcm_filter_many_iterations(dsc, gridc, dsGridc, scales, grid_vars_viscc, grid_vars_diffc, dx_minc, iterations, 'gauss')
        dsc = regridder(ds_temp)
        meanpi_coarse =ds_temp.energy_transfer.mean(dim=('i','j'))
        results.append(meanpi_coarse)
        #results.append(dsc)
        cs.append(coarsen_factor)
        # except:
        #     pass
    print(results)
    ds_coarse = xr.concat(results, dim='coarsen_factor')
    ds_coarse.coords['coarsen_factor'] = cs
    
    return ds_coarse
#%%
# filter using gaussian kernel
ds_fine = gcm_filter_many_iterations(ds, grid, dsGrid, scales, grid_vars_visc, grid_vars_diff, dx_min, iterations, 'gauss')
#%%
meanpi_fine = ds_fine.energy_transfer.mean(dim=('i','j'))

#%%
# coarsen data
# ds_coarse = gcm_filter_many_iterations_coarsening_factors(ds, grid, iterations, coarsen_factors, 'gauss')
# meanpi_coarse = grid.average(ds_coarse.energy_transfer, ['X','Y'])
meanpi_coarse = gcm_filter_many_iterations_coarsening_factors(ds, dsGrid, iterations, coarsen_factors, 'gauss')

#%%
# plot results

# calculate mean for plotting





sns.set_theme()
fig, ax = plt.subplots(figsize=(10,10))
colors = sns.color_palette("viridis", len(iterations)*2)
colors= ['red', 'blue', 'green', 'orange']
alphas = [0.3, 0.5, 0.7, 0.9]

#for i, n in enumerate([iterations[0]]):
for i, n in enumerate(iterations):
    
    pif = meanpi_fine.sel(n_iterations=n)

    ax.plot(scales/1e3,pif, marker='x', label=f'fine n={n}'
             #, color = colors[i]
             )

    #for j, c in enumerate([meanpi_coarse.coarsen_factor.values[0]]):
    for j, c in enumerate(meanpi_coarse.coarsen_factor.values):
        pic = meanpi_coarse.sel(n_iterations=n, coarsen_factor=c)

        ax.plot(scales/1e3,pic, marker='x', label=f'coarse n={n} c={c}'
                 #, color=colors[i]
                 #, alpha = alphas[j]
                 )

    

#ax1.plot(scales/1e3,meanpi_taper, marker='x', label='taper', color = 'red')
#ax2.plot(scales/1e3,meane_taper, marker='x', color = 'red')

#axes[0].set_yscale('log')
#ax.set_xscale('log')
#ax.set_ylim(-1.5e-10,0)
#ax1.set_xlim(0,300)
ax.set_ylim(-1.5e-10,0e-10)


ax.legend(ncol = 4)
#fig.savefig(figpath+'test_coarsening4.png')
# %%

# for key, item in grid_vars_visc.items():
#     itemc = grid_vars_viscc[key]
#     print(key)
#     print(item.sum(dim='i').values[0])
#     print(itemc.sum(dim='i').values[0])