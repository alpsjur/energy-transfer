#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:53:05 2022

@author: alsjur
"""
# add path to gcmFilterFunction so module can be imported
from FilterIndexSpace import FilterIndexSpace
import xarray as xr
import xgcm
import xesmf as xe
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj

import sys
sys.path.insert(0, '/home/alsjur/nird/energy-transfer/analysis')
from gcmFilterFunction import get_grid_vars, filter, calculate_energy_transfer, coarsen_grid, coarsen_data
#%%

datapath = '/home/alsjur/PhD/Data/test_data/LLC2160/'
figpath = '/home/alsjur/PhD/Figurer/EnergyTransfer/method/'

scales = np.geomspace(20000,150000,num=10,dtype=int)
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
def gcm_filter_many_iterations(ds, scales, grid_vars_visc, grid_vars_diff, dx_min, iterations, kernel):
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
ds_gauss = gcm_filter_many_iterations(ds, scales, grid_vars_visc, grid_vars_diff, dx_min, iterations, 'gauss')

# filter using taper kernel
#ds_taper = gcm_filter_many_iterations(ds, scales, grid_vars_visc, grid_vars_diff, dx_min, iterations=[1], kernel='taper')
#%%
# use scipy to filter data
tic = time.perf_counter()
U = dsU.U.values
V = dsV.V.values

ls = scales/1500

# make projection
lon = dsGrid.XC.values
lat = dsGrid.YC.values

project = Proj(proj='stere', lon_0=7, lat_0=90)
X, Y = project(lon, lat)

# regrid and filter
engine = FilterIndexSpace(X, Y, ls, kernel='gauss', dx=1500, dy=1500)

Ubari = engine.filter(U)
Vbari = engine.filter(V)
UUbari = engine.filter(U*U)
UVbari = engine.filter(U*V)
VVbari = engine.filter(V*V)

# regrid back to original grid, compute energytransfer
pis = []
for i in range(len(ls)):
    Ubar = engine.reverse_regrid(Ubari[i])
    Vbar = engine.reverse_regrid(Vbari[i])
    UUbar = engine.reverse_regrid(UUbari[i])
    UVbar = engine.reverse_regrid(UVbari[i])
    VVbar = engine.reverse_regrid(VVbari[i])
    
    dsbari = xr.Dataset(
        data_vars = dict(
            ubar = (['j','i_g'], Ubar),
            vbar = (['j_g','i'], Vbar),
            uubar = (['j','i_g'], UUbar),
            vvbar = (['j_g','i'], VVbar),
            uvbar = (['j','i'], UVbar),
            ),
        coords = ds_gauss.coords
        )
    pi = calculate_energy_transfer(dsbari, grid, calculate_energy=True)
    pis.append(pi)
    
dspii = xr.concat(pis, dim='L')
dspii.coords['L'] = scales

toc = time.perf_counter()
runtime = (toc-tic)/60
print(f"Runtime scipy {runtime:0.4f} minutes")
#%%
# # coarsen data 

# # first, filter data to intermediate scale. Use scipy to coarsen in index space?

# coarsen_factor = 3
# intermediate_scale = coarsen_factor*1500

# filter to intermediate scale
# assume regular grid, so filtered in index space, not physical space
# #dsbar = filter(ds, intermediate_scale, grid_vars_visc, grid_vars_diff, dx_min, n_iterations=1)

# dsGridc = coarsen_grid(dsGrid, coarsen_factor)
# dsc = coarsen_data(dsbar, coarsen_factor)

# dsc['u'] = dsc['u'].swap_dims({'i_g':'i'})
# dsc['uu'] = dsc['uu'].swap_dims({'i_g':'i'})
# dsc['v'] = dsc['v'].swap_dims({'j_g':'j'})
# dsc['vv'] = dsc['vv'].swap_dims({'j_g':'j'})

# gridc = xgcm.Grid(dsGridc
#                  , metrics=metrics
#                  , periodic=False
#                  )
 
# grid_vars_viscc, grid_vars_diffc, dx_minc = get_grid_vars(dsGridc)

# dsGridr = dsGrid.rename({"XC": "lon", "YC": "lat"})
# dsGridcr = dsGridc.rename({"XC": "lon", "YC": "lat"})

# regridder = xe.Regridder(dsGridcr, dsGridr, "bilinear")


# # filter for different number of iterations
# resultsc = []
# suptic = time.perf_counter()
# for n_iterations in iterations:
#     print(f'Starting on {n_iterations} iterations')
#     pis = []
#     tic = time.perf_counter()
#     for scale in scales[3:]:
#         #print('Starting scale',scale)
#         dsbarc = filter(dsc, scale, grid_vars_viscc, grid_vars_diffc, dx_minc, n_iterations)
        
#         # interpolate to original grid
#         dsbarc['ubar'] = gridc.interp(dsbarc.ubar, axis=['X'], boundary='fill')
#         dsbarc['uubar'] = gridc.interp(dsbarc.uubar, axis=['X'], boundary='fill')
#         dsbarc['vbar'] = gridc.interp(dsbarc.vbar, axis=['Y'], boundary='fill')
#         dsbarc['vvbar'] = gridc.interp(dsbarc.vvbar, axis=['Y'], boundary='fill')
        
#         dsbarc = dsbarc.rename({"XC": "lon", "YC": "lat"})
#         dsbar['ubar'] = grid.interp(regridder(dsbarc.ubar), axis=['X'], boundary='fill')
#         dsbar['uubar'] = grid.interp(regridder(dsbarc.uubar), axis=['X'], boundary='fill')
#         dsbar['vbar'] = grid.interp(regridder(dsbarc.vbar), axis=['Y'], boundary='fill')
#         dsbar['vvbar'] = grid.interp(regridder(dsbarc.vvbar), axis=['Y'], boundary='fill')
#         dsbar['uvbar'] = regridder(dsbarc.uvbar)
#         #dsbar = dsbar.rename({"lon": "XC", "lat": "YC"})
        
#         # dsbar['ubar'] = grid.interp(dsbar.ubar, axis=['X'], boundary='fill')
#         # dsbar['uubar'] = grid.interp(dsbar.uubar, axis=['X'], boundary='fill')
#         # dsbar['vbar'] = grid.interp(dsbar.vbar, axis=['Y'], boundary='fill')
#         # dsbar['vvbar'] = grid.interp(dsbar.vvbar, axis=['Y'], boundary='fill')
        
#         pi = calculate_energy_transfer(dsbar, grid, calculate_energy=True)
#         pis.append(pi)
        
#     dspi = xr.concat(pis, dim='L')
#     dspi.coords['L'] = scales[3:]
#     toc = time.perf_counter()
#     runtime = (toc-tic)/60
#     print(f"Runtime {n_iterations} iteration(s) {runtime:0.4f} minutes")
#     resultsc.append(dspi)
# suptac = time.perf_counter()
# supruntime = (suptac-suptic)/60
# print(f"Runtime full dataset {supruntime} minutes")    

# ds_gcmc = xr.concat(resultsc, dim='n_iterations')
# ds_gcmc.coords['n_iterations'] = iterations

# dsTemp = dsGrid.rename({'XC':'lon', 'YC':'lat', 'XG':'lon_g', 'YG':'lat_g'})
# dsTemp['u'] = grid.interp(dsU.U, axis=['X'], boundary='fill')
# dsTemp['v'] = grid.interp(dsV.V, axis=['Y'], boundary='fill')
# dsTemp['Depth'] = dsGrid.Depth
# dsTemp = dsTemp.drop('Theta')

# dsGrid_coars = dsGrid.sel(i=slice(i_start,i_stop,coarsen_factor),i_g=slice(i_start,i_stop,coarsen_factor),
#                      j=slice(j_start,j_stop,coarsen_factor),j_g=slice(j_start,j_stop,coarsen_factor)
#                      )
# dsGrid_coars = dsGrid_coars.rename({'XC':'lon', 'YC':'lat', 'XG':'lon_g', 'YG':'lat_g'})

# regridder = xe.Regridder(dsTemp, dsGrid_coars, "bilinear")
# ds_coars = regridder(dsTemp)
# ds_coars['Depth'] = regridder(dsTemp.Depth)

# # filter
# grid_vars_visc, grid_vars_diff, dx_min = get_grid_vars(ds_coars)




# # interpolate back to original grid 


#%%
# plot results

# calculate mean for plotting
meanpi_gauss = grid.average(ds_gauss.energy_transfer, ['X','Y'])
#meanpi_taper = grid.average(ds_taper.sel(n_iterations=1).energy_transfer, ['X','Y'])
meanpi_scipy = grid.average(dspii.energy_transfer, ['X','Y'])


meane_gauss = grid.average(ds_gauss.energy, ['X','Y'])
#meane_taper = grid.average(ds_taper.sel(n_iterations=1).energy, ['X','Y'])
meane_scipy = grid.average(dspii.energy, ['X','Y'])





sns.set_theme()
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,10))
colors = sns.color_palette("viridis", len(iterations))

for i, n in enumerate(iterations):
    
    pi = meanpi_gauss.sel(n_iterations=n)
    e = meane_gauss.sel(n_iterations=n)
    ax1.plot(scales/1e3,pi, marker='x', label=f'{n} iterations', color=colors[i])
    ax2.plot(scales/1e3,e, marker='x', color=colors[i])
    

#ax1.plot(scales/1e3,meanpi_taper, marker='x', label='taper', color = 'red')
#ax2.plot(scales/1e3,meane_taper, marker='x', color = 'red')

ax1.plot(ls*1.5,meanpi_scipy, marker='x', label='scipy', color='black')
ax2.plot(ls*1.5,meane_scipy, marker='x', color='black')
#axes[0].set_yscale('log')
#ax.set_xscale('log')
#ax.set_ylim(-1.5e-10,0)
#ax1.set_xlim(0,300)
ax1.set_ylim(-2.5e-10,1e-10)
ax2.set_ylim(0,1.5e-3)


ax1.legend(ncol = 4)

fig.savefig(figpath+'compare_gcm_scipy.png')