#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 16:46:34 2022

@author: alsjur
"""
import xarray as xr
import matplotlib.pyplot as plt
import glob
import xgcm
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs

datadir = '/projects/NS9869K/LLC2160/gcm_filtered/'
figdir = 'figures/'

idx = 1000
idy_start = 0
idy_stop = 2000

sns.set_theme()

files = sorted(glob.glob(datadir+'*'))
print(len(files))

data = xr.open_mfdataset(files, concat_dim='time', combine='nested')
gridData = xr.open_dataset('/projects/NS9869K/LLC2160/LLC2160_grid.nc')
bath = gridData.Depth

data = data.isel(i=idx,j=slice(idy_start,idy_stop))
area = gridData.rA.isel(i=idx,j=slice(idy_start,idy_stop))
bathc = bath.isel(i=idx,j=slice(idy_start,idy_stop))
#gridData = gridData.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop))

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

scales = 1/data.scale.values
lat = bathc.YC.values
x = np.arange(idy_stop-idy_start)  # !!! This should be distance between points

pi = data.energy_transfer
seasonal_pi = pi.groupby('time.season').mean()
#count = pi.isel(scale=0, j=0).groupby('time.season').count()
count = pi.groupby('time.season').count()

pi_mean = pi.mean(dim='time')

# projection used for plotting
projection = ccrs.NearsidePerspective(central_longitude=7
                                      , central_latitude=88.0
                                      , satellite_height = 5E6
                                      )

# fig, axd = plt.subplot_mosaic([['map', 'MAM', 'JJA'],
#                                ['all', 'SON', 'DJF']],
#                               figsize=(15, 10), constrained_layout=True)

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(15,10))

axd = {
       'map' : axes[0,0]
       ,'all' : axes[1,0]
       ,'MAM' : axes[0,1]
       ,'JJA' : axes[0,2]
       ,'SON' : axes[1,1]
       ,'DJF' : axes[1,2]
       }

axd['map'].remove()
with sns.axes_style("white"):
    axd['map'] = fig.add_subplot(2,3,1,projection=projection)

vmin = 0
vmax = bath.max()

# plot map
cm = axd['map'].contourf(bath.XC, bath.YC, bath.where(bath.XC>0)
           ,transform=ccrs.PlateCarree()
           ,cmap='Blues'
           ,vmin=vmin
           ,vmax=vmax
           #,zorder=5
           )
axd['map'].contourf(bath.XC, bath.YC, bath.where(bath.XC<0)
          ,transform=ccrs.PlateCarree()
          ,cmap='Blues'
          ,vmin=vmin
          ,vmax=vmax
          #,zorder=1
          )

# aestethics
#ax.gridlines(color='gray', linestyle='--')
axd['map'].coastlines()
axd['map'].set_extent([-180, 180, 70, 90], crs=ccrs.PlateCarree())

# remove spines
#ax.outline_patch.set_visible(False)
axd['map'].spines['geo'].set_visible(False)

# mark region on map
# !!! should plot all points, not just endpoints
lon = bath.XC.isel(i=idx,j=slice(idy_start,idy_stop))
lat = bath.YC.isel(i=idx,j=slice(idy_start,idy_stop))



axd['map'].plot(lon,lat, 
                transform=ccrs.PlateCarree(), 
                color='red'
                )
                     

# plot regional mean 
#pimax = float(np.abs(seasonal_pi).max().values)
pimax = 2e-10

axd['all'].pcolormesh(x, scales*1e3, pi_mean.values, 
                      vmin = -pimax,
                      vmax = pimax,
                      cmap = 'coolwarm',
                      shading='auto'
                      )
axd['all'].set_yscale('log')
axd['all'].set_title(f'All n={len(files)}')



# plot regional seasonal mean
for season in seasonal_pi.season.values:
    print(season)
    z = seasonal_pi.sel(season=season).values
    n = count.sel(season=season).values[-1,-1]
    
    axd[season].pcolormesh(x, scales*1e3,z, 
                          vmin = -pimax,
                          vmax = pimax,
                          cmap = 'coolwarm'
                          )
    axd[season].set_title(f'{season} n={n}')



fig.savefig(figdir+'test_transect.png')

