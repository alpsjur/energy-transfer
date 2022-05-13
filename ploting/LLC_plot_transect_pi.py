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

idx = 200
idy_start = 1100
idy_stop = 1500

sns.set_theme()

files = sorted(glob.glob(datadir+'*'))
print(len(files))

data = xr.open_mfdataset(files, concat_dim='time', combine='nested')
gridData = xr.open_dataset('/projects/NS9869K/LLC2160/LLC2160_grid.nc')
bath = gridData.Depth

data = data.isel(i=idx,j=slice(idy_start,idy_stop))
area = gridData.rA.isel(i=idx,j=slice(idy_start,idy_stop))
dx = gridData.dxF.isel(i=idx,j=slice(idy_start,idy_stop))
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
#x = np.arange(idy_stop-idy_start)  # !!! This should be distance between points
x = np.cumsum(dx.values)/1e3

pi = data.energy_transfer
pi_mean = pi.mean(dim='time')

# projection used for plotting
projection = ccrs.NearsidePerspective(central_longitude=7
                                      , central_latitude=88.0
                                      , satellite_height = 5E6
                                      )

fig = plt.figure(constrained_layout=True, figsize=(15,10))
gs = fig.add_gridspec(3,3)

ax0 = fig.add_subplot(gs[:-1,1:])
ax1 = fig.add_subplot(gs[-1,1:], sharex=ax0)

axd = {
       'pi' : ax0,
       'bathymetry' : ax1 
       }

#axd['map'].remove()
with sns.axes_style("white"):
    axd['map'] = fig.add_subplot(gs[0,0],projection=projection)

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

lon = bath.XC.isel(i=idx,j=slice(idy_start,idy_stop))
lat = bath.YC.isel(i=idx,j=slice(idy_start,idy_stop))


axd['map'].plot(lon,lat, 
                transform=ccrs.PlateCarree(), 
                color='red'
                )
                     

# plot regional mean 
#pimax = float(np.abs(seasonal_pi).max().values)
pimax = 5e-10

axd['pi'].pcolormesh(x, scales*1e3, pi_mean.values, 
                      vmin = -pimax,
                      vmax = pimax,
                      cmap = 'coolwarm',
                      shading='auto'
                      )

axd['pi'].set_yscale('log')
axd['pi'].set_title(f'n={len(files)}')
axd['pi'].invert_yaxis()
axd['pi'].invert_xaxis()
axd['pi'].get_xaxis().set_visible(False)

axd['bathymetry'].plot(x,bathc)
axd['bathymetry'].invert_yaxis()




fig.savefig(figdir+'test_transect.png')

