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

# idx_start = 900
# idx_stop = 1000
# # idy_start = 300
# # idy_stop = 400
# idy_start = 850
# idy_stop = 950
idx_start = 0
idx_stop = 2159

idy_start = 0
idy_stop = 2159


sns.set_theme()

files = sorted(glob.glob(datadir+'*'))
print(len(files))

data = xr.open_mfdataset(files, concat_dim='time', combine='nested')
gridData = xr.open_dataset('/projects/NS9869K/LLC2160/LLC2160_grid.nc')
bath = gridData.Depth

data = data.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop))
area = gridData.rA.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop))
#bathc = bath.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop))
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
pi = data.energy_transfer

seasonal_pi = pi.groupby('time.season').mean()
count = pi.groupby('time.season').count()
#print(count.values)

weights = area/area.sum(dim=('i', 'j'))

pi_mean = (pi.mean(dim='time')*weights).sum(dim=('i','j'))
seasonal_pi_mean = (seasonal_pi*weights).sum(dim=('i','j'))

pi_std = pi.std(dim=('i','j','time'))
seasonal_pi_std = seasonal_pi.std(dim=('i','j'))

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
#axd['map'].gridlines()
axd['map'].set_extent([-180, 180, 70, 90], crs=ccrs.PlateCarree())

# remove spines
#ax.outline_patch.set_visible(False)
axd['map'].spines['geo'].set_visible(False)

# mark region on map # !!! is this right?
cornerlon = bath.XC.isel(i=idx_start,j=idy_start)
cornerlat = bath.YC.isel(i=idx_start,j=idy_start)

regionsizex = bath.XC.isel(i=idx_stop,j=idy_stop)-cornerlon
regionsizey = bath.YC.isel(i=idx_stop,j=idy_stop)-cornerlat

# axd['map'].add_patch( mpatches.Rectangle((cornerlon,cornerlat),
#                         regionsizex, regionsizey,
#                         fc='none',
#                         color ='red',
#                         zorder = 21,
#                         #transform=ccrs.Geodetic()
#                         transform=ccrs.PlateCarree()
#                         )
                     
#                      )
# axd['map'].pcolormesh( area.XC.values, area.YC.values, np.ones(area.XC.values.shape),
#                         color ='red',
#                         vmin = 0,
#                         vmax = 1.5,
#                         alpha = 0.1,
#                         zorder = 21,
#                         transform=ccrs.PlateCarree()
#                      )

# plot regional mean 
axd['all'].plot(scales*1e3,pi_mean)
#axd['all'].fill_between(scales/1e3, pi_mean+pi_std, pi_mean-pi_std, alpha=0.2)
axd['all'].set_xscale('log')
axd['all'].set_title(f'All n={len(files)}')
axd['all'].hlines(0,np.min(scales*1e3),np.max(scales*1e3)
                 ,color='Gray'
                 ,ls='--'
                 )

# plot regional seasonal mean
for season in seasonal_pi.season.values:
    print(season)
    y = seasonal_pi_mean.sel(season=season)
    dy = seasonal_pi_std.sel(season=season)
    n = count.sel(season=season).values[-1,-1,-1]
    
    axd[season].plot(scales*1e3,y)
    axd[season].set_title(f'{season} n={n}')
    #axd['all'].fill_between(scales/1e3, y+dy, y-dy, alpha=0.2)
    axd[season].hlines(0,np.min(scales*1e3),np.max(scales*1e3)
                     ,color='Gray'
                     ,ls='--'
                     )

#fig.supxlabel('k [km-1]')
axd['all'].set_xlabel('k [km-1]')
axd['all'].set_ylabel('energy transfer')

fig.savefig(figdir+'full_seasonal.png')

