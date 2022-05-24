#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 08:51:44 2022

@author: alsjur
"""
import xarray as xr
import matplotlib.pyplot as plt
import glob
import xgcm
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
from matplotlib import patches as mpatches

sns.set_theme()

datadir = '/projects/NS9869K/LLC2160/gcm_filtered/'
figdir = '/nird/home/annals/figures_temp/'

levels = [16, 21, 28, 34, 39]


start_day = 0
stop_day = 778
step = 5
days = np.arange(start_day,stop_day,step)

# idx_start = 900
# idx_stop = 1000
# idy_start = 900
# idy_stop = 1000
idx_start = 1400
idx_stop = 1500
idy_start = 700
idy_stop = 800


# files = []
# for day in days:
#     for level in levels:
#         files.append(datadir+f'LLC2160_filtered_day{day:03n}_k{level:02n}.nc')

def spectrum_at_level(level, days, area, grid, scales):
    files = [datadir+f'LLC2160_filtered_day{day:03n}_k{level:02n}.nc' for day in days]
    data = xr.open_mfdataset(files, concat_dim=['time'], combine='nested', 
                             chunks={'scale':1},
                             data_vars=['energy_transfer', 'ubar', 'vbar']
                             )
    data = data.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop))
    
    u = grid.interp(data.ubar, axis=['X'], boundary='fill')
    v = grid.interp(data.vbar, axis=['Y'], boundary='fill')
    pi = data.energy_transfer

    scales = 1/data.scale.values

    E = ((u**2+v**2)/2)
    
    weights = area/area.sum(dim=('i', 'j'))
    meanpi = (pi.mean(dim='time')*weights).sum(dim=('i','j'))
    meanE = (E.mean(dim='time')*weights).sum(dim=('i','j'))
    meane = np.gradient(meanE.values, scales)
    
    return meanpi, meane
    

files = [[datadir+f'LLC2160_filtered_day{day:03n}_k{level:02n}.nc' for \
          level in levels] for day in days]

data = xr.open_mfdataset(files, concat_dim=['time','k'], combine='nested', 
                         chunks={'scale':1},
                         data_vars=['energy_transfer', 'ubar', 'vbar']
                         )
print(data)

gridData = xr.open_dataset('/projects/NS9869K/LLC2160/LLC2160_grid.nc')
bath = gridData.Depth

data = data.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop))
area = gridData.rA.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop))

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


u = grid.interp(data.ubar, axis=['X'], boundary='fill')
v = grid.interp(data.vbar, axis=['Y'], boundary='fill')
pi = data.energy_transfer

scales = 1/data.scale.values

E = ((u**2+v**2)/2)


weights = area/area.sum(dim=('i', 'j'))
meanpi = (pi.mean(dim='time')*weights).sum(dim=('i','j'))
meanE = (E.mean(dim='time')*weights).sum(dim=('i','j'))

stdpi = pi.std(dim=('time', 'i', 'j'))


# meanpi = pi.mean(dim=('time', 'i', 'j')) #grid.average(pi.mean(dim=('time')), ['X','Y'])
# meanE = E.mean(dim=('time', 'i', 'j')) #grid.average(e.mean(dim=('time')), ['X','Y'])


# projection used for plotting
projection = ccrs.NearsidePerspective(central_longitude=7
                                      , central_latitude=88.0
                                      , satellite_height = 5E6
                                      )

fig = plt.figure(constrained_layout=True, figsize=(15,10))
gs = fig.add_gridspec(2,2)


ax0 = fig.add_subplot(gs[:-1,1:])
ax1 = fig.add_subplot(gs[-1,1:], sharex=ax0)

axd = {
       'pi' : ax0,
       'e' : ax1 
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

axd['map'].pcolormesh( area.XC.values, area.YC.values, 
                      np.ones(area.XC.values.shape),
                        color ='red',
                        vmin = 0,
                        vmax = 1.5,
                        alpha = 0.1,
                        zorder = 21,
                        transform=ccrs.PlateCarree()
                      )

# aestethics
#ax.gridlines(color='gray', linestyle='--')
axd['map'].coastlines()
#axd['map'].gridlines()
axd['map'].set_extent([-180, 180, 70, 90], crs=ccrs.PlateCarree())

# remove spines
#ax.outline_patch.set_visible(False)
axd['map'].spines['geo'].set_visible(False)

# plot energy spectrum and flux
axd['pi'].plot(scales*1e3,np.zeros(scales.shape), ls='--', color='Gray')
axd['pi'].set_ylabel('Cross-scale energy transfer')


axd['e'].set_ylabel('Kinetic energy')
axd['e'].set_xlabel('K [km-1]')

axd['pi'].set_xscale('log')
axd['e'].set_xscale('log')

for level in levels:
    pi = meanpi.sel(k=level)
    E = meanE.sel(k=level)
    e = np.gradient(E.values, scales)
    
    axd['pi'].plot(scales*1e3, pi, label=f'Level {level}')
    axd['e'].plot(scales*1e3, e)
    
axd['pi'].legend()

fig.savefig(figdir+'depth_test.png')