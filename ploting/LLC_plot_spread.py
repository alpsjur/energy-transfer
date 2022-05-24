#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 18:11:56 2022

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

level = 21


start_day = 0
stop_day = 778
step = 50
days = np.arange(start_day,stop_day,step)

colors = sns.color_palette("viridis", len(days))

idx_start = 900
idx_stop = 905
idy_start = 1000
idy_stop = 1005



#files = sorted(glob.glob(datadir+f'*k{level:02n}.nc'))
files = [datadir+f'LLC2160_filtered_day{day:03n}_k{level:02n}.nc' for day in days]

data = xr.open_mfdataset(files, concat_dim='time', combine='nested', chunks={'scale':1})
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

meane = np.gradient(meanE.values, scales)


# projection used for plotting
projection = ccrs.NearsidePerspective(central_longitude=7
                                      , central_latitude=88.0
                                      , satellite_height = 5E6
                                      )

fig = plt.figure(constrained_layout=True, figsize=(15,10))
gs = fig.add_gridspec(2,1)


ax0 = fig.add_subplot(gs[1:])
#ax1 = fig.add_subplot(gs[-1,1:], sharex=ax0)

axd = {
       'pi' : ax0,
      # 'e' : ax1 
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

axd['map'].pcolormesh( area.XC.values, area.YC.values, np.ones(area.XC.values.shape),
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


# plot energy transfer and spectrum
for idx, color in enumerate(colors):
    for j in pi.j:
        for i in pi.i:
            axd['pi'].plot(scales*1e3, pi.isel(time=idx).sel(i=i,j=j), alpha=0.3, color=color)
            #axd['e'].plot(scales*1e3, e.isel(time=idx).sel(i=i,j=j), alpha=0.6)

axd['pi'].plot(scales*1e3, meanpi, color='Black')
axd['pi'].plot(scales*1e3,np.zeros(scales.shape), ls='--', color='Gray')
axd['pi'].set_ylabel('Cross-scale energy transfer')
#axd['pi'].fill_between(scales/1e3, meanpi+stdpi, meanpi-stdpi, alpha=0.2)
axd['pi'].set_xlabel('K [km-1]')
axd['pi'].set_xscale('log')

# axd['e'].plot(scales*1e3, meane)
# axd['e'].set_ylabel('Kinetic energy')
# axd['e'].set_xlabel('K [km-1]')


#axd['e'].set_xscale('log')

fig.savefig(figdir+'spread1.png')
