#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:28:11 2022

@author: alsjur
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import pickle

import sys
sys.path.insert(0, '/home/alsjur/nird/energy-transfer/analysis')
from LLC2A4 import readROMSfile, LLC2A4

figdir = '/home/alsjur/nird/figures_temp/transects/A4/'
datadir = '/home/alsjur/nird/data_temp/'

fontsize = 12

depth = 100

A4grid = readROMSfile('/home/alsjur/PhD/Data/test_data/A4/'+'ocean_avg_1827.nc')
LLCgrid = xr.open_dataset('/home/alsjur/PhD/Data/test_data/LLC2160/'+'LLC2160_grid.nc')

istart = int(sys.argv[1])
istop = int(sys.argv[2])
jstart = int(sys.argv[3])
jstop = int(sys.argv[4])
nr = int(sys.argv[5])

istart, jstart = LLC2A4([istart, jstart], A4grid, LLCgrid)
istop, jstop = LLC2A4([istop, jstop], A4grid, LLCgrid)


def find_indexes(istart, istop, jstart, jstop):
    if istop-istart == jstop-jstart:
        if istop > istart:
            ii = np.arange(istart, istop)
        else:
            ii = np.arange(istart, istop, -1)
        if jstop > jstart:
            jj = np.arange(jstart, jstop)
        else:
            jj = np.arange(jstart, jstop, -1)
    elif abs(istop-istart) > abs(jstop-jstart):
        if istop > istart:
            ii = np.arange(istart, istop)
        else:
            ii = np.arange(istart, istop, -1)
        jj = np.linspace(jstart, jstop, len(ii), dtype=int)
    else:
        if jstop > jstart:
            jj = np.arange(jstart, jstop)
        else:
            jj = np.arange(jstart, jstop, -1)
        ii = np.linspace(istart, istop, len(jj), dtype=int)
    return ii, jj

#sns.set_theme()

bath = A4grid.h

file = datadir+f'A4_depth{depth}_transect{nr}.pickle'

with open(file, 'rb') as f:
    data = pickle.load(f)


ii, jj = find_indexes(istart, istop, jstart, jstop)

lons = []
lats = []
bathc = []

for i, j in zip(ii, jj):
    lons.append(bath.lon_rho.sel(i=i,j=j))
    lats.append(bath.lat_rho.sel(i=i,j=j))
    bathc.append(bath.sel(i=i,j=j))


# projection used for plotting
projection = ccrs.NearsidePerspective(central_longitude=7
                                      , central_latitude=88.0
                                      , satellite_height = 5E6
                                      )

fig = plt.figure(constrained_layout=True, figsize=(15,10))
gs = fig.add_gridspec(2,3)

ax0 = fig.add_subplot(gs[:-1,1:])
ax1 = fig.add_subplot(gs[-1,1:], sharex=ax0, sharey=ax0)

axd = {
       'pi' : ax0,
       'dpi' : ax1,
       'bathymetry' : ax0.twinx()
       }

#axd['map'].remove()
#with sns.axes_style("white"):
axd['map'] = fig.add_subplot(gs[0,0],projection=projection)

vmin = 0
vmax = bath.max()

# plot map
cm = axd['map'].contourf(bath.lon_rho, bath.lat_rho, bath.where(bath.lon_rho>0)
            ,transform=ccrs.PlateCarree()
            ,cmap='Blues'
            ,vmin=vmin
            ,vmax=vmax
            #,zorder=5
            )
axd['map'].contourf(bath.lon_rho, bath.lat_rho, bath.where(bath.lon_rho<0)
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
axd['map'].plot(lons, lats
                , color='red'
                , lw=2
                , transform=ccrs.PlateCarree()
                )
# axd['map'].text(#lons[-1], lats[-1]
#                 lons[int(len(lons)/2)],lats[int(len(lats)/2)]
#                 , f'{nr}'
#                 , color = 'red'
#                 , fontsize = fontsize
#                 , va = 'center'
#                 , ha = 'center'
#                 , bbox=dict(facecolor='white', alpha=0.5)
#                 , transform=ccrs.PlateCarree()
#                 )
                     

pi = data['pi']
dpi = data['dpi']
scales = data['scales']

# plot regional mean 
pimax = float(np.abs(pi).max().values)*0.75
#pimax = 5e-9

cb = axd['pi'].pcolormesh(pi.x, scales*1e3, pi.values.transpose(), 
                      vmin = -pimax,
                      vmax = pimax,
                      cmap = 'coolwarm',
                      shading='auto'
                      )

fig.colorbar(cb, ax=axd['pi'], extend='both',fraction=0.02, pad =0.1)

axd['pi'].set_yscale('log')
#axd['pi'].set_title(f'n={len(files)}')
axd['pi'].set_ylabel('K [km-1]')
axd['pi'].invert_yaxis()
#axd['pi'].invert_xaxis()
axd['pi'].get_xaxis().set_visible(False)

dcb = axd['dpi'].pcolormesh(dpi.x, scales*1e3, dpi.values.transpose(), 
                      vmin = 0,#-pimax,
                      vmax = pimax,
                      cmap = 'Greens',
                      shading='auto'
                      )

fig.colorbar(dcb, ax=axd['dpi'], extend='max',fraction=0.02, pad =0.1)

axd['dpi'].set_ylabel('K [km-1]')

maxdepth = np.max(bathc)

depthticks = np.arange(0, maxdepth, 1000)

axd['bathymetry'].plot(pi.x,bathc, color='black')
axd['bathymetry'].set_ylim(-2*maxdepth, maxdepth*1.1)
axd['bathymetry'].invert_yaxis()
axd['bathymetry'].set_yticks(depthticks)

#axd['bathymetry'].set_ylabel('Bathymetry')


#plt.show()

fig.savefig(figdir+f'A4_pi_depth{depth}_transect{nr:02n}.png')
