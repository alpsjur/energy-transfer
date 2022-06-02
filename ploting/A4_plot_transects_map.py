#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:59:01 2022

@author: alsjur
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import json


import sys
sys.path.insert(0, '/home/alsjur/nird/energy-transfer/analysis')
from indexes_LLC2A4 import readROMSfile, LLC2A4

figdir = '/home/alsjur/nird/figures_temp/'

fontsize = 12

A4grid = readROMSfile('/home/alsjur/PhD/Data/test_data/A4/'+'ocean_avg_1827.nc')
LLCgrid = xr.open_dataset('/home/alsjur/PhD/Data/test_data/LLC2160/'+'LLC2160_grid.nc')

# read transect defininitions from file
with open('/home/alsjur/nird/energy-transfer/data/LLCtransects.txt') as f:
    data = f.read()
    LLCtransects = json.loads(data)

A4transects = {}

for name, data in LLCtransects.items():
    LLCistart = data['istart']
    LLCistop = data['istop']
    LLCjstart = data['jstart']
    LLCjstop = data['jstop']
    
    A4istart, A4jstart = LLC2A4([LLCistart, LLCjstart], A4grid, LLCgrid)
    A4istop, A4jstop = LLC2A4([LLCistop, LLCjstop], A4grid, LLCgrid)
    
    A4transects[name] = {
            'istart' : A4istart,
            'istop' : A4istop,
            'jstart' : A4jstart,
            'jstop' : A4jstop,
            'nr' : data['nr']
        }


bath = A4grid.h


# projection used for plotting
projection = ccrs.NearsidePerspective(central_longitude=7
                                      , central_latitude=88.0
                                      , satellite_height = 5E6
                                      )

fig = plt.figure(constrained_layout=True, figsize=(15,10))
gs = fig.add_gridspec(1,2)

axd = {}

axd['map'] = fig.add_subplot(gs[0], projection=projection)
axd['idmap'] = fig.add_subplot(gs[1])

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
#axd['map'].gridlines()
axd['map'].set_extent([-180, 180, 70, 90], crs=ccrs.PlateCarree())

# remove spines
#ax.outline_patch.set_visible(False)
axd['map'].spines['geo'].set_visible(False)

# plot index map
axd['idmap'].pcolormesh(bath.i, bath.j, bath
                       ,cmap='Blues'
                       ,vmin=vmin
                       ,vmax=vmax
                       ,shading='auto'
                       )
axd['idmap'].set_aspect('equal')

def plot_transect(bath, axd, istart, istop, jstart, jstop, nr):
    
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
        
    axd['idmap'].plot(ii, jj
                    , color='red'
                    , lw=2
                    )
    
    axd['idmap'].text(#istop, jstop
                      (istart+istop)/2, (jstart+jstop)/2
                      , f'{nr}'
                      , color = 'red'
                      , fontsize = fontsize
                      , va = 'center'
                      , ha = 'center'
                      , bbox=dict(facecolor='white', alpha=0.5)
                      )
    
    lons = []
    lats = []
    
    for i, j in zip(ii, jj):
        lons.append(bath.lon_rho.sel(i=i,j=j))
        lats.append(bath.lat_rho.sel(i=i,j=j))


    axd['map'].plot(lons, lats
                    , color='red'
                    , lw=2
                    , transform=ccrs.PlateCarree()
                    )
    axd['map'].text(#lons[-1], lats[-1]
                    lons[int(len(lons)/2)],lats[int(len(lats)/2)]
                    , f'{nr}'
                    , color = 'red'
                    , fontsize = fontsize
                    , va = 'center'
                    , ha = 'center'
                    , bbox=dict(facecolor='white', alpha=0.5)
                    , transform=ccrs.PlateCarree()
                    )

for transect, info in A4transects.items():
    istart = info['istart']
    istop = info['istop']
    jstart = info['jstart']
    jstop = info['jstop']
    nr = info['nr']
    
    plot_transect(bath, axd, istart, istop, jstart, jstop, nr) 

fig.savefig(figdir+'A4transects.png')

