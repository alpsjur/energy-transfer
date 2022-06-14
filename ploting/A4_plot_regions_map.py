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
from LLC2A4 import readROMSfile, LLC2A4

figdir = '/home/alsjur/nird/figures_temp/'

fontsize = 12

A4grid = readROMSfile('/home/alsjur/PhD/Data/test_data/A4/'+'ocean_avg_1827.nc')
LLCgrid = xr.open_dataset('/home/alsjur/PhD/Data/test_data/LLC2160/'+'LLC2160_grid.nc')

# read transect defininitions from file
with open('/home/alsjur/nird/energy-transfer/data/LLCregions.txt') as f:
    data = f.read()
    LLCtransects = json.loads(data)

A4transects = {}

for name, data in LLCtransects.items():
    LLCistart = data['idx_start']
    LLCistop = data['idx_stop']
    LLCjstart = data['idy_start']
    LLCjstop = data['idy_stop']
    
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

def plot_region(bath, axd, istart, istop, jstart, jstop, nr):
    idxs = [istart, istop, istop, istart, istart]
    idys = [jstart, jstart, jstop, jstop, jstart]
    
    axd['idmap'].plot(idxs, idys
                    , color='red'
                    , lw=2
                    )
    
    axd['idmap'].text((istart+istop)/2, (jstart+jstop)/2, f'{nr}'
                      , color = 'red'
                      , fontsize = fontsize
                      , va = 'center'
                      , ha = 'center'
                      )
    

    lon0 = float(bath.lon_rho.isel(i=istart,j=jstart).values)
    lat0 = float(bath.lat_rho.isel(i=istart,j=jstart).values)
    
    lon1 = float(bath.lon_rho.isel(i=istop,j=jstart).values)
    lat1 = float(bath.lat_rho.isel(i=istop,j=jstart).values)
    
    lon2 = float(bath.lon_rho.isel(i=istop,j=jstop).values)
    lat2 = float(bath.lat_rho.isel(i=istop,j=jstop).values)
    
    lon3 = float(bath.lon_rho.isel(i=istart,j=jstop).values)
    lat3 = float(bath.lat_rho.isel(i=istart,j=jstop).values)
    
    #fix strange behavior when crossing from lon -180 to lon 180
    if lon0*lon2 < 0 and lon2 < -90:
        if lon0 < 0:
            lon0 += 360
        if lon1 < 0:
            lon1 += 360
        if lon2 < 0:
            lon2 += 360
        if lon3 < 0:
            lon3 += 360
    
    lons = [lon0, lon1, lon2, lon3, lon0]
    lats = [lat0, lat1, lat2, lat3, lat0]
    

    axd['map'].plot(lons, lats
                    , color='red'
                    , lw=2
                    , transform=ccrs.PlateCarree()
                    )
    
    axd['map'].text((lon0+lon1+lon2+lon3)/4, (lat0+lat1+lat2+lat3)/4, f'{nr}'
                      , color = 'red'
                      , fontsize = fontsize
                      , transform=ccrs.PlateCarree()
                      , va = 'center'
                      , ha = 'center'
                      )

for transect, info in A4transects.items():
    istart = info['istart']
    istop = info['istop']
    jstart = info['jstart']
    jstop = info['jstop']
    nr = info['nr']
    
    plot_region(bath, axd, istart, istop, jstart, jstop, nr) 

fig.savefig(figdir+'A4regions.png')

