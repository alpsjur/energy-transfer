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
import pickle
import sys
sys.path.insert(0, '/home/alsjur/nird/energy-transfer/analysis')
from LLC2A4 import readROMSfile, LLC2A4

sns.set_theme()

datadir = '/home/alsjur/nird/data_temp/'
figdir = '/home/alsjur/nird/figures_temp/regions/LLC/'

levels = [16, 21, 28, 34, 39]
depths = [53, 100, 209, 352, 507]
#scales = 1/np.geomspace(2100,150000,num=20,dtype=int)

gridData = xr.open_dataset('/home/alsjur/PhD/Data/test_data/LLC2160/'+'LLC2160_grid.nc')

istart = int(sys.argv[1])
istop = int(sys.argv[2])
jstart = int(sys.argv[3])
jstop = int(sys.argv[4])
region = int(sys.argv[5])

#istart, jstart = LLC2A4([istart, jstart], gridData, LLCgrid)
#istop, jstop = LLC2A4([istop, jstop], gridData, LLCgrid)

bath = gridData.Depth

file = datadir+f'LLC_region{region}.pickle'

with open(file, 'rb') as f:
    data = pickle.load(f)

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

# mark region on map 
lon0 = bath.XC.sel(i=istart,j=jstart)
lat0 = bath.YC.sel(i=istart,j=jstart)

lon1 = bath.XC.sel(i=istop,j=jstart)
lat1 = bath.YC.sel(i=istop,j=jstart)

lon2 = bath.XC.sel(i=istop,j=jstop)
lat2 = bath.YC.sel(i=istop,j=jstop)

lon3 = bath.XC.sel(i=istart,j=jstop)
lat3 = bath.YC.sel(i=istart,j=jstop)

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

# aestethics
#ax.gridlines(color='gray', linestyle='--')
axd['map'].coastlines()
#axd['map'].gridlines()
axd['map'].set_extent([-180, 180, 70, 90], crs=ccrs.PlateCarree())

# remove spines
#ax.outline_patch.set_visible(False)
axd['map'].spines['geo'].set_visible(False)


colors = sns.color_palette("Set2", len(depths))
for level, color, depth in zip(levels, colors, depths):
    d = data[f'level{level}']
    scales = d['scales']
    pi = d['pi']
    dpi = d['dpi']
    
    e = d['e']
    de = d['dd']
    
    axd['pi'].plot(scales*1e3, pi, label=f'Depth {depth}', color=color)
    #axd['pi'].fill_between(scales*1e3, pi+dpi, pi-dpi, alpha=0.2, color=color)
    
    axd['e'].plot(scales*1e3, e, color=color)
    #axd['e'].fill_between(scales*1e3, e+de, e-de, alpha=0.2, color=color)
    
axd['pi'].plot(scales*1e3,np.zeros(scales.shape), ls='--', color='Gray')
axd['pi'].set_ylabel('Cross-scale energy transfer')


axd['e'].set_ylabel('Kinetic energy')
axd['e'].set_xlabel('K [km-1]')

axd['pi'].set_xscale('log')
axd['e'].set_xscale('log')    

axd['pi'].legend()

fig.savefig(figdir+f'LLC_pi_depth_region{region}.png')
