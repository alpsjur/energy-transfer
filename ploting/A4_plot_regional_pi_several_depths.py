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
from LLC2A4 import readROMSfile, LLC2A4, coarse2fine

sns.set_theme()

datadir = '/home/alsjur/nird/data_temp/'
figdir = '/home/alsjur/nird/figures_temp/regions/A4/'

depths = ['mean']
#scales = 1/np.geomspace(2100,150000,num=20,dtype=int)

coarsen = True
coarsen_factor = 3

gridData = readROMSfile('/home/alsjur/PhD/Data/test_data/A4/'+'ocean_avg_1827.nc')
LLCgrid = xr.open_dataset('/home/alsjur/PhD/Data/test_data/LLC2160/'+'LLC2160_grid.nc')

istart = int(sys.argv[1])
istop = int(sys.argv[2])
jstart = int(sys.argv[3])
jstop = int(sys.argv[4])
region = int(sys.argv[5])

istart, jstart, istop, jstop = coarse2fine(np.array([istart, jstart, istop, jstop]), coarsen_factor)

bath = gridData.h

#file = datadir+f'A4_region{region}.pickle'
file = datadir+f'A4_depthmean_region{region}.pickle'

with open(file, 'rb') as f:
    data = pickle.load(f)
    
if coarsen:
    filec = datadir+f'A4_depthmean_coarse{coarsen_factor}_region{region}.pickle'

    with open(filec, 'rb') as f:
        datac = pickle.load(f)    

# projection used for plotting
projection = ccrs.NearsidePerspective(central_longitude=-30
                                      , central_latitude=70.0
                                      #, satellite_height = 5E6
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
    axd['map'] = fig.add_subplot(gs[:,0],projection=projection)


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

# mark region on map 
lon0 = bath.lon_rho.sel(i=istart,j=jstart)
lat0 = bath.lat_rho.sel(i=istart,j=jstart)

lon1 = bath.lon_rho.sel(i=istop,j=jstart)
lat1 = bath.lat_rho.sel(i=istop,j=jstart)

lon2 = bath.lon_rho.sel(i=istop,j=jstop)
lat2 = bath.lat_rho.sel(i=istop,j=jstop)

lon3 = bath.lon_rho.sel(i=istart,j=jstop)
lat3 = bath.lat_rho.sel(i=istart,j=jstop)

#fix strange behavior when crossing from lon -180 to lon 180
# if lon0*lon2 < 0 and lon2 < -90:
#     if lon0 < 0:
#         lon0 += 360
#     if lon1 < 0:
#         lon1 += 360
#     if lon2 < 0:
#         lon2 += 360
#     if lon3 < 0:
#         lon3 += 360

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
axd['map'].set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())

# remove spines
#ax.outline_patch.set_visible(False)
axd['map'].spines['geo'].set_visible(False)


colors = sns.color_palette("Set2", len(depths))
for depth, color in zip(depths, colors):
    d = data[f'depth{depth}']

    scales = d['scales']
    smax = np.max(scales)
    pi = d['pi']
    dpi = d['dpi']
    
    e = d['e']
    de = d['de']
    
    axd['pi'].plot(scales*1e3, pi, label='Fine', color=color)
    axd['pi'].fill_between(scales*1e3, pi+dpi, pi-dpi, alpha=0.2, color=color)
    
    axd['e'].plot(scales*1e3, e, color=color)
    axd['e'].fill_between(scales*1e3, e+de, e-de, alpha=0.2, color=color)
    
    if coarsen:
        d = datac[f'depth{depth}']

        scales = d['scales']
        smin = np.min(scales)
        pi = d['pi']
        dpi = d['dpi']
        
        e = d['e']
        de = d['de']
        
        axd['pi'].plot(scales*1e3, pi, label='Coarse'
                       #, color=color
                       , color = 'red'
                       )
        axd['pi'].fill_between(scales*1e3, pi+dpi, pi-dpi, alpha=0.2
                                #, color=color
                                ,color='red'
                                )
        
        axd['e'].plot(scales*1e3, e
                      #, color=color
                      , color = 'red'
                      )
        axd['e'].fill_between(scales*1e3, e+de, e-de, alpha=0.2
                              #, color=color
                              ,color='red'
                              )
    
axd['pi'].plot((smax*1e3,smin*1e3),(0,0), ls='--', color='Gray')
axd['pi'].set_ylabel('Cross-scale energy transfer')


axd['e'].set_ylabel('Kinetic energy')
axd['e'].set_xlabel('K [km-1]')

axd['pi'].set_xscale('log')
axd['e'].set_xscale('log')    

axd['pi'].legend()

fig.savefig(figdir+f'A4_pi_depthmean_region{region}_spread.png')
#fig.savefig(figdir+f'A4_pi_depthmean_region{region}.png')
