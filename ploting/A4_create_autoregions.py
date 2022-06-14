#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:43:56 2022

@author: alsjur
"""
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import patches as mpatches
import sys
sys.path.insert(0, '/home/alsjur/nird/energy-transfer/analysis')
from LLC2A4 import readROMSfile, LLC2A4

fontsize = 12

outdir = '/home/alsjur/nird/energy-transfer/data/'
figdir = '/home/alsjur/nird/figures_temp/'

gridData = readROMSfile('/home/alsjur/PhD/Data/test_data/A4/'+'ocean_avg_1827.nc')
bath = gridData.h

#%%
step = 100
size = 50

count = 1
regions_auto = {}
for i in range(100, 1601-size, step):
    for j in range(0, 1201-size, step):
        bathc = bath.sel(i=slice(i,i+size),j=slice(j,j+size))
        print(bathc)
        if 20 in bathc.values:
            continue
        if bathc.mean(dim=('i', 'j')) < 100:
            continue
        if i > 350 and i < 850 and j > 700:
            continue
        regions_auto[f'region{count}'] = {
            'idx_start' : i,
            'idx_stop' : i+size,
            'idy_start' : j,
            'idy_stop' : j+size,
            'nr' : count
            }
        count += 1

print(count)
# projection used for plotting
projection = ccrs.NearsidePerspective(central_longitude=-30
                                      , central_latitude=70.0
                                      #, satellite_height = 5E6
                                      )

fig = plt.figure(figsize=(15,7))
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
axd['map'].set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())

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

def plot_region(bath, axd, idx_start, idx_stop, idy_start, idy_stop, nr):
    
    idxs = [idx_start, idx_stop, idx_stop, idx_start, idx_start]
    idys = [idy_start, idy_start, idy_stop, idy_stop, idy_start]
    
    axd['idmap'].plot(idxs, idys
                    , color='red'
                    , lw=2
                    )
    
    axd['idmap'].text((idx_start+idx_stop)/2, (idy_start+idy_stop)/2, f'{nr}'
                      , color = 'red'
                      , fontsize = fontsize
                      , va = 'center'
                      , ha = 'center'
                      )
    

    lon0 = float(bath.lon_rho.isel(i=idx_start,j=idy_start).values)
    lat0 = float(bath.lat_rho.isel(i=idx_start,j=idy_start).values)
    
    lon1 = float(bath.lon_rho.isel(i=idx_stop,j=idy_start).values)
    lat1 = float(bath.lat_rho.isel(i=idx_stop,j=idy_start).values)
    
    lon2 = float(bath.lon_rho.isel(i=idx_stop,j=idy_stop).values)
    lat2 = float(bath.lat_rho.isel(i=idx_stop,j=idy_stop).values)
    
    lon3 = float(bath.lon_rho.isel(i=idx_start,j=idy_stop).values)
    lat3 = float(bath.lat_rho.isel(i=idx_start,j=idy_stop).values)
    
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
    
for region, data in regions_auto.items():
    idx_start = data['idx_start']
    idx_stop = data['idx_stop']
    idy_start = data['idy_start']
    idy_stop = data['idy_stop']
    nr = data['nr']
    
    plot_region(bath, axd, idx_start, idx_stop, idy_start, idy_stop, nr)

fig.savefig(figdir+'A4regions.png')
plt.show()

# sace dictionary with regions

# open file for writing
f = open(outdir+'A4regions.txt','w')

# write file
f.write( str(regions_auto) )

# close file
f.close()
