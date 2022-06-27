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
from gcmFilterFunction import coarsen_grid, readROMSfile

fontsize = 12
coarsen_factor = 1

outdir = '/home/alsjur/nird/energy-transfer/data/'
figdir = '/home/alsjur/nird/figures_temp/'

dsGrid = readROMSfile('/home/alsjur/PhD/Data/test_data/A4/'+'ocean_avg_1827.nc')
#dsGridc = coarsen_grid(dsGrid, coarsen_factor)
#bath = dsGridc.h

bath = dsGrid.h

#%%
step = 30
size = 20

count = 1
regions_auto = {}
for i in range(60, 534-size, step):
    for j in range(0, 400-size, step):
        bathc = bath.sel(i=slice(i,i+size),j=slice(j,j+size))
        if 20 in bathc.values:
            continue
        if bathc.mean(dim=('i', 'j')) < 100:
            continue
        # if i > 350 and i < 850 and j > 700:
        #     continue
        regions_auto[f'region{count}'] = {
            'idx_start' : i,
            'idx_stop' : i+size,
            'idy_start' : j,
            'idy_stop' : j+size,
            'nr' : count
            }
        count += 1
        
#%% 
regions_auto = {
    'canada_baisin' : {
        'idx_start' : 1100,
        'idx_stop' : 1200,
        'idy_start' : 800,
        'idy_stop' : 900,
        'nr' : 1
        },
    'norwegian_baisin' : {
        'idx_start' : 100,
        'idx_stop' : 200,
        'idy_start' : 550,
        'idy_stop' : 650,
        'nr' : 2
        },
    'slope' : {
        'idx_start' : 1300,
        'idx_stop' : 1400,
        'idy_start' : 450,
        'idy_stop' : 550,
        'nr' : 3
        }
    }
#%%
print(count)
# projection used for plotting
projection = ccrs.NearsidePerspective(central_longitude=-30
                                      , central_latitude=70.0
                                      #, satellite_height = 5E6
                                      )

fig = plt.figure(figsize=(15*2,7*2))
gs = fig.add_gridspec(1,2)

axd = {}

axd['map'] = fig.add_subplot(gs[0], projection=projection)
axd['idmap'] = fig.add_subplot(gs[1])

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
    

    lon0 = float(bath.XC.isel(i=idx_start,j=idy_start).values)
    lat0 = float(bath.YC.isel(i=idx_start,j=idy_start).values)
    
    lon1 = float(bath.XC.isel(i=idx_stop,j=idy_start).values)
    lat1 = float(bath.YC.isel(i=idx_stop,j=idy_start).values)
    
    lon2 = float(bath.XC.isel(i=idx_stop,j=idy_stop).values)
    lat2 = float(bath.YC.isel(i=idx_stop,j=idy_stop).values)
    
    lon3 = float(bath.XC.isel(i=idx_start,j=idy_stop).values)
    lat3 = float(bath.YC.isel(i=idx_start,j=idy_stop).values)
    
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
#%%
# open file for writing
f = open(outdir+f'A4regions_c{coarsen_factor}.txt','w')

# write file
f.write( str(regions_auto) )

# close file
f.close()
