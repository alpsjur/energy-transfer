#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:29:28 2022

@author: alsjur
"""
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import patches as mpatches

fontsize = 12

figdir = '/nird/home/annals/figures_temp/'

gridData = xr.open_dataset('/projects/NS9869K/LLC2160/LLC2160_grid.nc')
bath = gridData.Depth

count = 1
regions_auto = {}
for i in range(75, 2060, 200):
    for j in range(0, 1900, 200):
        bathc = bath.sel(i=slice(i,i+100),j=slice(j,j+100))
        if 0 in bathc.values:
            continue
        if bathc.mean(dim=('i', 'j')) < 100:
            continue
        regions_auto[f'region{count}'] = {
            'idx_start' : i,
            'idx_stop' : i+100,
            'idy_start' : j,
            'idy_stop' : j+100,
            'nr' : count
            }
        count += 1

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
    
    # if nr == 95:
    #     print('Center')
    #     print(nr)
    #     print(lons)
    #     print(lats)
    
    # if nr in [220, 206, 192, 181, 172, 159, 95]:
    #     print('Top')
    #     print(nr)
    #     print(lons)
    #     print(lats)
    
    # if nr in [10,24,33,40,47,53,59,67,75,85,95]:
    #     print('Bottom')
    #     print(nr)
    #     print(lons)
    #     #print(lats)
    

    axd['map'].plot(lons, lats
                    , color='red'
                    , lw=2
                    , transform=ccrs.PlateCarree()
                    )
    
    # axd['map'].plot([lon0, lon1], [lat0, lat1],
    #     color='red'
    #     , lw=2
    #     , transform=ccrs.PlateCarree()
    #     )
    # axd['map'].plot([lon1, lon2], [lat1, lat2],
    #     color='green'
    #     , lw=2
    #     , transform=ccrs.PlateCarree()
    #     )
    # axd['map'].plot([lon2, lon3], [lat2, lat3],
    #     color='blue'
    #     , lw=2
    #     , transform=ccrs.PlateCarree()
    #     )
    # axd['map'].plot([lon3, lon0], [lat3, lat0],
    #     color='orange'
    #     , lw=2
    #     , transform=ccrs.PlateCarree()
    #     )
    
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

fig.savefig(figdir+'regions.png')