#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:29:28 2022

@author: alsjur
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib import patches as mpatches

figdir = '/nird/home/annals/figures_temp/'

gridData = xr.open_dataset('/projects/NS9869K/LLC2160/LLC2160_grid.nc')
bath = gridData.Depth

# projection used for plotting
projection = ccrs.NearsidePerspective(central_longitude=7
                                      , central_latitude=88.0
                                      , satellite_height = 5E6
                                      )

fig = plt.figure(constrained_layout=True, figsize=(15,10))
gs = fig.add_gridspec(2,1)

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

def plot_region(bath, axd, idx_start, idx_stop, idy_start, idy_stop):
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

    # axd['map'].pcolormesh( area.XC.values, area.YC.values, np.ones(area.XC.values.shape),
    #                         color ='red',
    #                         vmin = 0,
    #                         vmax = 1.5,
    #                         alpha = 0.1,
    #                         zorder = 21,
    #                         transform=ccrs.PlateCarree()
    #                       )
