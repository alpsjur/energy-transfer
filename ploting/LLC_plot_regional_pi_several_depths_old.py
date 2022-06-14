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

sns.set_theme()

datadir = '/projects/NS9869K/LLC2160/gcm_filtered/'
figdir = '/nird/home/annals/figures_temp/regions/'
outdir = '/nird/home/annals/data_temp/'

levels = [16, 21, 28, 34, 39]
#scales = 1/np.geomspace(2100,150000,num=20,dtype=int)


start_day = 0
# want exactly two years
stop_day = 365*2#778
step = 5
days = np.arange(start_day,stop_day,step)

idx_start = int(sys.argv[1])
idx_stop = int(sys.argv[2])
idy_start = int(sys.argv[3])
idy_stop = int(sys.argv[4])
region = int(sys.argv[5])

# idx_start = 1000
# idx_stop = 1100
# idy_start = 1000
# idy_stop = 1100
# region = 'test'

def spectrum_at_level(level, days, grid):
    dss = []
    files = [datadir+f'LLC2160_filtered_day{day:03n}_k{level:02n}.nc' for day in days]
    for file in files:
       ds = xr.open_dataset(file)#.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop),
                                 #   i_g=slice(idx_start,idx_stop),j_g=slice(idy_start,idy_stop)
                                 #                 )
                        
       ds = ds.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop),
                i_g=slice(idx_start,idx_stop),j_g=slice(idy_start,idy_stop)
                )                           
       dss.append(ds)
       
    data = xr.concat(dss, dim='time')
        
    # data = xr.open_mfdataset(files, concat_dim=['time'], combine='nested', 
    #                          #chunks={'scale':1},
    #                          #chunks={'i':5, 'j':5, 'time':1},
    #                          data_vars=['energy_transfer', 'ubar', 'vbar']
    #                          )#.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop),
    #                           #                i_g=slice(idx_start,idx_stop),j_g=slice(idy_start,idy_stop)
    #                           #                )
                   
    # data = data.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop),
    #                  i_g=slice(idx_start,idx_stop),j_g=slice(idy_start,idy_stop)
    #                  )
           
    #print(data)
    
    u = grid.interp(data.ubar, axis=['X'], boundary='fill')
    v = grid.interp(data.vbar, axis=['Y'], boundary='fill')
    pi = data.energy_transfer

    ls = data.scale.values

    E = ((u**2+v**2)/2)
    e = -E.differentiate('scale')#*scales**2
    
    meanpi = grid.average(pi.mean(dim=('time')), ['X','Y'])
    meane = grid.average(e.mean(dim=('time')), ['X','Y'])*(ls**2)
    
    stdpi = pi.std(dim=('time', 'i', 'j'))
    stde = e.std(dim=('time', 'i', 'j'))*(ls**2)
    
    #meane = np.gradient(meanE.values, scales)
    
    return meanpi.load(), stdpi.load(), meane.load(), stde.load()
    
files = [[datadir+f'LLC2160_filtered_day{day:03n}_k{level:02n}.nc' for \
          level in levels] for day in days]

# data = xr.open_mfdataset(files, concat_dim=['time','k'], combine='nested', 
#                          chunks={'scale':1},
#                          data_vars=['energy_transfer', 'ubar', 'vbar']
#                          )

gridData = xr.open_dataset('/projects/NS9869K/LLC2160/LLC2160_grid.nc')
bath = gridData.Depth
gridData = gridData.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop),
                 i_g=slice(idx_start,idx_stop),j_g=slice(idy_start,idy_stop)
                 )

# data = data.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop))
#area = gridData.rA.isel(i=slice(idx_start,idx_stop),j=slice(idy_start,idy_stop))

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


# u = grid.interp(data.ubar, axis=['X'], boundary='fill')
# v = grid.interp(data.vbar, axis=['Y'], boundary='fill')
# pi = data.energy_transfer

#scales = 1/data.scale.values

# E = ((u**2+v**2)/2)


# weights = area/area.sum(dim=('i', 'j'))
# meanpi = (pi.mean(dim='time')*weights).sum(dim=('i','j'))
# meanE = (E.mean(dim='time')*weights).sum(dim=('i','j'))

# stdpi = pi.std(dim=('time', 'i', 'j'))


# meanpi = pi.mean(dim=('time', 'i', 'j')) #grid.average(pi.mean(dim=('time')), ['X','Y'])
# meanE = E.mean(dim=('time', 'i', 'j')) #grid.average(e.mean(dim=('time')), ['X','Y'])


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
lon0 = bath.XC.sel(i=idx_start,j=idy_start)
lat0 = bath.YC.sel(i=idx_start,j=idy_start)

lon1 = bath.XC.sel(i=idx_stop,j=idy_start)
lat1 = bath.YC.sel(i=idx_stop,j=idy_start)

lon2 = bath.XC.sel(i=idx_stop,j=idy_stop)
lat2 = bath.YC.sel(i=idx_stop,j=idy_stop)

lon3 = bath.XC.sel(i=idx_start,j=idy_stop)
lat3 = bath.YC.sel(i=idx_start,j=idy_stop)

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



#colors = sns.color_palette("viridis", len(levels))
colors = sns.color_palette("Set2", len(levels))

datadict = {
    }
for level, color in zip(levels,colors):
    
    pi, dpi, e, de = spectrum_at_level(level, days, grid)
    
    scales = 1/pi.scale
    
    # pi = meanpi.sel(k=level)
    # E = meanE.sel(k=level)
    # e = np.gradient(E.values, scales)
    
    axd['pi'].plot(scales*1e3, pi, label=f'Level {level}', color=color)
    #axd['pi'].fill_between(scales*1e3, pi+dpi, pi-dpi, alpha=0.2, color=color)
    
    axd['e'].plot(scales*1e3, e, color=color)
    #axd['e'].fill_between(scales*1e3, e+de, e-de, alpha=0.2, color=color)
    
    # save results for later
    datadict[f'level{level}'] = {
        'pi' : pi,
        'e' : e,
        'dpi' : dpi,
        'dd' : de,
        'scales' : scales,
        'region' : region
        }

# plot energy spectrum and flux
axd['pi'].plot(scales*1e3,np.zeros(scales.shape), ls='--', color='Gray')
axd['pi'].set_ylabel('Cross-scale energy transfer')


axd['e'].set_ylabel('Kinetic energy')
axd['e'].set_xlabel('K [km-1]')

axd['pi'].set_xscale('log')
axd['e'].set_xscale('log')    

axd['pi'].legend()

fig.savefig(figdir+f'pi_depth_region{region}.png')

with open(outdir+f'data_region{region}.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(datadict, f, pickle.HIGHEST_PROTOCOL)