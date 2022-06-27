#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:46:19 2022

@author: alsjur
"""
import xarray as xr
import gcm_filters
import matplotlib.pyplot as plt
import numpy as np
import time
import xgcm
from scipy import interpolate, signal, ndimage
import pyproj
import cmocean
#from mpl_toolkits.basemap import Basemap
import sys
sys.path.insert(0, '/home/alsjur/nird/energy-transfer/analysis')
#sys.path.insert(0, '/home/alsjur/PhD/Kode/energy-transfer/analysis')
from gcmFilterFunction import get_grid_vars, filter, calculate_energy_transfer,  readROMSfile
#%%
datapath = '/home/alsjur/PhD/Data/test_data/A4/'
figpath = '/home/alsjur/PhD/Figurer/EnergyTransfer/method/scipyVSgcm/'

dx = 3500

#scalesi = np.geomspace(3,50,num=10,dtype=int)
scalesi = np.arange(5,50,5)
scales = scalesi*3500

iterations = []
for scale in scales:
    if scale <= 15000:
        iterations.append(1)  
    elif scale <= 30000:
        iterations.append(4)
    elif scale <= 60000:
        iterations.append(15)
    elif scale <= 100000:
        iterations.append(30)
    elif scale <= 200000:
        iterations.append(60)
    else:
        iterations.append(75)

regions = {
    'canada_baisin' : {
        'istart' : 1100,
        'istop' : 1200,
        'jstart' : 800,
        'jstop' : 900,
        'nr' : 1
        },
    'norwegian_baisin' : {
        'istart' : 100,
        'istop' : 200,
        'jstart' : 550,
        'jstop' : 650,
        'nr' : 2
        },
    'slope' : {
        'istart' : 1300,
        'istop' : 1400,
        'jstart' : 450,
        'jstop' : 550,
        'nr' : 3
        }
    }

region = 'canada_baisin'
#region = 'slope'
days = [1827]

#%% prepare grid
istart = regions[region]['istart']
istop = regions[region]['istop']#+100
jstart = regions[region]['jstart']
jstop = regions[region]['jstop']#+100

# ploti = int((istart+istop)/2)
# plotj = int((jstart+jstop)/2)


dsGrid = readROMSfile(datapath+'ocean_avg_1827.nc').squeeze()
dsGrid = dsGrid.sel(i=slice(istart,istop),i_g=slice(istart,istop),
                     j=slice(jstart,jstop),j_g=slice(jstart,jstop)
                     )

### Grid information ###
coords={'X':{'center':'i', 'left':'i_g'}, 
    'Y':{'center':'j', 'left':'j_g'}, 
    's':{'center':'s_rho', 'outer':'s_w'}}


grid = xgcm.Grid(dsGrid
                  , coords=coords
                  , periodic=False
                  )
#%% gcm_filters implementation

tic = time.perf_counter()

for day in days:
    ds_temp = readROMSfile(datapath+f'ocean_avg_{day:4n}.nc').squeeze()
    ds_temp = ds_temp.sel(i=slice(istart,istop),i_g=slice(istart,istop),
                         j=slice(jstart,jstop),j_g=slice(jstart,jstop)
                         )
    
    ds = xr.Dataset()
    ds['u'] = ds_temp.ubar
    ds['v'] = ds_temp.vbar
    utemp = grid.interp(ds.u, axis=['X'], boundary='fill')
    vtemp = grid.interp(ds.v, axis=['Y'], boundary='fill')
    ds['uv'] = utemp*vtemp
    ds['u'] = ds.u.swap_dims({'i_g':'i'})
    ds['v'] = ds.v.swap_dims({'j_g':'j'})
    ds['uu'] = ds.u*ds.u
    ds['vv'] = ds.v*ds.v
    ds['pm'] = ds_temp.pm
    ds['pn'] = ds_temp.pn

    grid_vars_visc, grid_vars_diff, dx_min = get_grid_vars(dsGrid, dsGrid.h.values, ROMS=True)
    
    pis = []
    for scale, iteration in zip(scales,iterations):
        dsbar = filter(ds, scale, grid_vars_visc, grid_vars_diff, dx_min, n_iterations=iteration, kernel='gauss')
        dsbar['pm'] = ds_temp.pm
        dsbar['pn'] = ds_temp.pn
        pi = calculate_energy_transfer(dsbar, grid, calculate_energy=True, ROMS=True)
        pis.append(pi)
    dspi_gcm = xr.concat(pis, dim='L')
    dspi_gcm.coords['L'] = scales
toc = time.perf_counter() 
runtime = (toc-tic)
print(f"Runtime {runtime:0.4f}")
#%% implement scipy


# do projection
lon = dsGrid.XC.values
lat = dsGrid.YC.values

u = ds_temp.ubar
v = ds_temp.vbar

u = grid.interp(u, axis=['X'], boundary='fill').values
v = grid.interp(v, axis=['Y'], boundary='fill').values

project = pyproj.Proj(proj='stere',
                      lon_0=58,
                      lat_0=90
                      )
#project = Proj('+proj=stere +lat_0=90 +lon_0=58 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs', preserve_units=False)

#print(project)
X, Y = project(lon, lat)   # X[i, j]

# m = Basemap(projection='npstere',boundinglat=50,lon_0=58,resolution='c')
# X, Y = m(lon, lat)

print(X.shape)

# create new grid

# find corners
xmax = np.max(X)
xmin = np.min(X)
ymax = np.max(Y)
ymin = np.min(Y)

print(xmax,xmin)

xs = np.arange(xmin, xmax, dx)
ys = np.arange(ymin, ymax, dx)

Xi, Yi = np.meshgrid(xs, ys)

#%%
tic = time.perf_counter()
for day in days:
    ds_temp = readROMSfile(datapath+f'ocean_avg_{day:4n}.nc').squeeze()
    ds_temp = ds_temp.sel(i=slice(istart,istop),i_g=slice(istart,istop),
                         j=slice(jstart,jstop),j_g=slice(jstart,jstop)
                         )
    
    
    
    # regrid
    ui = interpolate.griddata((X.flatten(), Y.flatten())
                              , u.flatten()
                              , (Xi, Yi)
                              , method='linear'
                              )
    vi = interpolate.griddata((X.flatten(), Y.flatten())
                              , v.flatten()
                              , (Xi, Yi)
                              , method='linear'
                              )
    
    ui[np.isnan(ui)] = 0
    vi[np.isnan(vi)] = 0
    
    
    print(ui.shape)
    # plot to check size
    fig, ax = plt.subplots()
    ax.imshow(ui)
    plt.show()
    
    uui = ui*ui
    vvi = vi*vi
    uvi = ui*vi
    
    # filter
    pis = []
    for i, scale in enumerate(scalesi):
        sigma = scale/np.sqrt(12)
        win1d = signal.windows.gaussian(scale*2,sigma)
        
        winX, winY = np.meshgrid(win1d,win1d)
        win = winX*winY
        win /= np.sum(win)                    # normalize
        
        # ut = signal.convolve(ui, win, 
        #                       mode='same', 
        #                       #method='direct'
        #                       )
        
        # print(scale, np.sum(np.isnan(ut)))

        
        ubari = signal.convolve2d(ui, win, mode='same')
        vbari = signal.convolve2d(vi, win, mode='same')
        uubari = signal.convolve2d(uui, win, mode='same')
        vvbari = signal.convolve2d(vvi, win, mode='same')
        uvbari = signal.convolve2d(uvi, win, mode='same')
        
        # grid back
        ubar = interpolate.griddata((Xi.flatten(), Yi.flatten())
                                  , ubari.flatten()
                                  , (X, Y)
                                  , method='linear'
                                  )
    
        vbar = interpolate.griddata((Xi.flatten(), Yi.flatten())
                                  , vbari.flatten()
                                  , (X, Y)
                                  , method='linear'
                                  )
        
        uubar = interpolate.griddata((Xi.flatten(), Yi.flatten())
                                  , uubari.flatten()
                                  , (X, Y)
                                  , method='linear'
                                  )
        
        vvbar = interpolate.griddata((Xi.flatten(), Yi.flatten())
                                  , vvbari.flatten()
                                  , (X, Y)
                                  , method='linear'
                                  )
        
        uvbar = interpolate.griddata((Xi.flatten(), Yi.flatten())
                                  , uvbari.flatten()
                                  , (X, Y)
                                  , method='linear'
                                  )
    
    
        dsbar = xr.Dataset()
        dsbar['pn'] = ds_temp['pn']
        dsbar['pm'] = ds_temp['pm']
        dsbar['temp'] = (('j','i'),ubar)
        dsbar['ubar'] = grid.interp(dsbar.temp, axis=['X'], boundary='fill')
        dsbar['temp'] = (('j','i'),uubar)
        dsbar['uubar'] = grid.interp(dsbar.temp, axis=['X'], boundary='fill')
        dsbar['temp'] = (('j','i'),vbar)
        dsbar['vbar'] = grid.interp(dsbar.temp, axis=['Y'], boundary='fill')
        dsbar['temp'] = (('j','i'),vvbar)
        dsbar['vvbar'] = grid.interp(dsbar.temp, axis=['Y'], boundary='fill')
        dsbar['uvbar'] = (('j','i'),uvbar)
        
        # calculate pi and energy
        pi = calculate_energy_transfer(dsbar, grid, calculate_energy=True, ROMS=True)
        pis.append(pi)
        
    dspi_scipy = xr.concat(pis, dim='L')
    dspi_scipy.coords['L'] = scales
    
toc = time.perf_counter() 
runtime = (toc-tic)
print(f"Runtime {runtime:0.4f}")

#%%

for scale in scales:
    fig, ax = plt.subplots(1,3)
    ax = ax.flatten()
    
    ax[0].set_title('gcm')
    ax[1].set_title('scipy')
    ax[2].set_title('diff*10')
    
    pi_gcm = dspi_gcm.sel(L=scale).energy_transfer
    pi_scipy = dspi_scipy.sel(L=scale).energy_transfer
    
    vmin = np.min([pi_gcm.min(), pi_scipy.min()])
    vmax = np.min([pi_gcm.max(), pi_scipy.max()])
    
    vv = np.max((vmin,vmax))
    
    ax[0].imshow(pi_gcm, vmin=-vv, vmax=vv, cmap=cmocean.cm.curl)
    ax[1].imshow(pi_scipy, vmin=-vv, vmax=vv, cmap=cmocean.cm.curl)
    ax[2].imshow((pi_gcm-pi_scipy)*10, vmin = -vv, vmax = vv, cmap=cmocean.cm.curl)
    
    fig.savefig(figpath+f'test_scale{scale}.png')

#%%
ploti = 40
plotj = 40

fig, [axpi, axe] = plt.subplots(2,1, sharex = True)

dspi_gcm.energy_transfer.sel(i=ploti,j=plotj).plot.line(ax=axpi, label='gcm_filter')
dspi_gcm.energy.sel(i=ploti,j=plotj).plot.line(ax=axe)

dspi_scipy.energy_transfer.sel(i=ploti,j=plotj).plot.line(ax=axpi, label='scipy')
dspi_scipy.energy.sel(i=ploti,j=plotj).plot.line(ax=axe)

axpi.legend()

#%%
fig, [axpi, axe] = plt.subplots(2,1, sharex = True)

dspi_gcm.energy_transfer.mean(dim=('i','j')).plot.line(ax=axpi, label='gcm_filter')
dspi_gcm.energy.mean(dim=('i','j')).plot.line(ax=axe)

dspi_scipy.energy_transfer.mean(dim=('i','j')).plot.line(ax=axpi, label='scipy')
dspi_scipy.energy.mean(dim=('i','j')).plot.line(ax=axe)

axpi.legend()