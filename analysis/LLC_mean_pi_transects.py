#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:31:24 2022

@author: alsjur
"""

import xarray as xr
import xgcm
import numpy as np
import sys
import pickle
from LLC2A4 import readROMSfile, LLC2A4

datadir = '/projects/NS9869K/LLC2160/gcm_filtered/'
outdir = '/nird/home/annals/data_temp/'


level = 21

istart = int(sys.argv[1])
istop = int(sys.argv[2])
jstart = int(sys.argv[3])
jstop = int(sys.argv[4])
nr = int(sys.argv[5])

def find_indexes(istart, istop, jstart, jstop):
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
    return ii, jj

start_day = 0
stop_day = 365*2
step = 5
days = np.arange(start_day,stop_day,step)

#files = sorted(glob.glob(datadir+'*'))
files = [datadir+f'LLC2160_filtered_day{day:03n}_k{level:02n}.nc' for day in days]

data = xr.open_mfdataset(files, concat_dim='time', combine='nested')
gridData = xr.open_dataset('/projects/NS9869K/LLC2160/LLC2160_grid.nc')
bath = gridData.Depth

dspi = data.energy_transfer

scales = 1/data.scale.values

ii, jj = find_indexes(istart, istop, jstart, jstop)

bathc = []
pi = []
dpi = []

for i, j in zip(ii, jj):
    bathc.append(bath.sel(i=i,j=j))
    pi.append(dspi.sel(i=i,j=j).mean(dim='time'))
    dpi.append(dspi.sel(i=i,j=j).std(dim='time'))

pi = xr.concat(pi, dim='x').load()
pi['x'] = np.arange(len(ii))

dpi = xr.concat(dpi, dim='x').load()
dpi['x'] = np.arange(len(ii))


# save results for later
datadict = {
    'pi' : pi,
    'dpi' : dpi,
    'scales' : scales,
    'level' : level,
    'transect' : nr,
    'bathymetry' : bathc 
    }

with open(outdir+f'data_level{level}_transect{nr}.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(datadict, f, pickle.HIGHEST_PROTOCOL)