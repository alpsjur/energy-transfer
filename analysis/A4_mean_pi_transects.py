#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:58:23 2022

@author: alsjur
"""

import xarray as xr
import xgcm
import numpy as np
import sys
import pickle
import glob
from LLC2A4 import readROMSfile, LLC2A4

datadir = '/projects/NS9869K/LLC2160/A4_filtered/'
outdir = '/nird/home/annals/data_temp/'


depth = 100

gridData = readROMSfile('/tos-project3/NS9081K/NORSTORE_OSL_DISK/NS9081K/shared/A4/A4_nudging_off/outputs/'+'ocean_avg_1827.nc')
LLCgrid = xr.open_dataset('/home/alsjur/PhD/Data/test_data/LLC2160/'+'LLC2160_grid.nc')

istart = int(sys.argv[1])
istop = int(sys.argv[2])
jstart = int(sys.argv[3])
jstop = int(sys.argv[4])
nr = int(sys.argv[5])

istart, jstart = LLC2A4([istart, jstart], gridData, LLCgrid)
istop, jstop = LLC2A4([istop, jstop], gridData, LLCgrid)

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


files = sorted(glob.glob(datadir+f'A4_filtered_day*_depth{depth:03n}.nc'))

data = xr.open_mfdataset(files, concat_dim='time', combine='nested')
bath = gridData.h

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
    'depth' : depth,
    'transect' : nr,
    'bathymetry' : bathc 
    }

with open(outdir+f'A4_depth{depth}_transect{nr}.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(datadict, f, pickle.HIGHEST_PROTOCOL)