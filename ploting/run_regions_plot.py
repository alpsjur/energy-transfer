#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:37:04 2022

@author: alsjur
"""
import os
import xarray as xr
import json

coarsen_factor = 3
# read regions defininitions from file
with open(f'/home/alsjur/nird/energy-transfer/data/A4regions_c{coarsen_factor}.txt') as f:
    data = f.read()
    regions = json.loads(data)

# with open('/home/alsjur/nird/energy-transfer/data/LLCregions.txt') as f:
#     data = f.read()
#     regions = json.loads(data)

for region, data in regions.items():
    idx_start = data['idx_start']
    idx_stop = data['idx_stop']
    idy_start = data['idy_start']
    idy_stop = data['idy_stop']
    nr = data['nr']
    
    print('Starting',region)
    #os.system(f'python LLC_plot_regional_pi_several_depths.py {idx_start} {idx_stop} {idy_start} {idy_stop} {nr}')
    #os.system(f'python LLC_plot_regional_pi.py {idx_start} {idx_stop} {idy_start} {idy_stop} {nr}')
    
    os.system(f'python A4_plot_regional_pi_several_depths.py {idx_start} {idx_stop} {idy_start} {idy_stop} {nr}')
