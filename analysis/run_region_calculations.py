#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:37:04 2022

@author: alsjur
"""
import os
import json

# read transect defininitions from file
#with open('../data/LLCregions.txt') as f:
with open('../data/A4regions_c3.txt') as f:    
    data = f.read()
    regions = json.loads(data)


for region, data in regions.items():
    idx_start = data['idx_start']
    idx_stop = data['idx_stop']
    idy_start = data['idy_start']
    idy_stop = data['idy_stop']
    nr = data['nr']
    
    print('Starting',region)
    #os.system(f'python LLC_mean_pi_regions.py {idx_start} {idx_stop} {idy_start} {idy_stop} {nr}')
    #os.system(f'python A4_mean_pi_regions.py {idx_start} {idx_stop} {idy_start} {idy_stop} {nr}')
    os.system(f'python A4_mean_pi_regions_coarse.py {idx_start} {idx_stop} {idy_start} {idy_stop} {nr}')
