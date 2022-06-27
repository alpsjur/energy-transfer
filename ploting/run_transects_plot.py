#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 09:15:44 2022

@author: alsjur
"""
import os
import json

# read transect defininitions from file
with open('../data/LLCtransects.txt') as f:
    data = f.read()
    transects = json.loads(data)

for transect, data in transects.items():
    istart = data['istart']
    istop = data['istop']
    jstart = data['jstart']
    jstop = data['jstop']
    nr = data['nr']
    
    if nr > 9:
    
        print('Starting',transect)
        #os.system(f'python LLC_plot_transect_pi.py {istart} {istop} {jstart} {jstop} {nr}')
        os.system(f'python A4_plot_transect_pi.py {istart} {istop} {jstart} {jstop} {nr}')