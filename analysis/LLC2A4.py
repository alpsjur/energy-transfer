#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:10:06 2022

@author: alsjur
"""
import xarray as xr
import cartopy.crs as ccrs
import numpy as np

def readROMSfile(filename):
    '''
    This functions makes sure that the dimensions match between i and i_g.
    This means that we need to remove one row and one column.
    Here, the first row and column is removed, menaing that 
    i_g and j_g is later spesified as left in xgcm coords.
    If the last clumn/row is removed instead, i_g and j_g is right in the 
    coordinates. 
    '''
    # read file with xarray
    ds_temp = xr.open_dataset(filename)
    # remove first column and row 
    ds = ds_temp.isel(eta_rho=slice(1,None),
                      xi_rho=slice(1,None),
                      eta_u=slice(1,None),
                      xi_v=slice(1,None)
                      )
    # rename variables to match MITgcm
    ds = ds.rename({'eta_rho': 'j'
                    , 'xi_rho': 'i'
                    , 'eta_u': 'j'
                    , 'xi_u': 'i_g'
                    , 'eta_v': 'j_g'
                    , 'xi_v' : 'i'
                    })
    return ds


def LLC2A4(indexes, A4grid, LLCgrid):
    i, j = indexes 
    
    # First, find lon/lat at grid point in LLC
    LLClon = LLCgrid.XC.sel(i=i,j=j).values
    LLClat = LLCgrid.YC.sel(i=i,j=j).values
    
    # Find the A4 index of the grid point nearest LLClon/LLClat.   
    abslat = np.abs(A4grid.YC-LLClat)
    abslon = np.abs(A4grid.XC-LLClon)
    c = np.maximum(abslon, abslat)
    
    ([A4j], [A4i]) = np.where(c == np.min(c))
    
    # Use index location to get the values at the j, i index
    # A4lon = A4grid.XC.sel(i=A4i,j=A4j).values
    # A4lat = A4grid.YC.sel(i=A4i,j=A4j).values
    
    return(A4i, A4j)

def fine2coarse(indexes, finegrid, coarsegrid):
    i, j = indexes 
    
    # First, find lon/lat at grid point in LLC
    lonf = finegrid.XC.sel(i=i,j=j).values
    latf = finegrid.YC.sel(i=i,j=j).values
    
    # Find the A4 index of the grid point nearest LLClon/LLClat.   
    abslat = np.abs(coarsegrid.YC-latf)
    abslon = np.abs(coarsegrid.XC-lonf)
    c = np.maximum(abslon, abslat)
    
    ([jc], [ic]) = np.where(c == np.min(c))
    
    # Use index location to get the values at the j, i index
    # A4lon = A4grid.XC.sel(i=A4i,j=A4j).values
    # A4lat = A4grid.YC.sel(i=A4i,j=A4j).values
    
    return(ic, jc)

def coarse2fine(indexc, coarsen_factor):
    start_index = int(np.floor(coarsen_factor/2))
    index = indexc*coarsen_factor+start_index
    return index
        
