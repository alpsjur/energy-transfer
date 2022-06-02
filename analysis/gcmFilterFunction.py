#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:01:46 2022

@author: alsjur
"""
import numpy as np
import xarray as xr
import gcm_filters
import time 
import xgcm
import xesmf as xe
from scipy import signal

def coarsen_grid(dsGrid, coarsen_factor):
    start_index = int(np.floor(coarsen_factor/2))
    print(start_index)

    dsGridc = dsGrid.isel(i=slice(start_index,None,coarsen_factor),i_g=slice(start_index,None,coarsen_factor),
                          j=slice(start_index,None,coarsen_factor),j_g=slice(start_index,None,coarsen_factor)
                          )
    

    # need to compute new grid information, need dx, dy and area for t-points, u-points and v-points
    # can use convolution with boxcar kernel, and then coarsen. NB! Every entry of the kernel must be 1
    # no need to use signal.window, just use np.ones() 
    # must make sure the right value is extracted later in the coarsening. Not completely trivial

    kerneli = np.ones((1,coarsen_factor))
    kernelj = np.ones((coarsen_factor,1))
    kernel2d = np.ones((coarsen_factor,coarsen_factor))

    dxFc = signal.convolve2d(dsGrid.dxF,kerneli,mode='same', boundary='fill')[start_index::coarsen_factor,start_index::coarsen_factor]
    dyFc = signal.convolve2d(dsGrid.dyF,kernelj,mode='same', boundary='fill')[start_index::coarsen_factor,start_index::coarsen_factor]
    rAc = signal.convolve2d(dsGrid.rA,kernel2d,mode='same', boundary='fill')[start_index::coarsen_factor,start_index::coarsen_factor]

    dxCc = signal.convolve2d(dsGrid.dxC,kerneli,mode='same', boundary='fill')[start_index::coarsen_factor,start_index::coarsen_factor]
    dyCc = signal.convolve2d(dsGrid.dyC,kernelj,mode='same', boundary='fill')[start_index::coarsen_factor,start_index::coarsen_factor]
    rAsc = signal.convolve2d(dsGrid.rAs,kernel2d,mode='same', boundary='fill')[start_index::coarsen_factor,start_index::coarsen_factor]

    dxGc = signal.convolve2d(dsGrid.dxG,kerneli,mode='same', boundary='fill')[start_index::coarsen_factor,start_index::coarsen_factor]
    dyGc = signal.convolve2d(dsGrid.dyG,kernelj,mode='same', boundary='fill')[start_index::coarsen_factor,start_index::coarsen_factor]
    rAwc = signal.convolve2d(dsGrid.rAw,kernel2d,mode='same', boundary='fill')[start_index::coarsen_factor,start_index
                                                                               ::coarsen_factor]

    dsGridc = dsGridc.assign(dxF=(['j','i'],dxFc), dyF=(['j','i'],dyFc), rA=(['j','i'],rAc),
                             dxC=(['j','i_g'],dxCc), dyC=(['j_g','i'],dyCc), rAs=(['j_g','i'],rAsc),
                             dxG=(['j_g','i'],dxGc), dyG=(['j','i_g'],dyGc), rAw=(['j','i_g'],rAwc)
                             )
    
    return dsGridc

def coarsen_data(dsbar, coarsen_factor):
    start_index = int(np.floor(coarsen_factor/2))
    dsbarc = dsbar.isel(i=slice(start_index,None,coarsen_factor), i_g=slice(start_index,None,coarsen_factor),
                        j=slice(start_index,None,coarsen_factor), j_g=slice(start_index,None,coarsen_factor)
                        )
    dsbarc = dsbarc.rename({'ubar':'u', 'vbar':'v', 'uubar':'uu', 'uvbar':'uv', 'vvbar':'vv'})
    
    return dsbarc
    

def get_grid_vars(dsGrid):
    mask_data = np.ones((len(dsGrid.j),len(dsGrid.i)))
    mask_data[dsGrid.Depth==0] = 0
    wet_mask = xr.DataArray(mask_data, dims=['j', 'i'])
    # grid info centered at T-points
    wet_mask_t = wet_mask
    dxT = dsGrid.dxF
    dyT = dsGrid.dyF
    area = dsGrid.rA

    # grid info centered at U-points
    dxCu = dsGrid.dxC.swap_dims({'i_g': 'i'})
    dyCu = dsGrid.dyG.swap_dims({'i_g': 'i'})
    area_u = dsGrid.rAw.swap_dims({'i_g': 'i'})
    # grid info centered at V-points
    dxCv = dsGrid.dxG.swap_dims({'j_g': 'j'})
    dyCv = dsGrid.dyC.swap_dims({'j_g': 'j'})
    area_v = dsGrid.rAs.swap_dims({'j_g': 'j'})
    # # grid info centered at vorticity points
    wet_mask_q = wet_mask
    dxBu = dsGrid.dxV.swap_dims({'i_g': 'i', 'j_g': 'j'})
    dyBu = dsGrid.dyU.swap_dims({'i_g': 'i', 'j_g': 'j'})

    dx_min = min(dxT.where(wet_mask_t).min(), dyT.where(wet_mask_t).min())
    dx_min = dx_min.values

    dx_max = max(dxT.max(), dyT.max(), dxCu.max(), dyCu.max(), dxCv.max(), \
                 dyCv.max(), dxBu.max(), dyBu.max())
    dx_max = dx_max.values

    kappa_iso = xr.ones_like(dxT)
    kappa_aniso = xr.zeros_like(dyT)

    kappa_w = xr.ones_like(dxCu)
    kappa_s = xr.ones_like(dxCu)

    grid_vars_visc={
        'wet_mask_t': wet_mask_t, 'wet_mask_q': wet_mask_q,
        'dxT': dxT, 'dyT': dyT, 
        'dxCu': dxCu, 'dyCu': dyCu, 'area_u': area_u, 
        'dxCv': dxCv, 'dyCv': dyCv, 'area_v': area_v,
        'dxBu': dxBu, 'dyBu': dyBu,
        'kappa_iso': kappa_iso, 'kappa_aniso': kappa_aniso
    }

    grid_vars_diff={
        'wet_mask': wet_mask_t, 
        'dxw' : dxCu, 'dyw' : dyCu,
        'dxs' : dxCv, 'dys' : dyCv,
        'area' : area, 'kappa_w' : kappa_w, 'kappa_s' : kappa_s
    }
    
    return grid_vars_visc, grid_vars_diff, dx_min

def filter(ds, filter_scale, grid_vars_visc, grid_vars_diff, dx_min, n_iterations, kernel='gauss'):
    
    if kernel == 'gauss':
        filter_shape = gcm_filters.FilterShape.GAUSSIAN
    elif kernel == 'taper':
        filter_shape = gcm_filters.FilterShape.TAPER
    
    # create filter
    filter_visc = gcm_filters.Filter(
        n_iterations=n_iterations,
        #n_steps=50,
        filter_scale=filter_scale,
        dx_min=dx_min,
        filter_shape=filter_shape,
        grid_type=gcm_filters.GridType.VECTOR_C_GRID,
        grid_vars=grid_vars_visc
    )

    filter_diff = gcm_filters.Filter(
        n_iterations=n_iterations,
        #n_steps=50,
        filter_scale=filter_scale,
        dx_min=dx_min,
        filter_shape=filter_shape,
        grid_type=gcm_filters.GridType.IRREGULAR_WITH_LAND,
        grid_vars=grid_vars_diff
    )
    
    (Ubar, Vbar) = filter_visc.apply_to_vector(ds.u, ds.v, dims=['j', 'i'])
    (UUbar, VVbar) = filter_visc.apply_to_vector(ds.uu, ds.vv, dims=['j', 'i'])
    UVbar = filter_diff.apply(ds.uv, dims=['j', 'i'])

    Ubar = Ubar.swap_dims({'i':'i_g'})
    UUbar = UUbar.swap_dims({'i':'i_g'})
    Vbar = Vbar.swap_dims({'j':'j_g'})
    VVbar = VVbar.swap_dims({'j':'j_g'})
    
    ds_out = xr.Dataset()
    ds_out['ubar'] = Ubar
    ds_out['vbar'] = Vbar
    ds_out['uubar'] = UUbar
    ds_out['vvbar'] = VVbar
    ds_out['uvbar'] = UVbar
    
    return ds_out

def calculate_energy_transfer(ds, grid, rho0=1, ROMS=False, calculate_energy=False):
    # shift variables to centre
    Ubar = grid.interp(ds.ubar, axis=['X'], boundary='fill')
    Vbar = grid.interp(ds.vbar, axis=['Y'], boundary='fill')

    # !!! this must be done differently for ROMS, see https://github.com/xgcm/xgcm/issues/108

    # calculate derivatives
    if not ROMS:
        dudx = grid.derivative(Ubar, 'X', boundary='fill')
        dudy = grid.derivative(Ubar,'Y', boundary='fill')
        dvdx = grid.derivative(Vbar, 'X', boundary='fill')
        dvdy = grid.derivative(Vbar,'Y', boundary='fill')
    else:
        dudx = grid.diff(Ubar, 'X', boundary='fill')*(ds.pm.swap_dims({'i':'i_g'}))
        dudy = grid.diff(Ubar, 'Y', boundary='fill')*(ds.pn.swap_dims({'j':'j_g'}))
        dvdx = grid.diff(Vbar, 'X', boundary='fill')*(ds.pm.swap_dims({'i':'i_g'}))
        dvdy = grid.diff(Vbar, 'Y', boundary='fill')*(ds.pn.swap_dims({'j':'j_g'}))
        
        #print(dudx, dudy, dvdx, dvdy)

    # shift variables to center
    dudx = grid.interp(dudx, axis=['X'], boundary='fill')
    dudy = grid.interp(dudy, axis=['Y'], boundary='fill')
    dvdx = grid.interp(dvdx, axis=['X'], boundary='fill')
    dvdy = grid.interp(dvdy, axis=['Y'], boundary='fill')
    UUbar = grid.interp(ds.uubar, axis=['X'], boundary='fill')
    VVbar = grid.interp(ds.vvbar, axis=['Y'], boundary='fill')
    UVbar = ds.uvbar

    ds_pi = xr.Dataset()
    ds_pi['energy_transfer'] = -(dudx*(UUbar-Ubar*Ubar)+(dudy+dvdx)*(UVbar-Ubar*Vbar)\
            +dvdy*(VVbar-Vbar*Vbar))*rho0
        
    if calculate_energy:
        ds_pi['energy'] = (Ubar**2+Vbar**2)*rho0/2
        
    return ds_pi
