#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:10:04 2022

@author: alsjur
"""
import numpy as np
import matplotlib.pyplot as plt
import gcm_filters
import xarray as xr

datapath = '/home/alsjur/PhD/Data/inverse_cascade/'
filename = 'sf.j1cbluank'
figpath = '/home/alsjur/PhD/Figurer/EnergyTransfer/method/'

nx = 512    # resolution in x and y
dx = 1      # grid size
nseg = 30   # number of segments
n = -2      # which segment to look at

# scales to filter for
scales = np.unique(np.geomspace(2,250,num=50,dtype=int))


iterations = []

for scale in scales:
    if scale <= 5:
        iterations.append(1)
    elif scale <= 10:
        iterations.append(5)
    elif scale <= 50:
        iterations.append(15)
    elif scale <= 100:
        iterations.append(25)
    elif scale <= 200:
        iterations.append(50)
    else:
        iterations.append(80)



def filter_for_scale(ds, scale, dx, n_iterations):
    # create filter
    filter = gcm_filters.Filter(
        n_iterations=n_iterations,
        #n_steps=50,
        filter_scale=scale,
        dx_min = dx,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR,
    )

    Ubar = filter.apply(ds.u, dims=['i', 'j'])
    Vbar = filter.apply(ds.v, dims=['i', 'j'])
    UUbar = filter.apply(ds.uu, dims=['i', 'j'])
    VVbar = filter.apply(ds.vv, dims=['i', 'j'])
    UVbar = filter.apply(ds.uv, dims=['i', 'j'])


    ds_out = xr.Dataset()
    ds_out['ubar'] = Ubar
    ds_out['vbar'] = Vbar
    ds_out['uubar'] = UUbar
    ds_out['vvbar'] = VVbar
    ds_out['uvbar'] = UVbar
    
    return ds_out

def calculate_pi(ds, dx, rho0=1):
    Ubar = ds.ubar
    Vbar = ds.vbar
    UUbar = ds. uubar
    VVbar = ds.vvbar 
    UVbar = ds.uvbar
    
    dudx = np.gradient(Ubar, dx, axis=0)
    dudy = np.gradient(Ubar, dx, axis=1)
    dvdx = np.gradient(Vbar, dx, axis=0)
    dvdy = np.gradient(Vbar, dx, axis=1)

    ds_pi = xr.Dataset()
    ds_pi['energy_transfer'] = -(dudx*(UUbar-Ubar*Ubar)+(dudy+dvdx)*(UVbar-Ubar*Vbar)\
            +dvdy*(VVbar-Vbar*Vbar))*rho0
        
    ds_pi['energy'] = (Ubar**2+Vbar**2)*rho0/2
        
    return ds_pi

# read streamfunction data from binary file using numpy
data = np.fromfile(datapath+filename, dtype=np.float32)

# reshape data
data = np.reshape(data,(nseg,nx,nx))

# select segment
psi = data[n]

# calculate velocities
u = -np.gradient(psi,dx, axis=0)
v = np.gradient(psi,dx, axis=1)

# plot streamfunction and velocities
fig, ax = plt.subplots(1,3) 
ax[0].imshow(psi)
ax[1].imshow(u)
ax[2].imshow(v)

# create xarray dataset
uu = u*u
vv = v*v
uv = u*v
i = np.arange(nx)
j = np.arange(nx)

ds = xr.Dataset(
    data_vars = dict(
        # u = (['i', 'j'], u),
        # v = (['i', 'j'], v),
        # uu = (['i', 'j'], uu),
        # vv = (['i', 'j'], vv),
        # uv = (['i', 'j'], uv)
        # ),
        u = (['j', 'i'], u),
        v = (['j', 'i'], v),
        uu = (['j', 'i'], uu),
        vv = (['j', 'i'], vv),
        uv = (['j', 'i'], uv)
        ),
    coords = dict(
        i = i,
        j = j
        )
    )

results = []
for scale, n_iterations in zip(scales, iterations):
    print(f'Starting scale {scale}')
    dsbar = filter_for_scale(ds, scale, dx, n_iterations)
    pi = calculate_pi(dsbar, dx)
    
    results.append(pi)
    
ds_pi =  xr.concat(results, dim='scale')
ds_pi.coords['scale'] = scales

#%%
energy = np.gradient(ds_pi.energy.values, 1/scales, axis=0)
meane = np.mean(energy, axis=(1,2))
meanpi = ds_pi.energy_transfer.mean(dim=('i','j'))

fig, axes =plt.subplots(2,1, sharex = True, figsize=(3,6))

axd = {'e' : axes[0],
       'pi' : axes[1]
       }

# theoretical slope
k = np.array([30, 100])
y = k**(-5/3)*5e-1


axd['e'].plot(1/scales*nx, meane)
axd['e'].set_xscale('log')
axd['e'].set_yscale('log')
axd['e'].set_ylabel('KE')
axd['e'].plot(k, y)

axd['pi'].plot(1/scales*nx, meanpi)
axd['pi'].set_ylabel('Energy transfer')



fig.supxlabel('k')

fig.tight_layout()

fig.savefig(figpath+'j1cflux_filtering.png')