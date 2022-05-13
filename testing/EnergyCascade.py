#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:11:06 2022

@author: alsjur
"""
from FilterIndexSpace import FilterIndexSpace
import numpy as np


class EnergyCascade(FilterIndexSpace):
    
    def __init__(self, X, Y, ls, kernel='tophat', dx=1500, dy=1500, 
                 interpolation_method='linear', rho0=1):

        self.rho0 = rho0

        FilterIndexSpace.__init__(self, X, Y, ls, kernel=kernel, dx=dx, dy=dy, 
                     interpolation_method=interpolation_method)
        
        
    def cascade(self, U, V):
        
        UU = U*U
        VV = V*V
        UV = U*V
        
        Ubar = self.filter(U)
        Vbar = self.filter(V)
        
        # Ubar, Uprim = self.filter(U,return_prime = True)
        # Vbar, Vprim = self.filter(V,return_prime = True)
        
        UUbar = self.filter(UU)
        VVbar = self.filter(VV)
        UVbar = self.filter(UV)
        
        dudx = np.gradient(Ubar, self.dx, axis=2)
        dudy = np.gradient(Ubar, self.dy, axis=1)

        dvdx = np.gradient(Vbar, self.dx, axis=2)
        dvdy = np.gradient(Vbar, self.dy, axis=1)
        
        pi = -(dudx*(UUbar-Ubar*Ubar)+(dudy+dvdx)*(UVbar-Ubar*Vbar)\
               +dvdy*(VVbar-Vbar*Vbar))*self.rho0
        
        # # Energy transfer from Frich 95 
        
        # dupdx = np.gradient(Uprim, self.dx, axis=2)
        # dupdy = np.gradient(Uprim, self.dy, axis=1)

        # dvpdx = np.gradient(Vprim, self.dx, axis=2)
        # dvpdy = np.gradient(Vprim, self.dy, axis=1)
        
        # piHT = ((Ubar+Uprim)*Ubar*dupdx+(Ubar+Uprim)*Vbar*dvpdx+(Vbar+Vprim)*Ubar*dupdy+(Vbar+Vprim)*Vbar*dvpdy)*self.rho0
        
        # print(piHT.mean(axis=(1,2)))
        
        return pi#, piHT

        
    
'''
From Nikki:
-----------------Ugly fortran code coming ——————
S11=dudx
S22=dvdy
S33=dwdz


S12=0.5d0*(dudy+dvdx)
S13=0.5d0*(dudz+dwdx)
S23=0.5d0*(dvdz+dwdy)

if (remove_trace_S == 1) then
   trace=(S11+S22+S33)/3.d0
   S11=S11-trace
   S22=S22-trace
   S33=S33-trace
end if

dissipation=-(T11*S11+T22*S22+T33*S33+2.d0*S12*T12+2.d0*S13*T13+2.d0*S23*T23)
'''
