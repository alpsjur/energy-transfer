#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:04:52 2022

@author: alsjur
"""
import numpy as np
from scipy import signal
from InterpolateRegularGrid import InterpolateRegularGrid

class FilterIndexSpace(InterpolateRegularGrid):
    # currently two different kernels are possible
    kernel_types = ['gauss', 'tophat']
    
    def __init__(self, X, Y, ls, kernel='tophat', dx=1500, dy=1500, 
                 interpolation_method='linear'):
        '''
        Constructs all necessary attributes for the FilterIndexSpace 
        object, and creates an regular grid to interpolate onto. 

        Parameters
        ----------
        X : 2D array
            x values of original grid.
        Y : 2D array
            y values of original grid.
        ls : list or 1D array
            List of length scales in index space to filter for.
        kernel : str, optional
            Name of kernel type. The default is 'tophat'.
        dx : float or int, optional
            Grid cell size in x direction for new grid. The default is 1500.
        dy : float of int, optional
            Grid cell size in y direction for new grid. The default is 1500.
        interpolation_method : str, optional
            Interpolation method used for grid. The default is 'linear'.

        Raises
        ------
        Exception
            Raised if not valid kernel is given.

        Returns
        -------
        None.

        '''
        
        # creates regular grid
        InterpolateRegularGrid.__init__(self, X, Y, dx=dx, dy=dy, 
                                        method=interpolation_method)
        
        # initialise cropped grid
        self.Xic = np.copy(self.Xi)
        self.Yic = np.copy(self.Yi)        
        # find the maximum length scale
        self.ls = ls
        self.maxl = np.max(ls)
        
        # indexes for cropping internally. These parameters may differ from 
        # the ones set by the user, since the algoritm need the domain to 
        # extend the region that is filtered. 
        self._idx_start = 0
        self._idx_stop = None
        
        self._idy_start = 0
        self._idy_stop = None
        
        self.cropx = False
        self.cropy = False
        
        # set the kernel used for convolution
        if kernel == 'tophat':
            self.kernel = self.tophat_kernel
        elif kernel == 'gauss':
            self.kernel = self.gauss_kernel


    def set_idxlim(self, idx_start, idx_stop):
        '''
        Method for spesifying x-range to preform filtering for.
        Helps to speed up the filtering. 

        Parameters
        ----------
        idx_start : int
            Lower x-index for cropped domain to be filtered.
        idx_stop : int
            Upper x-index for cropped domain to be filtered.

        Returns
        -------
        None.

        '''
        self.idx_start = idx_start
        self.idx_stop = idx_stop
        
        # crop grid
        self.Xic = self.Xic[:,idx_start:idx_stop] 
        self.Yic = self.Yic[:,idx_start:idx_stop] 
        
        self.cropx = True
        
        # if idx_start or idx_stop is futher from boundary than the largest 
        # scale to filter for, we spesify to crop internaly. Else, original 
        # boundary is kept. 
        if idx_start > self.maxl:
            self._idx_start = idx_start-self.maxl        
        if idx_stop + self.maxl < self.X.shape[1]:
            self._idx_stop = idx_stop + self.maxl
            
        
    def set_idylim(self, idy_start, idy_stop):
        '''
        Method for spesifying y-range to preform filtering for.
        Helps to speed up the filtering. 

        Parameters
        ----------
        idy_start : int
            Lower y-index for cropped domain to be filtered.
        idy_stop : int
            Upper y-index for cropped domain to be filtered.

        Returns
        -------
        None.

        '''
        self.idy_start = idy_start
        self.idy_stop = idy_stop
        
        # crop grid
        self.Xic = self.Xic[idy_start:idy_stop,:] 
        self.Yic = self.Yic[idy_start:idy_stop,:]
        
        self.cropy = True
        
        # if idy_start or idy_stop is futher from boundary than the largest 
        # scale to filter for, we spesify to crop internaly. Else, original 
        # boundary is kept.
        if idy_start > self.maxl:
            self._idy_start = idy_start-self.maxl        
        if idy_stop + self.maxl < self.X.shape[0]:
            self._idy_stop = idy_stop + self.maxl

    def gauss_kernel(self, l):
        '''
        Gaussian kernel creating a window with size l and standard deviation 
        l/4. Utilizes scipy.signal.gaussian.

        Parameters
        ----------
        l : int
            Width of kernel.

        Returns
        -------
        win : 2D array
            Gaussian window.

        '''
        sigma = l/(np.sqrt(12))
        winx = signal.windows.gaussian(l*2,sigma)
        winX, winY = np.meshgrid(winx,winx)
        win = winX*winY
        
        return win
    
    def tophat_kernel(self, l):
        '''
        Top hat kernel creating a window with size l. 

        Parameters
        ----------
        l : int
            Width of kernel.

        Returns
        -------
        win : 2D array
            Top hat window.

        '''
        # win = self.gauss_kernel(l)
        # win[win > win[l//2,0]] = 1
        # win[win < 1] = 0
        
        shift = (l-1)/2
        i = np.arange(l) - shift
        x, y = np.meshgrid(i,i)
        
        dist = np.sqrt(x**2+y**2)
        win = np.zeros((l,l))

        win[dist < l/2] = 1
        
        return win
    
    def filter(self, F, return_prime = False):
        '''
        Method for filtering. The filtering is calculated for each length scale
        as a 2D convolution between the variable F and the spesified kernel. 

        Parameters
        ----------
        F : 2D array
            Variable to be filtered.

        Returns
        -------
        Fbar : 3D array
            Resulting filtered variable, where the first dimention matches 
            number of lengthscales to filter for. If set_idxlim and/or 
            set_idy_lim is used, Fbar is cropped accordingly. 

        '''
        Fi = self.regrid(F)
        Ftemp = Fi[self._idy_start:self._idy_stop,self._idx_start:self._idx_stop]
        Fbar = np.zeros((len(self.ls), Ftemp.shape[0], Ftemp.shape[1]))
        
        for i, l in enumerate(self.ls): 
            win = self.kernel(l)
            
            # perform convolution
            Fbar[i] = signal.convolve2d(Ftemp,win
                                        , mode='same'
                                        , boundary = 'fill'
                                        ) / np.sum(win)
            
        if self.cropx:
            xstart = self.idx_start-self._idx_start
            xstop = xstart + (self.idx_stop-self.idx_start)
            
            Fbar = Fbar[:,:,xstart:xstop]
        if self.cropy:
            ystart = self.idy_start-self._idy_start
            ystop = ystart + (self.idy_stop-self.idy_start)
            
            Fbar = Fbar[:,ystart:ystop,:]
            
        if return_prime:
            Fexp = np.repeat(Fi[np.newaxis,:,:], len(self.ls), axis=0)
            if self.cropx:
                Fexp = Fexp[:,:,xstart:xstop]
            if self.cropy:
                Fexp = Fexp[:,ystart:ystop,:]
            Fprime = Fexp-Fbar
            
            return Fbar, Fprime
            
        else:
            return Fbar
    
        
    