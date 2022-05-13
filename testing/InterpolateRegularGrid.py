#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:33:46 2022

@author: alsjur
"""
import numpy as np
from scipy import interpolate

class InterpolateRegularGrid:
    '''
    A class for creating a regular grid from an possibly irregular grid, and 
    for interpolating variables onto the new regular grid.  
    
    Attributes
    ----------
    X : 2D array
        x values of original grid
    Y : 2D array
        y values of original grid
    dx : float or int, optional
        grid cell size in x direction for new grid. The default is 1500.
    dy : float of int, optional
        grid cell size in y direction for new grid. The default is 1500.
    method : str, optional
        interpolation method. The default is 'linear'.
    Xi : 2D array
        x values of new regular grid
    Yi : 2D array
        y values of new regular grid
    
    Methods
    -------
    regrid(F):
        Interpolates variable F onto the new regular grid.
    
    '''
    
    def __init__(self, X, Y, dx=1500, dy=1500, method = 'linear'):
        '''
        Constructs all necessary attributes for the InterpolateRegularGrid 
        object, and creates an regular grid to interpolate onto. 

        Parameters
        ----------
        X : 2D array
            x values of original grid
        Y : 2D array
            y values of original grid
        dx : float or int, optional
            Grid cell size in x direction for new grid. The default is 1500.
        dy : float of int, optional
            Grid cell size in y direction for new grid. The default is 1500.
        method : str, optional
            Interpolation method. The default is 'linear'.

        Returns
        -------
        None.

        '''
        self.X = np.copy(X)
        self.Y = np.copy(Y)
        
        self.dx = dx
        self.dy = dy
        
        self.method = method
        
        # creates regular grid from given parameters
        self._create_interpolation_grid()
        
    def _create_interpolation_grid(self):
        '''
        Creates new regular grid to interpolate onto. 

        Returns
        -------
        None.

        '''
        # find corners of new grid
        xmax = np.max(self.X)
        xmin = np.min(self.X)
        ymax = np.max(self.Y)
        ymin = np.min(self.Y)

        # create new grid
        xi = np.arange(xmin, xmax, self.dx)
        yi = np.arange(ymin, ymax, self.dy)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # set new grid as attribute
        self.Xi = Xi
        self.Yi = Yi

    def regrid(self, F):
        '''
        Interpolates the variable F from the original grid onto the new regular 
        grid.

        Parameters
        ----------
        F : 2D array
            Variable to be interpolated.

        Returns
        -------
        Fi : 2D array
            Resulting interpolated variable. 

        '''
        # interpolate F to new regular grid Xi Yi using scipy
        Fi = interpolate.griddata((self.X.flatten(), self.Y.flatten())
                                  , F.flatten()
                                  , (self.Xi, self.Yi)
                                  , method=self.method
                                  )
        return Fi
    
    def reverse_regrid(self,Fi):
        # interpolate Fi to old grid grid X Y using scipy
        F = interpolate.griddata((self.Xi.flatten(), self.Yi.flatten())
                                  , Fi.flatten()
                                  , (self.X, self.Y)
                                  , method=self.method
                                  )
        return F