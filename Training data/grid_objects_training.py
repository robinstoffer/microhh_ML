# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:48:25 2018

@author: Robin Stoffer
"""
#Developed for Python 3!
import numpy as np
import downsampling_training
import microhh_tools_robinst as tools

class Finegrid:
    """ Returns a single object that can read from binary files or explicitly define the grid and output variables.
        
        -The precision indicates the floating point precision of the fine resolution data, which is either 'single' or 'double'.
       
        -The fourth_order flag indicates the order of the needed spatial interpolation in later processing steps, making sure sufficient ghost cells are added.
         If true, ghost cells are implemented for fourth_order interpolation. If false (default), ghost cells are added for second order spatial interpolation.

        -The periodic_bc should indicate in a tuple for each spatial direction (z, y, x) whether periodic boundary conditions are assumed (True when present, False when not present).
        
        -The read_grid flag indicates whether the grid is read from an existing simulation (True) or explicitly defined for testing purposes (False).
         When read_grid flag is True, specify explicitly (using keywords) grid_filename and settings_filename.
            -If no grid filename is provided, grid.0000000 from the current directory is read.
            -If no settings filename is provided, a .ini file should be present in the current directory to read.
         When read_grid flag is False, specifiy explicitly (using keywords) each spatial coordinate (coordz,coordy,coordx) as a 1d numpy array, e.g. Finegrid(read_grid_flag = False, coordx = np.array([1,2,3]),xsize = 4, coordy = np.array([0.5,1.5,3.5,6]), ysize = 7, coordz = np.array([0.1,0.3,1]), zsize = 1.2).
         Furthermore, for each spatial dimension the length should be indicated (zsize,ysize,xsize), which has to be larger than the last spatial coordinate in each dimension. 
         Each coordinate should 1) refer to the center of the grid cell, 2) be larger than 0, 3) be larger than the previous coordinate in the same direction."""
    def __init__(self, read_grid_flag = True , precision = 'double', fourth_order = False, periodic_bc = (False, True, True), **kwargs):

        #Set correct precision and order of spatial interpolation with corresponding ghost cells
        self.prec  = tools._process_precision(precision)
        self.fourth_order = fourth_order
        
        if self.fourth_order:
            self.igc = 3
            self.jgc = 3
            self.kgc = 0
        else:
            self.igc = 1
            self.jgc = 1
            self.kgc = 0

        #Define empty dictionary to store grid and variables
        self.var = {}
        self.var['output'] = {}
        self.var['grid'] = {}
        self.var['time'] = {}
        
        #Check whether specified period_bc satisfies needed format and subsequently store it in object
        if not isinstance(periodic_bc, tuple):
            raise TypeError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
        if not len(periodic_bc) == 3:
            raise ValueError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
        if not any(isinstance(flag, bool) for flag in periodic_bc):
            raise ValueError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
        self.periodic_bc = periodic_bc

        #Read or define the grid depending on read_grid_flag
        if read_grid_flag:
            self.read_grid_flag = True
            self.settings_filename = kwargs.get('settings_filename', None)
            self.grid_filename = kwargs.get('grid_filename', 'grid.0000000')
            self.__read_settings(settings_filename = self.settings_filename)
            self.__read_binary_grid(grid_filename = self.grid_filename)
            self.define_grid_flag = False

        else:
            self.read_grid_flag = False
            self.define_grid_flag = False
            try:
                def __checks_coord(coord_name, dim):
                    if not(np.all(kwargs[coord_name][1:] > kwargs[coord_name][:-1]) and np.all(kwargs[coord_name][:] > 0) and \
                    (len(kwargs[coord_name][:].shape) == 1)):
                        raise ValueError("The coordinates should be 1-dimensional, strictly increasing, and consist of positive values only.") 
                    else:
                        self.var['grid'][dim] = kwargs[coord_name]

                __checks_coord('coordx', 'x')
                __checks_coord('coordy', 'y')
                __checks_coord('coordz', 'z')

                def __checks_size(size_name, coord_name):
                    if not kwargs[size_name] > kwargs[coord_name][-1]:
                        raise ValueError("The length of the coordinate should be larger than the last coordinate.")
                    else: 
                        self.var['grid'][size_name] = kwargs[size_name]

                __checks_size('xsize', 'coordx')
                __checks_size('ysize', 'coordy')
                __checks_size('zsize', 'coordz')

            except KeyError:
                print("The needed arguments were not correctly specified. Make sure that coordx, coordy and coordz are assinged to a 1d numpy array.")
                raise
            self.__define_grid(self.var['grid']['z'],self.var['grid']['y'],self.var['grid']['x'], self.var['grid']['zsize'], self.var['grid']['ysize'], self.var['grid']['xsize'])

    def __read_settings(self,settings_filename = None):
        """ Read settings from specified file (default: .ini file in current directory) """

        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read settings from an already existing settings file.")

        #Read settings of MicroHH simulation from specified settings_filename. 
        #If None, a .ini file should be present in the current directory to read.
        settings = tools.Read_namelist(settings_filename)

        self.var['grid']['itot'] = settings['grid']['itot']
        self.var['grid']['jtot'] = settings['grid']['jtot']
        self.var['grid']['ktot'] = settings['grid']['ktot']
        self.var['grid']['xsize'] = settings['grid']['xsize']
        self.var['grid']['ysize'] = settings['grid']['ysize']
        self.var['grid']['zsize'] = settings['grid']['zsize']

        #self.var['time']['starttime'] = settings['time']['starttime']
        self.var['time']['starttime'] = 0
        #self.var['time']['endtime'] = settings['time']['endtime']
        self.var['time']['endtime'] = 7200
        #self.var['time']['savetime'] = settings['time']['savetime']
        self.var['time']['savetime'] = 600
        #self.var['time']['timesteps'] = (self.var['time']['endtime'] - self.var['time']['starttime']) // self.var['time']['savetime']
        self.var['time']['timesteps'] = 13

    def __read_binary_grid(self, grid_filename):
        """ Read binary grid from specified file (default: grid.0000000). """

        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read the grid from an already existing binary grid file.")

        #Open binary file containing the grid information
        f = open(grid_filename,'rb')

        #Infer number of bytes from the used precision
        nbytes = 8 if self.prec == 'float64' else 4

        #Define function to read each coordinate
        def __read_coordinate(file_opened, nbytes, number_gridcells, dim):
            self.var['grid'][dim] = np.fromfile(f, dtype = self.prec, count = number_gridcells)

        #Call above defined function for all coordinates in the right order (the order in which they are stored in the file)
        __read_coordinate(f, nbytes, self.var['grid']['itot'], 'x') 
        __read_coordinate(f, nbytes, self.var['grid']['itot'], 'xh')
        __read_coordinate(f, nbytes, self.var['grid']['jtot'], 'y')
        __read_coordinate(f, nbytes, self.var['grid']['jtot'], 'yh')
        __read_coordinate(f, nbytes, self.var['grid']['ktot'], 'z')
        __read_coordinate(f, nbytes, self.var['grid']['ktot'], 'zh')
        f.close()
        
        #Add ghostcells to grid
        self.__add_ghostcells_grid()

    def read_binary_variables(self, variable_name, timestep, bool_edge_gridcell = (False, False, False)):
        """ Read specified variable at a certain time step from binary files of a MicroHH simulation. Furthermore, ghost cells are added consistent with the specified order of spatial interpolation."""
        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read settings from an already existing binary file.")

        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
            
        #Check that timestep is an integer
        if not isinstance(timestep, int):
            raise TypeError("Specified timestep should be an integer.")
            
        #Check whether specified bool_edge_gridcell satisfies needed format and subsequently store it in object
        if not isinstance(bool_edge_gridcell, tuple):
            raise TypeError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
        if not len(bool_edge_gridcell) == 3:
            raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
                
        if not any(isinstance(flag, bool) for flag in bool_edge_gridcell):
            raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")

        #Check wheter timestep is present in data
        if timestep > (self.var['time']['timesteps'] - 1):
            raise ValueError("Specified timestep not contained in data. Note that the timestep starts counting from 0.")

        #Open binary file to read variable at timestep
        time = self.var['time']['starttime'] + timestep*self.var['time']['savetime']
        var_filename = '{0}.{1:07d}'.format(variable_name,time)
        
        self.var['output'][variable_name] = {}
        self.var['output'][variable_name]['variable'] = np.fromfile(var_filename, dtype = self.prec).reshape((self.var['grid']['ktot'], self.var['grid']['jtot'], self.var['grid']['itot']))
        self.var['output'][variable_name]['orientation'] = bool_edge_gridcell
        
        #Add ghostcells depending on the order used for the spatial interpolation, add 1 additonal cell on downstream/top boundaries
        self.__add_ghostcells(variable_name)

    def __define_grid(self, coordz, coordy, coordx, sizez, sizey, sizex):
        """ Manually define grid stored in object. """

        if self.read_grid_flag:
            raise RuntimeError("Object defined to read the grid from binary files (via the read_binary_grid method). Create a new Finegrid object with read_grid_flag = False to manually define the settings of the grid.")

        #define lengths of specified grid
        self.var['grid']['itot'] = len(coordx)
        self.var['grid']['jtot'] = len(coordy)
        self.var['grid']['ktot'] = len(coordz)

        #create coordinates corresponding to edges grid cell
        self.var['grid']['xh'] = self.__edgegrid_from_centergrid(self.var['grid']['x'], self.var['grid']['itot'], sizex)
        self.var['grid']['yh'] = self.__edgegrid_from_centergrid(self.var['grid']['y'], self.var['grid']['jtot'], sizey)
        self.var['grid']['zh'] = self.__edgegrid_from_centergrid(self.var['grid']['z'], self.var['grid']['ktot'], sizez)
        
        #Add ghostcells to grid
        self.__add_ghostcells_grid()
        
        #Set define_grid_flag to True, such that the create_variables method can be executed
        self.define_grid_flag = True

    def create_variables(self, variable_name, output_field, bool_edge_gridcell = (False, False, False)):
        """ Manually define output variables for a manually defined grid. Furthermore, ghost cells are added consistent with the specified order of spatial interpolation. """

        if self.read_grid_flag:
            raise RuntimeError("Object defined to read variables from binary files (via the read_binary_variables method). Create a new Finegrid object with read_grid_flag = False to manually define the settings of the grid.")
            
        #Check that grid is already defined.
        if not self.define_grid_flag:
            raise RuntimeError("Object does not yet contain the coordinates for the output variable. Define them explicitly first via the __define_grid method")

        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
            
        #Check whether specified bool_edge_gridcell satisfies needed format and subsequently store it in object
        if not isinstance(bool_edge_gridcell, tuple):
            raise TypeError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
        if not len(bool_edge_gridcell) == 3:
            raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
                
        if not any(isinstance(flag, bool) for flag in bool_edge_gridcell):
            raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")

        #Check that shape output field is correct according to the coordinates specified via the define_grid method.
        if not (output_field.shape == (self.var['grid']['ktot'],self.var['grid']['jtot'],self.var['grid']['itot'])):
            raise RuntimeError("The shape corresponding to the specified coordinates (z,y,x) is not the same as the shape of the specified output field. Note that for the staggered grid the top/downstream boundaries do not have to be included, these are automatically added.")

        #Store output_field in object
        self.var['output'][variable_name] = {}
        self.var['output'][variable_name]['variable'] = output_field
        
        #Store orientation of output_field in object
        self.var['output'][variable_name]['orientation'] = bool_edge_gridcell

        #For convenience with the script func_generate_training.py, timesteps is set equal to 1.
        self.var['time']['timesteps'] = 1

        #Add ghostcells depending on the order used for the spatial interpolation, add 1 additonal cell on downstream/top boundaries
        self.__add_ghostcells(variable_name)
        
    def __add_ghostcells(self, variable_name):
        """ Add ghostcells (defined as grid cells located at the downstream/top boundary and outside the domain) to fine grid for variable_name. 
            The samples at the end of the array are repeated at the beginning and vice versa. Furthermore, the ghostcells are added to the coordinates as well."""
            
        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.var['output'].keys():
            raise KeyError("Specified variable_name not defined in object.")

        #Check that ghost cells are not negative
        if self.jgc < 0 or self.igc < 0 or self.kgc < 0:
            raise ValueError("The specified number of ghostcells cannot be negative.")
            
        #Check for each coordinate that no ghostcells are specified when no periodic bc are assumed. No ghost cells are yet implemented for other types of BC's.
        if self.kgc > 0 and not self.periodic_bc[0]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")
            
        if self.jgc > 0 and not self.periodic_bc[1]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        if self.igc > 0 and not self.periodic_bc[2]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        #Read variable from object
        s = self.var['output'][variable_name]['variable']

        #Determine size new output array including ghost cells
        icells = self.var['grid']['itot'] + 2*self.igc
        jcells = self.var['grid']['jtot'] + 2*self.jgc
      #  kcells = self.var['grid']['ktot'] + 2*kgc
        kcells = self.var['grid']['ktot'] + 2*self.kgc
        
        #Depending on bool_edge_gridcell, add one additional ghost cell at top/downstream boundaries independent of self.igc/jgc/kgc
        bkgc = 0
        bjgc = 0
        bigc = 0
        if self.var['output'][variable_name]['orientation'][0]:
            bkgc = 1
        if self.var['output'][variable_name]['orientation'][1]:
            bjgc = 1
        if self.var['output'][variable_name]['orientation'][2]:
            bigc = 1

        self.siend = self.igc + self.var['grid']['itot'] + bigc
        self.sjend = self.jgc + self.var['grid']['jtot'] + bjgc
        self.skend = self.kgc + self.var['grid']['ktot'] + bkgc
        
        #Check that there are not more ghost cells than grid cells. 
        if (self.igc + bigc) > (s.shape[2]) or (self.jgc + bjgc) > (s.shape[1]) or (self.kgc + bkgc) > (s.shape[0]):
            raise ValueError("The needed number of ghostcells is larger than the number of grid cells present in the object.") 
        
        #Initialize new output array including ghost cells
        sgc = np.zeros((kcells+bkgc, jcells+bjgc, icells+bigc))
        
        #Fill new initialzid array including ghost cells
        sgc[self.kgc:self.skend-bkgc, self.jgc:self.sjend-bjgc, self.igc:self.siend-bigc] = s[:,:,:].copy()
        sgc[:,:,0:self.igc] = sgc[:,:,self.siend-self.igc-bigc:self.siend-bigc] #Add ghostcell upstream x-direction
        sgc[:,:,self.siend-bigc:self.siend+self.igc] = sgc[:,:,self.igc:self.igc+self.igc+bigc] #Add ghostcell downstream x-direction
        sgc[:,0:self.jgc,:] = sgc[:,self.sjend-self.jgc-bjgc:self.sjend-bjgc,:] #Add ghostcell upstream y-direction
        sgc[:,self.sjend-bjgc:self.sjend+self.jgc,:] = sgc[:,self.jgc:self.jgc+self.jgc+bjgc,:] #Add ghostcell downstream y-direction
        sgc[0:self.kgc,:,:] = sgc[self.skend-self.kgc-bkgc:self.skend-bkgc,:,:] #Add ghostcell bottom z-direction
        sgc[self.skend-bkgc:self.skend+self.kgc] = sgc[self.kgc:self.kgc+self.kgc+bkgc,:,:] #Add ghostcell top z-direction

        #Store new fields in object
        self.var['output'][variable_name]['variable'] = sgc
        
    def __add_ghostcells_grid(self):
        '''Store new coordinates with ghost cells in object.'''

        #Get original coordinates
        z = self.var['grid']['z']
        zh = self.var['grid']['zh']
        y = self.var['grid']['y']
        yh = self.var['grid']['yh']
        x = self.var['grid']['x']
        xh = self.var['grid']['xh']

        #Check that the grid center and grid edges arrays defined above have the same shape (which is an implicit assumption in the code below).
        if (z.shape != zh.shape) or (y.shape != yh.shape) or (x.shape != xh.shape):
            raise RuntimeError("The shape of the arrays representing the grid centers and edges should be the same, but they are not while this implicitly assumed in the script." )
        
        #Check for each coordinate that no ghostcells are specified when no periodic bc are assumed. No ghost cells are yet implemented for other types of BC's.
        if self.kgc > 0 and not self.periodic_bc[0]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")
            
        if self.jgc > 0 and not self.periodic_bc[1]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        if self.igc > 0 and not self.periodic_bc[2]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        #Check that there are not more ghost cells than grid cells. 
        if (self.igc + 1) > (x.shape[0]) or (self.jgc + 1) > (y.shape[0]) or (self.kgc + 1) > (z.shape[0]):
            raise ValueError("The needed number of ghostcells is larger than the number of grid cells present in the object.") 
        
        #Determine size new output array including ghost cells
        icells = self.var['grid']['itot'] + 2*self.igc
        jcells = self.var['grid']['jtot'] + 2*self.jgc
      #  kcells = self.var['grid']['ktot'] + 2*kgc
        kcells = self.var['grid']['ktot'] + 2*self.kgc
        
        #Initialize new coordinates and store ghost cells in them
        xgc = np.zeros(icells)
        xhgc = np.zeros(icells+1)
        ygc = np.zeros(jcells)
        yhgc = np.zeros(jcells+1)
        zgc = np.zeros(kcells)
        zhgc = np.zeros(kcells+1)
        
        self.iend = self.igc + self.var['grid']['itot']
        self.ihend = self.igc + self.var['grid']['itot'] + 1
        self.jend = self.jgc + self.var['grid']['jtot']
        self.jhend = self.jgc + self.var['grid']['jtot'] + 1
        self.kend = self.kgc + self.var['grid']['ktot']
        self.khend = self.kgc + self.var['grid']['ktot'] + 1
        
        xgc  = self.__add_ghostcells_cor(xgc , 0,  self.igc, self.iend , x,  self.var['grid']['xsize'])
        xhgc = self.__add_ghostcells_cor(xhgc, 1,  self.igc, self.ihend, xh, self.var['grid']['xsize'])
        ygc  = self.__add_ghostcells_cor(ygc , 0,  self.jgc, self.jend , y,  self.var['grid']['ysize'])
        yhgc = self.__add_ghostcells_cor(yhgc, 1,  self.jgc, self.jhend, yh, self.var['grid']['ysize'])
        zgc  = self.__add_ghostcells_cor(zgc , 0,  self.kgc, self.kend , z,  self.var['grid']['zsize'])
        zhgc = self.__add_ghostcells_cor(zhgc, 1,  self.kgc, self.khend, zh, self.var['grid']['zsize'])

        self.var['grid']['z'] = zgc
        self.var['grid']['zh'] = zhgc
        self.var['grid']['y'] = ygc
        self.var['grid']['yh'] = yhgc
        self.var['grid']['x'] = xgc
        self.var['grid']['xh'] = xhgc
        
    def __add_ghostcells_cor(self, corgc, bgc, gc, endindex, cor, size):
        """ Add ghostcells to coordinates. """
        corgc[gc:endindex-bgc] = cor[:]
        if gc != 0:
            corgc[0:gc] = 0 - (size - corgc[endindex-bgc-gc:endindex-bgc])
        if (gc != 0) or (bgc != 0): 
            corgc[endindex-bgc:endindex+gc] = size + corgc[gc:gc+gc+bgc]
        return corgc
    
    def volume_integral(self, variable_name):
        """ Return volume integral for variable specified by variable name. (Defined for testing purposes.)"""
        
        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.var['output'].keys():
            raise KeyError("Specified variable_name not defined in object.")
            
        #Read in variable values and axes, remove ghostcells
        #NOTE: because in the next section np.trapz only integrates over the range specified by the coordinates, it is needed that all coordinates have the same range. This is done below by copying values of the variable.
        s = self.var['output'][variable_name]['variable']
        
        if self.var['output'][variable_name]['orientation'][0]:
            z = self.var['grid']['zh'][self.kgc:self.khend]
            s = s[self.kgc:self.khend,:,:].copy()
        else:
            z = self.var['grid']['z'][self.kgc:self.kend]
            s = s[self.kgc:self.kend,:,:].copy()
            z = np.insert(z, 0, 0.0)
            z = np.append(z, self.var['grid']['zsize'])
            s = np.insert(s, 0, s[0,:,:], axis = 0)
            s = np.append(s, s[np.newaxis,-1,:,:], axis = 0)
            
        if self.var['output'][variable_name]['orientation'][1]:
            y = self.var['grid']['yh'][self.jgc:self.jhend]
            s = s[:,self.jgc:self.jhend,:].copy()
        else:
            y = self.var['grid']['y'][self.jgc:self.jend]
            s = s[:,self.jgc:self.jend,:].copy()
            y = np.insert(y, 0, 0.0)
            y = np.append(y, self.var['grid']['ysize'])
            s = np.insert(s, 0, s[:,0,:], axis = 1)
            s = np.append(s, s[:,np.newaxis,-1,:], axis = 1)
        
        if self.var['output'][variable_name]['orientation'][2]:
            x = self.var['grid']['xh'][self.igc:self.ihend]
            s = s[:,:,self.igc:self.ihend].copy()
        else:
            x = self.var['grid']['x'][self.igc:self.iend]
            s = s[:,:,self.igc:self.iend].copy()
            x = np.insert(x, 0, 0.0)
            x = np.append(x, self.var['grid']['xsize'])
            s = np.insert(s, 0, s[:,:,0], axis = 2)
            s = np.append(s, s[:,:,np.newaxis,-1], axis = 2)
            
        #Integrate over variable by applying the composite trapezoïdal rule for each coordinate separately
        x_int = np.broadcast_to(x[np.newaxis,np.newaxis,:], (s.shape[0], s.shape[1], s.shape[2]))
        s_intx = np.trapz(s, x_int, axis = -1)
        y_int = np.broadcast_to(y[np.newaxis,:], (s.shape[0], s.shape[1]))
        s_intyx = np.trapz(s_intx, y_int, axis = -1)
        s_intzyx = np.trapz(s_intyx, z, axis = -1)
        
        return s_intzyx
    
    def __edgegrid_from_centergrid(self, coord_center, len_coord, size_coord):
        """ Define coordinates corresponding to the grid walls from coordinates corresponding to the centers of the grid. """

        dcoord = coord_center[1:] - coord_center[:-1]
        coord_edge = np.empty(len_coord)
        coord_edge[0] = 0
        for i in range(1,len_coord):
            coord_edge[i] = coord_center[i-1] + 0.5 * dcoord[i-1]
        return coord_edge

    def __getitem__(self, name):
        if name in self.var.keys():
            return self.var[name]
        else:
            raise RuntimeError('Can\'t find variable \"{}\" in object.')

class Coarsegrid:
    """ Returns a single object that defines a coarse grid based on a corresponding finegrid_object (instance of Finegrid class) and the dimensions of the new grid (dim_new_grid). 
        The dimensions of the coarse grid should be specified as a tuple (z,y,x).
        Furthermore, a method is provided to downsample variables already present in the finegrid_ojbect.
        NOTE: if the finegrid does not contain the variable to be downsampled at the time when the coarsegrid object is initialized, no downsampling is possible."""

    def __init__(self, dim_new_grid, finegrid_object):
        
        #Check whether dim_new_grid is a tuple of length 3 with only positive integers
        if not isinstance(dim_new_grid, tuple):
            raise TypeError("Dim_new_grid should be a tuple with length 3 (z, y, x), and consist only of positive integers.")
        
        if not len(dim_new_grid) == 3:
            raise ValueError("Dim_new_grid should be a tuple with length 3 (z, y, x), and consist only of positive integers.")
            
        if not any( (dim > 0 and isinstance(dim, int)) for dim in dim_new_grid):
            raise ValueError("Dim_new_grid should be a tuple with length 3 (z, y, x), and consist only of positive integers.")
        
        nz,ny,nx = dim_new_grid

        #Store finegrid_object in Coarsegrid object
        self.finegrid = finegrid_object

        #Store relevant settings from finegrid_object and dim_new_grid into coarsegrid_object
        self.var = {}
        self.var['grid'] = {}
        self.var['output'] = {} #After the coarsegrid object is created, output can be stored here. This is done in the script func_generate_training.py.
        self.igc = self.finegrid.igc
        self.jgc = self.finegrid.jgc
        self.kgc = self.finegrid.kgc
        self.periodic_bc = self.finegrid.periodic_bc

        #Boundary condition: the resolution of the coarse grid should indeed be coarser than the fine resolution
        if not (nz < finegrid_object['grid']['ktot'] and ny < finegrid_object['grid']['jtot'] and nx < finegrid_object['grid']['itot']):
            raise ValueError("The resolution of the coarse grid should be coarser than the fine resolution contained in the Finegrid object.")        

        self.var['grid']['ktot'] = nz
        self.var['grid']['jtot'] = ny
        self.var['grid']['itot'] = nx

        try:
            self.var['grid']['xsize'] = finegrid_object['grid']['xsize']
            self.var['grid']['ysize'] = finegrid_object['grid']['ysize']
            self.var['grid']['zsize'] = finegrid_object['grid']['zsize']

        except KeyError:
            raise KeyError("At least one of the needed settings was not contained in the finegrid_object. Check that the relevant methods have been called to initialize the grid.")
    
        self.var['grid']['x'] , self.var['grid']['xdist']  = self.__define_coarsegrid(self.var['grid']['itot'], self.var['grid']['xsize'], center_cell = True) 
        self.var['grid']['xh'], self.var['grid']['xhdist'] = self.__define_coarsegrid(self.var['grid']['itot'], self.var['grid']['xsize'], center_cell = False)
        self.var['grid']['y'] , self.var['grid']['ydist']  = self.__define_coarsegrid(self.var['grid']['jtot'], self.var['grid']['ysize'], center_cell = True)
        self.var['grid']['yh'], self.var['grid']['yhdist'] = self.__define_coarsegrid(self.var['grid']['jtot'], self.var['grid']['ysize'], center_cell = False)
        self.var['grid']['z'] , self.var['grid']['zdist']  = self.__define_coarsegrid(self.var['grid']['ktot'], self.var['grid']['zsize'], center_cell = True)
        self.var['grid']['zh'], self.var['grid']['zhdist'] = self.__define_coarsegrid(self.var['grid']['ktot'], self.var['grid']['zsize'], center_cell = False)
        
        #Add ghostcells to grid
        self.__add_ghostcells_grid()
        
    #Calculate new grid using the information stored above, assuming the grid is uniform
    def __define_coarsegrid(self, number_gridcells, gridsize, center_cell):
        """ Manually define the coarse grid. Note that the size of the domain is determined by the finegrid object used to initialize the coarsegrid object. """        

        dist_coarsecoord = gridsize / number_gridcells
        if center_cell:
            coarsecoord, step = np.linspace(0.5*dist_coarsecoord, gridsize-0.5*dist_coarsecoord, number_gridcells, endpoint=True, retstep=True)
        else:
            coarsecoord, step = np.linspace(0, gridsize, number_gridcells, endpoint=False, retstep=True)

        return coarsecoord, step
    
    def downsample(self, variable_name):
        
        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.finegrid['output'].keys():
            raise KeyError("Specified variable_name not defined in finegrid object on which this coarsegrid object is based. Not possible to do downsampling.")

        self.var['output'][variable_name] = {}
        self.var['output'][variable_name]['orientation'] = self.finegrid['output'][variable_name]['orientation']
        variable_downsampled = downsampling_training.downsample(finegrid = self.finegrid, coarsegrid = self, variable_name = variable_name, bool_edge_gridcell = self.var['output'][variable_name]['orientation'], periodic_bc = self.periodic_bc)
        self.var['output'][variable_name]['variable'] = variable_downsampled.copy()
        #Add ghostcells depending on the order used for the spatial interpolation, add 1 additonal cell on downstream/top boundaries
        self.__add_ghostcells(variable_name)
            
    def __add_ghostcells(self, variable_name):
        """ Add ghostcells (defined as grid cells located at the downstream/top boundary and outside the domain) to fine grid for variable_name. 
            The samples at the end of the array are repeated at the beginning and vice versa. Furthermore, the ghostcells are added to the coordinates as well."""

        #NOTE: slightly different from finegrid add_ghostcells method above, because here the additional points for the grid edges arrays are already added in the output field s. Remove them first to ensure added ghostcells are consistent.
            
        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.var['output'].keys():
            raise KeyError("Specified variable_name not defined in object.")

        #Check that ghost cells are not negative
        if self.jgc < 0 or self.igc < 0 or self.kgc < 0:
            raise ValueError("The specified number of ghostcells cannot be negative.")
            
        #Check for each coordinate that no ghostcells are specified when no periodic bc are assumed. No ghost cells are yet implemented for other types of BC's.
        if self.kgc > 0 and not self.periodic_bc[0]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")
            
        if self.jgc > 0 and not self.periodic_bc[1]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        if self.igc > 0 and not self.periodic_bc[2]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        #Read variable from object
        s = self.var['output'][variable_name]['variable']

        #Determine size new output array including ghost cells
        icells = self.var['grid']['itot'] + 2*self.igc
        jcells = self.var['grid']['jtot'] + 2*self.jgc
      #  kcells = self.var['grid']['ktot'] + 2*kgc
        kcells = self.var['grid']['ktot'] + 2*self.kgc
        
        #Depending on bool_edge_gridcell, add one additional ghost cell at top/downstream boundaries independent of self.igc/jgc/kgc
        bkgc = 0
        bjgc = 0
        bigc = 0
        if self.var['output'][variable_name]['orientation'][0]:
            bkgc = 1
            s = s[:-1,:,:].copy()
        if self.var['output'][variable_name]['orientation'][1]:
            bjgc = 1
            s = s[:,:-1,:].copy()
        if self.var['output'][variable_name]['orientation'][2]:
            bigc = 1
            s = s[:,:,:-1].copy()

        self.siend = self.igc + self.var['grid']['itot'] + bigc
        self.sjend = self.jgc + self.var['grid']['jtot'] + bjgc
        self.skend = self.kgc + self.var['grid']['ktot'] + bkgc
        
        #Check that there are not more ghost cells than grid cells. 
        if (self.igc + bigc) > (s.shape[2]) or (self.jgc + bjgc) > (s.shape[1]) or (self.kgc + bkgc) > (s.shape[0]):
            raise ValueError("The needed number of ghostcells is larger than the number of grid cells present in the object.") 
        
        #Initialize new output array including ghost cells
        sgc = np.zeros((kcells+bkgc, jcells+bjgc, icells+bigc))
        
        #Fill new initialzid array including ghost cells
        sgc[self.kgc:self.skend-bkgc, self.jgc:self.sjend-bjgc, self.igc:self.siend-bigc] = s[:,:,:].copy()
        sgc[:,:,0:self.igc] = sgc[:,:,self.siend-self.igc-bigc:self.siend-bigc] #Add ghostcell upstream x-direction
        sgc[:,:,self.siend-bigc:self.siend+self.igc] = sgc[:,:,self.igc:self.igc+self.igc+bigc] #Add ghostcell downstream x-direction
        sgc[:,0:self.jgc,:] = sgc[:,self.sjend-self.jgc-bjgc:self.sjend-bjgc,:] #Add ghostcell upstream y-direction
        sgc[:,self.sjend-bjgc:self.sjend+self.jgc,:] = sgc[:,self.jgc:self.jgc+self.jgc+bjgc,:] #Add ghostcell downstream y-direction
        sgc[0:self.kgc,:,:] = sgc[self.skend-self.kgc-bkgc:self.skend-bkgc,:,:] #Add ghostcell bottom z-direction
        sgc[self.skend-bkgc:self.skend+self.kgc] = sgc[self.kgc:self.kgc+self.kgc+bkgc,:,:] #Add ghostcell top z-direction

        #Store new fields in object
        self.var['output'][variable_name]['variable'] = sgc
        
    def __add_ghostcells_grid(self):
        '''Store new coordinates with ghost cells in object.'''

        #Get original coordinates
        z = self.var['grid']['z']
        zh = self.var['grid']['zh']
        y = self.var['grid']['y']
        yh = self.var['grid']['yh']
        x = self.var['grid']['x']
        xh = self.var['grid']['xh']

        #Check that the grid center and grid edges arrays defined above have the same shape (which is an implicit assumption in the code below).
        if (z.shape != zh.shape) or (y.shape != yh.shape) or (x.shape != xh.shape):
            raise RuntimeError("The shape of the arrays representing the grid centers and edges should be the same, but they are not while this implicitly assumed in the script." )
        
        #Check for each coordinate that no ghostcells are specified when no periodic bc are assumed. No ghost cells are yet implemented for other types of BC's.
        if self.kgc > 0 and not self.periodic_bc[0]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")
            
        if self.jgc > 0 and not self.periodic_bc[1]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        if self.igc > 0 and not self.periodic_bc[2]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        #Check that there are not more ghost cells than grid cells. 
        if (self.igc + 1) > (x.shape[0]) or (self.jgc + 1) > (y.shape[0]) or (self.kgc + 1) > (z.shape[0]):
            raise ValueError("The needed number of ghostcells is larger than the number of grid cells present in the object.") 
        
        #Determine size new output array including ghost cells
        icells = self.var['grid']['itot'] + 2*self.igc
        jcells = self.var['grid']['jtot'] + 2*self.jgc
      #  kcells = self.var['grid']['ktot'] + 2*kgc
        kcells = self.var['grid']['ktot'] + 2*self.kgc
        
        #Initialize new coordinates and store ghost cells in them
        xgc = np.zeros(icells)
        xhgc = np.zeros(icells+1)
        ygc = np.zeros(jcells)
        yhgc = np.zeros(jcells+1)
        zgc = np.zeros(kcells)
        zhgc = np.zeros(kcells+1)
        
        self.iend = self.igc + self.var['grid']['itot']
        self.ihend = self.igc + self.var['grid']['itot'] + 1
        self.jend = self.jgc + self.var['grid']['jtot']
        self.jhend = self.jgc + self.var['grid']['jtot'] + 1
        self.kend = self.kgc + self.var['grid']['ktot']
        self.khend = self.kgc + self.var['grid']['ktot'] + 1
        
        xgc  = self.__add_ghostcells_cor(xgc , 0,  self.igc, self.iend , x,  self.var['grid']['xsize'])
        xhgc = self.__add_ghostcells_cor(xhgc, 1,  self.igc, self.ihend, xh, self.var['grid']['xsize'])
        ygc  = self.__add_ghostcells_cor(ygc , 0,  self.jgc, self.jend , y,  self.var['grid']['ysize'])
        yhgc = self.__add_ghostcells_cor(yhgc, 1,  self.jgc, self.jhend, yh, self.var['grid']['ysize'])
        zgc  = self.__add_ghostcells_cor(zgc , 0,  self.kgc, self.kend , z,  self.var['grid']['zsize'])
        zhgc = self.__add_ghostcells_cor(zhgc, 1,  self.kgc, self.khend, zh, self.var['grid']['zsize'])

        self.var['grid']['z'] = zgc
        self.var['grid']['zh'] = zhgc
        self.var['grid']['y'] = ygc
        self.var['grid']['yh'] = yhgc
        self.var['grid']['x'] = xgc
        self.var['grid']['xh'] = xhgc
        self.define_ghost_grid = False
        
    def __add_ghostcells_cor(self, corgc, bgc, gc, endindex, cor, size):
        """ Add ghostcells to coordinates. """
        corgc[gc:endindex-bgc] = cor[:]
        if gc != 0:
            corgc[0:gc] = 0 - (size - corgc[endindex-bgc-gc:endindex-bgc])
        if (gc != 0) or (bgc != 0): 
            corgc[endindex-bgc:endindex+gc] = size + corgc[gc:gc+gc+bgc]
        return corgc
    
    def volume_integral(self, variable_name):
        """ Return volume integral for variable specified by variable name. (Defined for testing purposes.)"""
        
        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.var['output'].keys():
            raise KeyError("Specified variable_name not defined in object.")
            
        #Read in variable values and axes, remove ghostcells
        #NOTE: because in the next section np.trapz only integrates over the range specified by the coordinates, it is needed that all coordinates have the same range. This is done below by copying values of the variable.
        s = self.var['output'][variable_name]['variable']
        
        if self.var['output'][variable_name]['orientation'][0]:
            z = self.var['grid']['zh'][self.kgc:self.khend]
            s = s[self.kgc:self.khend,:,:].copy()
        else:
            z = self.var['grid']['z'][self.kgc:self.kend]
            s = s[self.kgc:self.kend,:,:].copy()
            z = np.insert(z, 0, 0.0)
            z = np.append(z, self.var['grid']['zsize'])
            s = np.insert(s, 0, s[0,:,:], axis = 0)
            s = np.append(s, s[np.newaxis,-1,:,:], axis = 0)
            
        if self.var['output'][variable_name]['orientation'][1]:
            y = self.var['grid']['yh'][self.jgc:self.jhend]
            s = s[:,self.jgc:self.jhend,:].copy()
        else:
            y = self.var['grid']['y'][self.jgc:self.jend]
            s = s[:,self.jgc:self.jend,:].copy()
            y = np.insert(y, 0, 0.0)
            y = np.append(y, self.var['grid']['ysize'])
            s = np.insert(s, 0, s[:,0,:], axis = 1)
            s = np.append(s, s[:,np.newaxis,-1,:], axis = 1)
        
        if self.var['output'][variable_name]['orientation'][2]:
            x = self.var['grid']['xh'][self.igc:self.ihend]
            s = s[:,:,self.igc:self.ihend].copy()
        else:
            x = self.var['grid']['x'][self.igc:self.iend]
            s = s[:,:,self.igc:self.iend].copy()
            x = np.insert(x, 0, 0.0)
            x = np.append(x, self.var['grid']['xsize'])
            s = np.insert(s, 0, s[:,:,0], axis = 2)
            s = np.append(s, s[:,:,np.newaxis,-1], axis = 2)
            
        #Integrate over variable by applying the composite trapezoïdal rule for each coordinate separately
        x_int = np.broadcast_to(x[np.newaxis,np.newaxis,:], (s.shape[0], s.shape[1], s.shape[2]))
        s_intx = np.trapz(s, x_int, axis = -1)
        y_int = np.broadcast_to(y[np.newaxis,:], (s.shape[0], s.shape[1]))
        s_intyx = np.trapz(s_intx, y_int, axis = -1)
        s_intzyx = np.trapz(s_intyx, z, axis = -1)
        
        return s_intzyx
    
    def __edgegrid_from_centergrid(self, coord_center, len_coord, size_coord):
        """ Define coordinates corresponding to the grid walls from coordinates corresponding to the centers of the grid. """

        dcoord = coord_center[1:] - coord_center[:-1]
        coord_edge = np.empty(len_coord)
        coord_edge[0] = 0
        for i in range(1,len_coord):
            coord_edge[i] = coord_center[i-1] + 0.5 * dcoord[i-1]
        return coord_edge
 
    def __getitem__(self, name):
        if name in self.var.keys():
            return self.var[name]
        else:
            raise RuntimeError('Can\'t find variable \"{}\" in object.')