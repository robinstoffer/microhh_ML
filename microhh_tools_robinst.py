import netCDF4 as nc
import numpy   as np
import struct  as st
import glob
import re

# -------------------------
# General help functions
# -------------------------

def _int_or_float_or_str(value):
    """ Helper function: convert a string to int/float/str """
    try:
        if ('.' in value):
            return float(value)
        else:
            return int(float(value))
    except:
        return value.rstrip()


def _convert_value(value):
    """ Helper function: convert namelist value or list """
    if ',' in value:
        value = value.split(',') 
        return [_int_or_float_or_str(val) for val in value]
    else:
        return _int_or_float_or_str(value)


def _find_namelist_file():
    """ Helper function: automatically find the .ini file in the current directory """
    namelist_file = glob.glob('*.ini')
    if len(namelist_file) == 0:
        raise RuntimeError('Can\'t find any .ini files in the current directory!') 
    if len(namelist_file) > 1:
        raise RuntimeError('There are multiple .ini files: {}'.format(namelist_file)) 
    else:
        return namelist_file[0]


def _process_endian(endian):
    if endian not in ['little', 'big']:
        raise ValueError('endian has to be \"little\" or \"big\"!')
    endian = '<' if endian == 'little' else '>'
    return endian


# -------------------------
# Classes and functions to read and write MicroHH things
# -------------------------

class Read_namelist:
    """ Reads a MicroHH .ini file to memory 
        All available variables are accessible as e.g.:
            nl = Read_namelist()    # with no name specified, it searches for a .ini file in the current dir
            itot = nl['grid']['itot']
            enttime = nl['time']['endtime']
            printing e.g. nl['grid'] provides an overview of the available variables in a group
    """
    def __init__(self, namelist_file=None):
        if (namelist_file is None):
            namelist_file = _find_namelist_file()
        
        self.groups = {}   # Dictionary holding all the data
        with open(namelist_file) as f:
            for line in f:
                lstrip = line.strip()
                if (len(lstrip) > 0 and lstrip[0] != "#"):
                    if lstrip[0] == '[' and lstrip[-1] == ']':
                        curr_group_name = lstrip[1:-1]
                        self.groups[curr_group_name] = {}
                    elif ("=" in line):
                        var_name = lstrip.split('=')[0]
                        value = _convert_value(lstrip.split('=')[1])
                        self.groups[curr_group_name][var_name] = value

    def __getitem__(self, name):
        if name in self.groups.keys():
            return self.groups[name]
        else:
            raise RuntimeError('Can\'t find group \"{}\" in .ini file'.format(name))

    def __repr__(self):
        return 'Available groups:\n{}'.format(', '.join(self.groups.keys()))


def replace_namelist_value(variable, new_value, namelist_file=None):
    """ Replace a variables value in an existing namelist """
    if namelist_file is None:
        namelist_file = _find_namelist_file()

    with open(namelist_file, "r") as source:
        lines = source.readlines()
    with open(namelist_file, "w") as source:
        for line in lines:
            source.write(re.sub(r'({}).*'.format(variable), r'\1={}'.format(new_value), line))


class Read_statistics:
    """ Read all the NetCDF statistics
        Example: 
        f = Read_statistics('drycblles.default.0000000.nc')
        print(f) prints a list with the available variables
        The data can be accessed as either f['th'] or f.th, which returns the numpy array with data
        The variable names can be accessed as f.names['th'], the units as f.units['th'], the dimensions as f.dimensions['th']
        This allows you to automatically format axis labels as e.g.:
        pl.xlabel("{0:} ({1:})".format(f.names['th'], f.units['th']))
        """
    def __init__(self, stat_file):
        f = nc4.Dataset(stat_file, 'r')

        # Dictionaries which hold the variable names, units, etc.
        self.data       = {}
        self.units      = {}
        self.names      = {}
        self.dimensions = {}
     
        # For each variable in the NetCDF file, read all the content and info 
        for var in f.variables:
            self.data[var]       = f.variables[var].__array__()
            self.units[var]      = f.variables[var].units
            self.names[var]      = f.variables[var].long_name
            self.dimensions[var] = f.variables[var].dimensions

        f.close()

    def __getitem__(self, name):
        if name in self.data.keys():
            return self.data[name]
        else:
            raise RuntimeError('Can\'t find variable \"{}\" in statistics file'.format(name))
    
    def __getattr__(self, name):
        if name in self.data.keys():
            return self.data[name]
        else:
            raise RuntimeError('Can\'t find variable \"{}\" in statistics file'.format(name))

    def __repr__(self):
        return 'Available variables:\n{}'.format(', '.join(self.names.keys()))


class Finegrid:
    """ Returns a single object that can read from binary files or explicitly define the grid and output variables.
        The read_grid flag indicates whether the grid is read from an existing simulation (True) or explicitly defined for testing purposes (False).
        When read_grid flag is True, specify explicitly (using keywords) grid_filename and settings_filename.
            -If no grid filename is provided, grid.0000000 from the current directory is read.
            -If no settings filename is provided, a .ini file should be present in the current directory to read.
        When read_grid flag is False, specifiy explicitly (using keywords) each spatial coordinate (coordx,coordy,coordz) as a 1d numpy array, e.g. Finegrid(read_grid_flag = False, coordx = np.array([1,2,3]),sizex = 4, coordy = np.array([0.5,1.5,3.5,6]), sizey = 7, coordz = np.array([0.1,0.3,1]), sizez = 1.2). 
        Furthermore, for each spatial dimension the length should be indicated (sizex,sizey,sizez), which has to be larger than the last spatial coordinate in each dimension. 
        Each coordinate should 1) refer to the center of the grid cell, 2) be larger than 0, 3) be larger than the previous coordinate in the same direction. """
    def __init__(self, read_grid_flag = True , endian = 'little', **kwargs):

        #Set correct endianness
        self.en  = _process_endian(endian)

        #Define empty dictionary to store grid and variables
        self.var = {}
        self.var['output'] = {}
        self.var['grid'] = {}
        self.var['time'] = {}

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
                if not (np.all(kwargs['coordx'][1:] > kwargs['coordx'][:-1]) and np.all(kwargs['coordx'][:] > 0) and (len(kwargs['coordx'][:].shape) == 1)):
                    raise ValueError("The coordinates in the x-direction should be 1-dimensional, strictly increaing, and consist of positive values only.") 
                self.var['grid']['x'] = kwargs['coordx']

                if not (np.all(kwargs['coordy'][1:] > kwargs['coordy'][:-1]) and np.all(kwargs['coordy'][:] > 0) and (len(kwargs['coordy'][:].shape) == 1)):
                    raise ValueError("The coordinates in the y-direction should be 1-dimensional, strictly increasing, and consist of positive values only.")
                self.var['grid']['y'] = kwargs['coordy']

                if not (np.all(kwargs['coordz'][1:] > kwargs['coordz'][:-1]) and np.all(kwargs['coordz'][:] > 0) and (len(kwargs['coordz'][:].shape) == 1)):
                    raise ValueError("The coordinates in the z-direction should be 1-dimensional, strictly increasing, and consist of positive values only.")
                self.var['grid']['z'] = kwargs['coordz']

                if not kwargs['sizex'] > kwargs['coordx'][-1]:
                    raise ValueError("The length of the x-coordinate should be larger than the last coordinate.")
                self.var['grid']['xsize'] = kwargs['sizex']

                if not kwargs['sizey'] > kwargs['coordy'][-1]:
                    raise ValueError("The length of the y-coordinate should be larger than the last coordinate.")
                self.var['grid']['ysize'] = kwargs['sizey']

                if not kwargs['sizez'] > kwargs['coordz'][-1]:
                    raise ValueError("The length of the z-coordinate should be larger than the last coordinate.")
                self.var['grid']['zsize'] = kwargs['sizez']

            except KeyError:
                print("The needed arguments were not correctly specified. Make sure that coordx, coordy and coordz are assinged to a 1d numpy array.")
                raise
            self.__define_grid(self.var['grid']['x'],self.var['grid']['y'],self.var['grid']['z'], self.var['grid']['xsize'], self.var['grid']['ysize'], self.var['grid']['zsize'])

    def __read_settings(self,settings_filename = None):
        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read settings from an already existing settings file.")

        #Read settings of MicroHH simulation from specified settings_filename. 
        #If None, a .ini file should be present in the current directory to read.
        settings = Read_namelist(settings_filename)

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
        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read the grid from an already existing binary grid file.")

        self.fin = open(grid_filename,'rb')
        self.var['grid']['x'] = np.array(st.unpack('{0}{1}d'.format(self.en, self.var['grid']['itot']), self.fin.read(self.var['grid']['itot']*8)))
        self.var['grid']['xh'] = np.array(st.unpack('{0}{1}d'.format(self.en, self.var['grid']['itot']), self.fin.read(self.var['grid']['itot']*8)))
        self.var['grid']['y'] = np.array(st.unpack('{0}{1}d'.format(self.en, self.var['grid']['jtot']), self.fin.read(self.var['grid']['jtot']*8)))
        self.var['grid']['yh'] = np.array(st.unpack('{0}{1}d'.format(self.en, self.var['grid']['jtot']), self.fin.read(self.var['grid']['jtot']*8)))
        self.var['grid']['z'] = np.array(st.unpack('{0}{1}d'.format(self.en, self.var['grid']['ktot']), self.fin.read(self.var['grid']['ktot']*8)))
        self.var['grid']['zh'] = np.array(st.unpack('{0}{1}d'.format(self.en, self.var['grid']['ktot']), self.fin.read(self.var['grid']['ktot']*8)))
        self.fin.close()

    def read_binary_variables(self, variable_name, timestep):
        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read settings from an already existing binary file.")

        #Check wheter timestep is present in data
        if timestep > (self.var['time']['timesteps'] - 1):
            raise ValueError("Specified timestep not contained in data. Note that the timestep starts counting from 0.")

        #Open binary file to read variable at timestep
        var_t = int(self.var['time']['starttime']  + timestep*self.var['time']['savetime'] )
        fin = open("%s.%07i"%(variable_name, var_t),"rb")
        self.var['output'][variable_name] = np.empty((self.var['grid']['ktot'],self.var['grid']['jtot'],self.var['grid']['itot'] ))
        for k in range(self.var['grid']['ktot']):
            raw = fin.read(self.var['grid']['jtot']*self.var['grid']['itot']*8)
            tmp = np.array(st.unpack('{0}{1}d'.format(self.en, self.var['grid']['jtot']*self.var['grid']['itot']), raw))
            self.var['output'][variable_name][k,:,:] = tmp.reshape((self.var['grid']['jtot'], self.var['grid']['itot']))
        fin.close()

    def read_nc_variables(self,variable_name,timestep): #Note: is made only for netcdf files from MicroHH binary files using the function binary_to_3d (see below)
        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read settings from an already existing netcdf file.")

        #Check wheter timestep is present in data
        if timestep > (self.var['time']['timesteps'] - 1):
            raise ValueError("Specified timestep not contained in data. Note that the timestep starts counting from 0.")
            
        nc_file = nc.Dataset("%s.nc"%(variable_name),"r")
        self.var['output'][variable_name] =  nc_file[variable_name][timestep,:,:,:]

    def __define_grid(self,coordx,coordy,coordz,sizex,sizey,sizez):
        if self.read_grid_flag:
            raise RuntimeError("Object defined to read the grid from binary files (via the read_binary_grid method). Create a new Finegrid object with read_grid_flag = False to manually define the settings of the grid.")

        #define lengths of specified grid
        self.var['grid']['itot'] = len(coordx)
        self.var['grid']['jtot'] = len(coordy)
        self.var['grid']['ktot'] = len(coordz)

        #create coordinates corresponding to edges grid cell
        self.var['grid']['xh'] = self.__edgegrid_from_centergrid(self.var['grid']['x'], self.var['grid']['itot'])
        self.var['grid']['yh'] = self.__edgegrid_from_centergrid(self.var['grid']['y'], self.var['grid']['jtot'])
        self.var['grid']['zh'] = self.__edgegrid_from_centergrid(self.var['grid']['z'], self.var['grid']['ktot'])
        
        #Set define_grid_flag to True, such that the create_variables method can be executed
        self.define_grid_flag = True

    def create_variables(self,variable_name, output_field):
        if self.read_grid_flag:
            raise RuntimeError("Object defined to read variables from binary files (via the read_binary_variables method). Create a new Finegrid object with read_grid_flag = False to manually define the settings of the grid.")


        #Check that grid is already defined.
        if not self.define_grid_flag:
            raise RuntimeError("Object does not yet contain the coordinates for the output variable. Define them explicitly first via the __define_grid method") 
        #Check that shape output field is correct according to the coordinates specified via the define_grid method.
        if not (output_field.shape == (self.var['grid']['ktot'],self.var['grid']['jtot'],self.var['grid']['itot'])):
            raise RuntimeError("The shape corresponding to the specified coordinates (z,y,x) is not the same as the shape of the specified output field.")

        #Store output_field in object
        self.var['output'][variable_name] = output_field

    def add_ghostcells(self,variable_name,number_ghostcells_hor, number_ghostcells_ver):
        """ Add ghostcells to coarse grid for variable_name when specified in object. In the horizontal direction (number_ghostcells_hor) the samples at the end of the array are repeated at the beginning and vice versa (adding an equal number of ghostcells at both sides of the domain i.e. number_ghostcells_hor should be even), in the vertical direction zeros are added at the top and bottom of the domain (but always one more at the top, so number_ghostcells_ver should be odd).  Note: since grid is defined when object is initialized, no checks are needed to ensure that the grid is present. """
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.var['output'].keys():
            raise RuntimeError("Specified variable_name not defined in object.")

        #Check that number_ghostcells_hor is even
        if number_ghostcells_hor % 2 != 0 or number_ghostcells_hor <= 0:
            raise ValueError("Specified number of ghost cells in the horizontal direction is not even or not larger than 0.")

        #Check that number_ghostcells_ver is odd
        if number_ghostcells_ver % 2 == 0 or number_ghostcells_ver <= 0:
            raise ValueError("Specified number of ghost cells in the vertical direction is not odd or not larger than 0.")

       pass

       # #Add specified ghostcells in horizontal direction
       # for i in range(int(number_ghostcells_hor * 0.5)):
       #     #Insert ghostcellls at the upstream sides of the domain
       #     self.var['output'][variable_name] = np.insert(self.var['output'][variable_name], 0, self.var['output'][variable_name][:,-1-i,:], axis = 1)
       #     self.var['output'][variable_name] = np.insert(self.var['output'][variable_name], 0, self.var['output'][variable_name][:,:,-1-i], axis = 2)

       #     #Insert ghostcells at the downstream sides of the domain
       #     self.var['output'][variable_name] = np.append(self.var['output'][variable_name],self.var['output'][variable_name][:,i,:], axis = 1)
       #     self.var['output'][variable_name] = np.append(self.var['output'][variable_name],self.var['output'][variable_name][:,:,i], axis = 2)

       # #Add specified ghostcells in horizontal direction
       # self.var['output'][variable_name] = np.append(self.var['output'][variable_name],0)
       # for i in range(int((number_ghostcells_ver-1) * 0.5)):
       #     #Insert ghostcellls at the bottom of the domain
       #     self.var['output'][variable_name] = np.insert(self.var['output'][variable_name], [0,:,:], 0, axis = 0)

       #     #Insert ghostcells at the top of the domain
       #     self.var['output'][variable_name] = np.append(self.var['output'][variable_name],0, axis = 0)

    def __edgegrid_from_centergrid(self, coord_center, len_coord):
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

    def __getattr__(self, name):
        if name in self.var.keys():
            return self.var[name]
        else:
            raise RuntimeError('Can\'t find variable \"{}\" in object.')

class Coarsegrid:
    """ Returns a single object that defines a coarse grid based on a corresponding finegrid_object (instance of Finegrid class) and the dimensions of the new grid (dim_new_grid). The dimensions of the coarse grid should be specified as a tuple (z,y,x)."""

    def __init__(self,dim_new_grid,finegrid_object):
        nz,ny,nx = dim_new_grid
        
        #Store relevant settings from finegrid_object and dim_new_grid into coarsegrid_object
        self.var = {}
        self.var['grid'] = {}
        self.var['grid']['ktot'] = nz
        self.var['grid']['jtot'] = ny
        self.var['grid']['itot'] = nx

        try:
            self.var['grid']['xsize'] = finegrid_object['grid']['xsize']
            self.var['grid']['ysize'] = finegrid_object['grid']['ysize']
            self.var['grid']['zsize'] = finegrid_object['grid']['zsize']
        except KeyError:
            print("At least one of the needed settings was not contained in the finegrid_object. Check that the relevant methods have been called to initialize the grid.")
            raise

        #Calculate new grid using the information stored above, assuming the grid is uniform
        self.var['grid']['x'] = self.__define_coarsegrid(self.var['grid']['itot'], self.var['grid']['xsize'], center_cell = True) 
        self.var['grid']['xh'] = self.__define_coarsegrid(self.var['grid']['itot'], self.var['grid']['xsize'],center_cell = False)
        self.var['grid']['y'] = self.__define_coarsegrid(self.var['grid']['jtot'], self.var['grid']['ysize'],center_cell = True)
        self.var['grid']['yh'] = self.__define_coarsegrid(self.var['grid']['jtot'], self.var['grid']['ysize'],center_cell = False)
        self.var['grid']['z'] = self.__define_coarsegrid(self.var['grid']['ktot'], self.var['grid']['zsize'],center_cell = True)
        self.var['grid']['zh'] = self.__define_coarsegrid(self.var['grid']['ktot'], self.var['grid']['zsize'],center_cell = False)

    def __define_coarsegrid(self,number_gridcells,gridsize,center_cell):
        dist_coarsecoord = float(gridsize) / number_gridcells #float(..) to ensure backwards compatibility with Python 2
        if center_cell:
            coarsecoord = np.linspace(0.5*dist_coarsecoord,gridsize-0.5*dist_coarsecoord,number_gridcells,True)
        else:
            coarsecoord = np.linspace(0,gridsize-dist_coarsecoord,number_gridcells,True)
        return coarsecoord

       
    def __getitem__(self, name):
        if name in self.var.keys():
            return self.var[name]
        else:
            raise RuntimeError('Can\'t find variable \"{}\" in object.')

    def __getattr__(self, name):
        if name in self.var.keys():
            return self.var[name]
        else:
            raise RuntimeError('Can\'t find variable \"{}\" in object.')

def read_restart_file(path, itot, jtot, ktot, endian='little'):
    """ Read a MicroHH restart file into a 3D (or 2D if ktot=1) numpy array 
        The returned array has the dimensions ordered as [z,y,x] """

    en = _process_endian(endian)
    
    f  = open(path, 'rb')
    if (ktot > 1):
        field = np.zeros((ktot, jtot, itot))
        for k in range(ktot):
            raw = f.read(itot*jtot*8)
            tmp = np.array(st.unpack('{0}{1}d'.format(en, itot*jtot), raw))
            field[k,:,:] = tmp.reshape((jtot, itot))[:,:]
        f.close()
    else:
        raw = f.read(itot*jtot*8)
        tmp = np.array(st.unpack('{0}{1}d'.format(en, itot*jtot), raw))
        field = tmp.reshape((jtot, itot))

    return field


def write_restart_file(data, itot, jtot, ktot, path, per_slice=True, endian='little'):
    """ Write a restart file in the format requires by MicroHH.
        The input array should be indexed as [z,y,x] """
    
    en = _process_endian(endian)

    if(per_slice): 
        # Write level by level (less memory hungry.....)
        fout  = open(path, "wb")
        for k in range(ktot):
            tmp  = data[k,:,:].reshape(itot*jtot)
            tmp2 = st.pack('{0}{1}d'.format(en, tmp.size), *tmp) 
            fout.write(tmp2)
        fout.close()
    else:
        # Write entire field at once (memory hungry....)
        tmp  = data.reshape(data.size)
        tmp2 = st.pack('{0}{1}d'.format(en, tmp.size), *tmp) 
        fout = open(path, "wb")
        fout.write(tmp2)
        fout.close()  


def get_cross_indices(variable, mode):
    """ Find the cross-section indices given a variable name and mode (in 'xy','xz','yz') """
    if mode not in ['xy','xz','yz']:
        raise ValueError('\"mode\" should be in {\"xy\", \"xz\", \"yz\"}')

    # Get a list of all the cross-section files
    files = glob.glob('{}.{}.*.*'.format(variable, mode))
    if len(files) == 0:
        raise Exception('Cannot find any cross-section')

    # Get a list with all the cross-section files for one time
    time  = files[0].split('.')[-1]
    files = glob.glob('{}.{}.*.{}'.format(variable, mode, time))

    # Get the indices
    indices = [int(f.split('.')[-2]) for f in files]
    indices.sort()
    return indices

def binary3d_to_nc(variable,nx,ny,nz,starttime,endtime,sampletime,endian = 'little', savetype = 'double'):

    # Set correct dimensions for savefile
    nxsave = nx
    nysave = ny
    nzsave = nz

    # Set the correct string for the endianness
    if (endian == 'little'):
        en = '<'
    elif (endian == 'big'):
        en = '>'
    else:
        raise RuntimeError("Endianness has to be little or big")

    # Set the correct string for the savetype
    if (savetype == 'double'):
        sa = 'f8'
    elif (savetype == 'float'):
        sa = 'f4'
    else:
        raise RuntimeError("The savetype has to be float or double")

    # Calculate the number of time steps
    nt = int((endtime - starttime) / sampletime + 1)

    # Read grid properties from grid.0000000
    n   = nx*ny*nz
    fin = open("grid.{:07d}".format(0),"rb")
    raw = fin.read(nx*8)
    x   = np.array(st.unpack('{0}{1}d'.format(en, nx), raw))
    raw = fin.read(nx*8)
    xh  = np.array(st.unpack('{0}{1}d'.format(en, nx), raw))
    raw = fin.read(ny*8)
    y   = np.array(st.unpack('{0}{1}d'.format(en, ny), raw))
    raw = fin.read(ny*8)
    yh  = np.array(st.unpack('{0}{1}d'.format(en, ny), raw))
    raw = fin.read(nz*8)
    z   = np.array(st.unpack('{0}{1}d'.format(en, nz), raw))
    raw = fin.read(nz*8)
    zh  = np.array(st.unpack('{0}{1}d'.format(en, nz), raw))
    fin.close()

    # Create netCDF file
    ncfile = nc.Dataset("%s.nc"%variable, "w")
    
    if  (variable=='u'): loc = [1,0,0]
    elif(variable=='v'): loc = [0,1,0]
    elif(variable=='w'): loc = [0,0,1]
    else:                loc = [0,0,0]
    
    locx = 'x' if loc[0] == 0 else 'xh'
    locy = 'y' if loc[1] == 0 else 'yh'
    locz = 'z' if loc[2] == 0 else 'zh'
    
    dim_x  = ncfile.createDimension(locx,  nxsave)
    dim_y  = ncfile.createDimension(locy,  nysave)
    dim_z  = ncfile.createDimension(locz,  nzsave)
    dim_t  = ncfile.createDimension('time', nt)
    
    var_x  = ncfile.createVariable(locx, sa, (locx,))
    var_y  = ncfile.createVariable(locy, sa, (locy,))
    var_z  = ncfile.createVariable(locz, sa, (locz,))
    var_t  = ncfile.createVariable('time', 'i4', ('time',))
    var_3d = ncfile.createVariable(variable, sa, ('time',locz, locy, locx,))
    
    var_t.units = "time units since start"
    
    # Write grid properties to netCDF
    var_x[:] = x[:nxsave] if locx=='x' else xh[:nxsave]
    var_y[:] = y[:nysave] if locy=='y' else yh[:nysave]
    var_z[:] = z[:nzsave] if locz=='z' else zh[:nzsave]
    
    # Loop through the files and read 3d field
    for t in range(nt):
        var_t[t] = int((starttime + t*sampletime))
        print("Processing t={:07d}".format(var_t[t]))
    
        fin = open("%s.%07i"%(variable, var_t[t]),"rb")
        for k in range(nzsave):
            raw = fin.read(nx*ny*8)
            tmp = np.array(st.unpack('{0}{1}d'.format(en, nx*ny), raw))
            var_3d[t,k,:,:] = tmp.reshape((ny, nx))[:nysave,:nxsave]
        fin.close()
        ncfile.sync()
    
    ncfile.close()

