import netCDF4 as nc
import numpy   as np
import struct  as st
import glob
import re

# Note: script only compatible with Python 3!
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

def _process_precision(precision):
    if precision not in ['single', 'double']:
        raise ValueError('precision has to be \"single\" or \"double\"!')
    precision = 'float32' if precision == 'single' else 'float64'
    return precision


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
    
    def __repr__(self):
        return 'Available variables:\n{}'.format(', '.join(self.names.keys()))



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


class Finegrid:
    """ Returns a single object that can read from binary files or explicitly define the grid and output variables.
        The read_grid flag indicates whether the grid is read from an existing simulation (True) or explicitly defined for testing purposes (False).
        When read_grid flag is True, specify explicitly (using keywords) grid_filename and settings_filename.
            -If no grid filename is provided, grid.0000000 from the current directory is read.
            -If no settings filename is provided, a .ini file should be present in the current directory to read.
        When read_grid flag is False, specifiy explicitly (using keywords) each spatial coordinate (coordz,coordy,coordx) as a 1d numpy array, e.g. Finegrid(read_grid_flag = False, coordx = np.array([1,2,3]),xsize = 4, coordy = np.array([0.5,1.5,3.5,6]), ysize = 7, coordz = np.array([0.1,0.3,1]), zsize = 1.2).
        Furthermore, for each spatial dimension the length should be indicated (zsize,ysize,xsize), which has to be larger than the last spatial coordinate in each dimension. 
        Each coordinate should 1) refer to the center of the grid cell, 2) be larger than 0, 3) be larger than the previous coordinate in the same direction. """
    def __init__(self, read_grid_flag = True , precision = 'double', **kwargs):

        #Set correct precision
        self.prec  = _process_precision(precision)

        #Define empty dictionary to store grid and variables
        self.var = {}
        self.var['output'] = {}
        self.var['grid'] = {}
        self.var['time'] = {}
        self.var['ghost'] = {}

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

        #Append size to zh, yh, xh such that all points within the domain are included.
        self.var['grid']['zh'] = np.append(self.var['grid']['zh'], self.var['grid']['zsize'])
        self.var['grid']['yh'] = np.append(self.var['grid']['yh'], self.var['grid']['ysize'])
        self.var['grid']['xh'] = np.append(self.var['grid']['xh'], self.var['grid']['xsize'])

    def read_binary_variables(self, variable_name, timestep, bool_edge_gridcell = (False, False, False)):
        """ Read specified variable at a certain time step from binary files of a MicroHH simulation. Furthermore, the boundaries are added for the wind velocities consistent with the staggered grid."""
        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read settings from an already existing binary file.")

        #Check wheter timestep is present in data
        if timestep > (self.var['time']['timesteps'] - 1):
            raise ValueError("Specified timestep not contained in data. Note that the timestep starts counting from 0.")

        #Remove ghostcells for this variable_name of it was stored before (most likely for another timestep)
        if variable_name in self.var['ghost'].keys():
            del self.var['ghost'][variable_name]

        #Open binary file to read variable at timestep
        time = self.var['time']['starttime'] + timestep*self.var['time']['savetime']
        var_filename = '{0}.{1:07d}'.format(variable_name,time)
        self.var['output'][variable_name] = np.fromfile(var_filename, dtype = self.prec).reshape((self.var['grid']['ktot'], self.var['grid']['jtot'], self.var['grid']['itot']))

        #Add boundaries for wind velocities. Trick with np.newaxis makes sure that the values being appended have the same number of dimensions as the input array.
        self.add_boundaries_variable(self, variable_name, bool_edge_gridcell)

    def read_nc_variables(self, variable_name, timestep, bool_edge_gridcell = (False, False, False)): #Note: is made only for netcdf files from MicroHH binary files using the function binary_to_3d (see below)
        """ Read specified variable at a certain time step from netCDF file of a MicroHH simulation generated with the function binary_to_3d. """

        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read settings from an already existing netcdf file.")

        #Check wheter timestep is present in data
        if timestep > (self.var['time']['timesteps'] - 1):
            raise ValueError("Specified timestep not contained in data. Note that the timestep starts counting from 0.")

        #Remove ghostcells for this variable_name of it was stored before (most likely for another timestep)
        if variable_name in self.var['ghost'].keys():
            del self.var['ghost'][variable_name]

        nc_file = nc.Dataset("{0}.nc".format(variable_name),"r")
        self.var['output'][variable_name] =  nc_file[variable_name][timestep,:,:,:]

        #Add boundaries for wind velocities. Trick with np.newaxis makes sure that the values being appended have the same number of dimensions as the input array.
        self.add_boundaries_variable(self, variable_name, bool_edge_gridcell)

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
        
        #Set define_grid_flag to True, such that the create_variables method can be executed
        self.define_grid_flag = True

    def create_variables(self,variable_name, output_field, bool_edge_gridcell = (False, False, False)):
        """ Manually define output variables for a manually defined grid. """

        if self.read_grid_flag:
            raise RuntimeError("Object defined to read variables from binary files (via the read_binary_variables method). Create a new Finegrid object with read_grid_flag = False to manually define the settings of the grid.")


        #Check that grid is already defined.
        if not self.define_grid_flag:
            raise RuntimeError("Object does not yet contain the coordinates for the output variable. Define them explicitly first via the __define_grid method")

        #Check that shape output field is correct according to the coordinates specified via the define_grid method.
        if not (output_field.shape == (self.var['grid']['ktot'],self.var['grid']['jtot'],self.var['grid']['itot'])):
            raise RuntimeError("The shape corresponding to the specified coordinates (z,y,x) is not the same as the shape of the specified output field. Note that for the staggered grid the top/downstream boundaries do not have to be included, these are automatically added.")

        #Remove ghostcells for this variable_name of it was stored before (most likely for another timestep)
        if variable_name in self.var['ghost'].keys():
            del self.var['ghost'][variable_name]

        #Store output_field in object
        self.var['output'][variable_name] = output_field

        #Add boundaries for wind velocities. Trick with np.newaxis makes sure that the values being appended have the same number of dimensions as the input array.
        self.add_boundaries_variable(self, variable_name, bool_edge_gridcell)

    def add_ghostcells_hor(self,variable_name, jgc, igc, bool_edge_gridcell = (False, False, False)):
        """ Add ghostcells (defined as grid cells located at the downstream boundary and outside the domain) in horizontal directions to fine grid for variable_name when it is present in the object. 
            igc, jgc should be integers specifying the number of ghostcells to be added at each horizontal side in the domain (e.g. if igc = 3, 3 ghostcells are added at both the upstream and downstream boundary of the domain).
            The samples at the end of the array are repeated at the beginning and vice versa. Furthermore, the ghostcells are added to the coordinates as well."""
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.var['output'].keys():
            raise KeyError("Specified variable_name not defined in object.")

        #Make sure ghostcells can not be defined two times
        if variable_name in self.var['ghost'].keys():
            raise RuntimeError("Ghostcells already defined in object. To redefine the ghostcells, create new object.")

        #Check that ghost cells are not negative
        if jgc < 0 or igc < 0:
            raise ValueError("The specified number of ghostcells cannot be negative.")

      #  #Check that kgc is 0, 1 or 2 for each specified variable_name except 'w'. For 'w' kgc should be equal to 0.
      #  if not kgc in [0,1,2]:
      #      raise ValueError("Number of ghostcells in the vertical direction should be either 0, 1 or 2.")


      #  if  variable_name == 'w':
      #      s = self.var['output'][variable_name][:-1,:,:]

        #Get original coordinates, remove samples at the downstream boundaries added in object for the wind velocities to prevent they are sampled multiple times
        s = self.var['output'][variable_name]
        z = self.var['grid']['z']
        zh = self.var['grid']['zh']
        y = self.var['grid']['y']
        yh = self.var['grid']['yh'][:-1]
        x = self.var['grid']['x']
        xh = self.var['grid']['xh'][:-1]

        if bool_edge_gridcell[1]:
            s = s[:,:-1,:]
        if bool_edge_gridcell[2]:
            s = s[:,:,:-1]
        
        #Initialize new arrays including ghostcells
        icells = s.shape[2] + 2*igc
        jcells = s.shape[1] + 2*jgc
      #  kcells = self.var['grid']['ktot'] + 2*kgc
        kcells = s.shape[0]
        sgc = np.zeros((kcells, jcells, icells))
        xgc = np.zeros(icells)
        xhgc = np.zeros(icells)
        ygc = np.zeros(jcells)
        yhgc = np.zeros(jcells)

        #Add ghostcells in horizontal directions to new initialized arrays
        iend = igc + s.shape[2]
        jend = jgc + s.shape[1]
        #kend = kgc + self.var['grid']['ktot']

      #  sgc[0:kend, jgc:jend, igc:iend] = s[:,:,:]
        sgc[:, jgc:jend, igc:iend] = s[:,:,:]
        sgc[:,:,0:igc] = sgc[:,:,iend-igc:iend] #Add ghostcell upstream x-direction
        sgc[:,:,iend:iend+igc] = sgc[:,:,igc:igc+igc] #Add ghostcell downstream x-direction
        sgc[:,0:jgc,:] = sgc[:,jend-jgc:jend,:] #Add ghostcell upstream y-direction
        sgc[:,jend:jend+jgc,:] = sgc[:,jgc:jgc+jgc,:] #Add ghostcell downstream y-direction

        def __add_ghostcells_cor(corgc, indexgc, endindex, cor, size):
            corgc[indexgc:endindex] = cor[:]
            if igc != 0:
                corgc[0:indexgc] = 0 - (size - corgc[endindex-indexgc:endindex])
                corgc[indexgc:endindex+indexgc] = size + corgc[indexgc:indexgc+indexgc]
            return xgc

        xgc  = __add_ghostcells_cor(xgc, igc, iend, x, self.var['grid']['xsize'])
        xhgc = __add_ghostcells_cor(xhgc, igc, iend, xh, self.var['grid']['xsize'])
        ygc  = __add_ghostcells_cor(ygc, jgc, jend, y, self.var['grid']['ysize'])
        yhgc = __add_ghostcells_cor(yhgc, jgc, jend, yh, self.var['grid']['ysize']) 

        #Store new fields in object
        self.var['ghost'][variable_name] = {}
        self.var['ghost'][variable_name]['variable'] = sgc
        self.var['ghost'][variable_name]['z'] = z
        self.var['ghost'][variable_name]['zh'] = zh
        self.var['ghost'][variable_name]['y'] = ygc
        self.var['ghost'][variable_name]['yh'] = yhgc
        self.var['ghost'][variable_name]['x'] = xgc
        self.var['ghost'][variable_name]['xh'] = xhgc

    def __edgegrid_from_centergrid(self, coord_center, len_coord, size_coord):
        """ Define coordinates corresponding to the grid walls from coordinates corresponding to the centers of the grid. """

        dcoord = coord_center[1:] - coord_center[:-1]
        coord_edge = np.empty(len_coord+1)
        coord_edge[0] = 0
        for i in range(1,len_coord):
            coord_edge[i] = coord_center[i-1] + 0.5 * dcoord[i-1]
        coord_edge[len_coord] = float(size_coord)
        return coord_edge

    def __add_boundaries_variable(self, variable_name, bool_edge_gridcell):
        """ Add top/downstream boundaries to specified variable."""

        if bool_edge_gridcell[0]:
            org_field = self.var['output'][variable_name][0,:,:]
            self.var['output'][variable_name] = np.append(self.var['output'][variable_name], org_field[np.newaxis,:,:],axis=0)

        elif bool_edge_gridcell[1]:
            org_field = self.var['output'][variable_name][:,0,:]
            self.var['output'][variable_name] = np.append(self.var['output'][variable_name], org_field[:,np.newaxis,:],axis=1)

        elif bool_edge_gridcell[2]:
            org_field = self.var['output'][variable_name][:,:,0]
            self.var['output'][variable_name] = np.append(self.var['output'][variable_name], org_field[:,:,np.newaxis],axis=2)

    def __getitem__(self, name):
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
        self.var['output'] = {} #After the coarsegrid object is created, output can be stored here.

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

        #Calculate new grid using the information stored above, assuming the grid is uniform
        self.var['grid']['x'] , self.var['grid']['xdist']  = self.__define_coarsegrid(self.var['grid']['itot'], self.var['grid']['xsize'], center_cell = True) 
        self.var['grid']['xh'], self.var['grid']['xhdist'] = self.__define_coarsegrid(self.var['grid']['itot'], self.var['grid']['xsize'], center_cell = False)
        self.var['grid']['y'] , self.var['grid']['ydist']  = self.__define_coarsegrid(self.var['grid']['jtot'], self.var['grid']['ysize'], center_cell = True)
        self.var['grid']['yh'], self.var['grid']['yhdist'] = self.__define_coarsegrid(self.var['grid']['jtot'], self.var['grid']['ysize'], center_cell = False)
        self.var['grid']['z'] , self.var['grid']['zdist']  = self.__define_coarsegrid(self.var['grid']['ktot'], self.var['grid']['zsize'], center_cell = True)
        self.var['grid']['zh'], self.var['grid']['zhdist'] = self.__define_coarsegrid(self.var['grid']['ktot'], self.var['grid']['zsize'], center_cell = False)

    def __define_coarsegrid(self,number_gridcells,gridsize,center_cell):
        """ Manually define the coarse grid. Note that the size of the domain is determined by the finegrid object used to initialize the coarsegrid object. """        

        dist_coarsecoord = gridsize / number_gridcells
        if center_cell:
            coarsecoord, step = np.linspace(0.5*dist_coarsecoord, gridsize-0.5*dist_coarsecoord, number_gridcells, endpoint=True, retstep=True)
        else:
            coarsecoord, step = np.linspace(0, gridsize, number_gridcells+1, endpoint=True, retstep=True)
        return coarsecoord, step

       
    def __getitem__(self, name):
        if name in self.var.keys():
            return self.var[name]
        else:
            raise RuntimeError('Can\'t find variable \"{}\" in object.')

