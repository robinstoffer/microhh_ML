#Script containing helper functions for generating training data NN
#From MicroHH scripts available on GitHub (https://github.com/microhh/microhh; van Heerwaarden et al., 2017), adapted by Robin Stoffer (robin.stoffer@wur.nl)

import netCDF4 as nc
import numpy   as np
import struct  as st
import glob
import re

# NOTE: script only compatible with Python 3!
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
        f = nc.Dataset(stat_file, 'r')

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
