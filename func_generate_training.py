#Developed for Python 3!
import numpy   as np
import netCDF4 as nc
import struct as st
import matplotlib as mpl
mpl.use('Agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
import matplotlib.pyplot as plt
import scipy.interpolate
#from microhh_tools_robins import *

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


def Read_namelist(namelist_file = "moser600.ini"):
    """ Reads a MicroHH .ini file to memory
        All available variables are accessible as e.g.:
            nl = Read_namelist()    # with no name specified, it searches for a .ini file in the current dir
            itot = nl['grid']['itot']
            endtime = nl['time']['endtime']
            printing e.g. nl['grid'] provides an overview of the available variables in a group
    """

    output = {}   # Dictionary holding all the data
    with open(namelist_file) as f:
        for line in f:
            lstrip = line.strip()
            if (len(lstrip) > 0 and lstrip[0] != "#"):
                if lstrip[0] == '[' and lstrip[-1] == ']':
                    curr_group_name = lstrip[1:-1]
                    output[curr_group_name] = {}
                elif ("=" in line):
                    var_name = lstrip.split('=')[0]
                    value = _convert_value(lstrip.split('=')[1])
                    output[curr_group_name][var_name] = value
        return output


def binary3d_to_nc(variable,nx,ny,nz,starttime,endtime,sampletime,endian = 'little', savetype = 'double'):

        # Set correct dimensions for savefil
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

#Boundary condition: fine grid must have a smaller resolution than the coarse grid
#def generate_training_data(dim_new_grid,u_file_dns = "u.nc", v_file_dns = "v.nc", w_file_dns = "w.nc",p_file_dns = "p.nc" ,grid_file = "grid.{:07d}".format(0) ,namelist_file = 'moser600.ini',endian = 'little'): #Filenames should be strings. Default input corresponds to names files from MicroHH and the provided scripts
training=True
if training:
    dim_new_grid = (32,16,64)
    u_file_dns = "u.nc"
    v_file_dns = "v.nc"
    w_file_dns = "w.nc"
    p_file_dns = "p.nc"
    name_output_file = "training_data.nc"
    grid_file = "grid.{:07d}".format(0)
    namelist_file = "moser600.ini"
    endian = "little"


    nxc,nyc,nzc = dim_new_grid # Tuple with dimensions of new grid
    
    f = nc.Dataset(u_file_dns, 'r+')
    g = nc.Dataset(v_file_dns, 'r+')
    h = nc.Dataset(w_file_dns, 'r+')
    l = nc.Dataset(p_file_dns, 'r+')
    
    #u = np.array(f.variables['u'])
    #v = np.array(g.variables['v'])
    #w = np.array(h.variables['w'])
    #p = np.array(l.variables['p'])
    
    #Read settings simulation
    settings = Read_namelist()
    nx = settings['grid']['itot']
    ny = settings['grid']['jtot']
    nz = settings['grid']['ktot']
    xsize = settings['grid']['xsize']
    ysize = settings['grid']['ysize']
    zsize = settings['grid']['zsize']
    starttime = settings['time']['starttime']
    endtime = settings['time']['endtime']
    savetime = settings['time']['savetime']
   # nt = int((endtime - starttime)//savetime)
    nt = 13
    
    # Set the correct string for the endianness
    if (endian == 'little'):
        en = '<'
    elif (endian == 'big'):
        en = '>'
    else:
        raise RuntimeError("Endianness has to be little or big")
    
    
    #Get grid dimensions and distances for both the fine and coarse grid, calculate distance to the middle of the channel for coarse grid
    # Read grid properties from grid.0000000
    n   = nx*ny*nz
    fin = open(grid_file,"rb")
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
    
    dist_xc = xsize / nxc
    dist_yc = ysize / nyc
    dist_zc = zsize / nzc
    xc = np.linspace(0.5*dist_xc,xsize-0.5*dist_xc,nxc,True) #Note: does not include edges
    xhc = np.linspace(dist_xc,xsize-dist_xc,nxc-1,True)
    yc = np.linspace(0.5*dist_yc,ysize-0.5*dist_yc,nyc,True)
    yhc = np.linspace(dist_yc,ysize-dist_yc,nyc-1,True)
    zc = np.linspace(0.5*dist_zc,zsize-0.5*dist_zc,nzc,True)
    zhc = np.linspace(dist_zc,zsize-dist_zc,nzc-1,True)
    dist_midchannel = np.absolute((zsize/2)-zc)

    create_variables = True
    create_file = True

 
    #Loop over timesteps
    #for t in range(nt): #Only works correctly in this script when whole simulation is saved with a constant time interval
    for t in range(nt):

        ##Downsampling from fine DNS data to user specified coarse grid and calculation total transport momentum ##
        ###########################################################################################################

        #Define empty arrays for storage, -1 to compensate for reduced length of grid dimensions due to definitions above
        u_c = np.zeros((nzc,nyc,nxc-1),dtype=float)
        v_c = np.zeros((nzc,nyc-1,nxc),dtype=float)
        w_c = np.zeros((nzc-1,nyc,nxc),dtype=float)
        p_c = np.zeros((nzc,nyc,nxc),dtype=float)
        total_tau_xu = np.zeros((nzc,nyc,nxc-1),dtype=float)
        total_tau_yu = np.zeros((nzc,nyc-1,nxc),dtype=float)
        total_tau_zu = np.zeros((nzc-1,nyc,nxc),dtype=float)
        total_tau_xv = np.zeros((nzc,nyc,nxc-1),dtype=float)
        total_tau_yv = np.zeros((nzc,nyc-1,nxc),dtype=float)
        total_tau_zv = np.zeros((nzc-1,nyc,nxc),dtype=float)
        total_tau_xw = np.zeros((nzc,nyc,nxc-1),dtype=float)
        total_tau_yw = np.zeros((nzc,nyc-1,nxc),dtype=float)
        total_tau_zw = np.zeros((nzc-1,nyc,nxc),dtype=float)
       
        #Extract variables from netCDF file for timestep
        u = np.array(f.variables['u'][t,:,:,:])
        v = np.array(g.variables['v'][t,:,:,:])
        w = np.array(h.variables['w'][t,:,:,:])
        p = np.array(l.variables['p'][t,:,:,:])

        #sample u on coarse grid
        izc=0
        for zc_cell_middle in zc:
            zhc_cell_bottom = zc_cell_middle - 0.5*dist_zc
            zhc_cell_top = zc_cell_middle + 0.5*dist_zc

            #Find points of fine grid located just outside the coarse grid cell considered in iteration
            zh_cell_bottom = zh[zh<=zhc_cell_bottom].max()
            if izc == (nzc - 1):
                zh_cell_top = zhc_cell_top
            else:
                zh_cell_top = zh[zh>=zhc_cell_top].min()

            #Select all points inside and just outside coarse grid cell
            points_indices_z = np.where(np.logical_and(zh_cell_bottom<=zh , zh<zh_cell_top))[0]
            zh_points = zh[points_indices_z] #Note: zh_points includes the bottom boundary (zh_cell_bottom), but not the top boundary (zh_cell_top).

            #Calculate weights for zh_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
            if len(points_indices_z) == 1:
                weights_zu = np.array([1])
            else:
                weights_zu = np.ones(len(points_indices_z))
                weights_zu[0] = 1-((zhc_cell_bottom - zh_points[0])/(zh_points[1]-zh_points[0])) #Should always be between 0 and 1
                weights_zu[-1] = 1-((zh_cell_top-zhc_cell_top)/(zh_cell_top-zh_points[-1])) #Should always be between 0 and 1

            #Extract corresponding u_points
            u_points_z = u[points_indices_z,:,:]

            iyc=0
            for yc_cell_middle in yc:
                yhc_cell_bottom = yc_cell_middle - 0.5*dist_yc
                yhc_cell_top = yc_cell_middle + 0.5*dist_yc

                #Find points of fine grid located just outside the coarse grid cell considered in iteration
                yh_cell_bottom = yh[yh<=yhc_cell_bottom].max() #Note:sometimes influenced by small rounding errors
                if iyc == (nyc -1):
                    yh_cell_top = yhc_cell_top
                else:
                    yh_cell_top = yh[yh>=yhc_cell_top].min()

                #Select all points inside and jdust outside coarse grid cell
                points_indices_y = np.where(np.logical_and(yh_cell_bottom<=yh , yh<yh_cell_top))[0]
                yh_points = yh[points_indices_y] #Note: yh_points includes the bottom boundary (yh_cell_bottom), but not the top boundary (yh_cell_top).

                #Calculate weights for yh_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
                if len(points_indices_y) == 1:
                    weights_yu = np.array([1])
                else:
                    weights_yu = np.ones(len(points_indices_y))
                    weights_yu[0] = 1-(yhc_cell_bottom - yh_points[0])/(yh_points[1]-yh_points[0]) #Should always be between 0 and 1
                    weights_yu[-1] = 1-((yh_cell_top-yhc_cell_top)/(yh_cell_top-yh_points[-1])) #Should always be between 0 and 1

                #Extract corresponding u_points
                u_points_zy = u_points_z[:,points_indices_y,:]

                ixc=0
                for xhc_cell_middle in xhc:
                    xc_cell_bottom = xhc_cell_middle - 0.5*dist_xc
                    xc_cell_top = xhc_cell_middle + 0.5*dist_xc

                    #Find points of fine grid located just outside the coarse grid cell considered in iteration
                    x_cell_bottom = x[x<=xc_cell_bottom].max()
                    x_cell_top = x[x>=xc_cell_top].min()

                    #Select all points inside and just outside coarse grid cell
                    points_indices_x = np.where(np.logical_and(x_cell_bottom<x , x<=x_cell_top))[0]
                    x_points = x[points_indices_x] #Note: x_points includes the bottom boundary (x_cell_bottom), but not the top boundary (x_cell_top).

                    #Calculate weights for x_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
                    if len(points_indices_x) == 1:
                        weights_xu = np.array([1])
                    else:
                        weights_xu = np.ones(len(points_indices_x))
                        weights_xu[0] = 1-(xc_cell_bottom - x_cell_bottom)/(x_points[0]-x_cell_bottom) #Should always be between 0 and 1
                        weights_xu[-1] = 1-((x_cell_top - xc_cell_top)/(x_cell_top-x_points[-2])) #Should always be between 0 and 1

                    #Calculate representative u on coarse grid from fine grid
                    u_points_zyx = u_points_zy[:,:,points_indices_x]
                    #weights_xu_points = np.tile(weights_xu,(len(weights_zu),len(weights_yu),1))
                    #weights_yu_points =  np.tile(weights_yu,(len(weights_zu),1,len(weights_xu)))
                    #weights_zu_points = np.tile(weights_zu,(1,len(weights_yu),len(weights_xu)))
                    #weights_u = weights_zu_points*weights_yu_points*weights_xu_points
                    weights_u =  weights_xu[np.newaxis,np.newaxis,:]*weights_yu[np.newaxis,:,np.newaxis]*weights_zu[:,np.newaxis,np.newaxis]
                    u_c[izc,iyc,ixc] = np.average(u_points_zyx,weights=weights_u)

                    ixc+=1
                iyc+=1
            izc+=1

        #Calculate TOTAL transport from fine grid in user-specified coarse grid cell. As a first step, fine grid-velocities interpolated to walls coarse grid cell.
        zc_int = np.ravel(np.broadcast_to(zc[:,np.newaxis,np.newaxis],(len(zc),len(yc),len(xhc))))
        yc_int = np.ravel(np.broadcast_to(yc[np.newaxis,:,np.newaxis],(len(zc),len(yc),len(xhc))))
        xhc_int = np.ravel(np.broadcast_to(xhc[np.newaxis,np.newaxis,:],(len(zc),len(yc),len(xhc))))

        #zh_int = np.broadcast_to(zh[:,np.newaxis,np.newaxis],(len(zh),len(y),len(x)))

        #z_int = np.ravel(np.broadcast_to(z[:,np.newaxis,np.newaxis],(len(z),len(y),len(xh))))
        #y_int = np.ravel(np.broadcast_to(y[np.newaxis,:,np.newaxis],(len(z),len(y),len(xh))))
        #xh_int = np.ravel(np.broadcast_to(xh[np.newaxis,np.newaxis,:],(len(z),len(y),len(xh))))
        #u_1d = np.ravel(u)
        #wifjwepoifjef
        #u_int = scipy.interpolate.griddata((z_int,y_int,xh_int),u_1d,(zc_int,yc_int,xhc_int),method='linear')

        u_int = np.reshape(scipy.interpolate.RegularGridInterpolator((z,y,xh),u,method='linear',bounds_error=False,fill_value=None)((zc_int,yc_int,xhc_int)),(len(zc),len(yc),len(xhc)))
        v_int = np.reshape(scipy.interpolate.RegularGridInterpolator((z,yh,x),v,method='linear',bounds_error=False,fill_value=None)((zc_int,yc_int,xhc_int)),(len(zc),len(yc),len(xhc)))
        w_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zh,y,x),w,method='linear',bounds_error=False,fill_value=None)((zc_int,yc_int,xhc_int)),(len(zc),len(yc),len(xhc)))
        #v_int = scipy.interpolate.griddata((z,yh,x),v,(zc,yc,xhc),method='linear')
        #w_int = scipy.interpolate.griddata((zh,y,x),w,(zc,yc,xhc),method='linear')

        total_tau_xu[:,:,:] = u_int ** 2
        total_tau_xv[:,:,:] = u_int * v_int
        total_tau_xw[:,:,:] = u_int * w_int

        ##Sample v on coarse grid
        izc=0
        for zc_cell_middle in zc:
            zhc_cell_bottom = zc_cell_middle - 0.5*dist_zc
            zhc_cell_top = zc_cell_middle + 0.5*dist_zc

            #Find points of fine grid located just outside the coarse grid cell considered in iteration
            zh_cell_bottom = zh[zh<=zhc_cell_bottom].max()
            if izc == (nzc - 1):
                zh_cell_top = zhc_cell_top
            else:
                zh_cell_top = zh[zh>=zhc_cell_top].min()

            #Select all points inside and just outside coarse grid cell
            points_indices_z = np.where(np.logical_and(zh_cell_bottom<=zh , zh<zh_cell_top))[0]
            zh_points = zh[points_indices_z] #Note: zh_points includes the bottom boundary (zh_cell_bottom), but not the top boundary (zh_cell_top).

            #Calculate weights for zh_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
            if len(points_indices_z) == 1:
                weights_zv = np.array([1])
            else:
                weights_zv = np.ones(len(points_indices_z))
                weights_zv[0] = 1-((zhc_cell_bottom - zh_points[0])/(zh_points[1]-zh_points[0])) #Should always be between 0 and 1
                weights_zv[-1] = 1-((zh_cell_top-zhc_cell_top)/(zh_cell_top-zh_points[-1])) #Should always be between 0 and 1

            #Extract corresponding v_points
            v_points_z = v[points_indices_z,:,:]

            iyc=0
            for yhc_cell_middle in yhc:
                yc_cell_bottom = yhc_cell_middle - 0.5*dist_yc
                yc_cell_top = yhc_cell_middle + 0.5*dist_yc

                #Find points of fine grid located just outside the coarse grid cell considered in iteration
                y_cell_bottom = y[y<=yc_cell_bottom].max() #Note:sometimes influenced by small rounding errors
                y_cell_top = y[y>=yc_cell_top].min()

                #Select all points inside and just outside coarse grid cell
                points_indices_y = np.where(np.logical_and(y_cell_bottom<y , y<=y_cell_top))[0]
                y_points = y[points_indices_y] #Note: y_points includes the bottom boundary (y_cell_bottom), but not the top boundary (y_cell_top).

                #Calculate weights for y_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
                if len(points_indices_y) == 1:
                    weights_yv = np.array([1])
                else:
                    weights_yv = np.ones(len(points_indices_y))
                    weights_yv[0] = 1-(yc_cell_bottom - y_cell_bottom)/(y_points[0]-y_cell_bottom) #Should always be between 0 and 1
                    weights_yv[-1] = 1-((y_cell_top - yc_cell_top)/(y_cell_top-y_points[-2])) #Should always be between 0 and 1


                #Extract corresponding v_points
                v_points_zy = v_points_z[:,points_indices_y,:]

                ixc=0
                for xc_cell_middle in xc:
                    xhc_cell_bottom = xc_cell_middle - 0.5*dist_xc
                    xhc_cell_top = xc_cell_middle + 0.5*dist_xc

                    #Find points of fine grid located just outside the coarse grid cell considered in iteration
                    xh_cell_bottom = xh[xh<=xhc_cell_bottom].max()
                    if ixc == (nxc -1):
                        xh_cell_top = xhc_cell_top
                    else:
                        xh_cell_top = xh[xh>=xhc_cell_top].min()

                    #Select all points inside and just outside coarse grid cell
                    points_indices_x = np.where(np.logical_and(xh_cell_bottom<=xh , xh<xh_cell_top))[0]
                    xh_points = xh[points_indices_x] #Note: xh_points includes the bottom boundary (xh_cell_bottom), but not the top boundary (xh_cell_top).

                    #Calculate weights for xh_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
                    if len(points_indices_x) == 1:
                        weights_xv = np.array([1])
                    else:
                        weights_xv = np.ones(len(points_indices_x))
                        weights_xv[0] = 1-(xhc_cell_bottom - xh_points[0])/(xh_points[1]-xh_points[0]) #Should always be between 0 and 1
                        weights_xv[-1] = 1-((xh_cell_top - xhc_cell_top)/(xh_cell_top-xh_points[-1])) #Should always be between 0 and 1

                    #Calculate representative v on coarse grid from fine grid
                    v_points_zyx = v_points_zy[:,:,points_indices_x]
                    #weights_xu_points = np.tile(weights_xu,(len(weights_zu),len(weights_yu),1))
                    #weights_yu_points =  np.tile(weights_yu,(len(weights_zu),1,len(weights_xu)))
                    #weights_zu_points = np.tile(weights_zu,(1,len(weights_yu),len(weights_xu)))
                    #weights_u = weights_zu_points*weights_yu_points*weights_xu_points
                    weights_v =  weights_xv[np.newaxis,np.newaxis,:]*weights_yv[np.newaxis,:,np.newaxis]*weights_zv[:,np.newaxis,np.newaxis]
                    v_c[izc,iyc,ixc] = np.average(v_points_zyx,weights=weights_v)

                    ixc+=1
                iyc+=1
            izc+=1

        #Calculate TOTAL transport from fine grid in user-specified coarse grid cell. As a first step, fine grid-velocities interpolated to walls coarse grid cell.
        zc_int = np.ravel(np.broadcast_to(zc[:,np.newaxis,np.newaxis],(len(zc),len(yhc),len(xc))))
        yhc_int = np.ravel(np.broadcast_to(yhc[np.newaxis,:,np.newaxis],(len(zc),len(yhc),len(xc))))
        xc_int = np.ravel(np.broadcast_to(xc[np.newaxis,np.newaxis,:],(len(zc),len(yhc),len(xc))))

        u_int = np.reshape(scipy.interpolate.RegularGridInterpolator((z,y,xh),u,method='linear',bounds_error=False,fill_value=None)((zc_int,yhc_int,xc_int)),(len(zc),len(yhc),len(xc)))
        v_int = np.reshape(scipy.interpolate.RegularGridInterpolator((z,yh,x),v,method='linear',bounds_error=False,fill_value=None)((zc_int,yhc_int,xc_int)),(len(zc),len(yhc),len(xc)))
        w_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zh,y,x),w,method='linear',bounds_error=False,fill_value=None)((zc_int,yhc_int,xc_int)),(len(zc),len(yhc),len(xc)))

        #u_int = scipy.interpolate.griddata((z,y,xh),u,(zc,yhc,xc),method='linear')
        #v_int = scipy.interpolate.griddata((z,yh,x),v,(zc,yhc,xc),method='linear')
        #w_int = scipy.interpolate.griddata((zh,y,x),w,(zc,yhc,xc),method='linear')

        total_tau_yu[:,:,:] = v_int * u_int
        total_tau_yv[:,:,:] = v_int ** 2
        total_tau_yw[:,:,:] = v_int * w_int

        ##Sample w on coarse grid
        izc=0
        for zhc_cell_middle in zhc:
            zc_cell_bottom = zhc_cell_middle - 0.5*dist_zc
            zc_cell_top = zhc_cell_middle + 0.5*dist_zc

            #Find points of fine grid located just outside the coarse grid cell considered in iteration
            z_cell_bottom = z[z<=zc_cell_bottom].max()
            z_cell_top = z[z>=zc_cell_top].min()

            #Select all points inside and just outside coarse grid cell
            points_indices_z = np.where(np.logical_and(z_cell_bottom<z , z<=z_cell_top))[0]
            z_points = z[points_indices_z] #Note: z_points includes the bottom boundary (z_cell_bottom), but not the top boundary (z_cell_top).

            #Calculate weights for z_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
            if len(points_indices_z) == 1:
                weights_zw = np.array([1])
            else:
                weights_zw = np.ones(len(points_indices_z))
                weights_zw[0] = 1-(zc_cell_bottom - z_cell_bottom)/(z_points[0]-z_cell_bottom) #Should always be between 0 and 1
                weights_zw[-1] = 1-((z_cell_top - zc_cell_top)/(z_cell_top-z_points[-2])) #Should always be between 0 and 1

            #Extract corresponding w_points
            w_points_z = w[points_indices_z,:,:]

            iyc=0


            for yc_cell_middle in yc:
                yhc_cell_bottom = yc_cell_middle - 0.5*dist_yc
                yhc_cell_top = yc_cell_middle + 0.5*dist_yc

                #Find points of fine grid located just outside the coarse grid cell considered in iteration
                yh_cell_bottom = yh[yh<=yhc_cell_bottom].max() #Note:sometimes influenced by small rounding errors
                if iyc == (nyc -1):
                    yh_cell_top = yhc_cell_top
                else:
                    yh_cell_top = yh[yh>=yhc_cell_top].min()

                #Select all points inside and just outside coarse grid cell
                points_indices_y = np.where(np.logical_and(yh_cell_bottom<=yh , yh<yh_cell_top))[0]
                yh_points = yh[points_indices_y] #Note: yh_points includes the bottom boundary (yh_cell_bottom), but not the top boundary (yh_cell_top).

                #Calculate weights for yh_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
                if len(points_indices_y) == 1:
                    weights_yw = np.array([1])
                else:
                    weights_yw = np.ones(len(points_indices_y))
                    weights_yw[0] = 1-(yhc_cell_bottom - yh_points[0])/(yh_points[1]-yh_points[0]) #Should always be between 0 and 1
                    weights_yw[-1] = 1-((yh_cell_top-yhc_cell_top)/(yh_cell_top-yh_points[-1])) #Should always be between 0 and 1

                #Extract corresponding w_points
                w_points_zy = w_points_z[:,points_indices_y,:]

                ixc=0

                for xc_cell_middle in xc:
                    xhc_cell_bottom = xc_cell_middle - 0.5*dist_xc
                    xhc_cell_top = xc_cell_middle + 0.5*dist_xc

                    #Find points of fine grid located just outside the coarse grid cell considered in iteration
                    xh_cell_bottom = xh[xh<=xhc_cell_bottom].max()
                    if ixc == (nxc -1):
                        xh_cell_top = xhc_cell_top
                    else:
                        xh_cell_top = xh[xh>=xhc_cell_top].min()

                    #Select all points inside and just outside coarse grid cell
                    points_indices_x = np.where(np.logical_and(xh_cell_bottom<=xh , xh<xh_cell_top))[0]
                    xh_points = xh[points_indices_x] #Note: xh_points includes the bottom boundary (xh_cell_bottom), but not the top boundary (xh_cell_top).

                    #Calculate weights for xh_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
                    if len(points_indices_x) == 1:
                        weights_xw = np.array([1])
                    else:
                        weights_xw = np.ones(len(points_indices_x))
                        weights_xw[0] = 1-(xhc_cell_bottom - xh_points[0])/(xh_points[1]-xh_points[0]) #Should always be between 0 and 1
                        weights_xw[-1] = 1-((xh_cell_top - xhc_cell_top)/(xh_cell_top-xh_points[-1])) #Should always be between 0 and 1

                    #Calculate representative w on coarse grid from fine grid
                    w_points_zyx = w_points_zy[:,:,points_indices_x]
                    #weights_xu_points = np.tile(weights_xu,(len(weights_zu),len(weights_yu),1))
                    #weights_yu_points =  np.tile(weights_yu,(len(weights_zu),1,len(weights_xu)))
                    #weights_zu_points = np.tile(weights_zu,(1,len(weights_yu),len(weights_xu)))
                    #weights_u = weights_zu_points*weights_yu_points*weights_xu_points
                    weights_w =  weights_xw[np.newaxis,np.newaxis,:]*weights_yw[np.newaxis,:,np.newaxis]*weights_zw[:,np.newaxis,np.newaxis]
                    w_c[izc,iyc,ixc] = np.average(w_points_zyx,weights=weights_w)

                    ixc+=1
                iyc+=1
            izc+=1

        #Calculate TOTAL transport from fine grid in user-specified coarse grid cell. As a first step, fine grid-velocities interpolated to walls coarse grid cell.
        zhc_int = np.ravel(np.broadcast_to(zhc[:,np.newaxis,np.newaxis],(len(zhc),len(yc),len(xc))))
        yc_int = np.ravel(np.broadcast_to(yc[np.newaxis,:,np.newaxis],(len(zhc),len(yc),len(xc))))
        xc_int = np.ravel(np.broadcast_to(xc[np.newaxis,np.newaxis,:],(len(zhc),len(yc),len(xc))))

        u_int = np.reshape(scipy.interpolate.RegularGridInterpolator((z,y,xh),u,method='linear',bounds_error=False,fill_value=None)((zhc_int,yc_int,xc_int)),(len(zhc),len(yc),len(xc)))
        v_int = np.reshape(scipy.interpolate.RegularGridInterpolator((z,yh,x),v,method='linear',bounds_error=False,fill_value=None)((zhc_int,yc_int,xc_int)),(len(zhc),len(yc),len(xc)))
        w_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zh,y,x),w,method='linear',bounds_error=False,fill_value=None)((zhc_int,yc_int,xc_int)),(len(zhc),len(yc),len(xc)))

       # u_int = scipy.interpolate.griddata((z,y,xh),u,(zhc,yc,xc),method='linear')
       # v_int = scipy.interpolate.griddata((z,yh,x),v,(zhc,yc,xc),method='linear')
       # w_int = scipy.interpolate.griddata((zh,y,x),w,(zhc,yc,xc),method='linear')

        total_tau_zu[:,:,:] = w_int * u_int
        total_tau_zv[:,:,:] = w_int * v_int
        total_tau_zw[:,:,:] = w_int **2

        ##Sample p on coarse grid
        izc=0
        for zc_cell_middle in zc:
            zhc_cell_bottom = zc_cell_middle - 0.5*dist_zc
            zhc_cell_top = zc_cell_middle + 0.5*dist_zc

            #Find points of fine grid located just outside the coarse grid cell considered in iteration
            zh_cell_bottom = zh[zh<=zhc_cell_bottom].max()
            if izc == (nzc - 1):
                zh_cell_top = zhc_cell_top
            else:
                zh_cell_top = zh[zh>=zhc_cell_top].min()

            #Select all points inside and just outside coarse grid cell
            points_indices_z = np.where(np.logical_and(zh_cell_bottom<=zh , zh<zh_cell_top))[0]
            zh_points = zh[points_indices_z] #Note: zh_points includes the bottom boundary (zh_cell_bottom), but not the top boundary (zh_cell_top).

            #Calculate weights for zh_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
            if len(points_indices_z) == 1:
                weights_zp = np.array([1])
            else:
                weights_zp = np.ones(len(points_indices_z))
                weights_zp[0] = 1-((zhc_cell_bottom - zh_points[0])/(zh_points[1]-zh_points[0])) #Should always be between 0 and 1
                weights_zp[-1] = 1-((zh_cell_top-zhc_cell_top)/(zh_cell_top-zh_points[-1])) #Should always be between 0 and 1

            #Extract corresponding p_points
            p_points_z = p[points_indices_z,:,:]

            iyc=0

            for yc_cell_middle in yc:
                yhc_cell_bottom = yc_cell_middle - 0.5*dist_yc
                yhc_cell_top = yc_cell_middle + 0.5*dist_yc

                #Find points of fine grid located just outside the coarse grid cell considered in iteration
                yh_cell_bottom = yh[yh<=yhc_cell_bottom].max() #Note:sometimes influenced by small rounding errors
                if iyc == (nyc -1):
                    yh_cell_top = yhc_cell_top
                else:
                    yh_cell_top = yh[yh>=yhc_cell_top].min()

                #Select all points inside and just outside coarse grid cell
                points_indices_y = np.where(np.logical_and(yh_cell_bottom<=yh , yh<yh_cell_top))[0]
                yh_points = yh[points_indices_y] #Note: yh_points includes the bottom boundary (yh_cell_bottom), but not the top boundary (yh_cell_top).

                #Calculate weights for yh_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
                if len(points_indices_y) == 1:
                    weights_yp = np.array([1])
                else:
                    weights_yp = np.ones(len(points_indices_y))
                    weights_yp[0] = 1-(yhc_cell_bottom - yh_points[0])/(yh_points[1]-yh_points[0]) #Should always be between 0 and 1
                    weights_yp[-1] = 1-((yh_cell_top-yhc_cell_top)/(yh_cell_top-yh_points[-1])) #Should always be between 0 and 1

                #Extract corresponding p_points
                p_points_zy = p_points_z[:,points_indices_y,:]

                ixc=0

                for xc_cell_middle in xc:
                    xhc_cell_bottom = xc_cell_middle - 0.5*dist_xc
                    xhc_cell_top = xc_cell_middle + 0.5*dist_xc

                    #Find points of fine grid located just outside the coarse grid cell considered in iteration
                    xh_cell_bottom = xh[xh<=xhc_cell_bottom].max()
                    if ixc == (nxc -1):
                        xh_cell_top = xhc_cell_top
                    else:
                        xh_cell_top = xh[xh>=xhc_cell_top].min()

                    #Select all points inside and just outside coarse grid cell
                    points_indices_x = np.where(np.logical_and(xh_cell_bottom<=xh , xh<xh_cell_top))[0]
                    xh_points = xh[points_indices_x] #Note: xh_points includes the bottom boundary (xh_cell_bottom), but not the top boundary (xh_cell_top).

                    #Calculate weights for xh_points in calculation representative velocity. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
                    if len(points_indices_x) == 1:
                        weights_xp = np.array([1])
                    else:
                        weights_xp = np.ones(len(points_indices_x))
                        weights_xp[0] = 1-(xhc_cell_bottom - xh_points[0])/(xh_points[1]-xh_points[0]) #Should always be between 0 and 1
                        weights_xp[-1] = 1-((xh_cell_top - xhc_cell_top)/(xh_cell_top-xh_points[-1])) #Should always be between 0 and 1

                    #Calculate representative p on coarse grid from fine grid
                    p_points_zyx = p_points_zy[:,:,points_indices_x]
                    #weights_xu_points = np.tile(weights_xu,(len(weights_zu),len(weights_yu),1))
                    #weights_yu_points =  np.tile(weights_yu,(len(weights_zu),1,len(weights_xu)))
                    #weights_zu_points = np.tile(weights_zu,(1,len(weights_yu),len(weights_xu)))
                    #weights_u = weights_zu_points*weights_yu_points*weights_xu_points
                    weights_p =  weights_xp[np.newaxis,np.newaxis,:]*weights_yp[np.newaxis,:,np.newaxis]*weights_zp[:,np.newaxis,np.newaxis]
                    p_c[izc,iyc,ixc] = np.average(p_points_zyx,weights=weights_p)

                    ixc+=1
                iyc+=1
            izc+=1

        ##Calculate resolved and unresolved transport user specified coarse grid ##
        ###########################################################################

        #Define empty variables for storage

        res_tau_xu = np.zeros((nzc,nyc,nxc-1),dtype=float)
        res_tau_yu = np.zeros((nzc,nyc-1,nxc),dtype=float)
        res_tau_zu = np.zeros((nzc-1,nyc,nxc),dtype=float)
        res_tau_xv = np.zeros((nzc,nyc,nxc-1),dtype=float)
        res_tau_yv = np.zeros((nzc,nyc-1,nxc),dtype=float)
        res_tau_zv = np.zeros((nzc-1,nyc,nxc),dtype=float)
        res_tau_xw = np.zeros((nzc,nyc,nxc-1),dtype=float)
        res_tau_yw = np.zeros((nzc,nyc-1,nxc),dtype=float)
        res_tau_zw = np.zeros((nzc-1,nyc,nxc),dtype=float)

        unres_tau_xu = np.zeros((nzc,nyc,nxc-1),dtype=float)
        unres_tau_yu = np.zeros((nzc,nyc-1,nxc),dtype=float)
        unres_tau_zu = np.zeros((nzc-1,nyc,nxc),dtype=float)
        unres_tau_xv = np.zeros((nzc,nyc,nxc-1),dtype=float)
        unres_tau_yv = np.zeros((nzc,nyc-1,nxc),dtype=float)
        unres_tau_zv = np.zeros((nzc-1,nyc,nxc),dtype=float)
        unres_tau_xw = np.zeros((nzc,nyc,nxc-1),dtype=float)
        unres_tau_yw = np.zeros((nzc,nyc-1,nxc),dtype=float)
        unres_tau_zw = np.zeros((nzc-1,nyc,nxc),dtype=float)

        #Calculate RESOLVED and UNRESOLVED transport of u-momentum for user-specified coarse grid. As a first step, the coarse grid-velocities are interpolated to the walls of the coarse grid cell.
        zc_int = np.ravel(np.broadcast_to(zc[:,np.newaxis,np.newaxis],(len(zc),len(yc),len(xhc))))
        yc_int = np.ravel(np.broadcast_to(yc[np.newaxis,:,np.newaxis],(len(zc),len(yc),len(xhc))))
        xhc_int = np.ravel(np.broadcast_to(xhc[np.newaxis,np.newaxis,:],(len(zc),len(yc),len(xhc))))

        u_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zc,yc,xhc),u_c[:,:,:],method='linear',bounds_error=False,fill_value=None)((zc_int,yc_int,xhc_int)),(len(zc),len(yc),len(xhc)))
        v_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zc,yhc,xc),v_c[:,:,:],method='linear',bounds_error=False,fill_value=None)((zc_int,yc_int,xhc_int)),(len(zc),len(yc),len(xhc)))
        w_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zhc,yc,xc),w_c[:,:,:],method='linear',bounds_error=False,fill_value=None)((zc_int,yc_int,xhc_int)),(len(zc),len(yc),len(xhc)))

        #u_int = scipy.interpolate.griddata((zc,yc,xhc),u_c[t,:,:,:],(zc,yc,xhc),method='linear')
        #v_int = scipy.interpolate.griddata((zc,yhc,xc),v_c[t,:,:,:],(zc,yc,xhc),method='linear')
        #w_int = scipy.interpolate.griddata((zhc,yc,xc),w_c[t,:,:,:],(zc,yc,xhc),method='linear')

        res_tau_xu[:,:,:] = u_int **2
        res_tau_xv[:,:,:] = u_int * v_int
        res_tau_xw[:,:,:] = u_int * w_int

        unres_tau_xu[:,:,:] = total_tau_xu[:,:,:]-res_tau_xu[:,:,:]
        unres_tau_xv[:,:,:] = total_tau_xv[:,:,:]-res_tau_xv[:,:,:]
        unres_tau_xw[:,:,:] = total_tau_xw[:,:,:]-res_tau_xw[:,:,:]

        #Calculate RESOLVED and UNRESOLVED transport of v-momentum for user-specified coarse grid. As a first step, the coarse grid-velocities are interpolated to the walls of the coarse grid cell.
        zc_int = np.ravel(np.broadcast_to(zc[:,np.newaxis,np.newaxis],(len(zc),len(yhc),len(xc))))
        yhc_int = np.ravel(np.broadcast_to(yhc[np.newaxis,:,np.newaxis],(len(zc),len(yhc),len(xc))))
        xc_int = np.ravel(np.broadcast_to(xc[np.newaxis,np.newaxis,:],(len(zc),len(yhc),len(xc))))

        u_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zc,yc,xhc),u_c[:,:,:],method='linear',bounds_error=False,fill_value=None)((zc_int,yhc_int,xc_int)),(len(zc),len(yhc),len(xc)))
        v_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zc,yhc,xc),v_c[:,:,:],method='linear',bounds_error=False,fill_value=None)((zc_int,yhc_int,xc_int)),(len(zc),len(yhc),len(xc)))
        w_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zhc,yc,xc),w_c[:,:,:],method='linear',bounds_error=False,fill_value=None)((zc_int,yhc_int,xc_int)),(len(zc),len(yhc),len(xc)))

        #u_int = scipy.interpolate.griddata((zc,yc,xhc),u_c[t,:,:,:],(zc,yhc,xc),method='linear')
        #v_int = scipy.interpolate.griddata((zc,yhc,xc),v_c[t,:,:,:],(zc,yhc,xc),method='linear')
        #w_int = scipy.interpolate.griddata((zhc,yc,xc),w_c[t,:,:,:],(zc,yhc,xc),method='linear')

        res_tau_yu[:,:,:] = v_int * u_int
        res_tau_yv[:,:,:] = v_int **2
        res_tau_yw[:,:,:] = v_int * w_int

        unres_tau_yu[:,:,:] = total_tau_yu[:,:,:]-res_tau_yu[:,:,:]
        unres_tau_yv[:,:,:] = total_tau_yv[:,:,:]-res_tau_yv[:,:,:]
        unres_tau_yw[:,:,:] = total_tau_yw[:,:,:]-res_tau_yw[:,:,:]

        #Calculate RESOLVED and UNRESOLVED transport of w-momentum for user-specified coarse grid. As a first step, the coarse grid-velocities are interpolated to the walls of the coarse grid cell.
        zhc_int = np.ravel(np.broadcast_to(zhc[:,np.newaxis,np.newaxis],(len(zhc),len(yc),len(xc))))
        yc_int = np.ravel(np.broadcast_to(yc[np.newaxis,:,np.newaxis],(len(zhc),len(yc),len(xc))))
        xc_int = np.ravel(np.broadcast_to(xc[np.newaxis,np.newaxis,:],(len(zhc),len(yc),len(xc))))

        u_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zc,yc,xhc),u_c[:,:,:],method='linear',bounds_error=False,fill_value=None)((zhc_int,yc_int,xc_int)),(len(zhc),len(yc),len(xc)))
        v_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zc,yhc,xc),v_c[:,:,:],method='linear',bounds_error=False,fill_value=None)((zhc_int,yc_int,xc_int)),(len(zhc),len(yc),len(xc)))
        w_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zhc,yc,xc),w_c[:,:,:],method='linear',bounds_error=False,fill_value=None)((zhc_int,yc_int,xc_int)),(len(zhc),len(yc),len(xc)))

        #u_int = scipy.interpolate.griddata((zc,yc,xhc),u_c[t,:,:,:],(zhc,yc,xc),method='linear')
        #v_int = scipy.interpolate.griddata((zc,yhc,xc),v_c[t,:,:,:],(zhc,yc,xc),method='linear')
        #w_int = scipy.interpolate.griddata((zhc,yc,xc),w_c[t,:,:,:],(zhc,yc,xc),method='linear')

        res_tau_zu[:,:,:] = w_int * u_int
        res_tau_zv[:,:,:] = w_int * v_int
        res_tau_zw[:,:,:] = w_int **2

        unres_tau_zu[:,:,:] = total_tau_zu[:,:,:]-res_tau_zu[:,:,:]
        unres_tau_zv[:,:,:] = total_tau_zv[:,:,:]-res_tau_zv[:,:,:]
        unres_tau_zw[:,:,:] = total_tau_zw[:,:,:]-res_tau_zv[:,:,:]

        ##Store flow fields coarse grid and unresolved transport ##
        ###########################################################

        #Create/open netCDF file
        if create_file:
            a = nc.Dataset(name_output_file, 'w')
            create_file = False
        else:
            a = nc.Dataset(name_output_file, 'r+')

        if create_variables:
            ##Extract time variable from u-file (should be identical to the one from v-,w-,or p-file)
            #time = np.array(f.variables['time'])

            #Create new dimensions
            dim_time = a.createDimension("time",None)
            dim_xh = a.createDimension("xhc",xhc.shape[0])
            dim_x = a.createDimension("xc",xc.shape[0])
            dim_yh = a.createDimension("yhc",yhc.shape[0])
            dim_y = a.createDimension("yc",yc.shape[0])
            dim_zh = a.createDimension("zhc",zhc.shape[0])
            dim_z = a.createDimension("zc",zc.shape[0])

            #Create coordinate variables and store values
            var_xhc = a.createVariable("xhc","f8",("xhc",))
            var_xc = a.createVariable("xc","f8",("xc",))
            var_yhc = a.createVariable("yhc","f8",("yhc",))
            var_yc = a.createVariable("yc","f8",("yc",))
            var_zhc = a.createVariable("zhc","f8",("zhc",))
            var_zc = a.createVariable("zc","f8",("zc",))
            var_dist_midchannel = a.createVariable("dist_midchannel","f8",("zc",))

            var_xhc[:] = xhc[:]
            var_xc[:] = xc[:]
            var_yhc[:] = yhc[:]
            var_yc[:] = yc[:]
            var_zhc[:] = zhc[:]
            var_zc[:] = zc[:]
            var_dist_midchannel[:] = dist_midchannel[:]

            #Create variables for coarse fields
            var_uc = a.createVariable("uc","f8",("time","zc","yc","xhc"))
            var_vc = a.createVariable("vc","f8",("time","zc","yhc","xc"))
            var_wc = a.createVariable("wc","f8",("time","zhc","yc","xc"))
            var_pc = a.createVariable("pc","f8",("time","zc","yc","xc"))

            var_total_tau_xu = a.createVariable("total_tau_xu","f8",("time","zc","yc","xhc"))
            var_res_tau_xu = a.createVariable("res_tau_xu","f8",("time","zc","yc","xhc"))
            var_unres_tau_xu = a.createVariable("unres_tau_xu","f8",("time","zc","yc","xhc"))

            var_total_tau_xv = a.createVariable("total_tau_xv","f8",("time","zc","yc","xhc"))
            var_res_tau_xv = a.createVariable("res_tau_xv","f8",("time","zc","yc","xhc"))
            var_unres_tau_xv = a.createVariable("unres_tau_xv","f8",("time","zc","yc","xhc"))

            var_total_tau_xw = a.createVariable("total_tau_xw","f8",("time","zc","yc","xhc"))
            var_res_tau_xw = a.createVariable("res_tau_xw","f8",("time","zc","yc","xhc"))
            var_unres_tau_xw = a.createVariable("unres_tau_xw","f8",("time","zc","yc","xhc"))

            var_total_tau_yu = a.createVariable("total_tau_yu","f8",("time","zc","yhc","xc"))
            var_res_tau_yu = a.createVariable("res_tau_yu","f8",("time","zc","yhc","xc"))
            var_unres_tau_yu = a.createVariable("unres_tau_yu","f8",("time","zc","yhc","xc"))

            var_total_tau_yv = a.createVariable("total_tau_yv","f8",("time","zc","yhc","xc"))
            var_res_tau_yv = a.createVariable("res_tau_yv","f8",("time","zc","yhc","xc"))
            var_unres_tau_yv = a.createVariable("unres_tau_yv","f8",("time","zc","yhc","xc"))

            var_total_tau_yw = a.createVariable("total_tau_yw","f8",("time","zc","yhc","xc"))
            var_res_tau_yw = a.createVariable("res_tau_yw","f8",("time","zc","yhc","xc"))
            var_unres_tau_yw = a.createVariable("unres_tau_yw","f8",("time","zc","yhc","xc"))

            var_total_tau_zu = a.createVariable("total_tau_zu","f8",("time","zhc","yc","xc"))
            var_res_tau_zu = a.createVariable("res_tau_zu","f8",("time","zhc","yc","xc"))
            var_unres_tau_zu = a.createVariable("unres_tau_zu","f8",("time","zhc","yc","xc"))

            var_total_tau_zv = a.createVariable("total_tau_zv","f8",("time","zhc","yc","xc"))
            var_res_tau_zv = a.createVariable("res_tau_zv","f8",("time","zhc","yc","xc"))
            var_unres_tau_zv = a.createVariable("unres_tau_zv","f8",("time","zhc","yc","xc"))

            var_total_tau_zw = a.createVariable("total_tau_zw","f8",("time","zhc","yc","xc"))
            var_res_tau_zw = a.createVariable("res_tau_zw","f8",("time","zhc","yc","xc"))
            var_unres_tau_zw = a.createVariable("unres_tau_zw","f8",("time","zhc","yc","xc"))

        create_variables = False #Make sure variables are only created once.

        #Store values coarse fields
        var_uc[t,:,:,:] = u_c[:,:,:]
        var_vc[t,:,:,:] = v_c[:,:,:]
        var_wc[t,:,:,:] = w_c[:,:,:]
        var_pc[t,:,:,:] = p_c[:,:,:]

        var_total_tau_xu[t,:,:,:] = total_tau_xu[:,:,:]
        var_res_tau_xu[t,:,:,:] = res_tau_xu[:,:,:]
        var_unres_tau_xu[t,:,:,:] = unres_tau_xu[:,:,:]

        var_total_tau_xv[t,:,:,:] = total_tau_xv[:,:,:]
        var_res_tau_xv[t,:,:,:] = res_tau_xv[:,:,:]
        var_unres_tau_xv[t,:,:,:] = unres_tau_xv[:,:,:]

        var_total_tau_xw[t,:,:,:] = total_tau_xw[:,:,:]
        var_res_tau_xw[t,:,:,:] = res_tau_xw[:,:,:]
        var_unres_tau_xw[t,:,:,:] = unres_tau_xw[:,:,:]

        var_total_tau_yu[t,:,:,:] = total_tau_yu[:,:,:]
        var_res_tau_yu[t,:,:,:] = res_tau_yu[:,:,:]
        var_unres_tau_yu[t,:,:,:] = unres_tau_yu[:,:,:]

        var_total_tau_yv[t,:,:,:] = total_tau_yv[:,:,:]
        var_res_tau_yv[t,:,:,:] = res_tau_yv[:,:,:]
        var_unres_tau_yv[t,:,:,:] = unres_tau_yv[:,:,:]

        var_total_tau_yw[t,:,:,:] = total_tau_yw[:,:,:]
        var_res_tau_yw[t,:,:,:] = res_tau_yw[:,:,:]
        var_unres_tau_yw[t,:,:,:] = unres_tau_yw[:,:,:]

        var_total_tau_zu[t,:,:,:] = total_tau_zu[:,:,:]
        var_res_tau_zu[t,:,:,:] = res_tau_zu[:,:,:]
        var_unres_tau_zu[t,:,:,:] = unres_tau_zu[:,:,:]

        var_total_tau_zv[t,:,:,:] = total_tau_zv[:,:,:]
        var_res_tau_zv[t,:,:,:] = res_tau_zv[:,:,:]
        var_unres_tau_zv[t,:,:,:] = unres_tau_zv[:,:,:]

        var_total_tau_zw[t,:,:,:] = total_tau_zw[:,:,:]
        var_res_tau_zw[t,:,:,:] = res_tau_zw[:,:,:]
        var_unres_tau_zw[t,:,:,:] = unres_tau_zw[:,:,:]

        #Close file
        a.close()




    #Close files
    #f.close()
    #g.close()
    #h.close()
    #l.close()

#binary3d_to_nc('u',768,384,256,starttime=0,endtime=7200,sampletime=600)
#binary3d_to_nc('v',768,384,256,starttime=0,endtime=7200,sampletime=600)
#binary3d_to_nc('w',768,384,256,starttime=0,endtime=7200,sampletime=600)
#binary3d_to_nc('p',768,384,256,starttime=0,endtime=7200,sampletime=600)

# generate_training_data((32,16,64))

