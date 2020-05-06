#Script that generates the training data for the NN
#Author: Robin Stoffer (robin.stoffer@wur.nl)
#NOTE: Developed for Python 3!
import numpy   as np
import netCDF4 as nc
import scipy.interpolate
from grid_objects_training import Finegrid
import matplotlib as mpl
mpl.use('agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
import matplotlib.pyplot as plt
import warnings

###############################################
#Actual functions to generate the training data
###############################################

def boxfilter_dns(input_directory, output_directory, reynolds_number_tau, filter_widths, precision = 'double', fourth_order = False, periodic_bc = (False, True, True), zero_w_topbottom = True, name_output_file = 'dns_boxfilter.nc', create_file = True, testing = False, settings_filepath = None, grid_filepath = 'grid.0000000'): 
    
    '''Applies box-filter to DNS fields without coarsening, used to compare with Smagorinsky (calculated on FINE grid as well!). '''

    #Check types input variables (not all of them, some are already checked in grid_objects_training.py.
    if not isinstance(output_directory,str):
        raise TypeError("Specified output directory should be a string.")
    
    if not (isinstance(settings_filepath,str) or (settings_filepath is None)):
        raise TypeError("Specified settings filepath should be a string or NoneType.")

    if not isinstance(grid_filepath,str):
        raise TypeError("Specified grid filepath should be a string.")

    if not (isinstance(reynolds_number_tau,int) or isinstance(reynolds_number_tau,float)):
        raise TypeError("Specified friction Reynolds number should be either a integer or float.")

    if not isinstance(name_output_file,str):
        raise TypeError("Specified name of output file should be a string.")

    if not isinstance(create_file,bool):
        raise TypeError("Specified create_file flag should be a boolean.")

    if not isinstance(testing,bool):
        raise TypeError("Specified testing flag should be a boolean.")

    if not isinstance(settings_filepath,str):
        raise TypeError("Specified settings filepath should be a string.")

    if not isinstance(grid_filepath,str):
        raise TypeError("Specified grid filepath should be a string.")

    #Define flag to ensure variables are only created once in netCDF file
    create_variables = True
    
    #Initialize finegrid object
    if testing:
        coordx = np.array([0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75]) #NOTE: small rounding errors in 12th decimal, in case in 9th decimal
        xsize = 10.0
        coordy = np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5])
        ysize = 20.0
        coordz = np.array([0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0,2.0,3.0,4.0,5.0,5.1,5.15,5.2,5.4,6.0,9.0,9.5,10.0,11.5,12.3,13.0,13.5,13.7,13.8,13.85,13.9,13.95])
        zsize = 14.0
        coordz = np.array([0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,4.9,5.1,5.3,5.5,5.7])
        zsize = 5.8
        nz,ny,nx = 29,20,20

        ##Define similar coordinates as real case
        #nz, ny, nx = 256, 384, 768 #Dimensions fine grid
        ##xsize = 6.283185307179586
        #xsize = 6.
        ##ysize = 3.141592653589793
        #ysize = 3.
        #zsize = 2.
        #dx = xsize / nx
        #dy = ysize / ny
        #coordxh = np.arange(0, xsize, dx)
        #coordyh = np.arange(0, ysize, dy)
        #coordx = np.arange(dx/2, xsize, dx)
        #coordy = np.arange(dy/2, ysize, dy)

        #with nc.Dataset("/projects/1/flowsim/simulation1/robin_moser_stats.nc", "r") as nc_file:
        #    coordz  = nc_file.variables["z" ][:]
        #    #coordzh = nc_file.variables["zh"][:]
        #    #Define for test arrays zh such that it is located exactly in between the grid centers, which is NOT the case for zh from the stats file.
        #    coordzh = np.zeros(len(coordz) + 1)
        #    for k in range(1, len(coordz)):
        #        coordzh[k] = coordz[k-1] + 0.5* (coordz[k] - coordz[k-1])
        #    coordzh[-1] = zsize
        #dz = coordzh[1:] - coordzh[:-1]
        
        finegrid = Finegrid(read_grid_flag = False, precision = precision, fourth_order = fourth_order, coordx = coordx, xsize = xsize, coordy = coordy, ysize = ysize, coordz = coordz, zsize = zsize, periodic_bc = periodic_bc, zero_w_topbottom = zero_w_topbottom, normalisation_grid = False)
            
        #Define arbritary values for scalars that the script below needs.
        finegrid['fields']['visc'] = 1e-5
        finegrid['grid']['channel_half_width'] = 1.

    else:
        finegrid = Finegrid(precision = precision, fourth_order = fourth_order, periodic_bc = periodic_bc, zero_w_topbottom = zero_w_topbottom, normalisation_grid = True, settings_filepath = settings_filepath, grid_filepath = grid_filepath) #Read settings and grid from .ini files produced by MicroHH, normalize grid with channel half width
 
    #Define orientation on grid for all the variables (z,y,x). True means variable is located on the sides in that direction, false means it is located in the grid centers.
    bool_edge_gridcell_u = (False, False, True)
    bool_edge_gridcell_v = (False, True, False)
    bool_edge_gridcell_w = (True, False, False)
    bool_edge_gridcell_p = (False, False, False)

    #Define reference friction velocity based on viscosity and channel half-width read from the settings files produced by Microhh, and the user-defined Reynolds number
    mvisc = finegrid['fields']['visc']
    channel_half_width = finegrid['grid']['channel_half_width']
    utau_ref = (reynolds_number_tau * mvisc) / channel_half_width
    
    #Define reference kinematic viscosity based on reference friction velocity and channel_half_width
    mvisc_ref = mvisc / (utau_ref * channel_half_width)

    #Define time steps corresponding to validation set
    tstart=27 #28th timestep
    tend = 29 #30th timestep

    #Loop over timesteps
    #for t in range(tstart, tend):
    #for t in range(finegrid['time']['timesteps']): #Only works correctly in this script when whole simulation is saved with a constant time interval. 
        #NOTE1: when testing, the # of timesteps should be set equal to 1.
    for t in range(tstart): #FOR TESTING PURPOSES ONLY!
        
        ##Read or define fine-resolution DNS data ##
        ############################################

        #Read variables from fine resolution data into finegrid or manually define them when testing
        if testing:
            output_shape  = (nz, ny, nx)
#            start_value   = 0
#            end_value     = start_value + output_shape[0]*output_shape[1]*output_shape[2]
#            output_array  = np.reshape(np.arange(start_value,end_value), output_shape)
            
            #Define output_array that has only a gradient in the z-direction.
            output_array  = np.zeros((nz, ny, nx))
            output_1level = np.ones((nz, ny))
            for i in range(1,nx+1):
                output_array[:,:,i-1] = (i ** 1.01)  * output_1level

            finegrid.create_variables('u', output_array, bool_edge_gridcell_u)
            finegrid.create_variables('v', output_array, bool_edge_gridcell_v)
            finegrid.create_variables('w', output_array, bool_edge_gridcell_w)
            finegrid.create_variables('p', output_array, bool_edge_gridcell_p)
            
        else:
            finegrid.read_binary_variables(input_directory, 'u', t, bool_edge_gridcell_u, normalisation_factor = utau_ref)
            finegrid.read_binary_variables(input_directory, 'v', t, bool_edge_gridcell_v, normalisation_factor = utau_ref)
            finegrid.read_binary_variables(input_directory, 'w', t, bool_edge_gridcell_w, normalisation_factor = utau_ref)

        ##Apply box filter to fine DNS data##
        ################################################################

        finegrid.boxfilter('u', filter_widths)
        print('Filtered u')
        finegrid.boxfilter('v', filter_widths)
        print('Filtered v')
        finegrid.boxfilter('w', filter_widths)
        print('Filtered w')

        ##Calculate total, filtered, and residual transport over fine grid filtered with box filter ##
        ###########################################################################

        #Initialize first arrays for total transport components and viscous transports on fine grid
        total_tau_xu_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot'],   finegrid['grid']['itot']), dtype=float)
        total_tau_yu_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot']+1, finegrid['grid']['itot']+1), dtype=float)
        total_tau_zu_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot'],   finegrid['grid']['itot']+1), dtype=float)
        total_tau_xv_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot']+1, finegrid['grid']['itot']+1), dtype=float)
        total_tau_yv_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot'], finegrid['grid']['itot']), dtype=float)
        total_tau_zv_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot']+1, finegrid['grid']['itot']), dtype=float)
        total_tau_xw_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot'],   finegrid['grid']['itot']+1), dtype=float)
        total_tau_yw_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot']+1, finegrid['grid']['itot']), dtype=float)
        total_tau_zw_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot'],   finegrid['grid']['itot']), dtype=float)
        #
        res_tau_xu_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot'],   finegrid['grid']['itot']), dtype=float)
        res_tau_yu_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot']+1, finegrid['grid']['itot']+1), dtype=float)
        res_tau_zu_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot'],   finegrid['grid']['itot']+1), dtype=float)
        res_tau_xv_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot']+1, finegrid['grid']['itot']+1), dtype=float)
        res_tau_yv_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot'], finegrid['grid']['itot']), dtype=float)
        res_tau_zv_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot']+1, finegrid['grid']['itot']), dtype=float)
        res_tau_xw_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot'],   finegrid['grid']['itot']+1), dtype=float)
        res_tau_yw_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot']+1, finegrid['grid']['itot']), dtype=float)
        res_tau_zw_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot'],   finegrid['grid']['itot']), dtype=float)
        #
        unres_tau_xu_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot'],   finegrid['grid']['itot']), dtype=float)
        unres_tau_yu_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot']+1, finegrid['grid']['itot']+1), dtype=float)
        unres_tau_zu_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot'],   finegrid['grid']['itot']+1), dtype=float)
        unres_tau_xv_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot']+1, finegrid['grid']['itot']+1), dtype=float)
        unres_tau_yv_turb = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot'], finegrid['grid']['itot']), dtype=float)
        unres_tau_zv_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot']+1, finegrid['grid']['itot']), dtype=float)
        unres_tau_xw_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot'],   finegrid['grid']['itot']+1), dtype=float)
        unres_tau_yw_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot']+1, finegrid['grid']['itot']), dtype=float)
        unres_tau_zw_turb = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot'],   finegrid['grid']['itot']), dtype=float)
  
        #Calculate transports
        tot_tau_xu_turb = finegrid['output']['u']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend] ** 2.
        tot_tau_yu_turb = finegrid['output']['u']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jend+1,finegrid.igc:finegrid.ihend] * finegrid['output']['v']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jhend,finegrid.igc:finegrid.iend+1]
        tot_tau_zu_turb = finegrid['output']['u']['variable'][finegrid.kgc_center:finegrid.kend+1,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.ihend] * finegrid['output']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend+1]
        tot_tau_xv_turb = finegrid['output']['v']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jhend,finegrid.igc:finegrid.iend+1] * finegrid['output']['u']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jend+1,finegrid.igc:finegrid.ihend]
        tot_tau_yv_turb = finegrid['output']['v']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend] ** 2.
        tot_tau_zv_turb = finegrid['output']['v']['variable'][finegrid.kgc_center:finegrid.kend+1,finegrid.jgc:finegrid.jhend,finegrid.igc:finegrid.iend] * finegrid['output']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend+1,finegrid.igc:finegrid.iend]
        tot_tau_xw_turb = finegrid['output']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend+1] * finegrid['output']['u']['variable'][finegrid.kgc_center:finegrid.kend+1,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.ihend]
        tot_tau_yw_turb = finegrid['output']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend+1,finegrid.igc:finegrid.iend] * finegrid['output']['v']['variable'][finegrid.kgc_center:finegrid.kend+1,finegrid.jgc:finegrid.jhend,finegrid.igc:finegrid.iend]
        tot_tau_zw_turb = finegrid['output']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend] ** 2.
        #
        res_tau_xu_turb = finegrid['boxfilter']['u']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend] ** 2.
        res_tau_yu_turb = finegrid['boxfilter']['u']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jend+1,finegrid.igc:finegrid.ihend] * finegrid['boxfilter']['v']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jhend,finegrid.igc:finegrid.iend+1]
        res_tau_zu_turb = finegrid['boxfilter']['u']['variable'][finegrid.kgc_center:finegrid.kend+1,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.ihend] * finegrid['boxfilter']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend+1]
        res_tau_xv_turb = finegrid['boxfilter']['v']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jhend,finegrid.igc:finegrid.iend+1] * finegrid['boxfilter']['u']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jend+1,finegrid.igc:finegrid.ihend]
        res_tau_yv_turb = finegrid['boxfilter']['v']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend] ** 2.
        res_tau_zv_turb = finegrid['boxfilter']['v']['variable'][finegrid.kgc_center:finegrid.kend+1,finegrid.jgc:finegrid.jhend,finegrid.igc:finegrid.iend] * finegrid['boxfilter']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend+1,finegrid.igc:finegrid.iend]
        res_tau_xw_turb = finegrid['boxfilter']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend+1] * finegrid['boxfilter']['u']['variable'][finegrid.kgc_center:finegrid.kend+1,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.ihend]
        res_tau_yw_turb = finegrid['boxfilter']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend+1,finegrid.igc:finegrid.iend] * finegrid['boxfilter']['v']['variable'][finegrid.kgc_center:finegrid.kend+1,finegrid.jgc:finegrid.jhend,finegrid.igc:finegrid.iend]
        res_tau_zw_turb = finegrid['boxfilter']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend] ** 2.
        #
        unres_tau_xu_turb = tot_tau_xu_turb - res_tau_xu_turb
        unres_tau_yu_turb = tot_tau_yu_turb - res_tau_yu_turb
        unres_tau_zu_turb = tot_tau_zu_turb - res_tau_zu_turb
        unres_tau_xv_turb = tot_tau_xv_turb - res_tau_xv_turb
        unres_tau_yv_turb = tot_tau_yv_turb - res_tau_yv_turb
        unres_tau_zv_turb = tot_tau_zv_turb - res_tau_zv_turb
        unres_tau_xw_turb = tot_tau_xw_turb - res_tau_xw_turb
        unres_tau_yw_turb = tot_tau_yw_turb - res_tau_yw_turb
        unres_tau_zw_turb = tot_tau_zw_turb - res_tau_zw_turb

        ##Store boxfiltered flow fields  and unresolved transports in netCDF-file ##
        ##########################################################################################
 
        #Create/open netCDF file
        if create_file:
            a = nc.Dataset(output_directory + name_output_file, 'w')
            create_file = False
        else:
            a = nc.Dataset(output_directory + name_output_file, 'r+')
 
        if create_variables:
            ##Extract time variable from u-file (should be identical to the one from v-,w-,or p-file)
            #time = np.array(f.variables['time'])
 
            #Create new dimensions
            a.createDimension("time",None)
            a.createDimension("xh",finegrid['grid']['itot']+1)
            a.createDimension("x", finegrid['grid']['itot'])
            a.createDimension("yh",finegrid['grid']['jtot']+1)
            a.createDimension("y", finegrid['grid']['jtot'])
            a.createDimension("zh",finegrid['grid']['ktot']+1)
            a.createDimension("z", finegrid['grid']['ktot'])
 
            #Create coordinate variables and store values
            var_xh          = a.createVariable("xh","f8",("xh",))
            var_x           = a.createVariable("x","f8",("x",))
            var_yh          = a.createVariable("yh","f8",("yh",))
            var_y           = a.createVariable("y","f8",("y",))
            var_zh          = a.createVariable("zh","f8",("zh",))
            var_z           = a.createVariable("z","f8",("z",))
            var_igc         = a.createVariable("igc","i",())
            var_jgc         = a.createVariable("jgc","i",())
            var_kgc_center  = a.createVariable("kgc_center","i",())
            var_kgc_edge    = a.createVariable("kgc_edge","i",())
            var_iend        = a.createVariable("iend","i",())
            var_ihend       = a.createVariable("ihend","i",())
            var_jend        = a.createVariable("jend","i",())
            var_jhend       = a.createVariable("jhend","i",())
            var_kend        = a.createVariable("kend","i",())
            var_khend       = a.createVariable("khend","i",())
            #
            var_reynolds_number_tau     = a.createVariable("reynolds_number_tau","f8",())
            var_mvisc                   = a.createVariable("mvisc","f8",())
            var_mvisc_ref               = a.createVariable("mvisc_ref","f8",())
            var_channel_half_width      = a.createVariable("channel_half_width","f8",())
            var_utau_ref                = a.createVariable("utau_ref","f8",())
            
            var_xh[:]         = finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]
            var_x[:]          = finegrid['grid']['x'][finegrid.igc:finegrid.iend]
            var_yh[:]         = finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend]
            var_y[:]          = finegrid['grid']['y'][finegrid.jgc:finegrid.jend]
            var_zh[:]         = finegrid['grid']['zh'][finegrid.kgc:finegrid.khend]
            var_z[:]          = finegrid['grid']['z'][finegrid.kgc:finegrid.kend]
            var_igc[:]        = finegrid.igc
            var_jgc[:]        = finegrid.jgc
            var_kgc_center[:] = finegrid.kgc
            var_kgc_edge[:]   = finegrid.kgc
            var_iend[:]       = finegrid.iend
            var_ihend[:]      = finegrid.ihend
            var_jend[:]       = finegrid.jend
            var_jhend[:]      = finegrid.jhend
            var_kend[:]       = finegrid.kend
            var_khend[:]      = finegrid.khend
            #
            var_reynolds_number_tau[:]     = reynolds_number_tau
            var_mvisc[:]                   = mvisc
            var_mvisc_ref[:]               = mvisc_ref
            var_channel_half_width[:]      = channel_half_width
            var_utau_ref[:]                = utau_ref
            
            #var_dist_midchannel[:] = dist_midchannel[:]
 
            #Create variables for coarse fields
            var_uboxfilt = a.createVariable("uc","f8",("time","z","y","xh"))
            var_vboxfilt = a.createVariable("vc","f8",("time","z","yh","x"))
            var_wboxfilt = a.createVariable("wc","f8",("time","zh","y","x"))
 
            var_total_tau_xu_turb = a.createVariable("total_tau_xu_turb","f8",("time","z","y","x"))
            var_res_tau_xu_turb   = a.createVariable("res_tau_xu_turb","f8",("time","z","y","x"))
            var_unres_tau_xu_turb = a.createVariable("unres_tau_xu_turb","f8",("time","z","y","x"))

            var_total_tau_xv_turb = a.createVariable("total_tau_xv_turb","f8",("time","z","yh","xh"))
            var_res_tau_xv_turb   = a.createVariable("res_tau_xv_turb","f8",("time","z","yh","xh"))
            var_unres_tau_xv_turb = a.createVariable("unres_tau_xv_turb","f8",("time","z","yh","xh"))

            var_total_tau_xw_turb = a.createVariable("total_tau_xw_turb","f8",("time","zh","y","xh"))
            var_res_tau_xw_turb   = a.createVariable("res_tau_xw_turb","f8",("time","zh","y","xh"))
            var_unres_tau_xw_turb = a.createVariable("unres_tau_xw_turb","f8",("time","zh","y","xh"))
 
            var_total_tau_yu_turb = a.createVariable("total_tau_yu_turb","f8",("time","z","yh","xh"))
            var_res_tau_yu_turb   = a.createVariable("res_tau_yu_turb","f8",("time","z","yh","xh"))
            var_unres_tau_yu_turb = a.createVariable("unres_tau_yu_turb","f8",("time","z","yh","xh"))
 
            var_total_tau_yv_turb = a.createVariable("total_tau_yv_turb","f8",("time","z","y","x"))
            var_res_tau_yv_turb   = a.createVariable("res_tau_yv_turb","f8",("time","z","y","x"))
            var_unres_tau_yv_turb = a.createVariable("unres_tau_yv_turb","f8",("time","z","y","x"))

            var_total_tau_yw_turb = a.createVariable("total_tau_yw_turb","f8",("time","zh","yh","x"))
            var_res_tau_yw_turb   = a.createVariable("res_tau_yw_turb","f8",("time","zh","yh","x"))
            var_unres_tau_yw_turb = a.createVariable("unres_tau_yw_turb","f8",("time","zh","yh","x"))
 
            var_total_tau_zu_turb = a.createVariable("total_tau_zu_turb","f8",("time","zh","y","xh"))
            var_res_tau_zu_turb   = a.createVariable("res_tau_zu_turb","f8",("time","zh","y","xh"))
            var_unres_tau_zu_turb = a.createVariable("unres_tau_zu_turb","f8",("time","zh","y","xh"))
 
            var_total_tau_zv_turb = a.createVariable("total_tau_zv_turb","f8",("time","zh","yh","x"))
            var_res_tau_zv_turb   = a.createVariable("res_tau_zv_turb","f8",("time","zh","yh","x"))
            var_unres_tau_zv_turb = a.createVariable("unres_tau_zv_turb","f8",("time","zh","yh","x"))

            var_total_tau_zw_turb = a.createVariable("total_tau_zw_turb","f8",("time","z","y","x"))
            var_res_tau_zw_turb   = a.createVariable("res_tau_zw_turb","f8",("time","z","y","x"))
            var_unres_tau_zw_turb = a.createVariable("unres_tau_zw_turb","f8",("time","z","y","x"))
 
        create_variables = False #Make sure variables are only created once.
 
        #Store values
        var_uc[t,:,:,:] = finegrid['output']['u']['variable']
        var_vc[t,:,:,:] = finegrid['output']['v']['variable']
        var_wc[t,:,:,:] = finegrid['output']['w']['variable']

        var_total_tau_xu_turb[t,:,:,:] = total_tau_xu_turb[:,:,:]
        var_res_tau_xu_turb[t,:,:,:]   = res_tau_xu_turb[:,:,:]
        var_unres_tau_xu_turb[t,:,:,:] = unres_tau_xu_turb[:,:,:]
 
        var_total_tau_xv_turb[t,:,:,:] = total_tau_xv_turb[:,:,:]
        var_res_tau_xv_turb[t,:,:,:]   = res_tau_xv_turb[:,:,:]
        var_unres_tau_xv_turb[t,:,:,:] = unres_tau_xv_turb[:,:,:]
       
        var_total_tau_xw_turb[t,:,:,:] = total_tau_xw_turb[:,:,:]
        var_res_tau_xw_turb[t,:,:,:]   = res_tau_xw_turb[:,:,:]
        var_unres_tau_xw_turb[t,:,:,:] = unres_tau_xw_turb[:,:,:]

        var_total_tau_yu_turb[t,:,:,:] = total_tau_yu_turb[:,:,:]
        var_res_tau_yu_turb[t,:,:,:]   = res_tau_yu_turb[:,:,:]
        var_unres_tau_yu_turb[t,:,:,:] = unres_tau_yu_turb[:,:,:]

        var_total_tau_yv_turb[t,:,:,:] = total_tau_yv_turb[:,:,:]
        var_res_tau_yv_turb[t,:,:,:]   = res_tau_yv_turb[:,:,:]
        var_unres_tau_yv_turb[t,:,:,:] = unres_tau_yv_turb[:,:,:]

        var_total_tau_yw_turb[t,:,:,:] = total_tau_yw_turb[:,:,:]
        var_res_tau_yw_turb[t,:,:,:]   = res_tau_yw_turb[:,:,:]
        var_unres_tau_yw_turb[t,:,:,:] = unres_tau_yw_turb[:,:,:]

        var_total_tau_zu_turb[t,:,:,:] = total_tau_zu_turb[:,:,:]
        var_res_tau_zu_turb[t,:,:,:]   = res_tau_zu_turb[:,:,:]
        var_unres_tau_zu_turb[t,:,:,:] = unres_tau_zu_turb[:,:,:]

        var_total_tau_zv_turb[t,:,:,:] = total_tau_zv_turb[:,:,:]
        var_res_tau_zv_turb[t,:,:,:]   = res_tau_zv_turb[:,:,:]
        var_unres_tau_zv_turb[t,:,:,:] = unres_tau_zv_turb[:,:,:]

        var_total_tau_zw_turb[t,:,:,:] = total_tau_zw_turb[:,:,:]
        var_res_tau_zw_turb[t,:,:,:]   = res_tau_zw_turb[:,:,:]
        var_unres_tau_zw_turb[t,:,:,:] = unres_tau_zw_turb[:,:,:]

        #Close file
        a.close()
