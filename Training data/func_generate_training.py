#Script that generates the training data for the NN
#Author: Robin Stoffer (robin.stoffer@wur.nl)
#NOTE: Developed for Python 3!
import numpy   as np
import netCDF4 as nc
import scipy.interpolate
from downsampling_training import generate_coarsecoord_centercell, generate_coarsecoord_edgecell
from grid_objects_training import Finegrid, Coarsegrid
import matplotlib as mpl
mpl.use('agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
import matplotlib.pyplot as plt

###############################################
#Actual functions to generate the training data
###############################################

def generate_training_data(dim_new_grid, input_directory, output_directory, size_samples = 5, precision = 'double', fourth_order = False, periodic_bc = (False, True, True), zero_w_topbottom = True, name_output_file = 'training_data.nc', create_file = True, testing = False, settings_filepath = None, grid_filepath = 'grid.0000000'): #Filenames should be strings. Default input corresponds to names files from MicroHH and the provided scripts. igc and jgc specify the amount of ghost cells to be added in the downsampled coarse grid for the x- and y-direction respectively.

    #Check types input variables
    if not isinstance(output_directory,str):
        raise TypeError("Specified output directory should be a string.")
        
    if not (isinstance(settings_filepath,str)) and not (settings_filepath is None):
        raise TypeError("Specified settings filepath should be a string or NoneType.")

    if not isinstance(grid_filepath,str):
        raise TypeError("Specified grid filepath should be a string.")

    #Check that size_samples is an integer, uneven, and larger than 1. This is necessary for the sampling strategy currently implemented
    if not isinstance(size_samples,int):
        raise TypeError("Specified size of samples should be an integer.")

    if (size_samples % 2) == 0 or (size_samples <= 1):
        raise ValueError("Specified size of samples should be uneven and larger than 1 for the sampling strategy currently implemented.")
    
    #Define number of ghost cells in horizontal directions (igc, jgc) based on size_samples
    cells_around_centercell = size_samples // 2 #Use floor division to calculate number of gridcells that are above and below the center gridcell of a given sample, which depends on the sample size. NOTE: sampling itself is done in sample_training_data_tfrecord.py.
    igc = cells_around_centercell
    jgc = cells_around_centercell
    
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
        #coordz = np.array([0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,4.9,5.1,5.3,5.5,5.7])
        #zsize = 5.8
        
        finegrid = Finegrid(read_grid_flag = False, precision = precision, fourth_order = fourth_order, coordx = coordx, xsize = xsize, coordy = coordy, ysize = ysize, coordz = coordz, zsize = zsize, periodic_bc = periodic_bc, zero_w_topbottom = zero_w_topbottom)
            
    else:
        finegrid = Finegrid(precision = precision, fourth_order = fourth_order, periodic_bc = periodic_bc, zero_w_topbottom = zero_w_topbottom, settings_filepath = settings_filepath, grid_filepath = grid_filepath) #Read settings and grid from .ini files produced by MicroHH
 
    #Define orientation on grid for all the variables (z,y,x). True means variable is located on the sides in that direction, false means it is located in the grid centers.
    bool_edge_gridcell_u = (False, False, True)
    bool_edge_gridcell_v = (False, True, False)
    bool_edge_gridcell_w = (True, False, False)
    bool_edge_gridcell_p = (False, False, False)
    
    #Loop over timesteps
    for t in range(finegrid['time']['timesteps']): #Only works correctly in this script when whole simulation is saved with a constant time interval. NOTE: when testing, the # of timesteps is by default set equal to 1.
 
        ##Downsampling from fine DNS data to user specified coarse grid and calculation total transport momentum ##
        ###########################################################################################################

        #Read variables from fine resolution data into finegrid or manually define them when testing
        if testing:
            output_shape  = (coordz.shape[0], coordy.shape[0], coordx.shape[0])
#            start_value   = 0
#            end_value     = start_value + output_shape[0]*output_shape[1]*output_shape[2]
#            output_array  = np.reshape(np.arange(start_value,end_value), output_shape)
            
#            output_1level = np.ones((output_shape[1], output_shape[2]))
#            output_1level = np.ones((output_shape[0], output_shape[1]))
            output_1level = np.ones((output_shape[0], output_shape[2]))
#            output_array  = np.stack([output_1level, 2*output_1level, 3*output_1level], axis = 0)
#            output_array  = np.stack([output_1level, 2*output_1level, 3*output_1level, 4*output_1level, 5*output_1level, 6*output_1level, 2*output_1level, 4*output_1level, 5*output_1level, 4*output_1level,3*output_1level,2*output_1level,1*output_1level,6*output_1level,1*output_1level,6*output_1level,1*output_1level,6*output_1level,5*output_1level,4*output_1level,3*output_1level,2*output_1level,1*output_1level,5*output_1level,2*output_1level,4*output_1level,1*output_1level,3*output_1level,6*output_1level])
#            output_array  = np.stack([output_1level, 2*output_1level, 3*output_1level, 4*output_1level, 5*output_1level, 6*output_1level, 7*output_1level, 8*output_1level, 9*output_1level, 10*output_1level,11*output_1level,12*output_1level,13*output_1level,14*output_1level,15*output_1level,16*output_1level,17*output_1level,18*output_1level,19*output_1level,20*output_1level,21*output_1level,22*output_1level,23*output_1level,24*output_1level,25*output_1level,26*output_1level,27*output_1level,28*output_1level,29*output_1level])
#            output_array  = np.stack([output_1level, 2*output_1level, 3*output_1level, 4*output_1level, 5*output_1level, 6*output_1level, 7*output_1level, 8*output_1level, 9*output_1level, 10*output_1level,11*output_1level,12*output_1level,13*output_1level,14*output_1level,15*output_1level,16*output_1level,17*output_1level,18*output_1level,19*output_1level,20*output_1level], axis=2)
            output_array  = np.stack([output_1level, 2*output_1level, 3*output_1level, 4*output_1level, 5*output_1level, 6*output_1level, 7*output_1level, 8*output_1level, 9*output_1level, 10*output_1level,11*output_1level,12*output_1level,13*output_1level,14*output_1level,15*output_1level,16*output_1level,17*output_1level,18*output_1level,19*output_1level,20*output_1level], axis=1)            
            
            finegrid.create_variables('u', output_array, bool_edge_gridcell_u)
            finegrid.create_variables('v', output_array, bool_edge_gridcell_v)
            finegrid.create_variables('w', output_array, bool_edge_gridcell_w)
            finegrid.create_variables('p', output_array, bool_edge_gridcell_p)
            
        else:
            finegrid.read_binary_variables(input_directory, 'u', t, bool_edge_gridcell_u)
            finegrid.read_binary_variables(input_directory, 'v', t, bool_edge_gridcell_v)
            finegrid.read_binary_variables(input_directory, 'w', t, bool_edge_gridcell_w)
            finegrid.read_binary_variables(input_directory, 'p', t, bool_edge_gridcell_p)
            
        #Initialize coarsegrid object
        coarsegrid = Coarsegrid(dim_new_grid, finegrid, igc = igc, jgc = jgc)

        #Calculate representative velocities for each coarse grid cell
        coarsegrid.downsample('u')
        coarsegrid.downsample('v')
        coarsegrid.downsample('w')
        coarsegrid.downsample('p')
 
        #Calculate total transport on coarse grid from fine grid, initialize first arrays
        total_tau_xu = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']), dtype=float)
        total_tau_yu = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_zu = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_xv = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_yv = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']), dtype=float)
        total_tau_zv = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']), dtype=float)
        total_tau_xw = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_yw = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']), dtype=float)
        total_tau_zw = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']), dtype=float)
 
        #Interpolate to side boundaries coarse gridcell
        def _interpolate_side_cell(variable, coord_variable_ghost, coord_boundary):
 
            #Define variables
            zghost, yghost, xghost = coord_variable_ghost
            zbound, ybound, xbound = coord_boundary
 
            #Interpolate to boundary
            z_int = np.ravel(np.broadcast_to(zbound[:,np.newaxis,np.newaxis],(len(zbound),len(ybound),len(xbound))))
            y_int = np.ravel(np.broadcast_to(ybound[np.newaxis,:,np.newaxis],(len(zbound),len(ybound),len(xbound))))
            x_int = np.ravel(np.broadcast_to(xbound[np.newaxis,np.newaxis,:],(len(zbound),len(ybound),len(xbound))))
 
            interpolator = scipy.interpolate.RegularGridInterpolator((zghost, yghost, xghost), variable, method = 'linear', bounds_error = True, fill_value = 0.) #Make sure that w is equal to 0 at the top and bottom boundary (where extrapolation is needed because no ghost cells are defined).
            interpolator_value = interpolator(np.transpose((z_int, y_int ,x_int))) #np.transpose added to reshape the arrays corresponding to the shape that is expected.
            var_int = np.reshape(interpolator_value,(len(zbound),len(ybound),len(xbound)))
 
            return var_int

        #NOTE: because the controle volume for the wind velocity component differs due to the staggered grid, the transport terms (total and resolved) need to be calculated on different location in the coarse grid depending on the wind component considered.
        
        #Controle volume u-momentum
        #NOTE: all transport terms need to be calculated on the upstream boundaries of the control volume
        #xz-boundary
        u_uxzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
        v_uxzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
 
        #yz-boundary
        #NOTE: At the upstream boundary 1 ghostcell needs to be implemented, which aligns the coordinates such that u_uyzint is located upstream of the center of the control volume. Make use of periodic BC's.
        u_uyzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        #u_uyzint = np.zeros((u_uyzint_noghost.shape[0], u_uyzint_noghost.shape[1], u_uyzint_noghost.shape[2]+1))
        #u_uyzint[:,:,1:] = u_uyzint_noghost.copy()
        #u_uyzint[:,:,0] = u_uyzint_noghost[:,:,-1]
 
        #xy-boundary
        u_uxyint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],   finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
        w_uxyint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],   finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
        
        #Controle volume v-momentum
        #NOTE: all transport terms need to be calculated on the upstream boundaries of the control volume
        #xz-boundary
        v_vxzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],   finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        #v_vxzint = np.zeros((v_vxzint_noghost.shape[0], v_vxzint_noghost.shape[1]+1, v_vxzint_noghost.shape[2]))
        #v_vxzint[:,1:,:] = v_vxzint_noghost.copy()
        #v_vxzint[:,0,:] = v_vxzint_noghost[:,-1,:]
    
        #yz-boundary
        u_vyzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        v_vyzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
 
        #xy-boundary
        v_vxyint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'],  finegrid['grid']['x']), (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],   finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        w_vxyint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],   finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        
        #Controle volume w-momentum
        #NOTE: all transport terms need to be calculated on the upstream boundaries of the control volume
        #xz-boundary
        v_wxzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        w_wxzint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'],  finegrid['grid']['y'], finegrid['grid']['x']),  (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
 
        #yz-boundary
        u_wyzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend])) 
        w_wyzint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'],  finegrid['grid']['y'],  finegrid['grid']['x']), (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
    
        #xy-boundary
        w_wxyint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend],    finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        #w_wxyint = np.zeros((w_wxyint_noghost.shape[0]+1, w_wxyint_noghost.shape[1], w_wxyint_noghost.shape[2]))
        #w_wxyint[1:,:,:] = w_wxyint_noghost.copy()
        #w_wxyint[0,:,:] = 0 - w_wxyint_noghost[0,:,:]

        #Calculate TOTAL transport of momentum over xz-, yz-, and xy-boundary. 
        #NOTE: only centercell functions needed because the boundaries are always located on the grid centers along the two directions over which the values have to be integrated
        #NOTE: at len+1 iteration for any given coordinate, only weights have to be known for other two coordinates. Furthermore, only part of the total transport terms need to be calculated (i.e. the terms located on the grid side boundaries for the coordinate in the len+1 iteration).
        for izc in range(len(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend])+1):
            
            zcor_c_middle_edge = coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend][izc]
            weights_z_edge, points_indices_z_edge = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], cor_c_middle = zcor_c_middle_edge, dist_corc = coarsegrid['grid']['zhdist'], finegrid = finegrid, periodic_bc = periodic_bc[0], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['zsize'])
            if izc != len(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend]):
                zcor_c_middle_center = coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend][izc]
                weights_z_center, points_indices_z_center = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], cor_c_middle = zcor_c_middle_center, dist_corc = coarsegrid['grid']['zdist'], finegrid = finegrid)
 
            for iyc in range(len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])+1):
                
                ycor_c_middle_edge = coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend][iyc]
                weights_y_edge, points_indices_y_edge = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['y'][finegrid.jgc:finegrid.jend], cor_c_middle = ycor_c_middle_edge, dist_corc = coarsegrid['grid']['yhdist'], finegrid = finegrid, periodic_bc = periodic_bc[1], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['ysize'])
                if iyc != len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend]):
                    ycor_c_middle_center = coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend][iyc]
                    weights_y_center, points_indices_y_center = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], cor_c_middle = ycor_c_middle_center, dist_corc = coarsegrid['grid']['ydist'], finegrid = finegrid)

                for ixc in range(len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])+1):
                    
                    xcor_c_middle_edge = coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend][ixc]
                    weights_x_edge, points_indices_x_edge = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['x'][finegrid.igc:finegrid.iend], cor_c_middle = xcor_c_middle_edge, dist_corc = coarsegrid['grid']['xhdist'], finegrid = finegrid, periodic_bc = periodic_bc[2], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['xsize'])
                    if ixc != len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]):
                        xcor_c_middle_center = coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend][ixc]
                        weights_x_center, points_indices_x_center = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['xh'][finegrid.igc:finegrid.ihend], cor_c_middle = xcor_c_middle_center, dist_corc = coarsegrid['grid']['xdist'], finegrid = finegrid)

                    #x,y,z: center coarse grid cell
                    if (izc != len(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend])) and (iyc != len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])) and (ixc != len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])): #Make sure this not evaluated for the len+1 iteration in the z-, y- and x-coordinates.
                        weights_y_center_z_center = weights_y_center[np.newaxis,:]*weights_z_center[:,np.newaxis]
                        total_tau_xu[izc,iyc,ixc] = np.sum(weights_y_center_z_center * u_uyzint[:,:,ixc][points_indices_z_center,:][:,points_indices_y_center] ** 2)

                        weights_x_center_z_center = weights_x_center[np.newaxis,:]*weights_z_center[:,np.newaxis]
                        total_tau_yv[izc,iyc,ixc] = np.sum(weights_x_center_z_center * v_vxzint[:,iyc,:][points_indices_z_center,:][:,points_indices_x_center] ** 2)
                        
                        weights_x_center_y_center = weights_x_center[np.newaxis,:]*weights_y_center[:,np.newaxis]
                        total_tau_zw[izc,iyc,ixc] = np.sum(weights_x_center_y_center * w_wxyint[izc,:,:][points_indices_y_center,:][:,points_indices_x_center] ** 2)
                        
                    #x,y: edge coarse grid cell; z: center coarse grid cell
                    if (izc != len(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend])): #Make sure this not evaluated for the len+1 iteration in the z-coordinates.
                        weights_y_edge_z_center   = weights_y_edge[np.newaxis,:]*weights_z_center[:,np.newaxis]
                        total_tau_xv[izc,iyc,ixc] = np.sum(weights_y_edge_z_center * u_vyzint[:,:,ixc][points_indices_z_center,:][:,points_indices_y_edge] * v_vyzint[:,:,ixc][points_indices_z_center,:][:,points_indices_y_edge])
                    
                        weights_x_edge_z_center   = weights_x_edge[np.newaxis,:]*weights_z_center[:,np.newaxis]
                        total_tau_yu[izc,iyc,ixc] = np.sum(weights_x_edge_z_center * v_uxzint[:,iyc,:][points_indices_z_center,:][:,points_indices_x_edge] * u_uxzint[:,iyc,:][points_indices_z_center,:][:,points_indices_x_edge])
                        
                    #x,z: edge coarse grid cell; y:center coarse grid cell
                    if (iyc != len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])): #Make sure this not evaluated for the len+1 iteration in the y-coordinates.
                        weights_y_center_z_edge   = weights_y_center[np.newaxis,:]*weights_z_edge[:,np.newaxis]
                        total_tau_xw[izc,iyc,ixc] = np.sum(weights_y_center_z_edge * u_wyzint[:,:,ixc][points_indices_z_edge,:][:,points_indices_y_center] * w_wyzint[:,:,ixc][points_indices_z_edge,:][:,points_indices_y_center])

                        weights_x_edge_y_center   = weights_x_edge[np.newaxis,:]*weights_y_center[:,np.newaxis]
                        total_tau_zu[izc,iyc,ixc] = np.sum(weights_x_edge_y_center * w_uxyint[izc,:,:][points_indices_y_center,:][:,points_indices_x_edge] * u_uxyint[izc,:,:][points_indices_y_center,:][:,points_indices_x_edge])

                    #y,z: edge coarse grid cell; x:center coarse grid cell
                    if (ixc != len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])): #Make sure this not evaluated for the len+1 iteration in the x-coordinates.
                        weights_x_center_z_edge   = weights_x_center[np.newaxis,:]*weights_z_edge[:,np.newaxis]
                        total_tau_yw[izc,iyc,ixc] = np.sum(weights_x_center_z_edge * v_wxzint[:,iyc,:][points_indices_z_edge,:][:,points_indices_x_center] * w_wxzint[:,iyc,:][points_indices_z_edge,:][:,points_indices_x_center])
                        
                        weights_x_center_y_edge   = weights_x_center[np.newaxis,:]*weights_y_edge[:,np.newaxis]
                        total_tau_zv[izc,iyc,ixc] = np.sum(weights_x_center_y_edge * w_vxyint[izc,:,:][points_indices_y_edge,:][:,points_indices_x_center] * v_vxyint[izc,:,:][points_indices_y_edge,:][:,points_indices_x_center])
                    
 
#                    #xz-boundary
#                    if (izc != len(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend])) and (ixc != len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])): #Make sure this not evaluated for the len+1 iteration in the z- and x-coordinates.
#                        weights_xz = weights_x[np.newaxis,:]*weights_z[:,np.newaxis]
#                        total_tau_yu[izc,iyc,ixc] = np.sum(weights_xz * v_uxzint[:,iyc,:][points_indices_z,:][:,points_indices_x] * u_uxzint[:,iyc,:][points_indices_z,:][:,points_indices_x])
#                        total_tau_yv[izc,iyc,ixc] = np.sum(weights_xz * v_vxzint[:,iyc,:][points_indices_z,:][:,points_indices_x] ** 2)
#                        total_tau_yw[izc,iyc,ixc] = np.sum(weights_xz * v_wxzint[:,iyc,:][points_indices_z,:][:,points_indices_x] * w_wxzint[:,iyc,:][points_indices_z,:][:,points_indices_x])
# 
#                    #yz-boundary
#                    if (izc != len(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend])) and (iyc != len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])): #Make sure this not evaluated for the len+1 iteration in the z- and y-coordinates.
#                        weights_yz = weights_y[np.newaxis,:]*weights_z[:,np.newaxis]
#                        total_tau_xu[izc,iyc,ixc] = np.sum(weights_yz * u_uyzint[:,:,ixc][points_indices_z,:][:,points_indices_y] ** 2)
#                        total_tau_xv[izc,iyc,ixc] = np.sum(weights_yz * u_vyzint[:,:,ixc][points_indices_z,:][:,points_indices_y] * v_vyzint[:,:,ixc][points_indices_z,:][:,points_indices_y])
#                        total_tau_xw[izc,iyc,ixc] = np.sum(weights_yz * u_wyzint[:,:,ixc][points_indices_z,:][:,points_indices_y] * w_wyzint[:,:,ixc][points_indices_z,:][:,points_indices_y])
# 
#                    #xy-boundary
#                    if (iyc != len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])) and (ixc != len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])): #Make sure this not evaluated for the len+1 iteration in the y- and x-coordinates.
#                        weights_xy = weights_x[np.newaxis,:]*weights_y[:,np.newaxis]
#                        total_tau_zu[izc,iyc,ixc] = np.sum(weights_xy * w_uxyint[izc,:,:][points_indices_y,:][:,points_indices_x] * u_uxyint[izc,:,:][points_indices_y,:][:,points_indices_x])
#                        total_tau_zv[izc,iyc,ixc] = np.sum(weights_xy * w_vxyint[izc,:,:][points_indices_y,:][:,points_indices_x] * v_vxyint[izc,:,:][points_indices_y,:][:,points_indices_x])
#                        total_tau_zw[izc,iyc,ixc] = np.sum(weights_xy * w_wxyint[izc,:,:][points_indices_y,:][:,points_indices_x] ** 2)
 
 
        ##Calculate resolved and unresolved transport user specified coarse grid ##
        ###########################################################################
        
        #NOTE: because the controle volume for the wind velocity component differs due to the staggered grid, the transport terms (total and resolved) need to be calculated on different location in the coarse grid depending on the wind component considered.
        
        #Control volume u-momentum
        #NOTE: all transport terms need to be calculated on the upstream boundaries of the control volume
        #xz-boundary
        uc_uxzint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['y'],  coarsegrid['grid']['xh']), (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        vc_uxzint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['yh'], coarsegrid['grid']['x']),  (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
 
        #yz-boundary
        uc_uyzint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['y'],  coarsegrid['grid']['xh']), (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],          coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        #uc_uyzint = np.zeros((uc_uyzint_noghost.shape[0], uc_uyzint_noghost.shape[1], uc_uyzint_noghost.shape[2]+1))
        #uc_uyzint[:,:,1:] = uc_uyzint_noghost.copy()
        #uc_uyzint[:,:,0] = uc_uyzint_noghost[:,:,-1]
 
        #xy-boundary
        uc_uxyint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['y'],  coarsegrid['grid']['xh']), (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],   coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        wc_uxyint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'], coarsegrid['grid']['y'],  coarsegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],   coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        
        #Control volume v-momentum
        #NOTE: all transport terms need to be calculated on the upstream boundaries of the control volume
        #xz-boundary
        vc_vxzint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['yh'], coarsegrid['grid']['x']),  (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],   coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        #vc_vxzint = np.zeros((vc_vxzint_noghost.shape[0], vc_vxzint_noghost.shape[1]+1, vc_vxzint_noghost.shape[2]))
        #vc_vxzint[:,1:,:] = vc_vxzint_noghost.copy()
        #vc_vxzint[:,0,:] = vc_vxzint_noghost[:,-1,:]
    
        #yz-boundary
        uc_vyzint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['y'],  coarsegrid['grid']['xh']), (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        vc_vyzint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['yh'], coarsegrid['grid']['x']),  (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
 
        #xy-boundary
        vc_vxyint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['yh'],  coarsegrid['grid']['x']), (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],   coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],          coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        wc_vxyint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'], coarsegrid['grid']['y'],  coarsegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],   coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],          coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        
        #Control volume w-momentum
        #NOTE: all transport terms need to be calculated on the upstream boundaries of the control volume
        #xz-boundary
        vc_wxzint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['yh'], coarsegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        wc_wxzint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'],  coarsegrid['grid']['y'], coarsegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
 
        #yz-boundary
        uc_wyzint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['y'],  coarsegrid['grid']['xh']), (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],         coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend])) 
        wc_wyzint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'],  coarsegrid['grid']['y'],  coarsegrid['grid']['x']), (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend],         coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
    
        #xy-boundary
        #NOTE: At the bottom boundary 1 ghostcell needs to be implemented, which aligns the coordinates such that w_wxyint is located below the center of the control volume. Make use of Dirichlet BC that w = 0.
        wc_wxyint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'], coarsegrid['grid']['y'],  coarsegrid['grid']['x']),  (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],    coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        #wc_wxyint = np.zeros((wc_wxyint_noghost.shape[0]+1, wc_wxyint_noghost.shape[1], wc_wxyint_noghost.shape[2]))
        #wc_wxyint[1:,:,:] = wc_wxyint_noghost.copy()
        #wc_wxyint[0,:,:] = 0 - wc_wxyint_noghost[0,:,:]

        #Calculate resolved and unresolved transport from interpolated velocities defined above
        #xz-boundary
        res_tau_yu = vc_uxzint * uc_uxzint
        res_tau_yv = vc_vxzint ** 2
        res_tau_yw = vc_wxzint * wc_wxzint
 
        unres_tau_yu = total_tau_yu - res_tau_yu
        unres_tau_yv = total_tau_yv - res_tau_yv
        unres_tau_yw = total_tau_yw - res_tau_yw

        #yz-boundary        
        res_tau_xu = uc_uyzint ** 2
        res_tau_xv = uc_vyzint * vc_vyzint
        res_tau_xw = uc_wyzint * wc_wyzint

        unres_tau_xu = total_tau_xu - res_tau_xu
        unres_tau_xv = total_tau_xv - res_tau_xv
        unres_tau_xw = total_tau_xw - res_tau_xw

        #xy-boundary
        res_tau_zu = wc_uxyint * uc_uxyint
        res_tau_zv = wc_vxyint * vc_vxyint
        res_tau_zw = wc_wxyint ** 2

        unres_tau_zu = total_tau_zu - res_tau_zu
        unres_tau_zv = total_tau_zv - res_tau_zv
        unres_tau_zw = total_tau_zw - res_tau_zw

        if testing: #Make plots to check calculations turbulent transport, make sure to use test arrays that ONLY differ in the vertical direction
            
            #Calculate RESOLVED and UNRESOLVED transport
            
            #NOTE: because the controle volume for the wind velocity component differs due to the staggered grid, the transport terms (total and resolved) need to be calculated on different location in the coarse grid depending on the wind component considered.
            
            #Controle volume u-momentum
            #NOTE: all transport terms need to be calculated on the upstream boundaries of the control volume
            #xz-boundary
            ut_uxzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],  finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
            vt_uxzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],  finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
     
            #yz-boundary
            ut_uyzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
            #ut_uyzint = np.zeros((ut_uyzint_noghost.shape[0], ut_uyzint_noghost.shape[1], ut_uyzint_noghost.shape[2]+1))
            #ut_uyzint[:,:,1:] = ut_uyzint_noghost.copy()
            #ut_uyzint[:,:,0] = ut_uyzint_noghost[:,:,-1]
     
            #xy-boundary
            ut_uxyint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],   finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
            wt_uxyint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],   finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
            
            #Controle volume v-momentum
            #NOTE: all transport terms need to be calculated on the upstream boundaries of the control volume
            #xz-boundary
            #NOTE: At the upstream boundary 1 ghostcell needs to be implemented, which aligns the coordinates such that v_vxzint is located upstream of the center of the control volume.Make use of periodic BC's.
            vt_vxzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend],   finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
            #vt_vxzint = np.zeros((vt_vxzint_noghost.shape[0], vt_vxzint_noghost.shape[1]+1, vt_vxzint_noghost.shape[2]))
            #vt_vxzint[:,1:,:] = vt_vxzint_noghost.copy()
            #vt_vxzint[:,0,:] = vt_vxzint_noghost[:,-1,:]
        
            #yz-boundary
            ut_vyzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
            vt_vyzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
     
            #xy-boundary
            vt_vxyint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'],  finegrid['grid']['x']), (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
            wt_vxyint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
            
            #Controle volume w-momentum
            #NOTE: all transport terms need to be calculated on the upstream boundaries of the control volume
            #xz-boundary
            vt_wxzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],  finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
            wt_wxzint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'],  finegrid['grid']['y'], finegrid['grid']['x']),  (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],  finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
     
            #yz-boundary
            ut_wyzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          finegrid['grid']['xh'][finegrid.igc:finegrid.ihend])) 
            wt_wyzint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'],  finegrid['grid']['y'],  finegrid['grid']['x']), (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
        
            #xy-boundary
            wt_wxyint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend],    finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
            #wt_wxyint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend],    finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
            #wt_wxyint = np.zeros((wt_wxyint_noghost.shape[0]+1, wt_wxyint_noghost.shape[1], wt_wxyint_noghost.shape[2]))
            #wt_wxyint[1:,:,:] = wt_wxyint_noghost.copy()
            #wt_wxyint[0,:,:] = 0 - wt_wxyint_noghost[0,:,:]
            
            #Calculatte TOTAL transport of momentum over xz-, yz-, and xy-boundary for ALL fine grid cells, initialize first arrays.
            totalt_tau_yu = vt_uxzint * ut_uxzint
            totalt_tau_yv = vt_vxzint ** 2
            totalt_tau_yw = vt_wxzint * wt_wxzint
            totalt_tau_xu = ut_uyzint ** 2
            totalt_tau_xv = ut_vyzint * vt_vyzint
            totalt_tau_xw = ut_wyzint * wt_wyzint
            totalt_tau_zu = wt_uxyint * ut_uxyint
            totalt_tau_zv = wt_vxyint * vt_vxyint
            totalt_tau_zw = wt_wxyint ** 2
            
#            plt.figure()
#            plt.plot(finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], totalt_tau_xu[:,0,0], 'C0o-', label = 'totalt_tau_xu')
#            plt.plot(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], total_tau_xu[:,0,0], 'C1o-', label = 'total_tau_xu')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], totalt_tau_xv[:,0,0], 'C0o-', label = 'totalt_tau_xv')
#            plt.plot(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], total_tau_xv[:,0,0], 'C1o-', label = 'total_tau_xv')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], totalt_tau_xw[:,0,0], 'C0o-', label = 'totalt_tau_xw')
#            plt.plot(coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend], total_tau_xw[:,0,0], 'C1o-', label = 'total_tau_xw')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], totalt_tau_yu[:,0,0], 'C0o-', label = 'totalt_tau_yu')
#            plt.plot(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], total_tau_yu[:,0,0], 'C1o-', label = 'total_tau_yu')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], totalt_tau_yv[:,0,0], 'C0o-', label = 'totalt_tau_yv')
#            plt.plot(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], total_tau_yv[:,0,0], 'C1o-', label = 'total_tau_yv')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], totalt_tau_yw[:,0,0], 'C0o-', label = 'totalt_tau_yw')
#            #plt.plot(coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend], total_tau_yw[:,0,0], 'C1o-', label = 'total_tau_yw')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], totalt_tau_zu[:,0,0], 'C0o-', label = 'totalt_tau_zu')
#            plt.plot(coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend], total_tau_zu[:,0,0], 'C1o-', label = 'total_tau_zu')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], totalt_tau_zv[:,0,0], 'C0o-', label = 'totalt_tau_zv')
#            plt.plot(coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend], total_tau_zv[:,0,0], 'C1o-', label = 'total_tau_zv')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], totalt_tau_zw[:,0,0], 'C0o-', label = 'totalt_tau_zw')
#            #plt.plot(finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], totalt_tau_zw[:,0,0], 'C0o-', label = 'totalt_tau_zw')
#            plt.plot(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], total_tau_zw[:,0,0], 'C1o-', label = 'total_tau_zw')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['x'][finegrid.igc:finegrid.iend], totalt_tau_xu[1,0,:], 'C0o-', label = 'totalt_tau_xu')
#            plt.plot(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend], total_tau_xu[1,0,:], 'C1o-', label = 'total_tau_xu')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['xh'][finegrid.igc:finegrid.ihend], totalt_tau_xv[1,0,:], 'C0o-', label = 'totalt_tau_xv')
#            plt.plot(coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend], total_tau_xv[1,0,:], 'C1o-', label = 'total_tau_xv')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['xh'][finegrid.igc:finegrid.ihend], totalt_tau_xw[1,0,:], 'C0o-', label = 'totalt_tau_xw')
#            plt.plot(coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend], total_tau_xw[1,0,:], 'C1o-', label = 'total_tau_xw')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['xh'][finegrid.igc:finegrid.ihend], totalt_tau_yu[1,0,:], 'C0o-', label = 'totalt_tau_yu')
#            plt.plot(coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend], total_tau_yu[1,0,:], 'C1o-', label = 'total_tau_yu')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['x'][finegrid.igc:finegrid.iend], totalt_tau_yv[1,0,:], 'C0o-', label = 'totalt_tau_yv')
#            plt.plot(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend], total_tau_yv[1,0,:], 'C1o-', label = 'total_tau_yv')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['x'][finegrid.igc:finegrid.iend], totalt_tau_yw[1,0,:], 'C0o-', label = 'totalt_tau_yw')
#            plt.plot(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend], total_tau_yw[1,0,:], 'C1o-', label = 'total_tau_yw')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['xh'][finegrid.igc:finegrid.ihend], totalt_tau_zu[1,0,:], 'C0o-', label = 'totalt_tau_zu')
#            plt.plot(coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend], total_tau_zu[1,0,:], 'C1o-', label = 'total_tau_zu')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['x'][finegrid.igc:finegrid.iend], totalt_tau_zv[1,0,:], 'C0o-', label = 'totalt_tau_zv')
#            plt.plot(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend], total_tau_zv[1,0,:], 'C1o-', label = 'total_tau_zv')
#            plt.legend()
#            plt.show()
#            
#            plt.figure()
#            plt.plot(finegrid['grid']['x'][finegrid.igc:finegrid.iend], totalt_tau_zw[1,0,:], 'C0o-', label = 'totalt_tau_zw')
#            #plt.plot(finegrid['grid']['x'][finegrid.igc:finegrid.iend], totalt_tau_zw[0,0,:], 'C0o-', label = 'totalt_tau_zw')
#            plt.plot(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend], total_tau_zw[1,0,:], 'C1o-', label = 'total_tau_zw')
#            plt.legend()
#            plt.show()
            
            plt.figure()
            plt.plot(finegrid['grid']['y'][finegrid.jgc:finegrid.jend], totalt_tau_xu[1,:,0], 'C0o-', label = 'totalt_tau_xu')
            plt.plot(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], total_tau_xu[1,:,0], 'C1o-', label = 'total_tau_xu')
            plt.legend()
            plt.show()
            
            plt.figure()
            plt.plot(finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], totalt_tau_xv[1,:,0], 'C0o-', label = 'totalt_tau_xv')
            plt.plot(coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], total_tau_xv[1,:,0], 'C1o-', label = 'total_tau_xv')
            plt.legend()
            plt.show()
            
            plt.figure()
            plt.plot(finegrid['grid']['y'][finegrid.igc:finegrid.jend], totalt_tau_xw[1,:,0], 'C0o-', label = 'totalt_tau_xw')
            plt.plot(coarsegrid['grid']['y'][coarsegrid.igc:coarsegrid.jend], total_tau_xw[1,:,0], 'C1o-', label = 'total_tau_xw')
            plt.legend()
            plt.show()
            
            plt.figure()
            plt.plot(finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], totalt_tau_yu[1,:,0], 'C0o-', label = 'totalt_tau_yu')
            plt.plot(coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], total_tau_yu[1,:,0], 'C1o-', label = 'total_tau_yu')
            plt.legend()
            plt.show()
            
            plt.figure()
            plt.plot(finegrid['grid']['y'][finegrid.jgc:finegrid.jend], totalt_tau_yv[1,:,0], 'C0o-', label = 'totalt_tau_yv')
            plt.plot(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], total_tau_yv[1,:,0], 'C1o-', label = 'total_tau_yv')
            plt.legend()
            plt.show()
            
            plt.figure()
            plt.plot(finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], totalt_tau_yw[1,:,0], 'C0o-', label = 'totalt_tau_yw')
            plt.plot(coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], total_tau_yw[1,:,0], 'C1o-', label = 'total_tau_yw')
            plt.legend()
            plt.show()
            
            plt.figure()
            plt.plot(finegrid['grid']['y'][finegrid.jgc:finegrid.jend], totalt_tau_zu[1,:,0], 'C0o-', label = 'totalt_tau_zu')
            plt.plot(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], total_tau_zu[1,:,0], 'C1o-', label = 'total_tau_zu')
            plt.legend()
            plt.show()
            
            plt.figure()
            plt.plot(finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], totalt_tau_zv[1,:,0], 'C0o-', label = 'totalt_tau_zv')
            plt.plot(coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], total_tau_zv[1,:,0], 'C1o-', label = 'total_tau_zv')
            plt.legend()
            plt.show()
            
            plt.figure()
            plt.plot(finegrid['grid']['y'][finegrid.jgc:finegrid.jend], totalt_tau_zw[1,:,0], 'C0o-', label = 'totalt_tau_zw')
            #plt.plot(finegrid['grid']['y'][finegrid.jgc:finegrid.jend], totalt_tau_zw[0,:,0], 'C0o-', label = 'totalt_tau_zw')
            plt.plot(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], total_tau_zw[1,:,0], 'C1o-', label = 'total_tau_zw')
            plt.legend()
            plt.show()

        ##Store flow fields coarse grid and unresolved transport ##
        ###########################################################
 
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
            a.createDimension("xhgc",coarsegrid['grid']['itot']+2*coarsegrid.igc+1)
            a.createDimension("xgc",coarsegrid['grid']['itot']+2*coarsegrid.igc)
            a.createDimension("yhgc",coarsegrid['grid']['jtot']+2*coarsegrid.jgc+1)
            a.createDimension("ygc",coarsegrid['grid']['jtot']+2*coarsegrid.jgc)
            a.createDimension("zhgc",coarsegrid['grid']['ktot']+2*coarsegrid.kgc_edge+1)
            a.createDimension("zgc",coarsegrid['grid']['ktot']+2*coarsegrid.kgc_center)
            a.createDimension("xhc",coarsegrid['grid']['itot']+1)
            a.createDimension("xc",coarsegrid['grid']['itot'])
            a.createDimension("yhc",coarsegrid['grid']['jtot']+1)
            a.createDimension("yc",coarsegrid['grid']['jtot'])
            a.createDimension("zhc",coarsegrid['grid']['ktot']+1)
            a.createDimension("zc",coarsegrid['grid']['ktot'])
 
            #Create coordinate variables and store values
            var_xhgc        = a.createVariable("xhgc","f8",("xhgc",))
            var_xgc         = a.createVariable("xgc","f8",("xgc",))
            var_yhgc        = a.createVariable("yhgc","f8",("yhgc",))
            var_ygc         = a.createVariable("ygc","f8",("ygc",))
            var_zhgc        = a.createVariable("zhgc","f8",("zhgc",))
            var_zgc         = a.createVariable("zgc","f8",("zgc",))
            var_xhc        = a.createVariable("xhc","f8",("xhc",))
            var_xc         = a.createVariable("xc","f8",("xc",))
            var_yhc        = a.createVariable("yhc","f8",("yhc",))
            var_yc         = a.createVariable("yc","f8",("yc",))
            var_zhc        = a.createVariable("zhc","f8",("zhc",))
            var_zc         = a.createVariable("zc","f8",("zc",))
            var_igc        = a.createVariable("igc","i",())
            var_jgc        = a.createVariable("jgc","i",())
            var_kgc_center = a.createVariable("kgc_center","i",())
            var_kgc_edge   = a.createVariable("kgc_edge","i",())
            var_iend       = a.createVariable("iend","i",())
            var_ihend      = a.createVariable("ihend","i",())
            var_jend       = a.createVariable("jend","i",())
            var_jhend      = a.createVariable("jhend","i",())
            var_kend       = a.createVariable("kend","i",())
            var_khend      = a.createVariable("khend","i",())
            var_size_samples = a.createVariable("size_samples","i",())
            var_cells_around_centercell = a.createVariable("cells_around_centercell","i",())
            
            #var_dist_midchannel = a.createVariable("dist_midchannel","f8",("zc",))
 
            var_xhgc[:]       = coarsegrid['grid']['xh']
            var_xgc[:]        = coarsegrid['grid']['x']
            var_yhgc[:]       = coarsegrid['grid']['yh']
            var_ygc[:]        = coarsegrid['grid']['y']
            var_zhgc[:]       = coarsegrid['grid']['zh']
            var_zgc[:]        = coarsegrid['grid']['z']
            var_xhc[:]        = coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]
            var_xc[:]         = coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]
            var_yhc[:]        = coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend]
            var_yc[:]         = coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend]
            var_zhc[:]        = coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend]
            var_zc[:]         = coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend]
            var_igc[:]        = coarsegrid.igc
            var_jgc[:]        = coarsegrid.jgc
            var_kgc_center[:] = coarsegrid.kgc_center
            var_kgc_edge[:]   = coarsegrid.kgc_edge
            var_iend[:]       = coarsegrid.iend
            var_ihend[:]      = coarsegrid.ihend
            var_jend[:]       = coarsegrid.jend
            var_jhend[:]      = coarsegrid.jhend
            var_kend[:]       = coarsegrid.kend
            var_khend[:]      = coarsegrid.khend
            var_size_samples[:] = size_samples
            var_cells_around_centercell[:] = cells_around_centercell
            #var_dist_midchannel[:] = dist_midchannel[:]
 
            #Create variables for coarse fields
            var_uc = a.createVariable("uc","f8",("time","zgc","ygc","xhgc"))
            var_vc = a.createVariable("vc","f8",("time","zgc","yhgc","xgc"))
            var_wc = a.createVariable("wc","f8",("time","zhgc","ygc","xgc"))
            var_pc = a.createVariable("pc","f8",("time","zgc","ygc","xgc"))
 
            var_total_tau_xu = a.createVariable("total_tau_xu","f8",("time","zc","yc","xc"))
            var_res_tau_xu   = a.createVariable("res_tau_xu","f8",("time","zc","yc","xc"))
            var_unres_tau_xu = a.createVariable("unres_tau_xu","f8",("time","zc","yc","xc"))
 
            var_total_tau_xv = a.createVariable("total_tau_xv","f8",("time","zc","yhc","xhc"))
            var_res_tau_xv   = a.createVariable("res_tau_xv","f8",("time","zc","yhc","xhc"))
            var_unres_tau_xv = a.createVariable("unres_tau_xv","f8",("time","zc","yhc","xhc"))
 
            var_total_tau_xw = a.createVariable("total_tau_xw","f8",("time","zhc","yc","xhc"))
            var_res_tau_xw   = a.createVariable("res_tau_xw","f8",("time","zhc","yc","xhc"))
            var_unres_tau_xw = a.createVariable("unres_tau_xw","f8",("time","zhc","yc","xhc"))
 
            var_total_tau_yu = a.createVariable("total_tau_yu","f8",("time","zc","yhc","xhc"))
            var_res_tau_yu   = a.createVariable("res_tau_yu","f8",("time","zc","yhc","xhc"))
            var_unres_tau_yu = a.createVariable("unres_tau_yu","f8",("time","zc","yhc","xhc"))
 
            var_total_tau_yv = a.createVariable("total_tau_yv","f8",("time","zc","yc","xc"))
            var_res_tau_yv   = a.createVariable("res_tau_yv","f8",("time","zc","yc","xc"))
            var_unres_tau_yv = a.createVariable("unres_tau_yv","f8",("time","zc","yc","xc"))
 
            var_total_tau_yw = a.createVariable("total_tau_yw","f8",("time","zhc","yhc","xc"))
            var_res_tau_yw   = a.createVariable("res_tau_yw","f8",("time","zhc","yhc","xc"))
            var_unres_tau_yw = a.createVariable("unres_tau_yw","f8",("time","zhc","yhc","xc"))
 
            var_total_tau_zu = a.createVariable("total_tau_zu","f8",("time","zhc","yc","xhc"))
            var_res_tau_zu   = a.createVariable("res_tau_zu","f8",("time","zhc","yc","xhc"))
            var_unres_tau_zu = a.createVariable("unres_tau_zu","f8",("time","zhc","yc","xhc"))
 
            var_total_tau_zv = a.createVariable("total_tau_zv","f8",("time","zhc","yhc","xc"))
            var_res_tau_zv   = a.createVariable("res_tau_zv","f8",("time","zhc","yhc","xc"))
            var_unres_tau_zv = a.createVariable("unres_tau_zv","f8",("time","zhc","yhc","xc"))
 
            var_total_tau_zw = a.createVariable("total_tau_zw","f8",("time","zc","yc","xc"))
            var_res_tau_zw   = a.createVariable("res_tau_zw","f8",("time","zc","yc","xc"))
            var_unres_tau_zw = a.createVariable("unres_tau_zw","f8",("time","zc","yc","xc"))
 
        create_variables = False #Make sure variables are only created once.
 
        #Store values coarse fields
        var_uc[t,:,:,:] = coarsegrid['output']['u']['variable']
        var_vc[t,:,:,:] = coarsegrid['output']['v']['variable']
        var_wc[t,:,:,:] = coarsegrid['output']['w']['variable']
        var_pc[t,:,:,:] = coarsegrid['output']['p']['variable']
 
        var_total_tau_xu[t,:,:,:] = total_tau_xu[:,:,:]
        var_res_tau_xu[t,:,:,:]   = res_tau_xu[:,:,:]
        var_unres_tau_xu[t,:,:,:] = unres_tau_xu[:,:,:]
 
        var_total_tau_xv[t,:,:,:] = total_tau_xv[:,:,:]
        var_res_tau_xv[t,:,:,:]   = res_tau_xv[:,:,:]
        var_unres_tau_xv[t,:,:,:] = unres_tau_xv[:,:,:]
 
        var_total_tau_xw[t,:,:,:] = total_tau_xw[:,:,:]
        var_res_tau_xw[t,:,:,:]   = res_tau_xw[:,:,:]
        var_unres_tau_xw[t,:,:,:] = unres_tau_xw[:,:,:]
 
        var_total_tau_yu[t,:,:,:] = total_tau_yu[:,:,:]
        var_res_tau_yu[t,:,:,:]   = res_tau_yu[:,:,:]
        var_unres_tau_yu[t,:,:,:] = unres_tau_yu[:,:,:]
 
        var_total_tau_yv[t,:,:,:] = total_tau_yv[:,:,:]
        var_res_tau_yv[t,:,:,:]   = res_tau_yv[:,:,:]
        var_unres_tau_yv[t,:,:,:] = unres_tau_yv[:,:,:]
 
        var_total_tau_yw[t,:,:,:] = total_tau_yw[:,:,:]
        var_res_tau_yw[t,:,:,:]   = res_tau_yw[:,:,:]
        var_unres_tau_yw[t,:,:,:] = unres_tau_yw[:,:,:]
 
        var_total_tau_zu[t,:,:,:] = total_tau_zu[:,:,:]
        var_res_tau_zu[t,:,:,:]   = res_tau_zu[:,:,:]
        var_unres_tau_zu[t,:,:,:] = unres_tau_zu[:,:,:]
 
        var_total_tau_zv[t,:,:,:] = total_tau_zv[:,:,:]
        var_res_tau_zv[t,:,:,:]   = res_tau_zv[:,:,:]
        var_unres_tau_zv[t,:,:,:] = unres_tau_zv[:,:,:]
 
        var_total_tau_zw[t,:,:,:] = total_tau_zw[:,:,:]
        var_res_tau_zw[t,:,:,:]   = res_tau_zw[:,:,:]
        var_unres_tau_zw[t,:,:,:] = unres_tau_zw[:,:,:]
 
        #Close file
        a.close()

