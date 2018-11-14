#Main script that generates the training data for the NN
#Author: Robin Stoffer (robin.stoffer@wur.nl)
#NOTE: Developed for Python 3!
import numpy   as np
import netCDF4 as nc
import scipy.interpolate
from downsampling_training import generate_coarsecoord_centercell
from grid_objects_training import Finegrid, Coarsegrid

###############################################
#Actual functions to generate the training data
###############################################

def generate_training_data(dim_new_grid, precision = 'double', fourth_order = False, periodic_bc = (False, True, True), name_output_file = 'training_data.nc', create_file = True, testing = False): #Filenames should be strings. Default input corresponds to names files from MicroHH and the provided scripts
 
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
        
        finegrid = Finegrid(read_grid_flag = False, fourth_order = False, coordx = coordx, xsize = xsize, coordy = coordy, ysize = ysize, coordz = coordz, zsize = zsize, periodic_bc = (False, True, True), no_slip = True)
            
    else:
        finegrid = Finegrid(precision = precision, fourth_order = fourth_order, periodic_bc = periodic_bc) #Read settings and grid from .ini files produced by MicroHH
 
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
            output_shape = (coordz.shape[0], coordy.shape[0], coordx.shape[0])
            start_value = 0
            end_value = start_value + output_shape[0]*output_shape[1]*output_shape[2]
            output_array = np.reshape(np.arange(start_value,end_value), output_shape)
            
            finegrid.create_variables('u', output_array, bool_edge_gridcell_u)
            finegrid.create_variables('v', output_array, bool_edge_gridcell_v)
            finegrid.create_variables('w', output_array, bool_edge_gridcell_w)
            finegrid.create_variables('p', output_array, bool_edge_gridcell_p)
            
        else:
            finegrid.read_binary_variables('u', t, bool_edge_gridcell_u)
            finegrid.read_binary_variables('v', t, bool_edge_gridcell_v)
            finegrid.read_binary_variables('w', t, bool_edge_gridcell_w)
            finegrid.read_binary_variables('p', t, bool_edge_gridcell_p)
            
        #Initialize coarsegrid object
        coarsegrid = Coarsegrid(dim_new_grid, finegrid)

        #Calculate representative velocities for each coarse grid cell
        coarsegrid.downsample('u')
        coarsegrid.downsample('v')
        coarsegrid.downsample('w')
        coarsegrid.downsample('p')
 
        #Calculate total transport on coarse grid from fine grid, initialize first arrays
        total_tau_xu = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_yu = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']), dtype=float)
        total_tau_zu = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']), dtype=float)
        total_tau_xv = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_yv = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']), dtype=float)
        total_tau_zv = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']), dtype=float)
        total_tau_xw = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_yw = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']), dtype=float)
        total_tau_zw = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']), dtype=float)
 
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
            interpolator_value = interpolator((z_int, y_int ,x_int))
            var_int = np.reshape(interpolator_value,(len(zbound),len(ybound),len(xbound)))
 
            return var_int

        #xz-boundary
        u_xzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],      coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        v_xzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],      coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        w_xzint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],      coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
 
        #yz-boundary
        u_yzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],      finegrid['grid']['y'][finegrid.jgc:finegrid.jend],         coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        v_yzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],      finegrid['grid']['y'][finegrid.jgc:finegrid.jend],         coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        w_yzint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],      finegrid['grid']['y'][finegrid.jgc:finegrid.jend],         coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
 
        #xy-boundary
        u_xyint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend],        finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        v_xyint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend],        finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        w_xyint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend],        finegrid['grid']['x'][finegrid.igc:finegrid.iend]))

        #Calculate TOTAL transport of momentum over xz-, yz-, and xy-boundary. 
        #Note: only centercell functions needed because the boundaries are always located on the grid centers along the two directions over which the values have to be integrated
        #Note: at len+1 iteration for any given coordinate, only weights have to be known for other two coordinates. Furthermore, only part of the total transport terms need to be calculated (i.e. the terms located on the grid side boundaries for the coordinate in the len+1 iteration).
        for izc in range(len(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend])+1):
            if izc != len(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend]):
                zcor_c_middle = coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend][izc]
                weights_z, points_indices_z = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], cor_c_middle = zcor_c_middle, dist_corc = coarsegrid['grid']['zdist'], finegrid = finegrid)
 
            for iyc in range(len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])+1):
                if iyc != len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend]):
                    ycor_c_middle = coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend][iyc]
                    weights_y, points_indices_y = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], cor_c_middle = ycor_c_middle, dist_corc = coarsegrid['grid']['ydist'], finegrid = finegrid)

                for ixc in range(len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])+1):
                    if ixc != len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]):
                        xcor_c_middle = coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend][ixc]
                        weights_x, points_indices_x = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['xh'][finegrid.igc:finegrid.ihend], cor_c_middle = xcor_c_middle, dist_corc = coarsegrid['grid']['xdist'], finegrid = finegrid)
 
                    #xz-boundary
                    if (izc != len(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend])) and (ixc != len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])): #Make sure this not evaluated for the len+1 iteration in the z- and x-coordinates.
                        weights_xz = weights_x[np.newaxis,:]*weights_z[:,np.newaxis]
                        total_tau_yu[izc,iyc,ixc] = np.sum(weights_xz * v_xzint[:,iyc,:][points_indices_z,:][:,points_indices_x] * u_xzint[:,iyc,:][points_indices_z,:][:,points_indices_x])
                        total_tau_yv[izc,iyc,ixc] = np.sum(weights_xz * v_xzint[:,iyc,:][points_indices_z,:][:,points_indices_x] ** 2)
                        total_tau_yw[izc,iyc,ixc] = np.sum(weights_xz * v_xzint[:,iyc,:][points_indices_z,:][:,points_indices_x] * w_xzint[:,iyc,:][points_indices_z,:][:,points_indices_x])
 
                    #yz-boundary
                    if (izc != len(coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend])) and (iyc != len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])): #Make sure this not evaluated for the len+1 iteration in the z- and y-coordinates.
                        weights_yz = weights_y[np.newaxis,:]*weights_z[:,np.newaxis]
                        total_tau_xu[izc,iyc,ixc] = np.sum(weights_yz * u_yzint[:,:,ixc][points_indices_z,:][:,points_indices_y] ** 2)
                        total_tau_xv[izc,iyc,ixc] = np.sum(weights_yz * u_yzint[:,:,ixc][points_indices_z,:][:,points_indices_y] * v_yzint[:,:,ixc][points_indices_z,:][:,points_indices_y])
                        total_tau_xw[izc,iyc,ixc] = np.sum(weights_yz * u_yzint[:,:,ixc][points_indices_z,:][:,points_indices_y] * w_yzint[:,:,ixc][points_indices_z,:][:,points_indices_y])
 
                    #xy-boundary
                    if (iyc != len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])) and (ixc != len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])): #Make sure this not evaluated for the len+1 iteration in the y- and x-coordinates.
                        weights_xy = weights_x[np.newaxis,:]*weights_y[:,np.newaxis]
                        total_tau_zu[izc,iyc,ixc] = np.sum(weights_xy * w_xyint[izc,:,:][points_indices_y,:][:,points_indices_x] * u_xyint[izc,:,:][points_indices_y,:][:,points_indices_x])
                        total_tau_zv[izc,iyc,ixc] = np.sum(weights_xy * w_xyint[izc,:,:][points_indices_y,:][:,points_indices_x] * v_xyint[izc,:,:][points_indices_y,:][:,points_indices_x])
                        total_tau_zw[izc,iyc,ixc] = np.sum(weights_xy * w_xyint[izc,:,:][points_indices_y,:][:,points_indices_x] ** 2)
 
 
        ##Calculate resolved and unresolved transport user specified coarse grid ##
        ###########################################################################
 
        #Define empty variables for storage
 
        res_tau_xu = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1),dtype=float)
        res_tau_yu = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']),dtype=float)
        res_tau_zu = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']),dtype=float)
        res_tau_xv = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1),dtype=float)
        res_tau_yv = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']),dtype=float)
        res_tau_zv = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']),dtype=float)
        res_tau_xw = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1),dtype=float)
        res_tau_yw = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']),dtype=float)
        res_tau_zw = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']),dtype=float)
 
        unres_tau_xu = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1),dtype=float)
        unres_tau_yu = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']),dtype=float)
        unres_tau_zu = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']),dtype=float)
        unres_tau_xv = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1),dtype=float)
        unres_tau_yv = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']),dtype=float)
        unres_tau_zv = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']),dtype=float)
        unres_tau_xw = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1),dtype=float)
        unres_tau_yw = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']),dtype=float)
        unres_tau_zw = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']),dtype=float)

#        #Add ghostcells to wind velocities on coarse grid, define short-hand notations
#        coarsegrid.add_ghostcells_hor('u', jgc=1, igc=1, bool_edge_gridcell = (False, False, True))
#        coarsegrid.add_ghostcells_hor('v', jgc=1, igc=1, bool_edge_gridcell = (False, True, False))
#        coarsegrid.add_ghostcells_hor('w', jgc=1, igc=1, bool_edge_gridcell = (True, False, False))
#        #coarsegrid.add_ghostcells_hor('p', jgc=1, igc=1, bool_edge_gridcell = (False, False, False))

        #Calculate RESOLVED and UNRESOLVED transport
        
        #xz-boundary
        uc_xzint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'], coarsegrid['grid']['y'], coarsegrid['grid']['xh']), (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        vc_xzint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'], coarsegrid['grid']['yh'], coarsegrid['grid']['x']), (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        wc_xzint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'], coarsegrid['grid']['y'], coarsegrid['grid']['x']), (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))

        res_tau_yu = vc_xzint * uc_xzint
        res_tau_yv = vc_xzint ** 2
        res_tau_yw = vc_xzint * wc_xzint
 
        unres_tau_yu = total_tau_yu - res_tau_yu
        unres_tau_yv = total_tau_yv - res_tau_yv
        unres_tau_yw = total_tau_yw - res_tau_yw

        #yz-boundary
        uc_yzint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'], coarsegrid['grid']['y'], coarsegrid['grid']['xh']), (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        vc_yzint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'], coarsegrid['grid']['yh'], coarsegrid['grid']['x']), (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        wc_yzint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'], coarsegrid['grid']['y'], coarsegrid['grid']['x']), (coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))

        res_tau_xu = uc_yzint ** 2
        res_tau_xv = uc_yzint * vc_yzint
        res_tau_xw = uc_yzint * wc_yzint

        unres_tau_xu = total_tau_xu - res_tau_xu
        unres_tau_xv = total_tau_xv - res_tau_xv
        unres_tau_xw = total_tau_xw - res_tau_xw

        #xy-boundary
        uc_xyint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'], coarsegrid['grid']['y'], coarsegrid['grid']['xh']), (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        vc_xyint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'], coarsegrid['grid']['yh'], coarsegrid['grid']['x']), (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        wc_xyint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'], coarsegrid['grid']['y'], coarsegrid['grid']['x']), (coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))

        res_tau_zu = wc_xyint * uc_xyint
        res_tau_zv = wc_xyint * vc_xyint
        res_tau_zw = wc_xyint ** 2

        unres_tau_zu = total_tau_zu - res_tau_zu
        unres_tau_zv = total_tau_zv - res_tau_zv
        unres_tau_zw = total_tau_zw - res_tau_zw

 
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
            dim_xh = a.createDimension("xhc",coarsegrid['grid']['itot']+1)
            dim_x = a.createDimension("xc",coarsegrid['grid']['itot'])
            dim_yh = a.createDimension("yhc",coarsegrid['grid']['jtot']+1)
            dim_y = a.createDimension("yc",coarsegrid['grid']['jtot'])
            dim_zh = a.createDimension("zhc",coarsegrid['grid']['ktot']+1)
            dim_z = a.createDimension("zc",coarsegrid['grid']['ktot'])
 
            #Create coordinate variables and store values
            var_xhc = a.createVariable("xhc","f8",("xhc",))
            var_xc = a.createVariable("xc","f8",("xc",))
            var_yhc = a.createVariable("yhc","f8",("yhc",))
            var_yc = a.createVariable("yc","f8",("yc",))
            var_zhc = a.createVariable("zhc","f8",("zhc",))
            var_zc = a.createVariable("zc","f8",("zc",))
            #var_dist_midchannel = a.createVariable("dist_midchannel","f8",("zc",))
 
            var_xhc[:] = coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]
            var_xc[:] = coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]
            var_yhc[:] = coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend]
            var_yc[:] = coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend]
            var_zhc[:] = coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend]
            var_zc[:] = coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend]
            #var_dist_midchannel[:] = dist_midchannel[:]
 
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
        var_uc[t,:,:,:] = coarsegrid['output']['u']['variable'][coarsegrid.kgc_center:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jend, coarsegrid.igc:coarsegrid.ihend]
        var_vc[t,:,:,:] = coarsegrid['output']['v']['variable'][coarsegrid.kgc_center:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jhend,coarsegrid.igc:coarsegrid.iend]
        var_wc[t,:,:,:] = coarsegrid['output']['w']['variable'][coarsegrid.kgc_edge:coarsegrid.khend, coarsegrid.jgc:coarsegrid.jend, coarsegrid.igc:coarsegrid.iend]
        var_pc[t,:,:,:] = coarsegrid['output']['p']['variable'][coarsegrid.kgc_center:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jend, coarsegrid.igc:coarsegrid.iend]
 
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

