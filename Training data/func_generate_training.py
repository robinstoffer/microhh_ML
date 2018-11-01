#Developed for Python 3!
import numpy   as np
import netCDF4 as nc
import scipy.interpolate
from downsampling_training import generate_coarsecoord_centercell
from grid_objects_training import Finegrid, Coarsegrid
#import struct as st
#import glob
#import re
#import matplotlib as mpl
#mpl.use('Agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
#import matplotlib.pyplot as plt
#from microhh_tools_robinst import Finegrid, Coarsegrid

#This script generates the training data for a developed NN, 
#which is subsequently sampled and stored in tfrecord-files using sample_training_data_tfrecord.py

##############################################
#Helper functions for generation training data
##############################################
#def add_ghostcells_finegrid(finegrid, coarsegrid, variable_name, bool_edge_gridcell):
#
#    #Determine how many ghostcells are needed (when variable is located on grid edges, otherwhise only 1 is needed). 
#    #NOTE: criterion should be based on grid centers (such that for both variables located on grid edges and centers enough ghost cells are present).
#    def _determine_ghostcell(ccor, dist, size):
#        #gc_upstream
#        bottom = 0 - 0.5*dist
#        gc_up = 0
#        dist_g = 0
#        while dist_g>bottom:
#            gc_up += 1
#            #Check that number of ghost cells does not become too large
#            if gc_up > len(ccor):
#                raise RuntimeError("Specified coarse cell too big: needed number of ghostcells exceeds number of grid points.")
#            dist_g = ccor[-gc_up] - size
#
#        #gc_downstream
#        top = size + 0.5*dist
#        gc_down = 0
#        dist_g = 0
#        while dist_g<top:
#             dist_g = size + ccor[gc_down]
#             gc_down += 1 #Appending ccor[0] to ccor is in fact adding 1 ghostcell, for this reason at least 1 is always to gc_down
#             #Check that number of ghost cells does not become too large
#             if gc_down > (len(ccor)-1):
#                raise RuntimeError("Specified coarse cell too big: needed number of ghostcells exceeds number of grid points.")
#
#        #Add maximum number of ghostcells needed to both sides (i.e. keep added ghostcells equal to both sides)
#        gc = max(gc_up, gc_down)
#
#        return gc
#
#    #y-direction
#    if bool_edge_gridcell[1]:
#        ycor = finegrid['grid']['y'][:-1] #Remove sample at downstream boundary since it is an already implemented ghost cell
#        jgc  = _determine_ghostcell(ycor, coarsegrid['grid']['yhdist'], finegrid['grid']['ysize'])
#    else:
#        jgc  = 1 #Only 1 ghostcell needed for interpolation total transport terms, none for downsampling
#
#    #x-direction
#    if bool_edge_gridcell[2]:
#        xcor = finegrid['grid']['x'][:-1] #Remove sample at downstream boundary since it is an already implemented ghost cell
#        igc  = _determine_ghostcell(xcor, coarsegrid['grid']['xhdist'], finegrid['grid']['xsize'])
#    else:
#        igc = 1 #Only 1 ghostcell needed for interpolation total transport terms, none for downsampling
#
#    #Add corresponding amount of ghostcells
#    finegrid.add_ghostcells_hor(variable_name, jgc, igc, bool_edge_gridcell)
#
#    return finegrid


###############################################
#Actual functions to generate the training data
###############################################

def generate_training_data(dim_new_grid, precision = 'double', fourth_order = False, periodic_bc = (False, True, True), name_output_file = 'training_data.nc', create_file = True, testing = False): #Filenames should be strings. Default input corresponds to names files from MicroHH and the provided scripts
 
    #Define flag to ensure variables are only created once in netCDF file
    create_variables = True
    
    #Initialize finegrid object
    if testing:
        finegrid = Finegrid(read_grid_flag = False, precision = precision, fourth_order = fourth_order, periodic_bc = periodic_bc, coordx = np.array([1,2.5,3]), xsize = 4, coordy = np.array([0.5,1.5,3.5,6]), ysize = 7, coordz = np.array([0.1,0.3,1]), zsize = 1.2)
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
            finegrid.create_variables('u', np.reshape(np.arange(0,36), (3,4,3)), bool_edge_gridcell_u)
            finegrid.create_variables('v', np.reshape(np.arange(0,36), (3,4,3)), bool_edge_gridcell_v)
            finegrid.create_variables('w', np.reshape(np.arange(0,36), (3,4,3)), bool_edge_gridcell_w)
            finegrid.create_variables('p', np.reshape(np.arange(0,36), (3,4,3)), bool_edge_gridcell_p)
            
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
        
        break #For now, do not yet execute code below but break out of for-loop
 
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
 
            #define variables
            zghost, yghost, xghost = coord_variable_ghost
            zbound, ybound, xbound = coord_boundary
 
            #Interpolate to boundary
            z_int = np.ravel(np.broadcast_to(zbound[:,np.newaxis,np.newaxis],(len(zbound),len(ybound),len(xbound))))
            y_int = np.ravel(np.broadcast_to(ybound[np.newaxis,:,np.newaxis],(len(zbound),len(ybound),len(xbound))))
            x_int = np.ravel(np.broadcast_to(xbound[np.newaxis,np.newaxis,:],(len(zbound),len(ybound),len(xbound))))
 
            interpolator = scipy.interpolate.RegularGridInterpolator((zghost, yghost, xghost), variable, method = 'linear', bounds_error = False, fill_value = 0.) #Make sure that w is equal to 0 at the top and bottom boundary (where extrapolation is needed because no ghost cells are defined).
            interpolator_value = interpolator((z_int, y_int ,x_int))
            var_int = np.reshape(interpolator_value,(len(zbound),len(ybound),len(xbound)))
 
            return var_int

        #xz-boundary
        u_xzint = _interpolate_side_cell(finegrid['ghost']['u']['variable'], (finegrid['ghost']['u']['z'],  finegrid['ghost']['u']['y'],  finegrid['ghost']['u']['xh']), (finegrid['grid']['z'],    coarsegrid['grid']['yh'], finegrid['grid']['x']))
        v_xzint = _interpolate_side_cell(finegrid['ghost']['v']['variable'], (finegrid['ghost']['v']['z'],  finegrid['ghost']['v']['yh'], finegrid['ghost']['v']['x']),  (finegrid['grid']['z'],    coarsegrid['grid']['yh'], finegrid['grid']['x']))
        w_xzint = _interpolate_side_cell(finegrid['ghost']['w']['variable'], (finegrid['ghost']['w']['zh'], finegrid['ghost']['w']['y'],  finegrid['ghost']['w']['x']),  (finegrid['grid']['z'],    coarsegrid['grid']['yh'], finegrid['grid']['x']))
 
        #yz-boundary
        u_yzint = _interpolate_side_cell(finegrid['ghost']['u']['variable'], (finegrid['ghost']['u']['z'],  finegrid['ghost']['u']['y'],  finegrid['ghost']['u']['xh']), (finegrid['grid']['z'],    finegrid['grid']['y'],    coarsegrid['grid']['xh']))
        v_yzint = _interpolate_side_cell(finegrid['ghost']['v']['variable'], (finegrid['ghost']['v']['z'],  finegrid['ghost']['v']['yh'], finegrid['ghost']['v']['x']),  (finegrid['grid']['z'],    finegrid['grid']['y'],    coarsegrid['grid']['xh']))
        w_yzint = _interpolate_side_cell(finegrid['ghost']['w']['variable'], (finegrid['ghost']['w']['zh'], finegrid['ghost']['w']['y'],  finegrid['ghost']['w']['x']),  (finegrid['grid']['z'],    finegrid['grid']['y'],    coarsegrid['grid']['xh']))
 
        #xy-boundary
        u_xyint = _interpolate_side_cell(finegrid['ghost']['u']['variable'], (finegrid['ghost']['u']['z'],  finegrid['ghost']['u']['y'],  finegrid['ghost']['u']['xh']), (coarsegrid['grid']['zh'], finegrid['grid']['y'],    finegrid['grid']['x']))
        v_xyint = _interpolate_side_cell(finegrid['ghost']['v']['variable'], (finegrid['ghost']['v']['z'],  finegrid['ghost']['v']['yh'], finegrid['ghost']['v']['x']),  (coarsegrid['grid']['zh'], finegrid['grid']['y'],    finegrid['grid']['x']))
        w_xyint = _interpolate_side_cell(finegrid['ghost']['w']['variable'], (finegrid['ghost']['w']['zh'], finegrid['ghost']['w']['y'],  finegrid['ghost']['w']['x']),  (coarsegrid['grid']['zh'], finegrid['grid']['y'],    finegrid['grid']['x']))

        #Calculate TOTAL transport of momentum over xz-, yz-, and xy-boundary. 
        #Note: only centercell functions needed because the boundaries are always located on the grid centers along the two directions over which the values have to be integrated
        #Note: at len+1 iteration for any given coordinate, only weights have to be known for other two coordinates. Furthermore, only part of the total transport terms need to be calculated (i.e. the terms located on the grid side boundaries for the coordinate in the len+1 iteration).
        for izc in range(len(coarsegrid['grid']['z'])+1):
            if izc != len(coarsegrid['grid']['z']):
                zcor_c_middle = coarsegrid['grid']['z'][izc]
                weights_z, points_indices_z = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['zh'], cor_c_middle = zcor_c_middle, dist_corc = coarsegrid['grid']['zdist'])
 
            for iyc in range(len(coarsegrid['grid']['y'])+1):
                if iyc != len(coarsegrid['grid']['y']):
                    ycor_c_middle = coarsegrid['grid']['y'][iyc]
                    weights_y, points_indices_y = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['yh'], cor_c_middle = ycor_c_middle, dist_corc = coarsegrid['grid']['ydist'])

                for ixc in range(len(coarsegrid['grid']['x'])+1):
                    if ixc != len(coarsegrid['grid']['x']):
                        xcor_c_middle = coarsegrid['grid']['x'][ixc]
                        weights_x, points_indices_x = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['xh'], cor_c_middle = xcor_c_middle, dist_corc = coarsegrid['grid']['xdist'])
 
                    #xz-boundary
                    if (izc != len(coarsegrid['grid']['z'])) and (ixc != len(coarsegrid['grid']['x'])): #Make sure this not evaluated for the len+1 iteration in the z- and x-coordinates.
                        weights_xz = weights_x[np.newaxis,:]*weights_z[:,np.newaxis]
                        total_tau_yu[izc,iyc,ixc] = np.sum(weights_xz * v_xzint[:,iyc,:][points_indices_z,:][:,points_indices_x] * u_xzint[:,iyc,:][points_indices_z,:][:,points_indices_x])
                        total_tau_yv[izc,iyc,ixc] = np.sum(weights_xz * v_xzint[:,iyc,:][points_indices_z,:][:,points_indices_x] ** 2)
                        total_tau_yw[izc,iyc,ixc] = np.sum(weights_xz * v_xzint[:,iyc,:][points_indices_z,:][:,points_indices_x] * w_xzint[:,iyc,:][points_indices_z,:][:,points_indices_x])
 
                    #yz-boundary
                    if (izc != len(coarsegrid['grid']['z'])) and (iyc != len(coarsegrid['grid']['y'])): #Make sure this not evaluated for the len+1 iteration in the z- and y-coordinates.
                        weights_yz = weights_y[np.newaxis,:]*weights_z[:,np.newaxis]
                        total_tau_xu[izc,iyc,ixc] = np.sum(weights_yz * u_yzint[:,:,ixc][points_indices_z,:][:,points_indices_y] ** 2)
                        total_tau_xv[izc,iyc,ixc] = np.sum(weights_yz * u_yzint[:,:,ixc][points_indices_z,:][:,points_indices_y] * v_yzint[:,:,ixc][points_indices_z,:][:,points_indices_y])
                        total_tau_xw[izc,iyc,ixc] = np.sum(weights_yz * u_yzint[:,:,ixc][points_indices_z,:][:,points_indices_y] * w_yzint[:,:,ixc][points_indices_z,:][:,points_indices_y])
 
                    #xy-boundary
                    if (iyc != len(coarsegrid['grid']['y'])) and (ixc != len(coarsegrid['grid']['x'])): #Make sure this not evaluated for the len+1 iteration in the y- and x-coordinates.
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

        #Add ghostcells to wind velocities on coarse grid, define short-hand notations
        coarsegrid.add_ghostcells_hor('u', jgc=1, igc=1, bool_edge_gridcell = (False, False, True))
        coarsegrid.add_ghostcells_hor('v', jgc=1, igc=1, bool_edge_gridcell = (False, True, False))
        coarsegrid.add_ghostcells_hor('w', jgc=1, igc=1, bool_edge_gridcell = (True, False, False))
        #coarsegrid.add_ghostcells_hor('p', jgc=1, igc=1, bool_edge_gridcell = (False, False, False))

        #Calculate RESOLVED and UNRESOLVED transport
        
        #xz-boundary
        uc_xzint = _interpolate_side_cell(coarsegrid['ghost']['u']['variable'], (coarsegrid['ghost']['u']['z'], coarsegrid['ghost']['u']['y'], coarsegrid['ghost']['u']['xh']), (coarsegrid['grid']['z'], coarsegrid['grid']['yh'], coarsegrid['grid']['x']))
        vc_xzint = _interpolate_side_cell(coarsegrid['ghost']['v']['variable'], (coarsegrid['ghost']['v']['z'], coarsegrid['ghost']['v']['yh'], coarsegrid['ghost']['v']['x']), (coarsegrid['grid']['z'], coarsegrid['grid']['yh'], coarsegrid['grid']['x']))
        wc_xzint = _interpolate_side_cell(coarsegrid['ghost']['w']['variable'], (coarsegrid['ghost']['w']['zh'], coarsegrid['ghost']['w']['y'], coarsegrid['ghost']['w']['x']), (coarsegrid['grid']['z'], coarsegrid['grid']['yh'], coarsegrid['grid']['x']))

        res_tau_yu = vc_xzint * uc_xzint
        res_tau_yv = vc_xzint ** 2
        res_tau_yw = vc_xzint * wc_xzint
 
        unres_tau_yu = total_tau_yu - res_tau_yu
        unres_tau_yv = total_tau_yv - res_tau_yv
        unres_tau_yw = total_tau_yw - res_tau_yw

        #yz-boundary
        uc_yzint = _interpolate_side_cell(coarsegrid['ghost']['u']['variable'], (coarsegrid['ghost']['u']['z'], coarsegrid['ghost']['u']['y'], coarsegrid['ghost']['u']['xh']), (coarsegrid['grid']['z'], coarsegrid['grid']['y'], coarsegrid['grid']['xh']))
        vc_yzint = _interpolate_side_cell(coarsegrid['ghost']['v']['variable'], (coarsegrid['ghost']['v']['z'], coarsegrid['ghost']['v']['yh'], coarsegrid['ghost']['v']['x']), (coarsegrid['grid']['z'], coarsegrid['grid']['y'], coarsegrid['grid']['xh']))
        wc_yzint = _interpolate_side_cell(coarsegrid['ghost']['w']['variable'], (coarsegrid['ghost']['w']['zh'], coarsegrid['ghost']['w']['y'], coarsegrid['ghost']['w']['x']), (coarsegrid['grid']['z'], coarsegrid['grid']['y'], coarsegrid['grid']['xh']))

        res_tau_xu = uc_yzint ** 2
        res_tau_xv = uc_yzint * vc_yzint
        res_tau_xw = uc_yzint * wc_yzint

        unres_tau_xu = total_tau_xu - res_tau_xu
        unres_tau_xv = total_tau_xv - res_tau_xv
        unres_tau_xw = total_tau_xw - res_tau_xw

        #xy-boundary
        uc_xyint = _interpolate_side_cell(coarsegrid['ghost']['u']['variable'], (coarsegrid['ghost']['u']['z'], coarsegrid['ghost']['u']['y'], coarsegrid['ghost']['u']['xh']), (coarsegrid['grid']['zh'], coarsegrid['grid']['y'], coarsegrid['grid']['x']))
        vc_xyint = _interpolate_side_cell(coarsegrid['ghost']['v']['variable'], (coarsegrid['ghost']['v']['z'], coarsegrid['ghost']['v']['yh'], coarsegrid['ghost']['v']['x']), (coarsegrid['grid']['zh'], coarsegrid['grid']['y'], coarsegrid['grid']['x']))
        wc_xyint = _interpolate_side_cell(coarsegrid['ghost']['w']['variable'], (coarsegrid['ghost']['w']['zh'], coarsegrid['ghost']['w']['y'], coarsegrid['ghost']['w']['x']), (coarsegrid['grid']['zh'], coarsegrid['grid']['y'], coarsegrid['grid']['x']))

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
 
            var_xhc[:] = coarsegrid['grid']['xh'][:]
            var_xc[:] = coarsegrid['grid']['x'][:]
            var_yhc[:] = coarsegrid['grid']['yh'][:]
            var_yc[:] = coarsegrid['grid']['y'][:]
            var_zhc[:] = coarsegrid['grid']['zh'][:]
            var_zc[:] = coarsegrid['grid']['z'][:]
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
        var_uc[t,:,:,:] = coarsegrid['output']['u'][:,:,:]
        var_vc[t,:,:,:] = coarsegrid['output']['v'][:,:,:]
        var_wc[t,:,:,:] = coarsegrid['output']['w'][:,:,:]
        var_pc[t,:,:,:] = coarsegrid['output']['p'][:,:,:]
 
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

