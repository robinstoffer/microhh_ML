#Developed for Python 3!
import numpy   as np
import netCDF4 as nc
import struct as st
import glob
import re
import matplotlib as mpl
mpl.use('Agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
import matplotlib.pyplot as plt
import scipy.interpolate
from microhh_tools_robinst import *

#This script generates the training data for a developed NN, 
#which is subsequently sampled and stored in tfrecord-files using sample_training_data_tfrecord.py

##############################################
#Helper functions for generation training data
##############################################
def add_ghostcells_finegrid(finegrid, coarsegrid, variable_name, bool_edge_gridcell):

    #Determine how many ghostcells are needed. NOTE: should be based on the same (not opposed!) coordinate, 1 point outside coarse grid cell needed for interpolation
    def _determine_ghostcell(ccor, dist, size):
        #gc_upstream
        bottom = 0 - 0.5*dist
        gc_up = 0
        dist = 0
        while dist>bottom:
            gc_up += 1
            dist = ccor[-gc_up] - size

        #gc_downstream
        top = size + 0.5*dist
        gc_down = 0
        dist = 0
        while dist<top:
             dist = size + ccor[gc_down]
             gc_down += 1 #Appending ccor[0] to ccor is in fact adding 1 ghostcell, for this reason at least 1 is always to gc_down

        #Add maximum number of ghostcells needed to both sides (i.e. keep added ghostcells equal to both sides)
        gc = max(gc_up, gc_down)

        return gc

    #y-direction
    if bool_edge_gridcell[1]:
        ycor = finegrid['grid']['yh'][:-1] #Remove sample at downstream boundary since it is an already implemented ghost cell
        jgc  = _determine_ghostcell(ycor, coarsegrid['grid']['ydist'], coarsegrid['grid']['ysize'])
    else:
        jgc  = 1 #Only 1 ghostcell needed for interpolation total transport terms, none for downsampling

    #x-direction
    if bool_edge_gridcell[2]:
        xcor = finegrid['grid']['xh'][:-1] #Remove sample at downstream boundary since it is an already implemented ghost cell
        igc  = _determine_ghostcell(xcor, coarsegrid['grid']['xdist'], coarsegrid['grid']['xsize'])
    else:
        igc = 1 #Only 1 ghostcell needed for interpolation total transport terms, none for downsampling

    #Add corresponding amount of ghostcells
    finegrid.add_ghostcells_hor(variable_name, jgc, igc, bool_edge_gridcell)

    return finegrid

def generate_coarsecoord_centercell(cor_edges, cor_c_middle, dist_corc,  vert_flag = False, size = 0): #For 'size' a default value is used that should not affect results as long as vert_flag = False.
    cor_c_bottom = cor_c_middle - 0.5*dist_corc
    cor_c_top = cor_c_middle + 0.5*dist_corc
    if vert_flag and (cor_c_bottom<0):
        cor_c_bottom = 0
    if vert_flag and (cor_c_top>size):
        cor_c_top = size
    
    #Find points of fine grid located just outside the coarse grid cell considered in iteration
    cor_bottom = cor_edges[cor_edges <= cor_c_bottom].max()
    #cor_top = cor_c_top if iteration == (len_coordinate - 1) else cor_edges[cor_edges >= cor_c_top].min()
    cor_top = cor_edges[cor_edges >= cor_c_top].min()    

    #Select all points inside and just outside coarse grid cell
    points_indices_cor = np.where(np.logical_and(cor_bottom <= cor_edges , cor_edges < cor_top))[0]
    cor_points = cor_edges[points_indices_cor] #Note: cor_points includes the bottom boundary (cor_bottom), but not the top boundary (cor_top).
    
    #Calculate weights for cor_points. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
    weights = np.zeros(len(points_indices_cor))
    if len(points_indices_cor) == 1:
        weights = np.array([1])

    else:
        for i in range(len(points_indices_cor):
            if i == 0:
                weights[0] = (cor_points[1] - cor_c_bottom)/(cor_c_top - cor_c_bottom)
            elif i == (len(points_indices_cor) - 1):
                weights[i] = (cor_c_top - cor_points[i])/(cor_c_top - cor_c_bottom)
            else:
                weights[i] = (cor_points[i+1] - cor_points[i])/(cor_c_top - cor_c_bottom)

#    #Select two additional points just outside coarse grid cell, which are needed for interpolation (in the total transport calculation) but not for calculating representative velocities.
#    #Consequently, the weights are set to 0.
#    points_indices_cor = np.insert(points_indices_cor, 0, points_indices_cor[0] - 1)
#    points_indices_cor = np.append(points_indices_cor, point_indices_cor[-1] + 1)
#    weights = np.insert(weights, 0, 0)
#    weights = np.append(weights, 0)

    return weights, points_indices_cor

def generate_coarsecoord_edgecell(cor_center, cor_c_middle, dist_corc, vert_flag = False, size = 0): #For 'size' a default value is used that should not affect results as long as vert_flag = False.
    cor_c_bottom = cor_c_middle - 0.5*dist_corc
    cor_c_top = cor_c_middle + 0.5*dist_corc
    if vert_flag and (cor_c_bottom<0):
        cor_c_bottom = 0
    if vert_flag and (cor_c_top>size):
        cor_c_top = size
    
    #Find points of fine grid located just outside the coarse grid cell considered in iteration
    cor_bottom = cor_center[cor_center <= cor_c_bottom].max()
    cor_top = cor_center[cor_center >= cor_c_top].min()
    
    #Select all points inside and just outside coarse grid cell
    points_indices_cor = np.where(np.logical_and(cor_bottom<cor_center , cor_center<=cor_top))[0]
    cor_points = cor_center[points_indices_cor] #Note: cor_points includes the bottom boundary (cor_bottom), but not the top boundary (cor_top).
	  
    #Calculate weights for cor_points. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
    weights = np.zeros(len(points_indices_cor))
    if len(points_indices_cor) == 1:
        weights = np.array([1])

    else:
        for i in range(len(points_indices_cor):
            if i == 0:
                weights[0] = (cor_points[0] - cor_c_bottom)/(cor_c_top - cor_c_bottom)
            elif i == (len(points_indices_cor) - 1):
                weights[i] = (cor_c_top - cor_points[i-1])/(cor_c_top - cor_c_bottom)
            else:
                weights[i] = (cor_points[i] - cor_points[i-1])/(cor_c_top - cor_c_bottom)

#    #Select one additional point just upstream/down of coarse grid cell, which is needed for interpolation (in the total transport calculation) but not for calculating representative velocities.
#    #Consequently, the weight of this point set to 0. Note: interpolation not required at bottom boundary, since boundaries of coarse and fine grid exactly coincide there. Since no ghostcells are added in the vertical direction, this would be even impossible (i.e. giving an error message).
#    if not (vert_flag and (cor_c_bottom == 0)):
#        points_indices_cor = np.insert(points_indices_cor, 0, points_indices_cor[0] - 1)
#        weights = np.insert(weights, 0, 0)
#   # if not (vert_flag and (cor_c_top == size)):
#   #     points_indices_cor = np.append(points_indices_cor, point_indices_cor[-1] + 1)
#   #     weights = np.append(weights, 0)
    
    return weights, points_indices_cor

def generate_coarse_data(finegrid, coarsegrid, variable_name, timestep, bool_edge_gridcell = (False,False,False)):
    """Function to generate coarse grid with variables and total transport of momentum for creation training data. Returns the specified variable on the coarse grid, together with the corresponding weights and coarse coordinates.
    Variable_name specifies the variable to calculate on the coarse grid.
    Timestep is the time step that will be read from the fine resolution data.
    Coordinates should contain a tuple with the three spatial dimensions from the fine resolution (x,y,z).
    Len_coordinates should contain a tuple indicating the spatial distance for each coordinate (x,y,z).
    Edge coordinates should contain a tuple with the coordinates that form the edges of the top-hat filter applied for the variable specified by variable_name.
    Bool_edge_gridcell indicates for each coordinate (x,y,z) whether they should be aligned at the center of the grid cells (False) or the edges (True). """

    #Read in the right coarse coordinates determined by bool_edge_gridcell.
    #z-direction
    if bool_edge_gridcell[0]:
        zcor_c     = coarsegrid['grid']['zh']
        dist_zc    = coarsegrid['grid']['zhdist']
    else:
        zcor_c     = coarsegrid['grid']['z']
        dist_zc    = coarsegrid['grid']['zdist']

    #y-direction
    if bool_edge_gridcell[1]:
        ycor_c     = coarsegrid['grid']['yh']
        dist_yc    = coarsegrid['grid']['yhdist']
    else:
        ycor_c     = coarsegrid['grid']['y']
        dist_yc    = coarsegrid['grid']['ydist']

    #x-direction
    if bool_edge_gridcell[2]:
        xcor_c     = coarsegrid['grid']['xh']
        dist_xc    = coarsegrid['grid']['xhdist']
    else:
        xcor_c     = coarsegrid['grid']['x']
        dist_xc    = coarsegrid['grid']['xdist']

    if variable_name not in self.var['output'].keys(): #Read only when variable_name is not defined. Allows for testing with previously defined arrays.
        finegrid.read_binary_variables(variable_name, timestep, bool_edge_gridcell)

    var_c = np.zeros(len(zcor_c), len(ycor_c), len(xcor_c), dtype=float)
    #weights_c = np.zeros(len(zcor_c), len(ycor_c), len(xcor_c), dtype=(object, object, object))
    #points_indices_c = np.zeros(len(zcor_c), len(ycor_c), len(xcor_c), dtype=(object, object, object))

    #Add needed ghostcells to finegrid object for the downsampling and calculation total transport
    finegrid = add_ghostcells_finegrid(finegrid, coarsegrid, variable_name, bool_edge_gridcell)

    #Loop over coordinates for downsampling and calculation total transport
    izc = 0
    for zcor_c_middle in zcor_c:
    #for izc in range(coarsegrid['grid']['ktot'])
        if bool_edge_gridcell[0]:
            weights_z, points_indices_z = generate_coarsecoord_edgecell(cor_center = finegrid['ghost'][variable_name]['z'], cor_c_middle = zcor_c_middle, dist_corc = dist_zc[izc], vert_flag = True, size = finegrid['grid']['zsize'])
        else:
            weights_z, points_indices_z = generate_coarsecoord_centercell(cor_edges = finegrid['ghost'][variable_name]['zh'], cor_c_middle = zcor_c_middle, dist_corc = dist_zc[izc], vert_flag = True, size = finegrid['grid']['zsize'])

        var_finez = finegrid['ghost'][variable_name]['variable'][points_indices_z,:,:].copy()

        iyc = 0
	
        for ycor_c_middle in ycor_c:
            if bool_edge_gridcell[1]:
                weights_y, points_indices_y = generate_coarsecoord_edgecell(cor_center = finegrid['ghost'][variable_name]['y'], cor_c_middle = ycor_c_middle, dist_corc = dist_yc[iyc])
            else:
                weights_y, points_indices_y = generate_coarsecoord_centercell(cor_edges = finegrid['ghost'][variable_name]['yh'], cor_c_middle = ycor_c_middle, dist_corc = dist_yc[iyc])
   
            var_finezy = var_finez[:,points_indices_y,:].copy()

            ixc = 0
				
            for xcor_c_middle in xcor_c:
                if bool_edge_gridcell[2]:
                    weights_x, points_indices_x = generate_coarsecoord_edgecell(cor_center = finegrid['ghost'][variable_name]['x'], cor_c_middle = xcor_c_middle, dist_corc = dist_xc[ixc])
                else:
                    weights_x, points_indices_x = generate_coarsecoord_centercell(cor_edges = finegrid['ghost'][variable_name]['xh'], cor_c_middle = xcor_c_middle, dist_corc = dist_xc[ixc])

                var_finezyx = var_finezy[:,:,points_indices_x].copy()

                #Calculate downsampled variable on coarse grid using the selected points in var_finezyx and the fractions defined in the weights variables
                weights =  weights_x[np.newaxis,np.newaxis,:]*weights_y[np.newaxis,:,np.newaxis]*weights_z[:,np.newaxis,np.newaxis]
                var_c[izc,iyc,ixc] = np.sum(np.multiply(weights, var_finezyx)
 
                #weights_c[izc,iyc,ixc] = (weights_z,weights_y,weights_x)
                #points_indices_c[izc,iyc,ixc] = (points_indices_z,points_indices_y,points_indices_x)

                ixc += 1
            iyc += 1
        izc += 1
    
    #Store downsampled variable in coarsegrid object
    coarsegrid['output'][variable_name] = var_c
 
    return finegrid, coarsegrid
    
	
###############################################
#Actual functions to generate the training data
###############################################

def generate_training_data(finegrid, coarsegrid, name_output_file = 'training_data.nc'): #Filenames should be strings. Default input corresponds to names files from MicroHH and the provided scripts
 
    create_variables = True
    create_file = True
 
    #Loop over timesteps
    for t in range(finegrid['grid']['timesteps']): #Only works correctly in this script when whole simulation is saved with a constant time interval
 
        ##Downsampling from fine DNS data to user specified coarse grid and calculation total transport momentum ##
        ###########################################################################################################
 
        #Calculate representative velocities for each coarse grid cell
        finegrid, coarsegrid = generate_coarse_data(finegrid, coarsegrid, variable_name = 'u', timestep = t, bool_edge_gridcell = (False, False, True))
        finegrid, coarsegrid = generate_coarse_data(finegrid, coarsegrid, variable_name = 'v', timestep = t, bool_edge_gridcell = (False, True, False))
        finegrid, coarsegrid = generate_coarse_data(finegrid, coarsegrid, variable_name = 'w', timestep = t, bool_edge_gridcell = (True, False, False))
        finegrid, coarsegrid = generate_coarse_data(finegrid, coarsegrid, variable_name = 'p', timestep = t, bool_edge_gridcell = (False, False, False))
 
        #Calculate total transport on coarse grid from fine grid, initialize first arrays
        total_tau_xu = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']), dtype=float)
        total_tau_yu = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']), dtype=float)
        total_tau_zu = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']), dtype=float)
        total_tau_xv = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']), dtype=float)
        total_tau_yv = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']), dtype=float)
        total_tau_zv = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']), dtype=float)
        total_tau_xw = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']), dtype=float)
        total_tau_yw = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']), dtype=float)
        total_tau_zw = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']), dtype=float)
 
        #Interpolate to side boundaries coarse gridcell
        def _interpolate_side_cell(variable_name, coord_variable_ghost, coord_boundary):
 
            #define variables
            zghost, yghost, xghost = coord_variable_ghost
            zbound, ybound, xbound = coord_boundary
 
            #Interpolate to boundary
            z_int = np.ravel(np.broadcast_to(zbound[:,np.newaxis,np.newaxis],(len(zbound),len(ybound),len(xbound))))
            y_int = np.ravel(np.broadcast_to(ybound[np.newaxis,:,np.newaxis],(len(zbound),len(ybound),len(xbound))))
            x_int = np.ravel(np.broadcast_to(xbound[np.newaxis,np.newaxis,:],(len(zbound),len(ybound,len(xbound))))
 
            var_int = np.reshape(scipy.interpolate.RegularGridInterpolator((zghost, yghost, xghost), finegrid['ghost'][variable_name]['variable'], method='linear')((z_int, y_int ,x_int)),(len(z_int),len(y_int),len(x_int)))
 
            return var_int

        #xz-boundary
        u_xzint = _interpolate_side_cell('u', (finegrid['ghost']['u']['z'], finegrid['ghost']['u']['y'], finegrid['ghost']['u']['xh']), (finegrid['grid']['z'], coarsegrid['grid']['yh'], finegrid['grid']['x']))
        v_xzint = _interpolate_side_cell('v', (finegrid['ghost']['v']['z'], finegrid['ghost']['v']['yh'], finegrid['ghost']['v']['x']), (finegrid['grid']['z'], coarsegrid['grid']['yh'], finegrid['grid']['x']))
        w_xzint = _interpolate_side_cell('w', (finegrid['ghost']['w']['zh'], finegrid['ghost']['w']['y'], finegrid['ghost']['w']['x']), (finegrid['grid']['z'], coarsegrid['grid']['yh'], finegrid['grid']['x']))
 
        #yz-boundary
        u_yzint = _interpolate_side_cell('u', (finegrid['ghost']['u']['z'], finegrid['ghost']['u']['y'], finegrid['ghost']['u']['xh']), (finegrid['grid']['z'], finegrid['grid']['y'], coarsegrid['grid']['xh']))
        v_yzint = _interpolate_side_cell('v', (finegrid['ghost']['v']['z'], finegrid['ghost']['v']['yh'], finegrid['ghost']['v']['x']), (finegrid['grid']['z'], finegrid['grid']['y'], coarsegrid['grid']['xh']))
        w_yzint = _interpolate_side_cell('w', (finegrid['ghost']['w']['zh'], finegrid['ghost']['w']['y'], finegrid['ghost']['w']['x']), (finegrid['grid']['z'], finegrid['grid']['y'], coarsegrid['grid']['xh']))
 
        #xy-boundary
        u_xyint = _interpolate_side_cell('u', (finegrid['ghost']['u']['z'], finegrid['ghost']['u']['y'], finegrid['ghost']['u']['xh']), (coarsegrid['grid']['zh'], finegrid['grid']['y'], finegrid['grid']['x']))
        v_xyint = _interpolate_side_cell('v', (finegrid['ghost']['v']['z'], finegrid['ghost']['v']['yh'], finegrid['ghost']['v']['x']), (coarsegrid['grid']['zh'], finegrid['grid']['y'], finegrid['grid']['x']))
        w_xyint = _interpolate_side_cell('w', (finegrid['ghost']['w']['zh'], finegrid['ghost']['w']['y'], finegrid['ghost']['w']['x']), (coarsegrid['grid']['zh'], finegrid['grid']['y'], finegrid['grid']['x']))

        #Calculate TOTAL transport of momentum over xz-, yz-, and xy-boundary. 
        #Note: only centercell functions needed because the boundaries are always located on the grid centers along the two directions over which the values have to be integrated
        izc = 0
        for zcor_c_middle in coarsegrid['grid']['z']:
            weights_z, points_indices_z = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['zh'], cor_c_middle = zcor_c_middle, dist_corc = coarsegrid['zdist'][izc])
 
            iyc = 0
            for ycor_c_middle in coarsegrid['grid']['y']:
                weights_y, points_indices_y = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['yh'], cor_c_middle = ycor_c_middle, dist_corc = coarsegrid['ydist'][iyc])
 
                ixc = 0
                for xcor_c_middle in coarsegrid['grid']['x']:
                    weights_x, points_indices_x = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['xh'], cor_c_middle = xcor_c_middle, dist_corc = coarsegrid['xdist'][ixc])
 
                    #xz-boundary
                    weights_xz = weights_x[np.newaxis,;]*weights_z[:,np.newaxis]
                    total_transport_yu[izc,iyc,ixc] = np.sum(weights_xz * v_xzint[:,iyc,:][points_indices_z,:,:][:,:,points_indices_x] * u_xzint[:,iyc,:][points_indices_z,:,:][:,:,points_indices_x])
                    total_transport_yv[izc,iyc,ixc] = np.sum(weights_xz * v_xzint[:,iyc,:][points_indices_z,:,:][:,:,points_indices_x] ** 2)
                    total_transport_yw[izc,iyc,ixc] = np.sum(weights_xz * v_xzint[:,iyc,:][points_indices_z,:,:][:,:,points_indices_x] * w_xzint[:,iyc,:][points_indices_z,:,:][:,:,points_indices_x])
 
                    #yz-boundary
                    weights_yz = weights_y[np.newaxis,:]*weights_z[:,np.newaxis]
                    total_transport_xu[izc,iyc,ixc] = np.sum(weights_yz * u_yzint[:,:,ixc][points_indices_z,:,:][:,points_indices_y,:] ** 2)
                    total_transport_xv[izc,iyc,ixc] = np.sum(weights_yz * u_yzint[:,:,ixc][points_indices_z,:,:][:,points_indices_y,:] * v_yzint[:,:,ixc][points_indices_z,:,:][:,points_indices_y,:])
                    total_transport_xw[izc,iyc,ixc] = np.sum(weights_yz * u_yzint[:,:,ixc][points_indices_z,:,:][:,points_indices_y,:] * w_yzint[:,:,ixc][points_indices_z,:,:][:,points_indices_y,:])
 
                    #xy-boundary
                    weights_xy = weights_x[np.newaxis,:]*weights_y[:,np.newaxis]
                    total_transport_zu[izc,iyc,ixc] = np.sum(weights_xy * w_xyint[izc,:,:][:,points_indices_y,:][:,:,points_indices_x] * u_xyint[izc,:,:][:,points_indices_y,:][:,:,points_indices_x])
                    total_transport_zv[izc,iyc,ixc] = np.sum(weights_xy * w_xyint[izc,:,:][:,points_indices_y,:][:,:,points_indices_x] * v_xyint[izc,:,:][:,points_indices_y,:][:,:,points_indices_x])
                    total_transport_zw[izc,iyc,ixc] = np.sum(weights_xy * w_xyint[izc,:,:][:,points_indices_y,:][:,:,points_indices_x] ** 2)
 
                    ixc +=1
                iyc += 1
            izc += 1
 
 
        ##Calculate resolved and unresolved transport user specified coarse grid ##
        ###########################################################################
 
        #Define empty variables for storage
 
        res_tau_xu = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        res_tau_yu = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        res_tau_zu = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        res_tau_xv = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        res_tau_yv = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        res_tau_zv = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        res_tau_xw = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        res_tau_yw = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        res_tau_zw = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
 
        unres_tau_xu = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        unres_tau_yu = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        unres_tau_zu = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        unres_tau_xv = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        unres_tau_yv = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        unres_tau_zv = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        unres_tau_xw = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        unres_tau_yw = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
        unres_tau_zw = np.zeros((coarsegrid['grid']['ktot'], coarsegrid['grid']['jtot'], coarsegrid['grid']['itot']),dtype=float)
	
        #Add ghostcells to wind velocities on coarse grid, define short-hand notations
        coarsegrid.add_ghostcells_hor('u', jgc=1, igc=1, bool_edge_gridcell = (False, False, True))
        coarsegrid.add_ghostcells_hor('v', jgc=1, igc=1, bool_edge_gridcell = (False, True, False))
        coarsegrid.add_ghostcells_hor('w', jgc=1, igc=1, bool_edge_gridcell = (True, False, False))
        #coarsegrid.add_ghostcells_hor('p', jgc=1, igc=1, bool_edge_gridcell = (False, False, False))

        #Calculate RESOLVED and UNRESOLVED transport
        
        #xz-boundary
        uc_xzint = _interpolate_side_cell('u', (coarsegrid['ghost']['u']['z'], coarsegrid['ghost']['u']['y'], coarsegrid['ghost']['u']['xh']), (coarsegrid['grid']['z'], coarsegrid['grid']['yh'], coarsegrid['grid']['x']))
        vc_xzint = _interpolate_side_cell('v', (coarsegrid['ghost']['v']['z'], coarsegrid['ghost']['v']['yh'], coarsegrid['ghost']['v']['x']), (coarsegrid['grid']['z'], coarsegrid['grid']['yh'], coarsegrid['grid']['x']))
        wc_xzint = _interpolate_side_cell('w', (coarsegrid['ghost']['w']['zh'], coarsegrid['ghost']['w']['y'], coarsegrid['ghost']['w']['x']), (coarsegrid['grid']['z'], coarsegrid['grid']['yh'], coarsegrid['grid']['x']))

        res_tau_yu = vc_xzint * uc_xzint
        res_tau_yv = vc_xzint ** 2
        res_tau_yw = vc_xzint * wc_xzint
 
        unres_tau_yu = total_tau_yu - res_tau_yu
        unres_tau_yv = total_tau_yv - res_tau_yv
        unres_tau_yw = total_tau_yw - res_tau_yw

        #yz-boundary
        uc_yzint = _interpolate_side_cell('u', (coarsegrid['ghost']['u']['z'], coarsegrid['ghost']['u']['y'], coarsegrid['ghost']['u']['xh']), (coarsegrid['grid']['z'], coarsegrid['grid']['y'], coarsegrid['grid']['xh']))
        vc_yzint = _interpolate_side_cell('v', (coarsegrid['ghost']['v']['z'], coarsegrid['ghost']['v']['yh'], coarsegrid['ghost']['v']['x']), (coarsegrid['grid']['z'], coarsegrid['grid']['y'], coarsegrid['grid']['xh']))
        wc_yzint = _interpolate_side_cell('w', (coarsegrid['ghost']['w']['zh'], coarsegrid['ghost']['w']['y'], coarsegrid['ghost']['w']['x']), (coarsegrid['grid']['z'], coarsegrid['grid']['y'], coarsegrid['grid']['xh']))

        res_tau_xu = uc_yzint ** 2
        res_tau_xv = uc_yzint * vc_yzint
        res_tau_xw = uc_yzint * wc_yzint

        unres_tau_xu = total_tau_xu - res_tau_xu
        unres_tau_xv = total_tau_xv - res_tau_xv
        unres_tau_xw = total_tau_xw - res_tau_xw

        #xy-boundary
        uc_xyint = _interpolate_side_cell('u', (coarsegrid['ghost']['u']['z'], coarsegrid['ghost']['u']['y'], coarsegrid['ghost']['u']['xh']), (coarsegrid['grid']['zh'], coarsegrid['grid']['y'], coarsegrid['grid']['x']))
        vc_xyint = _interpolate_side_cell('v', (coarsegrid['ghost']['v']['z'], coarsegrid['ghost']['v']['yh'], coarsegrid['ghost']['v']['x']), (coarsegrid['grid']['zh'], coarsegrid['grid']['y'], coarsegrid['grid']['x']))
        wc_xyint = _interpolate_side_cell('w', (coarsegrid['ghost']['w']['zh'], coarsegrid['ghost']['w']['y'], coarsegrid['ghost']['w']['x']), (coarsegrid['grid']['zh'], coarsegrid['grid']['y'], coarsegrid['grid']['x']))

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

