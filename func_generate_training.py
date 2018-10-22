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
    #y-direction
    if bool_edge_gridcell[1]:
        ycor = finegrid['grid']['yh']
        #jgc_upstream
        bottom = 0 - 0.5*coarsegrid['grid']['ydist']
        jgc_up = 0
        dist = 0
        while dist>bottom:
            jgc_up += 1
            dist = ycor[-jgc_up] - finegrid['grid']['ysize']

        #jgc_downstream
        top = coarsegrid['grid']['ysize'] + 0.5*coarsegrid['grid']['ydist']
        jgc_down = 0
        dist = 0
        while dist<top:
             jgc_down += 1
             dist = coarsegrid['grid']['ysize'] + ycor[jgc_down - 1] #Appending ycor[0] to ycor is in fact adding 1 ghostcell, for this reason 1 is substracted from jgc_down

        #Add maximum number of ghostcells needed to both sides (i.e. keep added ghostcells equal to both sides)
        jgc = max(jgc_up, jgc_down)

    else:
       jgc = 1 #Only 1 ghostcell needed for interpolation total transport terms, none for downsampling

    #x-direction
    if bool_edge_gridcell[2]:
        xcor = finegrid['grid']['xh'][:-1] #Remove sample at downstream boundary since it is an already implemented ghost cell
        #igc_upstream
        bottom = 0 - 0.5*coarsegrid['grid']['xdist']
        igc_up = 0
        dist = 0
        while dist>bottom:
            igc_up += 1
            dist = xcor[-igc_up] - finegrid['grid']['xsize']

        #igc_downstream
        top = coarsegrid['grid']['xsize'] + 0.5*coarsegrid['grid']['xdist']
        igc_down = 0
        dist = 0
        while dist<top:
             igc_down += 1
             dist = coarsegrid['grid']['xsize'] + xcor[igc_down - 1] #Appending ycor[0] to ycor is in fact adding 1 ghostcell, for this reason 1 is substracted from jgc_down

        #Add maximum number of ghostcells needed to both sides (i.e. keep added ghostcells equal to both sides)
        igc = max(igc_up, igc_down)

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

##############################################
#Actual functions to generate the training data
##############################################

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
        if bool_edge_gridcell[2]:
            weights_z, points_indices_z = generate_coarsecoord_edgecell(cor_center = finegrid['ghost'][variable_name]['z'], cor_c_middle = zcor_c_middle, dist_corc = dist_zc[izc], vert_flag = True, size = finegrid['grid']['zsize'])
        else:
            weights_z, points_indices_z = generate_coarsecoord_centercell(cor_edges = finegrid['ghost'][variable_name]['zh'], cor_c_middle = zcor_c_middle, dist_corc = dist_zc[izc], vert_flag = True, size = finegrid['grid']['zsize'])

        var_finez = finegrid['ghost'][variable_name]['variable'][points_indices_z,:,:].copy()

        izc += 1
        iyc = 0
	
        for ycor_c_middle in ycor_c:
            if bool_edge_gridcell[1]:
                weights_y, points_indices_y = generate_coarsecoord_edgecell(cor_center = finegrid['ghost'][variable_name]['y'], cor_c_middle = ycor_c_middle, dist_corc = dist_yc[iyc])
            else:
                weights_y, points_indices_y = generate_coarsecoord_centercell(cor_edges = finegrid['ghost'][variable_name]['yh'], cor_c_middle = ycor_c_middle, dist_corc = dist_yc[iyc])
   
            var_finezy = var_finez[:,points_indices_y,:].copy()

            iyc += 1
            ixc = 0
				
            for xcor_c_middle in xcor_c:
                if bool_edge_gridcell[0]:
                    weights_x, points_indices_x = generate_coarsecoord_edgecell(cor_center = finegrid['ghost'][variable_name]['x'], cor_c_middle = xcor_c_middle, dist_corc = dist_xc[ixc])
                else:
                    weights_x, points_indices_x = generate_coarsecoord_centercell(cor_edges = finegrid['ghost'][variable_name]['xh'], cor_c_middle = xcor_c_middle, dist_corc = dist_xc[ixc])

                var_finezyx = var_finezy[:,:,points_indices_x].copy()

                ixc += 1
	
                #Calculate downsampled variable on coarse grid using the selected points in var_finezyx and the fractions defined in the weights variables
                weights =  weights_x[np.newaxis,np.newaxis,:]*weights_y[np.newaxis,:,np.newaxis]*weights_z[:,np.newaxis,np.newaxis]
                var_c[izc,iyc,ixc] = np.sum(np.multiply(weights, var_finezyx)
 
                #weights_c[izc,iyc,ixc] = (weights_z,weights_y,weights_x)
                #points_indices_c[izc,iyc,ixc] = (points_indices_z,points_indices_y,points_indices_x)
	    
    return var_c, finegrid, coarsegrid
    
	
#Boundary condition: fine grid must have a smaller resolution than the coarse grid

def generate_training_data(finegrid, coarsegrid, name_output_file = 'training_data.nc', training = True): #Filenames should be strings. Default input corresponds to names files from MicroHH and the provided scripts
    if training:
 
        create_variables = True
        create_file = True

        #Loop over timesteps
        for t in range(finegrid['grid']['timesteps']): #Only works correctly in this script when whole simulation is saved with a constant time interval
    
            ##Downsampling from fine DNS data to user specified coarse grid and calculation total transport momentum ##
            ###########################################################################################################
    
            #Define short-hand notations for the relevant variables and create empty arrays for storage
            nzc = coarsegrid['grid']['ktot']
            nyc = coarsegrid['grid']['jtot']
            nxc = coarsegrid['grid']['itot']
            xc  = coarsegrid['grid']['x']
            xhc = coarsegrid['grid']['xh']
            yc  = coarsegrid['grid']['y']
            yhc = coarsegrid['grid']['yh']
            zc  = coarsegrid['grid']['z']
            zhc = coarsegrid['grid']['zh']

            nz  = finegrid['grid']['ktot']
            ny  = finegrid['grid']['jtot']
            nx  = finegrid['grid']['itot']
            x   = finegrid['grid']['x']
            xh  = finegrid['grid']['xh']
            y   = finegrid['grid']['y']
            yh  = finegrid['grid']['yh']
            z   = finegrid['grid']['z']
            zh  = finegrid['grid']['zh']
            xsize = finegrid['grid']['xsize'] #The size of the coarse grid should be the same
            ysize = finegrid['grid']['ysize']
            zsize = finegrid['grid']['zsize']

            u_c = np.zeros((nzc,nyc,nxc), dtype=float)
            v_c = np.zeros((nzc,nyc,nxc), dtype=float)
            w_c = np.zeros((nzc,nyc,nxc), dtype=float)
            p_c = np.zeros((nzc,nyc,nxc), dtype=float)

            finegrid.read_binary_variables('u', t)
            finegrid.read_binary_variables('v', t)
            finegrid.read_binary_variables('w', t)
            finegrid.read_binary_variables('p', t)
            u   = finegrid['output']['u']
            v   = finegrid['output']['v']
            w   = finegrid['output']['w']
            p   = finegrid['output']['p']

            total_tau_xu = np.zeros((nzc,nyc,nxc), dtype=float)
            total_tau_yu = np.zeros((nzc,nyc,nxc), dtype=float)
            total_tau_zu = np.zeros((nzc,nyc,nxc), dtype=float)
            total_tau_xv = np.zeros((nzc,nyc,nxc), dtype=float)
            total_tau_yv = np.zeros((nzc,nyc,nxc), dtype=float)
            total_tau_zv = np.zeros((nzc,nyc,nxc), dtype=float)
            total_tau_xw = np.zeros((nzc,nyc,nxc), dtype=float)
            total_tau_yw = np.zeros((nzc,nyc,nxc), dtype=float)
            total_tau_zw = np.zeros((nzc,nyc,nxc), dtype=float)
           
            #Calculate representative velocities for each coarse grid cell
            u_c, finegrid, coarsegrid = generate_coarse_data(finegrid, coarsegrid, variable_name = 'u', timestep = t, bool_edge_gridcell = (True, False, False))
            v_c, finegrid, coarsegrid = generate_coarse_data(finegrid, coarsegrid, variable_name = 'v', timestep = t, bool_edge_gridcell = (False, True, False))
            w_c, finegrid, coarsegrid = generate_coarse_data(finegrid, coarsegrid, variable_name = 'w', timestep = t, bool_edge_gridcell = (False, False, True))
            p_c, finegrid, coarsegrid = generate_coarse_data(finegrid, coarsegrid, variable_name = 'p', timestep = t, bool_edge_gridcell = (False, False, False))

            #Calculate representative velocities on coarse grid from fine grid
            for izc in range(nzc):
                for iyc in range(nyc):
                    for ixc in range(nxc):
                        u_finegrid_indexz, u_finegrid_indexy, u_finegrid_indexx = points_indices_u[izc,iyc,ixc]
                        u_finegrid_points = u[u_finegrid_indexz,:,:][:, u_finegrid_indexy, :][:, :, u_finegrid_indexx]
                        u_finegrid_weightsz, u_finegrid_weightsy, u_finegrid_weightsx = weights_u[izc,iyc,ixc]
                        u_finegrid_weights = u_finegrid_weightsx[np.newaxis,np.newaxis,:]*u_finegrid_weightsy[np.newaxis,:,np.newaxis]*u_finegrid_weightsz[:,np.newaxis,np.newaxis]
                        if not ixc == (nxc - 1):
                            u_c[izc,iyc,ixc] = np.sum(np.multiply(u_finegrid_weights, u_finegrid_points))

                        v_finegrid_indexz, v_finegrid_indexy, v_finegrid_indexx = points_indices_v[izc,iyc,ixc]
                        v_finegrid_points = v[v_finegrid_indexz,:,:][:, v_finegrid_indexy, :][:, :, v_finegrid_indexx]
                        v_finegrid_weightsz, v_finegrid_weightsy, v_finegrid_weightsx = weights_v[izc,iyc,ixc]
                        v_finegrid_weights = v_finegrid_weightsx[np.newaxis,np.newaxis,:]*v_finegrid_weightsy[np.newaxis,:,np.newaxis]*v_finegrid_weightsz[:,np.newaxis,np.newaxis]
                        if not iyc == (nyc - 1):
                            v_c[izc,iyc,ixc] = np.sum(np.multiply(v_finegrid_weights, v_finegrid_points))

                        w_finegrid_indexz, w_finegrid_indexy, w_finegrid_indexx = points_indices_w[izc,iyc,ixc]
                        w_finegrid_points = w[w_finegrid_indexz,:,:][:, w_finegrid_indexy, :][:, :, w_finegrid_indexx]
                        w_finegrid_weightsz, w_finegrid_weightsy, w_finegrid_weightsx = weights_w[izc,iyc,ixc]
                        w_finegrid_weights = w_finegrid_weightsx[np.newaxis,np.newaxis,:]*w_finegrid_weightsy[np.newaxis,:,np.newaxis]*w_finegrid_weightsz[:,np.newaxis,np.newaxis]
                        if not izc == (nzc - 1):
                            w_c[izc,iyc,ixc] = np.sum(np.multiply(w_finegrid_weights, w_finegrid_points))

                        p_finegrid_indexz, p_finegrid_indexy, p_finegrid_indexx = points_indices_p[izc,iyc,ixc]
                        p_finegrid_points = p[p_finegrid_indexz,:,:][:, p_finegrid_indexy, :][:, :, p_finegrid_indexx]
                        p_finegrid_weightsz, p_finegrid_weightsy, p_finegrid_weightsx = weights_p[izc,iyc,ixc]
                        p_finegrid_weights = p_finegrid_weightsx[np.newaxis,np.newaxis,:]*p_finegrid_weightsy[np.newaxis,:,np.newaxis]*p_finegrid_weightsz[:,np.newaxis,np.newaxis]
                        p_c[izc,iyc,ixc] = np.sum(np.multiply(p_finegrid_weights, p_finegrid_points))

                        #Calculate TOTAL transport from fine grid in user-specified coarse grid cell.
                        #yz-side boundary
                        if not ixc == (nxc - 1):
                            ~,~, coord_c_u_x = coord_c_u
                            coord_c_u_x_point = coord_c_u_x[ixc]

                            x_int = np.ravel(np.broadcast_to(coord_c_u_x_point[np.newaxis,np.newaxis,:],(len(u_finegrid_indexz),len(u_finegrid_indexy),len(coord_c_u_x_point))))
                            y_int = np.ravel(np.broadcast_to(u_finegrid_indexy[np.newaxis,:,np.newaxis],(len(u_finegrid_indexz),len(u_finegrid_indexy),len(coord_c_u_x_point))))
                            z_int = np.ravel(np.broadcast_to(u_finegrid_indexx[:,np.newaxis,np.newaxis],(len(u_finegrid_indexz),len(u_finegrid_indexy),len(coord_c_u_x_point))))

                            u_int = np.reshape(scipy.interpolate.RegularGridInterpolator((u_finegrid_indexz,u_finegrid_indexy,u_finegrid_indexx),u_finegrid_points[:,:,:],method='linear',bounds_error=False,fill_value=None)((z_int,y_int,x_int)),(len(z_int),len(y_int),len(x_int)))
                            v_int = np.reshape(scipy.interpolate.RegularGridInterpolator((v_finegrid_indexz,v_finegrid_indexy,v_finegrid_indexx),v_finegrid_points[:,:,:],method='linear',bounds_error=False,fill_value=None)((z_int,y_int,x_int)),(len(z_int),len(y_int),len(x_int)))
                            w_int = np.reshape(scipy.interpolate.RegularGridInterpolator((w_finegrid_indexz,w_finegrid_indexy,w_finegrid_indexx),w_finegrid_points[:,:,:],method='linear',bounds_error=False,fill_value=None)((z_int,y_int,x_int)),(len(z_int),len(y_int),len(x_int)))

                            weights_u_yz = (u_finegrid_weightsy[np.newaxis,:]*u_finegrid_weightsz[:,np.newaxis])[:,:,np.newaxis]
                            total_tau_xu[izc,iyc,ixc] = np.sum(np.multiply(weights_u_yz, u_int**2))
                            total_tau_xv[izc,iyc,ixc] = np.sum(np.multiply(weights_u_yz, np.multiply(u_int,v_int))
                            total_tau_xw[izc,iyc,ixc] = np.sum(np.multiply(weights_u_yz, np.multiply(u_int,w_int))

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
			
            #Calculate velocities on coarse grid
    
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

# binary3d_to_nc('u',768,384,256,starttime=0,endtime=7200,sampletime=600)
# binary3d_to_nc('v',768,384,256,starttime=0,endtime=7200,sampletime=600)
# binary3d_to_nc('w',768,384,256,starttime=0,endtime=7200,sampletime=600)
# binary3d_to_nc('p',768,384,256,starttime=0,endtime=7200,sampletime=600)

finegrid = Finegrid()
coarsegrid = Coarsegrid((32,16,64), finegrid)
generate_training_data(finegrid, coarsegrid)

