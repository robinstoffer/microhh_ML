# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:50:03 2018

Author: Robin Stoffer
"""

#Developed for Python 3!
import numpy as np

def generate_coarsecoord_centercell(cor_edges, cor_c_middle, dist_corc):
    cor_c_bottom = cor_c_middle - 0.5*dist_corc
    cor_c_top = cor_c_middle + 0.5*dist_corc
    
    #Find points of fine grid located just outside the coarse grid cell considered in iteration
    cor_bottom = cor_edges[cor_edges <= cor_c_bottom].max()
    #cor_top = cor_c_top if iteration == (len_coordinate - 1) else cor_edges[cor_edges >= cor_c_top].min()
    cor_top = cor_edges[cor_edges >= cor_c_top].min()    

    #Select all points inside and just outside coarse grid cell
    points_indices_cor = np.logical_and(cor_bottom <= cor_edges , cor_edges < cor_top)
    cor_points = cor_edges[points_indices_cor] #Note: cor_points includes the bottom boundary (cor_bottom), but not the top boundary (cor_top).
    
    #Calculate weights for cor_points. 
    #NOTE: only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
    #NOTE: since cor_edges is one point longer than cor_center, the lengths of points_indices cor should be 1 shorter.
    weights = np.zeros(len(cor_points))
    points_indices_cor = points_indices_cor[:-1].copy()
    if len(cor_points) == 1:
        weights = np.array([1])

    else:
        for i in range(len(cor_points)):
            if i == 0:
                weights[0] = (cor_points[1] - cor_c_bottom)/(cor_c_top - cor_c_bottom)
            elif i == (len(cor_points) - 1):
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

def generate_coarsecoord_edgecell(cor_center, cor_c_middle, dist_corc, periodic_bc = True, size = 0): #For 'size' a default value is used that should not affect results as long as vert_flag = False.
    cor_c_bottom = cor_c_middle - 0.5*dist_corc
    cor_c_top = cor_c_middle + 0.5*dist_corc

    #Find points of fine grid located just outside the coarse grid cell considered in iteration, except when the top and bottom boundary in the vertical direction are considered (since in that direction no ghost cells are added).
    
    #Define function that selects correct points for coarse grid cells where periodic bc are not used for the downsampling from the fine grid.
    def _select_points(cor_center, cor_c_bottom, cor_c_top, cor_bottom_defined = False, cor_top_defined = False):

        if not cor_bottom_defined:
            cor_bottom = cor_center[cor_center <= cor_c_bottom].max()
        else:
            cor_bottom = cor_c_bottom
            
        if not cor_top_defined:
            cor_top = cor_center[cor_center >= cor_c_top].min()
        else:
            cor_top = cor_c_top
            
        #Select all points inside and just outside coarse grid cell
        points_indices_cor = np.logical_and(cor_bottom<cor_center , cor_center<=cor_top)
        
        #Set cells where fractions need to be calculated
        index_bottom_coarse_cell = 0
        index_top_coarse_cell = len(cor_center[points_indices_cor]) - 1
        coord_bottom_coarse_cell = cor_c_bottom
        coord_top_coarse_cell = cor_c_top
        
        return points_indices_cor, index_bottom_coarse_cell, index_top_coarse_cell, coord_bottom_coarse_cell, coord_top_coarse_cell
    
    #Select points of fine grid depending on coarse grid configuration en periodic bc.
    if periodic_bc and (cor_c_bottom<0):
        #Select two different regions of coarse grid
        cor_c_bottom1 = 0
        cor_bottom1 = cor_c_bottom1
        cor_c_top1 = cor_c_top
        cor_top1 = cor_center[cor_center >= cor_c_top1].min()
        
        cor_c_bottom2 = cor_c_bottom + size
        cor_bottom2 = cor_center[cor_center <= cor_c_bottom2].max()
        cor_c_top2 = size
        cor_top2 = cor_c_top2
        
        points_indices_cor1 = np.logical_and(cor_bottom1 < cor_center, cor_center <= cor_top1)
        points_indices_cor2 = np.logical_and(cor_bottom2 < cor_center, cor_center <= cor_top2)
        points_indices_cor = np.logical_or(points_indices_cor1, points_indices_cor2)
        index_bottom_coarse_cell = np.where(points_indices_cor2)[0][0] - len(np.where(np.logical_and(np.logical_not(points_indices_cor1), np.logical_not(points_indices_cor2)))[0]) #Compensate for the indices that are not selected, cor_points has fewer indices than cor_center!
        index_top_coarse_cell = np.where(points_indices_cor1)[0][-1]
        coord_bottom_coarse_cell = cor_c_bottom2
        coord_top_coarse_cell = cor_c_top1
    
    elif cor_c_bottom < 0:
        cor_c_bottom = 0
        points_indices_cor, index_bottom_coarse_cell, index_top_coarse_cell, coord_bottom_coarse_cell, coord_top_coarse_cell = _select_points(cor_center, cor_c_bottom, cor_c_top, cor_bottom_defined = True)
        
    elif periodic_bc and (cor_c_top > size):
        #Select two different regions of coarse grid
        cor_c_bottom1 = 0
        cor_bottom1 = cor_c_bottom1
        cor_c_top1 = cor_c_top - size
        cor_top1 = cor_center[cor_center >= cor_c_top1].min()
        
        cor_c_bottom2 = cor_c_bottom
        cor_bottom2 = cor_center[cor_center <= cor_c_bottom2].max()
        cor_c_top2 = size
        cor_top2 = cor_c_top2
        
        points_indices_cor1 = np.logical_and(cor_bottom1 < cor_center, cor_center <= cor_top1)
        points_indices_cor2 = np.logical_and(cor_bottom2 < cor_center, cor_center <= cor_top2)
        points_indices_cor = np.logical_or(points_indices_cor1, points_indices_cor2)
        index_bottom_coarse_cell = np.where(points_indices_cor2)[0][0]  - len(np.where(np.logical_and(np.logical_not(points_indices_cor1), np.logical_not(points_indices_cor2)))[0]) #Compensate for the indices that are not selected, cor_points has fewer indices than cor_center!
        index_top_coarse_cell = np.where(points_indices_cor1)[0][-1]
        coord_bottom_coarse_cell = cor_c_bottom2
        coord_top_coarse_cell = cor_c_top1
        
        
    elif cor_c_top > size:
        cor_c_top = size
        points_indices_cor, index_bottom_coarse_cell, index_top_coarse_cell, coord_bottom_coarse_cell, coord_top_coarse_cell = _select_points(cor_center, cor_c_bottom, cor_c_top, cor_top_defined = True)
       
    else:
        points_indices_cor, index_bottom_coarse_cell, index_top_coarse_cell, coord_bottom_coarse_cell, coord_top_coarse_cell = _select_points(cor_center, cor_c_bottom, cor_c_top)

#    #Find points of fine grid located just outside the coarse grid cell considered in iteration
#    cor_bottom = cor_center[cor_center <= cor_c_bottom].max()
#    cor_top = cor_center[cor_center >= cor_c_top].min()
    
    #Select all cells that are within coarse grid cell
    cor_points = cor_center[points_indices_cor] #Note: cor_points includes the bottom boundary (cor_bottom), but not the top boundary (cor_top).
	  
    #Calculate weights for cor_points. Note that only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
    weights = np.zeros(len(cor_points))
    if len(cor_points) == 1:
        weights = np.array([1])

    else:
        for i in range(len(cor_points)):
            if i == index_bottom_coarse_cell:
                weights[i] = (cor_points[i] - coord_bottom_coarse_cell)/(cor_c_top - cor_c_bottom)
            elif i == index_top_coarse_cell:
                weights[i] = (coord_top_coarse_cell - cor_points[i-1])/(cor_c_top - cor_c_bottom)
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

def downsample(finegrid, coarsegrid, variable_name, bool_edge_gridcell = (False, False, False), periodic_bc = (False, True, True)):
    """Function to generate coarse grid with variables and total transport of momentum for creation training data. Returns the specified variable on the coarse grid, together with the corresponding weights and coarse coordinates.
    Variable_name specifies the variable to calculate on the coarse grid.
    Bool_edge_gridcell indicates in a tuple for each spatial direction (z, y, x) whether they should be aligned at the center of the grid cells (False) or the edges (True).
    Periodic_bc indicates in a tuple for each spatial direction (z, y, x) whether periodic boundary conditions are assumed (True when present, False when not present)."""

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
        
    #Check that variable_name is a string
    if not isinstance(variable_name, str):
        raise TypeError("Specified variable_name should be a string.")

    #Check whether variable to be downsampled is contained in finegrid object associated with coarsegrid object.
    if variable_name not in finegrid['output'].keys(): 
        raise KeyError("Specified variable_name not defined in finegrid object on which this coarsegrid object is based.")

    #Check whether specified period_bc satisfies needed format and subsequently store it in object.
    if not isinstance(periodic_bc, tuple):
        raise TypeError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
    if not len(periodic_bc) == 3:
        raise ValueError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
    if not any(isinstance(flag, bool) for flag in periodic_bc):
        raise ValueError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
    #Check whether specified bool_edge_gridcell satisfies needed format and subsequently store it in object.
    if not isinstance(bool_edge_gridcell, tuple):
        raise TypeError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
    if not len(bool_edge_gridcell) == 3:
        raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
    if not any(isinstance(flag, bool) for flag in bool_edge_gridcell):
        raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")

    var_c = np.zeros((len(zcor_c), len(ycor_c), len(xcor_c)), dtype=float)
    #weights_c = np.zeros(len(zcor_c), len(ycor_c), len(xcor_c), dtype=(object, object, object))
    #points_indices_c = np.zeros(len(zcor_c), len(ycor_c), len(xcor_c), dtype=(object, object, object))

    #Add needed ghostcells to finegrid object for the downsampling and calculation total transport
    #finegrid = add_ghostcells_finegrid(finegrid, coarsegrid, variable_name, bool_edge_gridcell)

    #Loop over coordinates for downsampling
    izc = 0
    for zcor_c_middle in zcor_c:
    #for izc in range(coarsegrid['grid']['ktot'])
        if bool_edge_gridcell[0]:
            weights_z, points_indices_z = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['z'][finegrid.kgc:finegrid.khend], cor_c_middle = zcor_c_middle, dist_corc = dist_zc, periodic_bc = periodic_bc[0], size = finegrid['grid']['zsize'])
            var_finez = finegrid['output'][variable_name]['variable'][finegrid.kgc:finegrid.khend, :, :]
        else:
            weights_z, points_indices_z = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['zh'][finegrid.kgc:finegrid.khend], cor_c_middle = zcor_c_middle, dist_corc = dist_zc)
            var_finez = finegrid['output'][variable_name]['variable'][finegrid.kgc:finegrid.kend, :, :]

        var_finez = var_finez[points_indices_z,:,:]

        iyc = 0
	
        for ycor_c_middle in ycor_c:
            if bool_edge_gridcell[1]:
                weights_y, points_indices_y = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['y'][finegrid.jgc:finegrid.jend], cor_c_middle = ycor_c_middle, dist_corc = dist_yc, periodic_bc = periodic_bc[1], size = finegrid['grid']['ysize'])
                var_finezy = var_finez[:, finegrid.jgc:finegrid.jhend,:]
            else:
                weights_y, points_indices_y = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], cor_c_middle = ycor_c_middle, dist_corc = dist_yc)
                var_finezy = var_finez[:, finegrid.jgc:finegrid.jend,:]

            var_finezy = var_finezy[:,points_indices_y,:]

            ixc = 0
				
            for xcor_c_middle in xcor_c:
                if bool_edge_gridcell[2]:
                    weights_x, points_indices_x = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['x'][finegrid.igc:finegrid.iend], cor_c_middle = xcor_c_middle, dist_corc = dist_xc, periodic_bc = periodic_bc[2], size = finegrid['grid']['xsize'])
                    var_finezyx = var_finezy[:, :, finegrid.igc:finegrid.ihend]
                else:
                    weights_x, points_indices_x = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['xh'][finegrid.igc:finegrid.ihend], cor_c_middle = xcor_c_middle, dist_corc = dist_xc)
                    var_finezyx = var_finezy[:, :, finegrid.igc:finegrid.ihend]
                    
                var_finezyx = var_finezyx[:,:,points_indices_x]

                #Calculate downsampled variable on coarse grid using the selected points in var_finezyx and the fractions defined in the weights variables
                weights =  weights_x[np.newaxis,np.newaxis,:]*weights_y[np.newaxis,:,np.newaxis]*weights_z[:,np.newaxis,np.newaxis]
                var_c[izc,iyc,ixc] = np.sum(np.multiply(weights, var_finezyx))
 
                #weights_c[izc,iyc,ixc] = (weights_z,weights_y,weights_x)
                #points_indices_c[izc,iyc,ixc] = (points_indices_z,points_indices_y,points_indices_x)

                ixc += 1
            iyc += 1
        izc += 1
    
#    #Store downsampled variable in coarsegrid object
#    coarsegrid['output'][variable_name] = var_c
 
    return var_c
    