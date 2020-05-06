#Script to downsample finegrid data to coarsegrid, which is needed for the generation of the training data for the NN.
#Author: Robin Stoffer (robin.stoffer@wur.nl)

#Developed for Python 3!
import numpy as np
import warnings

def generate_filtercoord_centercell(cor_edges, cor_f_middle, dist_corf, finegrid):
    """ Function that filters a variable in finegrid, located at the center of the grid cells, to a fine grid cell centered around cor_f_middle, with filter width dist_corf. Since the control volume of such a variable is defined by the edges of the grid cells, the cor_edges are used to select the correct cells of the fine grid and to calculate the corresponding weights (or to be more precise, fractions)."""

    dist_corf = np.round(dist_corf, finegrid.sgn_digits)
    cor_f_bottom = cor_f_middle - 0.5*dist_corf
    cor_f_top = cor_f_middle + 0.5*dist_corf
    
    #Determine machine precision specified by finegrid with corresponding significant decimal digits
    sgn_digits = finegrid.sgn_digits
    cor_f_bottom = np.round(cor_f_bottom, sgn_digits)
    cor_f_top = np.round(cor_f_top, sgn_digits)
    cor_edges = np.round(cor_edges, sgn_digits)

    #Check whether domain is exceeded, raise error if this happens
    if cor_f_bottom < 0 or cor_f_top > cor_edges[-1]: #cor_edges[-1] should correspond to the size of the domain
        raise ValueError("For variables located on the grid centers, the filter should not extend beyond the domain.")

    #Find points of fine grid located just outside the coarse grid cell considered in iteration
    cor_bottom = np.round(cor_edges[cor_edges <= cor_f_bottom].max(), sgn_digits)
    #cor_top = cor_f_top if iteration == (len_coordinate - 1) else cor_edges[cor_edges >= cor_f_top].min()
    cor_top = np.round(cor_edges[cor_edges >= cor_f_top].min(), sgn_digits)    

    #Select all points inside and just outside coarse grid cell
    points_indices_cor = np.logical_and(cor_bottom <= cor_edges , cor_edges < cor_top) #NOTE: cor_points includes the bottom boundary (cor_bottom), but not the top boundary (cor_top).
    cor_points = cor_edges[points_indices_cor]
    
    #Calculate weights for cor_points. 
    #NOTE: only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell.
    #NOTE: since cor_edges is one point longer than cor_center, the lengths of weights and points_indices_cor should be 1 shorter. 
    points_indices_cor = points_indices_cor[:-1].copy()
    len_weights = len(np.where(points_indices_cor)[0])
    weights = np.zeros(len_weights)
    if len(cor_points) == 1:
        weights = np.array([1])
        warnings.warn("Note that at certain coordinate interval(s) the defined fine grid has either the same or even a lower resolution than the defined coarse grid. The script applies in these instances a nearest neighbour procedure to fill in the values of the coarse grid.", RuntimeWarning)


    else:
        for i in range(len_weights):
            if i == 0:
                weights[0] = (cor_points[1] - cor_f_bottom)/(cor_f_top - cor_f_bottom)
            elif i == (len_weights - 1):
                weights[i] = (cor_f_top - cor_points[i])/(cor_f_top - cor_f_bottom)
            else:
                weights[i] = (cor_points[i+1] - cor_points[i])/(cor_f_top - cor_f_bottom)

#    #Select two additional points just outside coarse grid cell, which are needed for interpolation (in the total transport calculation) but not for calculating representative velocities.
#    #Consequently, the weights are set to 0.
#    points_indices_cor = np.insert(points_indices_cor, 0, points_indices_cor[0] - 1)
#    points_indices_cor = np.append(points_indices_cor, point_indices_cor[-1] + 1)
#    weights = np.insert(weights, 0, 0)
#    weights = np.append(weights, 0)

    return weights, points_indices_cor

def generate_filtercoord_edgecell(cor_center, cor_f_middle, dist_corf, finegrid, periodic_bc = True, zero_w_topbottom = True, size = 0): #For 'size' a default value is used that should not affect results as long as vert_flag = False.
    """ Function that filters a variable in finegrid, located at the edge of the grid cells, to a fine grid cell centered around cor_f_middle, with filter width dist_corf. Since the control volume of such a variable is defined by the centers of the grid cells, the cor_center are used to select the correct cells of the fine grid and to calculate the corresponding weights (or to be more precise, fractions). NOTE: to deal with the edges of the domain, it needs to be specified whether periodic bc and zero_w_topbottom are present (True) or not (False). Furthermore, the size of the domain needs to be specified as well. """
    
    dist_corf = np.round(dist_corf, finegrid.sgn_digits)
    cor_f_bottom = cor_f_middle - 0.5*dist_corf
    cor_f_top = cor_f_middle + 0.5*dist_corf
    
    #Determine machine precision specified by finegrid with corresponding significant decimal digits
    sgn_digits = finegrid.sgn_digits
    cor_f_bottom = np.round(cor_f_bottom, sgn_digits)
    cor_f_top = np.round(cor_f_top, sgn_digits)
    cor_center = np.round(cor_center, sgn_digits)

    #Find points of fine grid located just outside the coarse grid cell considered in iteration, except when the top and bottom boundary in the vertical direction are considered (since in that direction no ghost cells are added).

    #Define function that adds an additional point to points_indices_cor, which is needed because cor_center is 1 point smaller than cor_edges
    def _add_points(cor_center, cor_f_top, points_indices_cor):
        if cor_f_top > cor_center[-1]: #Select top/downstream boundary when coarse grid top/downstream boundary exceeds the boundaries of the fine grid cell centers
            points_indices_cor = np.append(points_indices_cor, True)
        else:
            points_indices_cor = np.append(points_indices_cor, False)
        return points_indices_cor
    
    #Define function that selects correct points for box filter in all locations where periodic bc are not applied.
    def _select_points(cor_center, cor_f_bottom, cor_f_top, sgn_digits, cor_bottom_defined = False, cor_top_defined = False):

        if not cor_bottom_defined and np.any(cor_center <= cor_f_bottom):
            cor_bottom = np.round(cor_center[cor_center <= cor_f_bottom].max(), sgn_digits)
        else:
            cor_bottom = cor_f_bottom
            
        if (not cor_top_defined) and np.any(cor_center >= cor_f_top):
            cor_top = np.round(cor_center[cor_center >= cor_f_top].min(), sgn_digits)
        else:
            cor_top = cor_f_top
            
        #Select all points inside and just outside coarse grid cell
        points_indices_cor = np.logical_and(cor_bottom<cor_center , cor_center<=cor_top) #NOTE: points_indices_cor includes the top boundary (cor_top), but not the bottom boundary (cor_bottom). 
        points_indices_cor = _add_points(cor_center, cor_f_top, points_indices_cor)
        
        return points_indices_cor
    
    #Select points of fine grid depending on fine grid configuration and periodic bc.
    if periodic_bc and (cor_f_bottom < 0): #NOTE: this is the only exception where the code is allowed to extend beyond the fine grid domain. This is done to ensure that the relevant domain for a possible sampling on the coarse grid is fully covered.
        #Select two different regions of fine grid
        cor_f_bottom1 = 0.0
        cor_bottom1 = cor_f_bottom1
        cor_f_top1 = cor_f_top
        cor_top1 = np.round(cor_center[cor_center >= cor_f_top1].min(), sgn_digits)
        
        cor_f_bottom2 = np.round(cor_f_bottom + size, sgn_digits)
        cor_bottom2 = np.round(cor_center[cor_center <= cor_f_bottom2].max(), sgn_digits)
        cor_f_top2 = np.round(size, sgn_digits)
        cor_top2 = cor_f_top2
        
        points_indices_cor1 = np.logical_and(cor_bottom1 < cor_center, cor_center <= cor_top1)
        points_indices_cor1 = _add_points(cor_center, cor_f_top1, points_indices_cor1)
        points_indices_cor2 = np.logical_and(cor_bottom2 < cor_center, cor_center <= cor_top2)
        points_indices_cor2 = _add_points(cor_center, cor_f_top2, points_indices_cor2)
        points_indices_cor = np.logical_or(points_indices_cor1, points_indices_cor2)
        index_bottom_coarse_cell = np.where(points_indices_cor2)[0][0] - len(np.where(np.logical_and(np.logical_not(points_indices_cor1), np.logical_not(points_indices_cor2)))[0]) #Compensate for the indices that are not selected, cor_points has fewer indices than cor_center!
        index_top_coarse_cell = np.where(points_indices_cor1)[0][-1]
        two_boxes = True
        dirichlet_bc = False
    
    elif cor_f_bottom < 0:
        raise ValueError("The filter should not extend beyond the domain when no periodic BCs are present.")
        #cor_f_bottom = 0.0
        #points_indices_cor = _select_points(cor_center, cor_f_bottom, cor_f_top, sgn_digits, cor_bottom_defined = True)
        #two_boxes = False
        #dirichlet_bc = True
        
    elif periodic_bc and (cor_f_top > size): #NOTE: this is the only exception where the code is allowed to extend beyond the fine grid domain. This is done to ensure that the relevant domain for a possible sampling on the coarse grid is fully covered.
        #Select two different regions of coarse grid
        cor_f_bottom1 = 0.0
        cor_bottom1 = cor_f_bottom1
        cor_f_top1 = np.round(cor_f_top - size, sgn_digits)
        cor_top1 = np.round(cor_center[cor_center >= cor_f_top1].min(), sgn_digits)
        
        cor_f_bottom2 = cor_f_bottom
        cor_bottom2 = np.round(cor_center[cor_center <= cor_f_bottom2].max(), sgn_digits)
        cor_f_top2 = np.round(size, sgn_digits)
        cor_top2 = cor_f_top2
        
        points_indices_cor1 = np.logical_and(cor_bottom1 < cor_center, cor_center <= cor_top1)
        points_indices_cor1 = _add_points(cor_center, cor_f_top1, points_indices_cor1)
        points_indices_cor2 = np.logical_and(cor_bottom2 < cor_center, cor_center <= cor_top2)
        points_indices_cor2 = _add_points(cor_center, cor_f_top2, points_indices_cor2)
        points_indices_cor = np.logical_or(points_indices_cor1, points_indices_cor2)
        index_bottom_coarse_cell = np.where(points_indices_cor2)[0][0]  - len(np.where(np.logical_and(np.logical_not(points_indices_cor1), np.logical_not(points_indices_cor2)))[0]) #Compensate for the indices that are not selected, cor_points has fewer indices than cor_center!
        index_top_coarse_cell = np.where(points_indices_cor1)[0][-1]
        two_boxes = True
        dirichlet_bc = False
                
    elif cor_f_top > size:
        raise ValueError("The filter should not extend beyond the domain when no periodic BCs are present.")
        #cor_f_top = size
        #points_indices_cor = _select_points(cor_center, cor_f_bottom, cor_f_top, sgn_digits, cor_top_defined = True)
        #two_boxes = False
        #dirichlet_bc = True
        
    else:
        points_indices_cor = _select_points(cor_center, cor_f_bottom, cor_f_top, sgn_digits)
        two_boxes = False
        dirichlet_bc = False

#    #Find points of fine grid located just outside the coarse grid cell considered in iteration
#    cor_bottom = cor_center[cor_center <= cor_f_bottom].max()
#    cor_top = cor_center[cor_center >= cor_f_top].min()
    
    #Select all cells that are within coarse grid cell
    cor_points = cor_center[points_indices_cor[:-1]] #NOTE: cor_center is one shorter than cor_edges, meaning last index point in array (i.e. [-1]) included earlier should be removed for correct selection cor_center.
    
    #Calculate weights for cor_points. 
    #NOTE: only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
    #NOTE: since cor_center is one point shorter than cor_edges, the lengths of weights and points_indices_cor should be 1 longer. The additional point in points_indices_cor is already added via the _add_points function.
    len_weights = len(np.where(points_indices_cor)[0])
    weights = np.zeros(len_weights)
    if zero_w_topbottom and dirichlet_bc:
        pass #Keep the weights at zero
    
    elif len_weights == 1:
        weights = np.array([1])
        warnings.warn("Note that at certain coordinate interval(s) the defined fine grid has either the same or even a lower resolution than the defined coarse grid. The script applies in these instances a nearest neighbour procedure to fill in the values of the coarse grid.", RuntimeWarning)
    
    elif two_boxes: #Deal with the cases where the selected fine grid cells consist of two separate regions (which occurs when periodic bc are imposed and the center of the coarse grid cell is located on the bottom/top of the domain).
        for i in range(len_weights):
            if i == 0 and i == index_top_coarse_cell:
                weights[i] = (cor_f_top1 - cor_f_bottom1)/(cor_f_top - cor_f_bottom)
            elif i == 0:
                weights[i] = (cor_points[i] - cor_f_bottom1)/(cor_f_top - cor_f_bottom)
            elif i == index_top_coarse_cell:
                weights[i] = (cor_f_top1 - cor_points[i-1])/(cor_f_top - cor_f_bottom)
            elif i == (len_weights - 1) and i == index_bottom_coarse_cell:
                weights[i] = (cor_f_top2 - cor_f_bottom2)/(cor_f_top - cor_f_bottom)
            elif i == (len_weights - 1):
                weights[i] = (cor_f_top2 - cor_points[i-1])/(cor_f_top - cor_f_bottom)
            elif i == index_bottom_coarse_cell:
                weights[i] = (cor_points[i] - cor_f_bottom2)/(cor_f_top - cor_f_bottom)
            else:
                weights[i] = (cor_points[i] - cor_points[i-1])/(cor_f_top - cor_f_bottom)

    else:
        for i in range(len_weights):
            if i == 0:
                weights[i] = (cor_points[i] - cor_f_bottom)/(cor_f_top - cor_f_bottom)
            elif i == (len_weights - 1):
                weights[i] = (cor_f_top - cor_points[i-1])/(cor_f_top - cor_f_bottom)
            else:
                weights[i] = (cor_points[i] - cor_points[i-1])/(cor_f_top - cor_f_bottom)

    return weights, points_indices_cor

def boxfilter(filter_widths, finegrid, variable_name, bool_edge_gridcell = (False, False, False), periodic_bc = (False, True, True), zero_w_topbottom = True):
    """Function to apply box filter on fine grid. Returns the specified variable filtered on the fine grid. The inputs are as follows:
    
    -filter_widths: tuple with three floats, which indicate for each spatial direction (z,y,x) the filter width
    
    -finegrid: finegrid object defined in grid_objects_training.py
    
    -variable_name: string that specifies the variable to calculate on the coarse grid.
    
    -bool_edge_gridcell: a tuple with a boolean for each spatial direction (z, y, x), indicating whether they should be aligned at the center of the grid cells (False) or the edges (True).
    
    -periodic_bc: a tuple with a boolean for each spatial direction (z, y, x), indicating  whether periodic boundary conditions are assumed (True when present, False when not present).
    
    -zero_w_topbottom: boolean specifying whether the vertical wind velocity is 0 at the bottom and top levels of the domain or not."""

    #Read in the right coarse coordinates determined by bool_edge_gridcell.
    #z-direction
    if bool_edge_gridcell[0]:
        zcor_f     = finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend]
        dist_zf    = filter_widths[0] 
    else:
        zcor_f     = finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend]
        dist_zf    = filter_widths[0]

    #y-direction
    if bool_edge_gridcell[1]:
        ycor_f     = finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend]
        dist_yf    = filter_widths[1]
    else:
        ycor_f     = finegrid['grid']['y'][finegrid.jgc:finegrid.jend]
        dist_yf    = filter_widths[1]

    #x-direction
    if bool_edge_gridcell[2]:
        xcor_f     = finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]
        dist_xf    = filter_widths[2]
    else:
        xcor_f     = finegrid['grid']['x'][finegrid.igc:finegrid.iend]
        dist_xf    = filter_widths[2]
        
    #Check that variable_name is a string.
    if not isinstance(variable_name, str):
        raise TypeError("Specified variable_name should be a string.")

    #Check whether variable to be filtered is contained in finegrid object associated with coarsegrid object.
    if variable_name not in finegrid['output'].keys(): 
        raise KeyError("Specified variable_name not defined in finegrid object on which this coarsegrid object is based.")

    #Check whether specified period_bc satisfies needed format.
    if not isinstance(periodic_bc, tuple):
        raise TypeError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
    if not len(periodic_bc) == 3:
        raise ValueError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
    if not any(isinstance(flag, bool) for flag in periodic_bc):
        raise ValueError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
    #Check whether specified bool_edge_gridcell satisfies needed format.
    if not isinstance(bool_edge_gridcell, tuple):
        raise TypeError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
    if not len(bool_edge_gridcell) == 3:
        raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
    if not any(isinstance(flag, bool) for flag in bool_edge_gridcell):
        raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")

    #Check that zero_w_topbottom is a boolean.
    if not isinstance(zero_w_topbottom, bool):
        raise TypeError('The zero_w_topbottom flag should be a boolean (True/False).')

    #Define filtered variable
    var_f = np.zeros((len(zcor_f), len(ycor_f), len(xcor_f)), dtype=float)

    #Loop over coordinates for box-filtering
    izc = 0
    for zcor_f_middle in zcor_f:
        outside_domain = False
    #for izc in range(coarsegrid['grid']['ktot'])
        if ((np.round((zcor_f_middle - dist_zf),finegrid.sgn_digits) < 0.) or ((np.round(zcor_f_middle + dist_zf, finegrid.sgn_digits)) > (np.round(finegrid['grid']['zsize'], finegrid.sgn_digits)))): #Don't filter when filter width extends beyond grid domain
            outside_domain = True
        elif bool_edge_gridcell[0]:
            weights_z, points_indices_z = generate_filtercoord_edgecell(cor_center = finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], cor_f_middle = zcor_f_middle, dist_corf = dist_zf, finegrid = finegrid, periodic_bc = periodic_bc[0], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['zsize'])
            var_finez = finegrid['output'][variable_name]['variable'][finegrid.kgc_edge:finegrid.khend, :, :]
            var_finez = var_finez[points_indices_z,:,:]
        else:
            weights_z, points_indices_z = generate_filtercoord_centercell(cor_edges = finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], cor_f_middle = zcor_f_middle, dist_corf = dist_zf, finegrid = finegrid)
            var_finez = finegrid['output'][variable_name]['variable'][finegrid.kgc_center:finegrid.kend, :, :]
            var_finez = var_finez[points_indices_z,:,:]

        iyc = 0
	
        for ycor_f_middle in ycor_f:
            if ((np.round((ycor_f_middle - dist_yf),finegrid.sgn_digits) < 0.) or ((np.round(ycor_f_middle + dist_yf, finegrid.sgn_digits)) > (np.round(finegrid['grid']['ysize'], finegrid.sgn_digits))) or outside_domain): #Don't filter when filter width extends beyond grid domain
                outside_domain = True
            elif bool_edge_gridcell[1]:
                weights_y, points_indices_y = generate_filtercoord_edgecell(cor_center = finegrid['grid']['y'][finegrid.jgc:finegrid.jend], cor_f_middle = ycor_f_middle, dist_corf = dist_yf, finegrid = finegrid, periodic_bc = periodic_bc[1], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['ysize'])
                var_finezy = var_finez[:, finegrid.jgc:finegrid.jhend,:]
                var_finezy = var_finezy[:,points_indices_y,:]
            else:
                weights_y, points_indices_y = generate_filtercoord_centercell(cor_edges = finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], cor_f_middle = ycor_f_middle, dist_corf = dist_yf, finegrid = finegrid)
                var_finezy = var_finez[:, finegrid.jgc:finegrid.jend,:]
                var_finezy = var_finezy[:,points_indices_y,:]

            ixc = 0
				
            for xcor_f_middle in xcor_f:
                if ((np.round((xcor_f_middle - dist_xf),finegrid.sgn_digits) < 0.) or ((np.round(xcor_f_middle + dist_xf, finegrid.sgn_digits)) > (np.round(finegrid['grid']['xsize'], finegrid.sgn_digits))) or outside_domain): #Don't filter when filter width extends beyond grid domain
                    outside_domain = True
                elif bool_edge_gridcell[2]:
                    weights_x, points_indices_x = generate_filtercoord_edgecell(cor_center = finegrid['grid']['x'][finegrid.igc:finegrid.iend], cor_f_middle = xcor_f_middle, dist_corf = dist_xf, finegrid = finegrid, periodic_bc = periodic_bc[2], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['xsize'])
                    var_finezyx = var_finezy[:, :, finegrid.igc:finegrid.ihend]
                    var_finezyx = var_finezyx[:,:,points_indices_x]
                else:
                    weights_x, points_indices_x = generate_filtercoord_centercell(cor_edges = finegrid['grid']['xh'][finegrid.igc:finegrid.ihend], cor_f_middle = xcor_f_middle, dist_corf = dist_xf, finegrid = finegrid)
                    var_finezyx = var_finezy[:, :, finegrid.igc:finegrid.iend]
                    var_finezyx = var_finezyx[:,:,points_indices_x]

                if outside_domain:
                    var_f[izc,iyc,ixc] = np.nan
                else:
                    #Calculate downsampled variable on coarse grid using the selected points in var_finezyx and the fractions defined in the weights variables
                    weights =  weights_x[np.newaxis,np.newaxis,:]*weights_y[np.newaxis,:,np.newaxis]*weights_z[:,np.newaxis,np.newaxis]
                    var_f[izc,iyc,ixc] = np.sum(np.multiply(weights, var_finezyx))
 
                ixc += 1
            iyc += 1
        izc += 1
    
    return var_f
    
