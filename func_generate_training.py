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
from microhh_tools_robins import *

#This scripts generates the training data for a developed NN, 
#which is subsequently sampled and stored in tfrecord-files using sample_training_data_tfrecord.py

##############################################
#Helper functions for generation training data
##############################################

def generate_coarsecoord_centercell(cor_edges,cor_c_middle,dist_corc,iteration,len_cor):
    cor_c_bottom = cor_c_middle - 0.5*dist_corc
    cor_c_top = cor_c_middle + 0.5*dist_corc
    
    #Find points of fine grid located just outside the coarse grid cell considered in iteration
    cor_bottom = cor_edges[cor_edges <= cor_c_bottom].max()
    cor_top = cor_c_top if iteration == (len_coordinate - 1) else cor_edges[cor_edges >= cor_c_top].min()
    
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
    
    return weights, points_indices_cor
	
def generate_coarsecoord_edgecell(cor_center,cor_c_middle,dist_corc):	
    cor_c_bottom = cor_c_middle - 0.5*dist_corc
    cor_c_top = cor_c_middle + 0.5*dist_corc
    
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
    
    return weights, points_indices_cor

##############################################
#Actual function to generate the training data
##############################################

def generate_weights(timestep,variable_name, variable_filename, coordinates, len_coordinates, edge_coordinates, dim_new_grid, bool_edge_gridcell = (False,False,False)):
    """Function to generate coarse grid for creation training data. Returns the specified variable on the coarse grid, together with the corresponding weights and coarse coordinates.
    Variable_name specifies the variable to calculate on the coarse grid.
    Variable_filename specifies the file in which the variable specified 
    Timesteps is the number of time steps present in the fine resolution data.
    Coordinates should contain a tuple with the three spatial dimensions from the fine resolution (x,y,z).
    Len_coordinates should contain a tuple indicating the spatial distance for each coordinate (x,y,z).
    Edge coordinates should contain a tuple with the coordinates that form the edges of the top-hat filter applied for the variable specified by variable_name.
    Bool_edge grid cell indicates for each coordinate (x,y,z) whether the weights should be aligned at the center of the grid cells (False) or the edges (True). """
    xcor,ycor,zcor = coordinates
    xcor_edges,ycor_edges,zcor_edges = edge_coordinates
    nxc,nyc,nzc = dim_new_grid
    xsize,ysize,zsize = len_coordinates

    #Define grid distance coarse grid, assuming it is uniform for all coordinates
    dist_xc = xsize / nxc
    dist_yc = ysize / nyc
    dist_zc = zsize / nzc

    #Define coordinates for coarse grid depending on alignment specified by bool_edge_gridcell. Note: does not include grid edges
    xcor_c = np.linspace(dist_xc,xsize-dist_xc,nxc-1,True) if bool_edge_gridcell[0] else np.linspace(0.5*dist_xc,xsize-0.5*dist_xc,nxc,True)
    ycor_c = np.linspace(dist_yc,ysize-dist_yc,nyc-1,True) if bool_edge_gridcell[1] else np.linspace(0.5*dist_yc,ysize-0.5*dist_yc,nyc,True)
    zcor_c = np.linspace(dist_zc,zsize-dist_zc,nzc-1,True) if bool_edge_gridcell[2] else np.linspace(0.5*dist_zc,zsize-0.5*dist_zc,nzc,True)
    coord_c = (zcor_c,ycor_c,xcor_c)
    
    izc = izc - 1 if bool_edge_gridcell[2]
    iyc = iyc - 1 if bool_edge_gridcell[1]
    ixc = ixc - 1 if bool_edge_gridcell[0]

    #Define a numpy array containing numpy arrays to store all weights (z,y,x) of the fine grid for all coarse grid cells. 
    #Furthermore, the indices of the fine grid cells contained are stored for each coarse grid cell.
    weights = np.empty((nzc,nyc,nxc),dtype = (object,object,object))
    points_indices = np.empty((nzc,nyc,nxc),dtype = (object,object,object))

    izc = 0
    for zcor_c_middle in zcor_c:
        if bool_edge_gridcell[2]:
            weights_z, points_indices_z = generate_coarsecoord_edgecell(cor_center = zcor, cor_c_middle = zcor_c_middle, dist_corc = dist_zc)
        else:
            weights_z, points_indices_z = generate_coarsecoord_centercell(cor_edges = zcor_edges, cor_c_middle = zcor_c_middle, dist_corc = dist_zc, iteration = izc, len_cor = nzc)

        izc += 1
        iyc = 0
	
        for ycor_c_middle in ycor_c:
            if bool_edge_gridcell[1]:
                weights_y, points_indices_y = generate_coarsecoord_edgecell(cor_center = ycor, cor_c_middle = ycor_c_middle, dist_corc = dist_yc)
            else:
                weights_y, points_indices_y = generate_coarsecoord_centercell(cor_edges = ycor_edges, cor_c_middle = ycor_c_middle, dist_corc = dist_yc, iteration = iyc, len_cor = nyc)
   
            iyc += 1
            ixc = 0
				
            for xcor_c_middle in xcor_c:
                if bool_edge_gridcell[0]:
                    weights_x, points_indices_x = generate_coarsecoord_edgecell(cor_center = xcor, cor_c_middle = xcor_c_middle, dist_corc = dist_xc)
                else:
                    weights_x, points_indices_x = generate_coarsecoord_centercell(cor_edges = xcor_edges, cor_c_middle = xcor_c_middle, dist_corc = dist_xc, iteration = ixc, len_cor = nxc)
		        
                ixc += 1
	
                #Calculate weights and points_indices
                #weights =  weights_x[np.newaxis,np.newaxis,:]*weights_y[np.newaxis,:,np.newaxis]*weights_z[:,np.newaxis,np.newaxis]
                weights[izc,iyc,ixc] = (weights_z,weights_y,weights_x)
                points_indices[izc,iyc,ixc] = (points_indices_z,points_indices_y,points_indices_x)
	    
    return weights, points_indices, coord_c
    
	
#Boundary condition: fine grid must have a smaller resolution than the coarse grid

def generate_training_data(finegrid, coarsegrid, name_output_file = 'training_data.nc', training=True): #Filenames should be strings. Default input corresponds to names files from MicroHH and the provided scripts
    if training:

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
    
        create_variables = True
        create_file = True
    
     
        #Loop over timesteps
        for t in range(nt): #Only works correctly in this script when whole simulation is saved with a constant time interval
    
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
           
            #Calculate weights and points_indices for each coarse grid cell
            weights_u, points_indices_u, coord_c_u = generate_weights(timestep = t, variable_name = 'u',variable_filename = u_file_dns,coordinates = (xh,y,z), len_coordinates = (xsize,ysize,zsize), edge_coordinates = (x,yh,zh), dim_new_grid = dim_new_grid, bool_edge_gridcell = (True,False,False))
            weights_v, points_indices_v, coord_c_v = generate_weights(timestep = t, variable_name = 'v',variable_filename = v_file_dns,coordinates = (x,yh,z), len_coordinates = (xsize,ysize,zsize), edge_coordinates = (xh,y,zh), dim_new_grid = dim_new_grid, bool_edge_gridcell = (False,True,False))
            weights_w, points_indices_w, coord_c_w = generate_weights(timestep = t, variable_name = 'w',variable_filename = w_file_dns,coordinates = (x,y,zh), len_coordinates = (xsize,ysize,zsize), edge_coordinates = (xh,yh,z), dim_new_grid = dim_new_grid, bool_edge_gridcell = (False,False,True))
            weights_p, points_indices_p, coord_c_p = generate_weights(timestep = t, variable_name = 'p',variable_filename = p_file_dns,coordinates = (x,y,z), len_coordinates = (xsize,ysize,zsize), edge_coordinates = (xh,yh,zh), dim_new_grid = dim_new_grid, bool_edge_gridcell = (False,False,False))

            #Calculate representative velocities on coarse grid from fine grid
            nzc,nyc,nxc = dim_new_grid
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

