#Function to calculate tendencies for unresolved momentum flux
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import tensorflow as tf
import numpy as np
import netCDF4 as nc
from load_frozen_graph import load_graph

def diff_U(u, v, w, utau_ref, frozen_graph_filename, dzi, dzhi, grid, MLP, b):
         
    #Load frozen graph
    graph = load_graph(frozen_graph_filename)
 
    ##List ops in graph
    #for op in graph.get_operations():
    #    print(op.name)
 
    #Access input and output nodes
    #NOTE: specify ':0' to select the correct output of the ops and get the tensors themselves
    input_u               = graph.get_tensor_by_name('input_u:0')
    input_v               = graph.get_tensor_by_name('input_v:0')
    input_w               = graph.get_tensor_by_name('input_w:0')
    #input_flag_topwall    = graph.get_tensor_by_name('flag_topwall:0')
    #input_flag_bottomwall = graph.get_tensor_by_name('flag_bottomwall:0')
    input_utau_ref        = graph.get_tensor_by_name('input_utau_ref:0')
    output                = graph.get_tensor_by_name('output_layer_denorm:0')
    
    #Initialize zeros arrays for tendencies
    #NOTE: for wt, the initial zero value of the bottom layer is on purpose not changed. This is ONLY valid for a no-slip BC.
    ut = np.zeros((grid.ktot,grid.jtot,grid.itot))
    vt = np.zeros((grid.ktot,grid.jtot,grid.itot))
    wt = np.zeros((grid.ktot,grid.jtot,grid.itot))
    
    #NOTE: several expand_dims included to account for batch dimension
    #Reshape 1d arrays to 3d, which is much more convenient for the slicing below.
    u = np.reshape(u, (grid.kcells,  grid.jcells, grid.icells))
    v = np.reshape(v, (grid.kcells,  grid.jcells, grid.icells))
    w = np.reshape(w, (grid.khcells, grid.jcells, grid.icells))
 
    #Extract friction velocity
    input_utau_ref_val = utau_ref
 
    #Calculate inverse height differences
    dxi = 1./grid.dx
    dyi = 1./grid.dy
    
    #NOTE: offset factors are defined to ensure alternate inference
    for k in range(grid.kstart,grid.kend,1): 
        k_offset = k % 2
        for j in range(grid.jstart,grid.jend,1):
            if k_offset != 0:
                offset = int(j % 2 != 0) #Do offset for odd columns
            else:
                offset = int(j % 2 == 0) #Do offset for even columns
            for i in range(grid.istart+offset,grid.iend,2):
                
                ##Calculate additional needed loop indices
                #ij = i + j*jj
                #ijk = i + j*jj + k*kk

                #Extract grid box flow fields
                input_u_val = np.expand_dims(u[k-b:k+b+1,j-b:j+b+1,i-b:i+b+1].flatten(), axis=0) #Flatten and expand dims arrays for MLP
                #print(input_u_val)
                input_v_val = np.expand_dims(v[k-b:k+b+1,j-b:j+b+1,i-b:i+b+1].flatten(), axis=0)
                #print(input_v_val)
                input_w_val = np.expand_dims(w[k-b:k+b+1,j-b:j+b+1,i-b:i+b+1].flatten(), axis=0)
                #print(input_w_val)
                #raise RuntimeError("Stop run")

                #Execute MLP once for selected grid box
                result = MLP.predict(input_u_val, input_v_val, input_w_val, input_utau_ref_val)

                #Store results in initialized arrays in nc-file
                #NOTE1: compensate indices for lack of ghost cells
                #NOTE2: flatten 'result' matrix to have consistent shape for output arrays
                result = result.flatten()
                i_nogc = i - grid.istart
                j_nogc = j - grid.jstart
                k_nogc = k - grid.kstart
                k_1gc  = k_nogc + 1
                #unres_tau_xu_CNN[k_nogc  ,j_nogc  ,i_nogc]       = result[0] #xu_upstream
                #unres_tau_xu_CNN[k_nogc  ,j_nogc  ,i_nogc+1]     = result[1] #xu_downstream
                #unres_tau_yu_CNN[k_nogc  ,j_nogc  ,i_nogc]       = result[2] #yu_upstream
                #unres_tau_yu_CNN[k_nogc  ,j_nogc+1,i_nogc]       = result[3] #yu_downstream
                #if not (k == grid.kstart): #Don't assign the bottom wall
                #    unres_tau_zu_CNN[k_nogc  ,j_nogc  ,i_nogc]   = result[4] #zu_upstream
                #if not (k == (grid.kend - 1)): #Don't assign the top wall
                #    unres_tau_zu_CNN[k_nogc+1,j_nogc  ,i_nogc]   = result[5] #zu_downstream
                #unres_tau_xv_CNN[k_nogc  ,j_nogc  ,i_nogc]       = result[6] #xv_upstream
                #unres_tau_xv_CNN[k_nogc  ,j_nogc  ,i_nogc+1]     = result[7] #xv_downstream
                #unres_tau_yv_CNN[k_nogc  ,j_nogc  ,i_nogc]       = result[8] #yv_upstream
                #unres_tau_yv_CNN[k_nogc  ,j_nogc+1,i_nogc]       = result[9] #yv_downstream
                #if not (k == grid.kstart): #Don't assign the bottom wall
                #    unres_tau_zv_CNN[k_nogc  ,j_nogc  ,i_nogc]   = result[10] #zv_upstream
                #if not (k == (grid.kend - 1)): #Don't assign the top wall
                #    unres_tau_zv_CNN[k_nogc+1,j_nogc  ,i_nogc]   = result[11] #zv_downstream
                #if not (k == grid.kstart): #Don't assign the bottom wall
                #    unres_tau_xw_CNN[k_nogc,  j_nogc  ,i_nogc]   = result[12] #xw_upstream
                #    unres_tau_xw_CNN[k_nogc,  j_nogc  ,i_nogc+1] = result[13] #xw_downstream
                #    unres_tau_yw_CNN[k_nogc,  j_nogc  ,i_nogc]   = result[14] #yw_upstream
                #    unres_tau_yw_CNN[k_nogc,  j_nogc+1,i_nogc]   = result[15] #yw_downstream
                #    unres_tau_zw_CNN[k_nogc-1,  j_nogc  ,i_nogc] = result[16] #zw_upstream
                #    unres_tau_zw_CNN[k_nogc,j_nogc  ,i_nogc]     = result[17] #zw_downstream
                #    #NOTE: although the last one line does not change zw at the bottom layer, it is still not included for k=0 to keep consistency between the top and bottom of the domain.
                ##
                ###Account for all tendencies affected by calculated transport components (two tendencies for each of the predicted 18 components)###
                #Check whether a horizontal boundary is reached, and if so make use of horizontal periodic BCs. This adjustment is only needed at the downstream side of the domain, since at the upstream side the index is already automatically converted to -1.
                if i == (grid.iend - 1):
                    i_nogc_bound = 0
                else:
                    i_nogc_bound = i_nogc + 1

                if j == (grid.jend - 1):
                    j_nogc_bound = 0
                else:
                    j_nogc_bound = j_nogc + 1
                
                #xu_upstream
                ut[k_nogc  ,j_nogc  ,i_nogc]           += -result[0] * dxi
                ut[k_nogc  ,j_nogc  ,i_nogc-1]         +=  result[0] * dxi
                #xu_downstream
                ut[k_nogc  ,j_nogc  ,i_nogc]           +=  result[1] * dxi
                ut[k_nogc  ,j_nogc  ,i_nogc_bound]     += -result[1] * dxi
                #yu_upstream
                ut[k_nogc  ,j_nogc  ,i_nogc]           += -result[2] * dyi
                ut[k_nogc  ,j_nogc-1,i_nogc]           +=  result[2] * dyi
                #yu_downstream
                ut[k_nogc  ,j_nogc  ,i_nogc]           +=  result[3] * dyi
                ut[k_nogc  ,j_nogc_bound,i_nogc]       += -result[3] * dyi
                #zu_upstream
                if not (k == grid.kstart): #1) zu_upstream is in this way implicitly set to 0 at bottom layer (no-slip BC), and 2) ghost cell is not assigned.
                    ut[k_nogc-1,j_nogc  ,i_nogc]       +=  result[4] * dzi[k_1gc-1]
                    ut[k_nogc  ,j_nogc  ,i_nogc]       += -result[4] * dzi[k_1gc]
                #zu_downstream
                if not (k == (grid.kend - 1)): #1) zu_downstream is in this way implicitly set to 0 at top layer, and 2) ghost cell is not assigned.
                    ut[k_nogc  ,j_nogc  ,i_nogc]       +=  result[5] * dzi[k_1gc]
                    ut[k_nogc+1,j_nogc  ,i_nogc]       += -result[5] * dzi[k_1gc+1]
                #xv_upstream
                vt[k_nogc  ,j_nogc  ,i_nogc]           += -result[6] * dxi
                vt[k_nogc  ,j_nogc  ,i_nogc-1]         +=  result[6] * dxi
                #xv_downstream
                vt[k_nogc  ,j_nogc  ,i_nogc]           +=  result[7] * dxi
                vt[k_nogc  ,j_nogc  ,i_nogc_bound]     += -result[7] * dxi
                #yv_upstream
                vt[k_nogc  ,j_nogc  ,i_nogc]           += -result[8] * dyi
                vt[k_nogc  ,j_nogc-1,i_nogc]           +=  result[8] * dyi
                #yv_downstream
                vt[k_nogc  ,j_nogc  ,i_nogc]           +=  result[9] * dyi
                vt[k_nogc  ,j_nogc_bound,i_nogc]       += -result[9] * dyi
                #zv_upstream
                if not (k == grid.kstart): #1) zv_upstream is in this way implicitly set to 0 at bottom layer (no-slip BC), and 2) ghost cell is not assigned.
                    vt[k_nogc-1,j_nogc  ,i_nogc]       +=  result[10] * dzi[k_1gc-1]
                    vt[k_nogc  ,j_nogc  ,i_nogc]       += -result[10] * dzi[k_1gc]
                #zv_downstream
                if not (k == (grid.kend - 1)): #1) zv_downstream is in this way implicitly set to 0 at top layer (no-slip BC), and 2) ghost cell is not assigned.
                    vt[k_nogc  ,j_nogc  ,i_nogc]       +=  result[11] * dzi[k_1gc]
                    vt[k_nogc+1,j_nogc  ,i_nogc]       += -result[11] * dzi[k_1gc+1]
                #xw_upstream
                if not (k == grid.kstart): #Don't adjust wt for bottom layer, should stay 0
                    wt[k_nogc  ,j_nogc  ,i_nogc]       += -result[12] * dxi
                    wt[k_nogc  ,j_nogc  ,i_nogc-1]     +=  result[12] * dxi
                    #xw_downstream
                    wt[k_nogc  ,j_nogc  ,i_nogc]       +=  result[13] * dxi
                    wt[k_nogc  ,j_nogc  ,i_nogc_bound] += -result[13] * dxi
                    #yw_upstream
                    wt[k_nogc  ,j_nogc  ,i_nogc]       += -result[14] * dyi
                    wt[k_nogc  ,j_nogc-1,i_nogc]       +=  result[14] * dyi
                    #yw_downstream
                    wt[k_nogc  ,j_nogc      ,i_nogc]   +=  result[15] * dyi
                    wt[k_nogc  ,j_nogc_bound,i_nogc]   += -result[15] * dyi
                    #zw_upstream
                    if not (k == (grid.kstart + 1)): #Don't adjust wt for bottom layer, should stay 0
                        wt[k_nogc-1,j_nogc  ,i_nogc]   +=  result[16] * dzhi[k_1gc-1]
                    wt[k_nogc  ,j_nogc      ,i_nogc]   += -result[16] * dzhi[k_1gc] 
                    #zw_downstream
                    wt[k_nogc  ,j_nogc      ,i_nogc]   +=  result[17] * dzhi[k_1gc]
                    if not (k == (grid.kend - 1)):
                        wt[k_nogc+1,j_nogc  ,i_nogc]   += -result[17] * dzhi[k_1gc-1] #NOTE: although this does not change wt at the bottom layer, it is still not included for k=0 to keep consistency between the top and bottom of the domain.
                 
                #Execute for each iteration in the first layer above the bottom layer, and for each iteration in the top layer, the MLP for a second grid cell to calculate 'missing' zw-values.
                if (k == (grid.kend - 1)) or (k == (grid.kstart+1)):
                    #Determine the second grid cell for which the MLP has to be evaluated based on the offset
                    if offset == 1:
                        i_2grid = i - 1
                    elif offset == 0:
                        i_2grid = i + 1
                    else:
                        raise RuntimeError("The offset used to do the inference with the MLP has an invalid value.")
                    #Select input values for second grid cell
                    input_u_val2 = np.expand_dims(u[k-b:k+b+1,j-b:j+b+1,i_2grid-b:i_2grid+b+1].flatten(),axis=0)
                    input_v_val2 = np.expand_dims(v[k-b:k+b+1,j-b:j+b+1,i_2grid-b:i_2grid+b+1].flatten(),axis=0)
                    input_w_val2 = np.expand_dims(w[k-b:k+b+1,j-b:j+b+1,i_2grid-b:i_2grid+b+1].flatten(),axis=0)

                    #Execute MLP for selected second grid cell
                    result2 = MLP.predict(input_u_val2, input_v_val2, input_w_val2, input_utau_ref_val)

                    #Store results in initialized arrays in nc-file
                    #NOTE1: compensate indices for lack of ghost cells
                    #NOTE2: flatten 'result' matrix to have consistent shape for output arrays
                    result2 = result2.flatten()
                    i_nogc2 = i_2grid - grid.istart
                    j_nogc2 = j - grid.jstart
                    k_nogc2 = k - grid.kstart
                    k_1gc2  = k_nogc2 + 1

                    if k == (grid.kstart+1):
                    #    unres_tau_zw_CNN[k_nogc2-1 ,j_nogc2 ,i_nogc2] =  result2[16] #zw_upstream
                        wt[k_nogc2,j_nogc2,i_nogc2]                  += -result2[16] * dzhi[k_1gc2]
                    else:
                    #    unres_tau_zw_CNN[k_nogc2   ,j_nogc2 ,i_nogc2] =  result2[17] #zw_downstream
                        wt[k_nogc2,j_nogc2,i_nogc2]                  +=  result2[17] * dzhi[k_1gc2]
    return ut, vt, wt
