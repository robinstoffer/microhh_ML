#Script to load frozen model and do inference. Parts of the code are adopted from: 'https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc' (11 July 2019).
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import argparse
import tensorflow as tf
import numpy as np
import netCDF4 as nc

def load_graph(frozen_graph_filename):
    #Load protopub file from disk (i.e. the frozen graph) and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #Import graph_def into a new Graph and return it
    with tf.Graph().as_default() as graph: #Make sure to define a new graph
        # The name var will prefix every op/nodes in the graph.
        # Since everything is loaded in a new graph, this is not needed.
        tf.import_graph_def(graph_def, name='')
    return graph

if __name__ == '__main__':
    # Pass filenames and batch size as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_graph_filename", default="frozen_inference_graph.pb")
    parser.add_argument("--training_filename", default="training_data.nc")
    parser.add_argument("--inference_filename", default="inference_reconstructed_fields.nc")
    #parser.add_argument("--batch_size", default=1000)
    args = parser.parse_args()
    #batch_size = int(args.batch_size)

    #Load graph
    graph = load_graph(args.frozen_graph_filename)

    ##List ops in graph
    #for op in graph.get_operations():
    #    print(op.name)

    ###Extract flow fields and from netCDF file###
    #Specify time steps NOTE: SHOULD BE 27 TO 30 to access validation fields, CHECK WHETHER THIS IS STILL CONSISTENT!
    #NOTE: flow fields were normalized stored, UNDO first normalisation!
    tstart = 27
    tend   = 30
    tstep_unique = np.linspace(tstart,tend-1, num=3)
    nt = tend - tstart
    #
    flowfields = nc.Dataset(args.training_filename)
    utau_ref_channel = np.array(flowfields['utau_ref'][:],dtype='f4')
    u = np.array(flowfields['uc'][tstart:tend,:,:,:]) * utau_ref_channel
    v = np.array(flowfields['vc'][tstart:tend,:,:,:]) * utau_ref_channel
    w = np.array(flowfields['wc'][tstart:tend,:,:,:]) * utau_ref_channel
    utau_ref = 0.2 #in [m/s]
    #
    unres_tau_xu = np.array(flowfields['unres_tau_xu_turb'] [tstart:tend,:,:,:]) * (utau_ref_channel ** 2)
    unres_tau_yu = np.array(flowfields['unres_tau_yu_turb'] [tstart:tend,:,:,:]) * (utau_ref_channel ** 2) 
    unres_tau_zu = np.array(flowfields['unres_tau_zu_turb'] [tstart:tend,:,:,:]) * (utau_ref_channel ** 2) 
    unres_tau_xv = np.array(flowfields['unres_tau_xv_turb'] [tstart:tend,:,:,:]) * (utau_ref_channel ** 2) 
    unres_tau_yv = np.array(flowfields['unres_tau_yv_turb'] [tstart:tend,:,:,:]) * (utau_ref_channel ** 2) 
    unres_tau_zv = np.array(flowfields['unres_tau_zv_turb'] [tstart:tend,:,:,:]) * (utau_ref_channel ** 2) 
    unres_tau_xw = np.array(flowfields['unres_tau_xw_turb'] [tstart:tend,:,:,:]) * (utau_ref_channel ** 2) 
    unres_tau_yw = np.array(flowfields['unres_tau_yw_turb'] [tstart:tend,:,:,:]) * (utau_ref_channel ** 2) 
    unres_tau_zw = np.array(flowfields['unres_tau_zw_turb'] [tstart:tend,:,:,:]) * (utau_ref_channel ** 2) 

    #
    #Extract coordinates, shape fields, and ghost cells
    zc       = np.array(flowfields['zc'][:])
    zgc      = np.array(flowfields['zgc'][:])
    nz       = len(zc)
    zhc      = np.array(flowfields['zhc'][:])
    zgcextra = np.array(flowfields['zgcextra'][:])
    yc       = np.array(flowfields['yc'][:])
    ny       = len(yc)
    yhc      = np.array(flowfields['yhc'][:])
    ygcextra = np.array(flowfields['ygcextra'][:])
    xc       = np.array(flowfields['xc'][:])
    nx       = len(xc)
    xhc      = np.array(flowfields['xhc'][:])
    xgcextra = np.array(flowfields['xgcextra'][:])
    zhcless  = zhc[:-1]
    yhcless  = yhc[:-1]
    xhcless  = xhc[:-1]
    igc      = int(flowfields['igc'][:])
    jgc      = int(flowfields['jgc'][:])
    kgc      = int(flowfields['kgc_center'][:])
    iend     = int(flowfields['iend'][:])
    ihend    = int(flowfields['ihend'][:])
    jend     = int(flowfields['jend'][:])
    jhend    = int(flowfields['jhend'][:])
    kend     = int(flowfields['kend'][:])
    khend    = int(flowfields['khend'][:])
    #

    #Calculate grid distances, ASSUMING a second-order numerical scheme for calculation of the heights
    xsize   = xhc[-1]
    itot    = len(xc)
    dx      = xsize / itot
    dxi     = 1. / dx
    #
    ysize   = yhc[-1]
    jtot    = len(yc)
    dy      = ysize / jtot
    dyi     = 1. / dy
    #
    ktot    = len(zc)
    zgc1    = zgc[kgc-1:kend+1] #Include one ghost cell at each side
    dzh     = np.zeros(ktot + 2) #Include two ghost cells for heights
    dzh[1:] = zgc1[1:] - zgc1[:-1]
    dzh[0]  = dzh[2]
    dzhi    = 1. / dzh
    #
    dz      = np.zeros(ktot + 2)
    dz[1:-1]  = zhc[1:]  - zhc[:-1]
    dz[0]   = dz[1]
    dz[-1]  = dz[-2]
    dzi     = 1. / dz
    
    ###Create file for inference results###
    inference = nc.Dataset(args.inference_filename, 'w')

    #Create dimensions for storage in nc-file
    inference.createDimension("zc", len(zc))
    inference.createDimension("zgcextra", len(zgcextra))
    inference.createDimension("zhc",len(zhc))
    inference.createDimension("zhcless",len(zhcless))
    inference.createDimension("yc", len(yc))
    inference.createDimension("ygcextra", len(ygcextra))
    inference.createDimension("yhc",len(yhc))
    inference.createDimension("yhcless",len(yhcless))
    inference.createDimension("xc", len(xc))
    inference.createDimension("xgcextra", len(xgcextra))
    inference.createDimension("xhc",len(xhc))
    inference.createDimension("xhcless",len(xhcless))
    inference.createDimension("tstep_unique",len(tstep_unique))

    #Create variables for dimensions and store them
    var_zc           = inference.createVariable("zc",           "f8", ("zc",))
    var_zgcextra     = inference.createVariable("zgcextra",     "f8", ("zgcextra",))
    var_zhc          = inference.createVariable("zhc",          "f8", ("zhc",))
    var_zhcless      = inference.createVariable("zhcless",      "f8", ("zhcless",))
    var_yc           = inference.createVariable("yc",           "f8", ("yc",))
    var_ygcextra     = inference.createVariable("ygcextra",     "f8", ("ygcextra",))
    var_yhc          = inference.createVariable("yhc",          "f8", ("yhc",))
    var_yhcless      = inference.createVariable("yhcless",      "f8", ("yhcless",))
    var_xc           = inference.createVariable("xc",           "f8", ("xc",))
    var_xgcextra     = inference.createVariable("xgcextra",     "f8", ("xgcextra",))
    var_xhc          = inference.createVariable("xhc",          "f8", ("xhc",))
    var_xhcless      = inference.createVariable("xhcless",      "f8", ("xhcless",))
    var_tstep_unique = inference.createVariable("tstep_unique", "f8", ("tstep_unique",))

    var_zc[:]            = zc
    var_zgcextra[:]      = zgcextra
    var_zhc[:]           = zhc
    var_zhcless[:]       = zhcless
    var_yc[:]            = yc
    var_ygcextra[:]      = ygcextra
    var_yhc[:]           = yhc
    var_yhcless[:]       = yhcless
    var_xc[:]            = xc
    var_xgcextra[:]      = xgcextra
    var_xhc[:]           = xhc
    var_xhcless[:]       = xhcless
    var_tstep_unique[:]  = tstep_unique
    
    #Initialize variables for storage inference results
    var_unres_tau_xu_CNN = inference.createVariable("unres_tau_xu_CNN","f8",("tstep_unique","zc","yc","xgcextra"))      
    var_unres_tau_xv_CNN = inference.createVariable("unres_tau_xv_CNN","f8",("tstep_unique","zc","yhcless","xhc"))     
    var_unres_tau_xw_CNN = inference.createVariable("unres_tau_xw_CNN","f8",("tstep_unique","zhcless","yc","xhc"))     
    var_unres_tau_yu_CNN = inference.createVariable("unres_tau_yu_CNN","f8",("tstep_unique","zc","yhc","xhcless"))
    var_unres_tau_yv_CNN = inference.createVariable("unres_tau_yv_CNN","f8",("tstep_unique","zc","ygcextra","xc"))
    var_unres_tau_yw_CNN = inference.createVariable("unres_tau_yw_CNN","f8",("tstep_unique","zhcless","yhc","xc"))
    var_unres_tau_zu_CNN = inference.createVariable("unres_tau_zu_CNN","f8",("tstep_unique","zhc","yc","xhcless"))
    var_unres_tau_zv_CNN = inference.createVariable("unres_tau_zv_CNN","f8",("tstep_unique","zhc","yhcless","xc"))
    var_unres_tau_zw_CNN = inference.createVariable("unres_tau_zw_CNN","f8",("tstep_unique","zc","yc","xc"))
    #
    var_ut = inference.createVariable("u_tendency","f8",("tstep_unique","zc","yc","xhcless"))
    var_vt = inference.createVariable("v_tendency","f8",("tstep_unique","zc","yhcless","xc"))
    var_wt = inference.createVariable("w_tendency","f8",("tstep_unique","zhcless","yc","xc"))
    #
    #var_unres_tau_xu_lbls = inference.createVariable("unres_tau_xu_lbls","f8",("tstep_unique","zc","yc","xgcextra"))
    #var_unres_tau_xv_lbls = inference.createVariable("unres_tau_xv_lbls","f8",("tstep_unique","zc","yhcless","xhc"))
    #var_unres_tau_xw_lbls = inference.createVariable("unres_tau_xw_lbls","f8",("tstep_unique","zhcless","yc","xhc"))
    #var_unres_tau_yu_lbls = inference.createVariable("unres_tau_yu_lbls","f8",("tstep_unique","zc","yhc","xhcless"))
    #var_unres_tau_yv_lbls = inference.createVariable("unres_tau_yv_lbls","f8",("tstep_unique","zc","ygcextra","xc"))
    #var_unres_tau_yw_lbls = inference.createVariable("unres_tau_yw_lbls","f8",("tstep_unique","zhcless","yhc","xc"))
    #var_unres_tau_zu_lbls = inference.createVariable("unres_tau_zu_lbls","f8",("tstep_unique","zhc","yc","xhcless"))
    #var_unres_tau_zv_lbls = inference.createVariable("unres_tau_zv_lbls","f8",("tstep_unique","zhc","yhcless","xc"))
    #var_unres_tau_zw_lbls = inference.createVariable("unres_tau_zw_lbls","f8",("tstep_unique","zgcextra","yc","xc"))

    ##Generate random input matrices
    #input_u_val               = np.ones((batch_size, 125))
    #input_v_val               = np.ones((batch_size, 125))
    #input_w_val               = np.ones((batch_size, 125))
    ##input_flag_topwall_val    = np.squeeze(np.ones((batch_size, 1)), axis=1) #Mask everything, squeeze to get correct shape
    #input_flag_topwall_val    = np.squeeze(np.zeros((batch_size, 1)), axis=1) #Don't mask anything, squeeze to get correct shape
    ##input_flag_bottomwall_val = np.squeeze(np.ones((batch_size, 1)), axis=1) #Mask everything, squeeze to get correct shape
    #input_flag_bottomwall_val = np.squeeze(np.zeros((batch_size, 1)), axis=1) #Don't mask anything, squeeze to get correct shape
    #input_utau_ref_val        = utau_ref

    #Access input and output nodes
    #NOTE: specify ':0' to select the correct output of the ops and get the tensors themselves
    input_u               = graph.get_tensor_by_name('input_u:0')
    input_v               = graph.get_tensor_by_name('input_v:0')
    input_w               = graph.get_tensor_by_name('input_w:0')
    #input_flag_topwall    = graph.get_tensor_by_name('flag_topwall:0')
    #input_flag_bottomwall = graph.get_tensor_by_name('flag_bottomwall:0')
    input_utau_ref        = graph.get_tensor_by_name('input_utau_ref:0')
    output                = graph.get_tensor_by_name('output_layer_denorm:0')
    
    #Calculate grid dimensions and start indices
    icells  = iend + igc
    #ihcells = ihend + igc
    jcells  = jend + jgc
    #jhcells = jhend + jgc
    ijcells = icells * jcells
    kcells  = kend + kgc
    khcells = khend + kgc
    #ncells  = ijcells * kcells
    istart  = igc
    jstart  = jgc
    kstart  = kgc
    
    with tf.Session(graph=graph) as sess:
    #Loop over flow fields, for each time step in tstep_unique (giving 4 loops in total).
    #For each alternating grid cell, store transport components by calling the 'frozen' MLP within a tf.Session().
        for t in range(nt):
            
            #Initialize empty arrays for temporary storage transport components
            unres_tau_xu_CNN = np.full((len(zc),len(yc),len(xgcextra)), np.nan, dtype=np.float32)
            unres_tau_xv_CNN = np.full((len(zc),len(yhcless),len(xhc)), np.nan, dtype=np.float32)
            unres_tau_xw_CNN = np.full((len(zhcless),len(yc),len(xhc)), np.nan, dtype=np.float32)
            unres_tau_yu_CNN = np.full((len(zc),len(yhc),len(xhcless)), np.nan, dtype=np.float32)
            unres_tau_yv_CNN = np.full((len(zc),len(ygcextra),len(xc)), np.nan, dtype=np.float32)
            unres_tau_yw_CNN = np.full((len(zhcless),len(yhc),len(xc)), np.nan, dtype=np.float32)
            unres_tau_zu_CNN = np.full((len(zhc),len(yc),len(xhcless)), np.nan, dtype=np.float32)
            unres_tau_zv_CNN = np.full((len(zhc),len(yhcless),len(xc)), np.nan, dtype=np.float32)
            unres_tau_zw_CNN = np.full((len(zc),len(yc),len(xc)),       np.nan, dtype=np.float32)
            #
            #unres_tau_xu_lbls = np.full((len(zc),len(yc),len(xgcextra)), np.nan, dtype=np.float32)
            #unres_tau_xv_lbls = np.full((len(zc),len(yhcless),len(xhc)), np.nan, dtype=np.float32)
            #unres_tau_xw_lbls = np.full((len(zhcless),len(yc),len(xhc)), np.nan, dtype=np.float32)
            #unres_tau_yu_lbls = np.full((len(zc),len(yhc),len(xhcless)), np.nan, dtype=np.float32)
            #unres_tau_yv_lbls = np.full((len(zc),len(ygcextra),len(xc)), np.nan, dtype=np.float32)
            #unres_tau_yw_lbls = np.full((len(zhcless),len(yhc),len(xc)), np.nan, dtype=np.float32)
            #unres_tau_zu_lbls = np.full((len(zhc),len(yc),len(xhcless)), np.nan, dtype=np.float32)
            #unres_tau_zv_lbls = np.full((len(zhc),len(yhcless),len(xc)), np.nan, dtype=np.float32)
            #unres_tau_zw_lbls = np.full((len(zc),len(yc),len(xc)),       np.nan, dtype=np.float32)

            #Select flow fields of time step
            u_singletimestep = u[t,:,:,:-1].flatten()#Flatten and remove ghost cells in horizontal staggered dimensions to make shape consistent to arrays in MicroHH
            v_singletimestep = v[t,:,:-1,:].flatten()
            w_singletimestep = w[t,:,:,:].flatten()
            unres_tau_xu_singletimestep = unres_tau_xu[t,:,:,:]
            unres_tau_yu_singletimestep = unres_tau_yu[t,:,:,:]
            unres_tau_zu_singletimestep = unres_tau_zu[t,:,:,:]
            unres_tau_xv_singletimestep = unres_tau_xv[t,:,:,:]
            unres_tau_yv_singletimestep = unres_tau_yv[t,:,:,:]
            unres_tau_zv_singletimestep = unres_tau_zv[t,:,:,:]
            unres_tau_xw_singletimestep = unres_tau_xw[t,:,:,:]
            unres_tau_yw_singletimestep = unres_tau_yw[t,:,:,:]
            unres_tau_zw_singletimestep = unres_tau_zw[t,:,:,:]
           
            #Define loop indices
            #ii = 1
            #jj = icells
            #kk = ijcells
            blocksize = 5 #size of block used as input for MLP
            b = blocksize // 2 

            ###The code block starting below  is roughly the only part that has to be executed when doing inference in MicroHH###
            
            #Initialize zeros arrays for tendencies
            ut = np.zeros((len(zc),len(yc),len(xc)))
            vt = np.zeros((len(zc),len(yc),len(xc)))
            wt = np.zeros((len(zc),len(yc),len(xc)))
            
            #NOTE: several expand_dims included to account for batch dimension
            #Reshape 1d arrays to 3d, which is much more convenient for the slicing below.
            u_singletimestep = np.reshape(u_singletimestep, (kcells,jcells,icells))
            v_singletimestep = np.reshape(v_singletimestep, (kcells,jcells,icells))
            w_singletimestep = np.reshape(w_singletimestep, (khcells,jcells,icells))

            #Extract friction velocity
            input_utau_ref_val = utau_ref_channel
            
            #NOTE: offset factors are defined to ensure alternate sampling
            for k in range(kstart,kend,1):
                
                ##Test whether grid cells are located at the bottom or top wall
                #if k == kstart:
                #    input_flag_bottomwall_val = 1
                #    input_flag_topwall_val    = 0
                #elif k == kend - 1:
                #    input_flag_bottomwall_val = 0
                #    input_flag_topwall_val    = 1
                #else:
                #    input_flag_bottomwall_val = 0
                #    input_flag_topwall_val    = 0
                ##
                #input_flag_bottomwall_val = np.expand_dims(input_flag_bottomwall_val, axis=0)
                #input_flag_topwall_val    = np.expand_dims(input_flag_topwall_val, axis=0)
                #
                k_offset = k % 2
                for j in range(jstart,jend,1):
                    if k_offset != 0:
                        offset = int(j % 2 != 0) #Do offset for odd columns
                    else:
                        offset = int(j % 2 == 0) #Do offset for even columns
                    for i in range(istart+offset,iend,2):
                    #for i in range(istart,iend,2):
                        
                        ##Calculate additional needed loop indices
                        #ij = i + j*jj
                        #ijk = i + j*jj + k*kk

                        #Extract grid box flow fields
                        input_u_val = np.expand_dims(u_singletimestep[k-b:k+b+1,j-b:j+b+1,i-b:i+b+1].flatten(), axis=0) #Flatten and expand dims arrays for MLP
                        #print(input_u_val)
                        input_v_val = np.expand_dims(v_singletimestep[k-b:k+b+1,j-b:j+b+1,i-b:i+b+1].flatten(), axis=0)
                        #print(input_v_val)
                        input_w_val = np.expand_dims(w_singletimestep[k-b:k+b+1,j-b:j+b+1,i-b:i+b+1].flatten(), axis=0)
                        #print(input_w_val)
                        #raise RuntimeError("Stop run")

                        #Execute MLP once for selected grid box
                        #NOTE: no need to initialize/restore anything as there are only constants in the graph.
                        result = sess.run(output, feed_dict={
                            input_u:               input_u_val,
                            input_v:               input_v_val,
                            input_w:               input_w_val,
                            #input_flag_topwall:    input_flag_topwall_val,
                            #input_flag_bottomwall: input_flag_bottomwall_val,
                            input_utau_ref:        input_utau_ref_val
                            })

                        #Store results in initialized arrays in nc-file
                        #NOTE1: compensate indices for lack of ghost cells
                        #NOTE2: flatten 'result' matrix to have consistent shape for output arrays
                        result = result.flatten()
                        i_nogc = i - istart
                        j_nogc = j - jstart
                        k_nogc = k - kstart
                        k_1gc  = k_nogc + 1
                        unres_tau_xu_CNN[k_nogc  ,j_nogc  ,i_nogc]   = result[0] #xu_upstream
                        unres_tau_xu_CNN[k_nogc  ,j_nogc  ,i_nogc+1] = result[1] #xu_downstream
                        unres_tau_yu_CNN[k_nogc  ,j_nogc  ,i_nogc]   = result[2] #yu_upstream
                        unres_tau_yu_CNN[k_nogc  ,j_nogc+1,i_nogc]   = result[3] #yu_downstream
                        unres_tau_zu_CNN[k_nogc  ,j_nogc  ,i_nogc]   = result[4] #zu_upstream
                        unres_tau_zu_CNN[k_nogc+1,j_nogc  ,i_nogc]   = result[5] #zu_downstream
                        unres_tau_xv_CNN[k_nogc  ,j_nogc  ,i_nogc]   = result[6] #xv_upstream
                        unres_tau_xv_CNN[k_nogc  ,j_nogc  ,i_nogc+1] = result[7] #xv_downstream
                        unres_tau_yv_CNN[k_nogc  ,j_nogc  ,i_nogc]   = result[8] #yv_upstream
                        unres_tau_yv_CNN[k_nogc  ,j_nogc+1,i_nogc]   = result[9] #yv_downstream
                        unres_tau_zv_CNN[k_nogc  ,j_nogc  ,i_nogc]   = result[10] #zv_upstream
                        unres_tau_zv_CNN[k_nogc+1,j_nogc  ,i_nogc]   = result[11] #zv_downstream
                        unres_tau_xw_CNN[k_nogc,  j_nogc  ,i_nogc]   = result[12] #xw_upstream
                        unres_tau_xw_CNN[k_nogc,  j_nogc  ,i_nogc+1] = result[13] #xw_downstream
                        unres_tau_yw_CNN[k_nogc,  j_nogc  ,i_nogc]   = result[14] #yw_upstream
                        unres_tau_yw_CNN[k_nogc,  j_nogc+1,i_nogc]   = result[15] #yw_downstream
                        unres_tau_zw_CNN[k_nogc,  j_nogc  ,i_nogc]   = result[16] #zw_upstream
                        unres_tau_zw_CNN[k_nogc+1,j_nogc  ,i_nogc]   = result[17] #zw_downstream
                        #
                        #Account for all tendencies affected by calculated transport components (two tendencies for each of the predicted 18 components)
                        #Check whether a horizontal boundary is reached, and if so make use of horizontal periodic BCs. This adjustment is only needed at the downstream side of the domain, since at the upstream side the index is already automatically converted to -1.
                        if i == (iend - 1):
                            i_nogc_bound = 0
                        else:
                            i_nogc_bound = i_nogc + 1

                        if j == (jend - 1):
                            j_nogc_bound = 0
                        else:
                            j_nogc_bound = j_nogc + 1
                        #xu_upstream
                        ut[k_nogc  ,j_nogc  ,i_nogc  ] += -result[0] * dxi
                        ut[k_nogc  ,j_nogc  ,i_nogc-1] +=  result[0] * dxi
                        #xu_downstream
                        ut[k_nogc  ,j_nogc  ,i_nogc  ] +=  result[1] * dxi
                        ut[k_nogc  ,j_nogc  ,i_nogc_bound] += -result[1] * dxi
                        #yu_upstream
                        ut[k_nogc  ,j_nogc  ,i_nogc  ] += -result[2] * dyi
                        ut[k_nogc  ,j_nogc-1,i_nogc  ] +=  result[2] * dyi
                        #yu_downstream
                        ut[k_nogc  ,j_nogc  ,i_nogc  ] +=  result[3] * dyi
                        ut[k_nogc  ,j_nogc_bound,i_nogc  ] += -result[3] * dyi
                        #zu_upstream
                        if not (k_nogc == 0): #1) zu_upstream is in this way implicitly set to 0 at bottom layer (no-slip BC), and 2) ghost cell is not assigned.
                            ut[k_nogc-1,j_nogc  ,i_nogc  ] +=  result[4] * dzi[k_nogc-1]
                            ut[k_nogc  ,j_nogc  ,i_nogc  ] += -result[4] * dzi[k_nogc]
                        #zu_downstream
                        if not (k_nogc == (kend - 1)): #1) zu_downstream is in this way implicitly set to 0 at top layer, and 2) ghost cell is not assigned.
                            ut[k_nogc  ,j_nogc  ,i_nogc  ] +=  result[5] * dzi[k_nogc]
                            ut[k_nogc+1,j_nogc  ,i_nogc  ] += -result[5] * dzi[k_nogc+1]
                        #xv_upstream
                        vt[k_nogc  ,j_nogc  ,i_nogc  ] += -result[6] * dxi
                        vt[k_nogc  ,j_nogc  ,i_nogc-1] +=  result[6] * dxi
                        #xv_downstream
                        vt[k_nogc  ,j_nogc  ,i_nogc  ] +=  result[7] * dxi
                        vt[k_nogc  ,j_nogc  ,i_nogc_bound] += -result[7] * dxi
                        #yv_upstream
                        vt[k_nogc  ,j_nogc  ,i_nogc  ] += -result[8] * dyi
                        vt[k_nogc  ,j_nogc-1,i_nogc  ] +=  result[8] * dyi
                        #yv_downstream
                        vt[k_nogc  ,j_nogc  ,i_nogc  ] +=  result[9] * dyi
                        vt[k_nogc  ,j_nogc_bound,i_nogc  ] += -result[9] * dyi
                        #zv_upstream
                        if not (k_nogc == 0): #1) zu_upstream is in this way implicitly set to 0 at bottom layer (no-slip BC), and 2) ghost cell is not assigned.
                            vt[k_nogc-1,j_nogc  ,i_nogc  ] +=  result[10] * dzi[k_1gc-1]
                            vt[k_nogc  ,j_nogc  ,i_nogc  ] += -result[10] * dzi[k_1gc]
                        #zv_downstream
                        if not (k_nogc == (kend - 1)): #1) zu_downstream is in this way implicitly set to 0 at top layer (no-slip BC), and 2) ghost cell is not assigned.
                            vt[k_nogc  ,j_nogc  ,i_nogc  ] +=  result[11] * dzi[k_1gc]
                            vt[k_nogc+1,j_nogc  ,i_nogc  ] += -result[11] * dzi[k_1gc+1]
                        #xw_upstream
                        if not (k_nogc == 0): #Don't adjust wt for bottom layer, should stay 0
                            wt[k_nogc  ,j_nogc  ,i_nogc  ] += -result[12] * dxi
                            wt[k_nogc  ,j_nogc  ,i_nogc-1] +=  result[12] * dxi
                            #xw_downstream
                            wt[k_nogc  ,j_nogc  ,i_nogc  ] +=  result[13] * dxi
                            wt[k_nogc  ,j_nogc  ,i_nogc_bound] += -result[13] * dxi
                            #yw_upstream
                            wt[k_nogc  ,j_nogc  ,i_nogc  ] += -result[14] * dyi
                            wt[k_nogc  ,j_nogc-1,i_nogc  ] +=  result[14] * dyi
                            #yw_downstream
                            wt[k_nogc  ,j_nogc      ,i_nogc  ] +=  result[15] * dyi
                            wt[k_nogc  ,j_nogc_bound,i_nogc  ] += -result[15] * dyi
                            #zw_upstream
                            if not (k_nogc == 1): #Don't adjust wt for bottom layer, should stay 0
                                wt[k_nogc-1,j_nogc      ,i_nogc  ] +=  result[16] * dzhi[k_1gc-1]
                            wt[k_nogc  ,j_nogc      ,i_nogc  ] += -result[16] * dzhi[k_1gc] 
                            #zw_downstream
                            wt[k_nogc  ,j_nogc      ,i_nogc  ] +=  result[17] * dzhi[k_1gc]
                            wt[k_nogc+1,j_nogc      ,i_nogc  ] += -result[17] * dzhi[k_1gc-1] #NOTE: although this does not change wt at the bottom layer, it is still not included for k=0 to keep consistency between the top and bottom of the domain.


                        #unres_tau_xu_lbls[k_nogc ,j_nogc, i_nogc]   = unres_tau_xu_singletimestep[k_nogc ,j_nogc, i_nogc]
                        #unres_tau_xu_lbls[k_nogc ,j_nogc, i_nogc+1] = unres_tau_xu_singletimestep[k_nogc ,j_nogc, i_nogc+1]
                        #unres_tau_yu_lbls[k_nogc ,j_nogc, i_nogc]   = unres_tau_yu_singletimestep[k_nogc ,j_nogc, i_nogc]
                        #unres_tau_yu_lbls[k_nogc ,j_nogc+1, i_nogc] = unres_tau_yu_singletimestep[k_nogc ,j_nogc+1, i_nogc]
                        #unres_tau_zu_lbls[k_nogc ,j_nogc, i_nogc]   = unres_tau_zu_singletimestep[k_nogc ,j_nogc, i_nogc]
                        #unres_tau_zu_lbls[k_nogc+1 ,j_nogc, i_nogc] = unres_tau_zu_singletimestep[k_nogc+1 ,j_nogc, i_nogc]
                        #unres_tau_xv_lbls[k_nogc ,j_nogc, i_nogc]   = unres_tau_xv_singletimestep[k_nogc ,j_nogc, i_nogc]
                        #unres_tau_xv_lbls[k_nogc ,j_nogc, i_nogc+1] = unres_tau_xv_singletimestep[k_nogc ,j_nogc, i_nogc+1]
                        #unres_tau_yv_lbls[k_nogc ,j_nogc, i_nogc]   = unres_tau_yv_singletimestep[k_nogc ,j_nogc, i_nogc]
                        #unres_tau_yv_lbls[k_nogc ,j_nogc+1, i_nogc] = unres_tau_yv_singletimestep[k_nogc ,j_nogc+1, i_nogc]
                        #unres_tau_zv_lbls[k_nogc ,j_nogc, i_nogc]   = unres_tau_zv_singletimestep[k_nogc ,j_nogc, i_nogc]
                        #unres_tau_zv_lbls[k_nogc+1 ,j_nogc, i_nogc] = unres_tau_zv_singletimestep[k_nogc+1 ,j_nogc, i_nogc]
                        #unres_tau_xw_lbls[k_nogc ,j_nogc, i_nogc]   = unres_tau_xw_singletimestep[k_nogc ,j_nogc, i_nogc]
                        #unres_tau_xw_lbls[k_nogc ,j_nogc, i_nogc+1] = unres_tau_xw_singletimestep[k_nogc ,j_nogc, i_nogc+1]
                        #unres_tau_yw_lbls[k_nogc ,j_nogc, i_nogc]   = unres_tau_yw_singletimestep[k_nogc ,j_nogc, i_nogc]
                        #unres_tau_yw_lbls[k_nogc ,j_nogc+1, i_nogc] = unres_tau_yw_singletimestep[k_nogc ,j_nogc+1, i_nogc]
                        #unres_tau_zw_lbls[k_nogc ,j_nogc, i_nogc]   = unres_tau_zw_singletimestep[k_nogc ,j_nogc, i_nogc]
                        #unres_tau_zw_lbls[k_nogc+1 ,j_nogc, i_nogc] = unres_tau_zw_singletimestep[k_nogc+1 ,j_nogc, i_nogc]

            ###End code block which has to be executed in MicroHH###

            #Store temporary arrays containing transport components in nc-file, rescale with realistic friction velocity
            var_unres_tau_xu_CNN[t,:,:,:] =  unres_tau_xu_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            var_unres_tau_yu_CNN[t,:,:,:] =  unres_tau_yu_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            var_unres_tau_zu_CNN[t,:,:,:] =  unres_tau_zu_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            var_unres_tau_xv_CNN[t,:,:,:] =  unres_tau_xv_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            var_unres_tau_yv_CNN[t,:,:,:] =  unres_tau_yv_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            var_unres_tau_zv_CNN[t,:,:,:] =  unres_tau_zv_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            var_unres_tau_xw_CNN[t,:,:,:] =  unres_tau_xw_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            var_unres_tau_yw_CNN[t,:,:,:] =  unres_tau_yw_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            var_unres_tau_zw_CNN[t,:,:,:] =  unres_tau_zw_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #
            var_ut[t,:,:,:] = ut[:,:,:]
            var_vt[t,:,:,:] = vt[:,:,:]
            var_wt[t,:,:,:] = wt[:,:,:]
            #
            #var_unres_tau_xu_lbls[t,:,:,:] =  unres_tau_xu_lbls[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_yu_lbls[t,:,:,:] =  unres_tau_yu_lbls[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_zu_lbls[t,:,:,:] =  unres_tau_zu_lbls[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_xv_lbls[t,:,:,:] =  unres_tau_xv_lbls[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_yv_lbls[t,:,:,:] =  unres_tau_yv_lbls[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_zv_lbls[t,:,:,:] =  unres_tau_zv_lbls[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_xw_lbls[t,:,:,:] =  unres_tau_xw_lbls[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_yw_lbls[t,:,:,:] =  unres_tau_yw_lbls[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_zw_lbls[t,:,:,:] =  unres_tau_zw_lbls[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))


    #Close inference file
    inference.close()
