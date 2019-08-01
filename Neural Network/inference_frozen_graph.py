#Script to load frozen model and do inference. Parts of the code are adopted from: 'https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc' (11 July 2019).
#Author: Robin Stoffer (robin.stoffer@wur.nl
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
    parser.add_argument("--frozen_graph_filename", default="frozen_graph.pb")
    parser.add_argument("--training_filename", default="training_data.nc")
    parser.add_argument("--inference_filename", default="inference_reconstructed_fields.nc")
    parser.add_argument("--batch_size", default=1000)
    args = parser.parse_args()
    batch_size = int(args.batch_size)

    #Load graph
    graph = load_graph(args.frozen_graph_filename)

    #List ops in graph
    for op in graph.get_operations():
        print(op.name)

    ###Extract flow fields and from netCDF file###
    #Specify time steps NOTE: SHOULD BE 27 TO 30 to access validation fields, CHECK WHETHER THIS IS STILL CONSISTENT!
    tstart = 27
    tend   = 30
    tstep_unique = np.linspace(tstart+1,tend, num=3)
    print(tstep_unique)
    nt = tend - tstart
    #
    flowfields = nc.Dataset(args.training_filename)
    u = np.array(flowfields['uc'][tstart:tend,:,:,:])
    v = np.array(flowfields['vc'][tstart:tend,:,:,:])
    w = np.array(flowfields['wc'][tstart:tend,:,:,:])
    utau_ref = np.array(flowfields['utau_ref'][:],dtype='f4')
    #
    #Extract coordinates, shape fields, and ghost cells
    zc       = np.array(flowfields['zc'][:])
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
    igc        = int(a['igc'][:])
    jgc        = int(a['jgc'][:])
    kgc        = int(a['kgc_center'][:])
    iend       = int(a['iend'][:])
    jend       = int(a['jend'][:])
    kend       = int(a['kend'][:])
    #
    
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
    var_unres_tau_xu_CNN = d.createVariable("unres_tau_xu_CNN","f8",("tstep_unique","zc","yc","xgcextra"))      
    var_unres_tau_xv_CNN = d.createVariable("unres_tau_xv_CNN","f8",("tstep_unique","zc","yhcless","xhc"))     
    var_unres_tau_xw_CNN = d.createVariable("unres_tau_xw_CNN","f8",("tstep_unique","zhcless","yc","xhc"))     
    var_unres_tau_yu_CNN = d.createVariable("unres_tau_yu_CNN","f8",("tstep_unique","zc","yhc","xhcless"))
    var_unres_tau_yv_CNN = d.createVariable("unres_tau_yv_CNN","f8",("tstep_unique","zc","ygcextra","xc"))
    var_unres_tau_yw_CNN = d.createVariable("unres_tau_yw_CNN","f8",("tstep_unique","zhcless","yhc","xc"))
    var_unres_tau_zu_CNN = d.createVariable("unres_tau_zu_CNN","f8",("tstep_unique","zhc","yc","xhcless"))
    var_unres_tau_zv_CNN = d.createVariable("unres_tau_zv_CNN","f8",("tstep_unique","zhc","yhcless","xc"))
    var_unres_tau_zw_CNN = d.createVariable("unres_tau_zw_CNN","f8",("tstep_unique","zgcextra","yc","xc")) 

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
    input_flag_topwall    = graph.get_tensor_by_name('flag_topwall:0')
    input_flag_bottomwall = graph.get_tensor_by_name('flag_bottomwall:0')
    input_utau_ref        = graph.get_tensor_by_name('input_utau_ref:0')
    output                = graph.get_tensor_by_name('output_layer_denorm:0')
    
    with tf.Session(graph=graph) as sess:
    #Loop over flow fields, for each time step in tstep_unique (giving 4 loops in total).
    #For each alternating grid cell, store transport components by calling the 'frozen' MLP within a tf.Session().
        for t in range(tstep_unique):
            for k in range(kgc,kend,2):
                for j in range(jgc,jend,2):
                    for i in range(igc,iend,2):
                        input_u_val = u[t,k-2:k+3,j-2:j+3,i-2:i+3]
                        input_v_val = v[t,k-2:k+3,j-2:j+3,i-2:i+3]
                        input_w_val = w[t,k-2:k+3,j-2:j+3,i-2:i+3]


        #Execute MLP once for selected grid cell
        #NOTE: no need to initialize/restore anything as there are only constants in the graph.
        result = sess.run(output, feed_dict={
            input_u:               input_u_val,
            input_v:               input_v_val,
            input_w:               input_w_val,
            input_flag_topwall:    input_flag_topwall_val,
            input_flag_bottomwall: input_flag_bottomwall_val,
            input_utau_ref:        input_utau_ref_val
            })

        print(result.shape)
        print(result)
