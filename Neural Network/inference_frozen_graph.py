#Script to load frozen model and do inference. Parts of the code are adopted from: 'https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc' (11 July 2019).
#Author: Robin Stoffer (robin.stoffer@wur.nl
import argparse
import tensorflow as tf
import numpy as np
import netCDF4 as nc

def load_graph(frozen_graph_filename):
    #Load protopub file from disk (i.e. the frozen graph) and parse itto to to retrieve the unserialized graph_def
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
    # Pass filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_graph_filename", default="frozen_graph.pb")
    parser.add_argument("--training_filename", default="training_data.nc")
    parser.add_argument("--batch_size", default=1000)
    args = parser.parse_args()
    batch_size = int(args.batch_size)

    #Load graph
    graph = load_graph(args.frozen_graph_filename)

    #List ops in graph
    for op in graph.get_operations():
        print(op.name)

    ###Extract flow fields from netCDF file###
    #Specify time steps NOTE: SHOULD BE 27 TO 30 to access validation fields, CHECK WHETHER THIS IS STILL CONSISTENT!
    tstart = 27
    tend   = 30
    #
    flowfields = nc.Dataset(args.training_filename)
    #u = np.array(flowfields['uc'][tstart:tend,:,:,:])
    #v = np.array(flowfields['vc'][tstart:tend,:,:,:])
    #w = np.array(flowfields['wc'][tstart:tend,:,:,:])
    utau_ref = np.array(flowfields['utau_ref'][:],dtype='f4')


    #Generate random input matrices
    input_u_val               = np.ones((batch_size, 125))
    input_v_val               = np.ones((batch_size, 125))
    input_w_val               = np.ones((batch_size, 125))
    #input_flag_topwall_val    = np.squeeze(np.ones((batch_size, 1)), axis=1) #Mask everything, squeeze to get correct shape
    input_flag_topwall_val    = np.squeeze(np.zeros((batch_size, 1)), axis=1) #Don't mask anything, squeeze to get correct shape
    #input_flag_bottomwall_val = np.squeeze(np.ones((batch_size, 1)), axis=1) #Mask everything, squeeze to get correct shape
    input_flag_bottomwall_val = np.squeeze(np.zeros((batch_size, 1)), axis=1) #Don't mask anything, squeeze to get correct shape
    input_utau_ref_val        = utau_ref

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
