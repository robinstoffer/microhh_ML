#Script to load frozen tensorflow graph or extract variables from it, code partly taken from: 'https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc' (11 July 2019).
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import numpy as np

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

def extract_weights_graph(frozen_graph_filename):
    #Load protopub file from disk (i.e. the frozen graph) and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #Import graph_def into a new Graph and return it
    with tf.Graph().as_default() as graph: #Make sure to define a new graph
        # The name var will prefix every op/nodes in the graph.
        # Since everything is loaded in a new graph, this is not needed.
        tf.import_graph_def(graph_def, name='')
        const_nodes = [n for n in graph_def.node if n.op == 'Const']

    for n in range(len(const_nodes)):
        print(n, const_nodes[n].name)

    #List of all nodes/layers in const_nodes that need to be saved in text files for manual implementation of the MLP.
    nodes_idx_name = {
            'means_inputs'       : 0,
            'stdevs_inputs'      : 1,
            'means_labels'       : 2,
            'stdevs_labels'      : 3,
            'MLPu_hidden_kernel' : 33,
            'MLPu_hidden_bias'   : 34,
            'MLPu_hidden_alpha'  : 35,
            'MLPu_output_kernel' : 36,
            'MLPu_output_bias'   : 37,
            'MLPv_hidden_kernel' : 49,
            'MLPv_hidden_bias'   : 50,
            'MLPv_hidden_alpha'  : 51,
            'MLPv_output_kernel' : 52,
            'MLPv_output_bias'   : 53,
            'MLPw_hidden_kernel' : 65,
            'MLPw_hidden_bias'   : 66,
            'MLPw_hidden_alpha'  : 67,
            'MLPw_output_kernel' : 68,
            'MLPw_output_bias'   : 69,
            'output_denorm_utau2': 71,
            }

    #Loop over all nodes/layers specified above to change them into numpy arrays and subsequently store them in text files.
    for i in nodes_idx_name.keys():
        values = tensor_util.MakeNdarray(const_nodes[nodes_idx_name[i]].attr['value'].tensor)
        #if len(values.shape) == 2:
        #    print(values.shape)
        #    #Flatten weight arrays to ease the manual implementation of the MLP
        #    values = values.flatten()
        #Make sure all arrays are at least 1D (needed for np.savetxt below)
        values = np.atleast_1d(values)
        np.savetxt(i+'.txt', values)

