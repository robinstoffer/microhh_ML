#Script to load frozen model and do inference. Parts of the code are adopted from: 'https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc' (11 July 2019).
#Author: Robin Stoffer (robin.stoffer@wur.nl
import argparse
import tensorflow as tf

def load_graph(frozen_graph_filename):
    #Load protopub file from disk (i.e. the frozen graph) and parse itto to to retrieve the unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #Import graph_def into a new Graph and return it
    with tf.Graph().as_default(): #Make sure to define a new graph
        # The name var will prefix every op/nodes in the graph.
        # Since everything is loaded in a new graph, this is not needed.
        tf.import_graph_def(graph_def)

if __name__ == '__main__':
    # Pass filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_graph_filename", default="frozen_graph.pb")
    args = parser.parse_args()

    #Load graph
    graph = load_graph(args.frozen_graph_filename)

    #Verify ops in graph
    for op in graph.get_operations():
        print(op.name)
