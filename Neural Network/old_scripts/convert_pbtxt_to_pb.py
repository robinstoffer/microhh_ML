#Author: Robin Stoffer (robin.stoffer@wur.nl
import argparse
import tensorflow as tf
from google.protobuf import text_format

def convert_pbtxt_to_pb(model_dir, output_dir):
    with open(model_dir + "graph.pbtxt") as f:
        text_graph = f.read()
    graph_def = text_format.Parse(text_graph, tf.GraphDef())
    tf.train.write_graph(graph_def, output_dir, "graph.pb", as_text=False)

if __name__ == '__main__':
    # Pass filenames as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    convert_pbtxt_to_pb(args.model_dir, args.output_dir)

