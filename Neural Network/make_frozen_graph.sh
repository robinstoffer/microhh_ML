#!/bin/bash
#Command used to create frozen graph:
cd ~/microhh/cases/moser600/git_repository/Neural\ Network/
python3 optimize_for_inference.py --input /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP7/graph.pbtxt --output /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP7/optimized_graph.pbtxt --input_names input_u,input_v,input_w,input_p,input_height,utau_ref --output_names output_denorm,save/restore_all --frozen_graph=False
python3 freeze_graph.py --input_graph /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP7/optimized_graph.pbtxt --input_checkpoint /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP7/model.ckpt-0 --output_graph /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP7/frozen_graph.pb --output_node_names output_denorm
#Get remark of new intialized thread pool with default settings for inter_op_parallelism_threads

#GO TO LISA, and execute commands below
#convert-to-uff convert-to-uff ~/inference_firstMLP/frozen_graph/MLP1.pb -o uff/MLP1.uff -O outputs/BiasAdd
#REMARKS BELOW ONLY FOR CNN:
#Get remark DEBUG: convert reshape to flatten node
#Get remark warning: No conversion function registered for layer: Conv3D yet. Converting conv1/Conv3D as custom op: Conv3D.
