#!/bin/bash
export MLPNUM=53
#Command used to create frozen graph:
cd ~/microhh/cases/moser600/git_repository/Neural\ Network/
python3 optimize_for_inference.py --input /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP${MLPNUM}/inference_graph.pbtxt --output /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP${MLPNUM}/optimized_inference_graph.pb --input_names input_u,input_v,input_w --placeholder_type_enum 1,1,1 --output_names output_layer_denorm,save/restore_all --frozen_graph=False #integers refer to data types in DataType enum; 1 is tf.float32, 9 is tf.int64
python3 freeze_graph.py --input_graph /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP${MLPNUM}/optimized_inference_graph.pb --input_checkpoint /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP${MLPNUM}/model.ckpt-2000000 --output_graph /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP${MLPNUM}/frozen_inference_graph.pb --output_node_names output_layer_denorm
#Get remark of new intialized thread pool with default settings for inter_op_parallelism_threads
#Extract weights from frozen graph for manual inference
python3 extract_weights_frozen_graph.py --frozen_graph_filename /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP${MLPNUM}/frozen_inference_graph.pb --variables_filepath /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP${MLPNUM}/


#GO TO LISA, and execute commands below
#convert-to-uff convert-to-uff ~/inference_firstMLP/frozen_graph/MLP1.pb -o uff/MLP1.uff -O outputs/BiasAdd
#REMARKS BELOW ONLY FOR CNN:
#Get remark DEBUG: convert reshape to flatten node
#Get remark warning: No conversion function registered for layer: Conv3D yet. Converting conv1/Conv3D as custom op: Conv3D.
