#!/bin/bash

#Visualize frozen graph MLP
python3 import_pb_to_tensorboard.py --model_dir /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP10/frozen_inference_graph.pb --log_dir /home/robinst/microhh/cases/moser600/git_repository/Neural\ Network/predictions_real_data_MLP10

##Visualize full graph MLP
#python3 convert_pbtxt_to_pb.py --model_dir /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP9/ --output_dir /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP9/
#python3 import_pb_to_tensorboard.py --model_dir /home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP9/graph.pb --log_dir /home/robinst/microhh/cases/moser600/git_repository/Neural\ Network/tensorboard_full_graph/
