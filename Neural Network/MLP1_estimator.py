#Script that contains MLP for turbulent channel flow case without distributed learning, making use of the tf.Estimator and tf.Dataset API.
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import numpy as np
import netCDF4 as nc
import tensorflow as tf
#import horovod.tensorflow as hvd
import random
import os
import subprocess
import glob
import argparse
import matplotlib
matplotlib.use('agg')
from tensorflow.python import debug as tf_debug

#Set logging info
tf.logging.set_verbosity(tf.logging.INFO)

##Enable eager execution
#tf.enable_eager_execution()

#Amount of cores on node
ncores = int(subprocess.check_output(["nproc", "--all"]))

# Instantiate the parser
parser = argparse.ArgumentParser(description='microhh_ML')
parser.add_argument('--checkpoint_dir', type=str, default='/projects/1/flowsim/simulation1/CNN_checkpoints',
                    help='directory where checkpoints are stored')
parser.add_argument('--input_dir', type=str, default='/projects/1/flowsim/simulation1/',
                    help='directory where tfrecord files are located')
parser.add_argument('--stored_means_stdevs_filepath', type=str, default='/projects/1/flowsim/simulation1/means_stdevs_allfields.nc', \
        help='filepath for stored means and standard deviations of input variables, which should refer to a nc-file created as part of the training data')
parser.add_argument('--training_filepath', type=str, default='/projects/1/flowsim/simulation1/training_data.nc', \
        help='filepath for stored training file which should contain the friction velocity and be in netCDF-format.')
parser.add_argument('--gradients', default=None, \
        action='store_true', \
        help='Wind velocity gradients are used as input for the NN when this is true, otherwhise absolute wind velocities are used.')
parser.add_argument('--benchmark', dest='benchmark', default=None, \
        action='store_true', \
        help='Do fullrun when benchmark is false, which includes producing and storing of preditions. Furthermore, in a fullrun more variables are stored to facilitate reconstruction of the corresponding transport fields. When the benchmark flag is true, the scripts ends immidiately after calculating the validation loss to facilitate benchmark tests.')
parser.add_argument('--debug', default=None, \
        action='store_true', \
        help='Run script in debug mode to inspect tensor values while the Estimator is in training mode.')
parser.add_argument('--intra_op_parallelism_threads', type=int, default=ncores-1, \
        help='intra_op_parallelism_threads')
parser.add_argument('--inter_op_parallelism_threads', type=int, default=1, \
        help='inter_op_parallelism_threads')
parser.add_argument('--num_steps', type=int, default=10000, \
        help='Number of steps, i.e. number of batches times number of epochs')
parser.add_argument('--batch_size', type=int, default=100, \
        help='Number of samples selected in each batch')
parser.add_argument('--profile_steps', type=int, default=10000, \
        help='Every nth step, a profile measurement is performed that is stored in a JSON-file.')
parser.add_argument('--summary_steps', type=int, default=100, \
        help='Every nth step, a summary is written for Tensorboard visualization')
parser.add_argument('--checkpoint_steps', type=int, default=10000, \
        help='Every nth step, a checkpoint of the model is written')
args = parser.parse_args()

#Define settings
batch_size = int(args.batch_size)
num_steps = args.num_steps #Number of steps, i.e. number of batches times number of epochs
num_labels = 9 #Number of predicted transport components
random_seed = 1234

##Define function for making variables dimensionless and standardized.
##NOTE: uses global variable utau_ref
#def _standardization(variable, mean, standard_dev):
#    variable = variable / utau_ref
#    standardized_variable = (variable - mean) / standard_dev
#    return standardized_variable

#Define parse function for tfrecord files, which gives for each component in the example_proto 
#the output in format (dict(features),tensor(labels)) and normalizes according to specified means and variances.
def _parse_function(example_proto,means,stdevs):

    if args.gradients is None:

        if args.benchmark is None:
            keys_to_features = {
                'uc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'vc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'wc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'pc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'unres_tau_xu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample' :tf.FixedLenFeature([],tf.float32),
                'x_sample_size':tf.FixedLenFeature([],tf.int64),
                'y_sample_size':tf.FixedLenFeature([],tf.int64),
                'z_sample_size':tf.FixedLenFeature([],tf.int64),
                'tstep_sample':tf.FixedLenFeature([],tf.int64),
                'xloc_sample':tf.FixedLenFeature([],tf.float32),
                'xhloc_sample':tf.FixedLenFeature([],tf.float32),
                'yloc_sample':tf.FixedLenFeature([],tf.float32),
                'yhloc_sample':tf.FixedLenFeature([],tf.float32),
                'zloc_sample':tf.FixedLenFeature([],tf.float32),
                'zhloc_sample':tf.FixedLenFeature([],tf.float32)
            }

        else:
            keys_to_features = {
                'uc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'vc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'wc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'pc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'unres_tau_xu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample' :tf.FixedLenFeature([],tf.float32),
                'zhloc_sample':tf.FixedLenFeature([],tf.float32)
            }

            
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        #parsed_features['uc_sample'] = _standardization(parsed_features['uc_sample'], means['uc'], stdevs['uc'])
        #parsed_features['vc_sample'] = _standardization(parsed_features['vc_sample'], means['vc'], stdevs['vc'])
        #parsed_features['wc_sample'] = _standardization(parsed_features['wc_sample'], means['wc'], stdevs['wc'])
        #parsed_features['pc_sample'] = _standardization(parsed_features['pc_sample'], means['pc'], stdevs['pc'])

    else:

        if args.benchmark is None:
            keys_to_features = {
                'ugradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'ugrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'ugradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'pgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'pgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'pgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'unres_tau_xu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample' :tf.FixedLenFeature([],tf.float32),
                'x_sample_size':tf.FixedLenFeature([],tf.int64),
                'y_sample_size':tf.FixedLenFeature([],tf.int64),
                'z_sample_size':tf.FixedLenFeature([],tf.int64),
                'tstep_sample':tf.FixedLenFeature([],tf.int64),
                'xloc_sample':tf.FixedLenFeature([],tf.float32),
                'xhloc_sample':tf.FixedLenFeature([],tf.float32),
                'yloc_sample':tf.FixedLenFeature([],tf.float32),
                'yhloc_sample':tf.FixedLenFeature([],tf.float32),
                'zloc_sample':tf.FixedLenFeature([],tf.float32),
                'zhloc_sample':tf.FixedLenFeature([],tf.float32)
            }
        else:
            keys_to_features = {
                'ugradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'ugrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'ugradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'pgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'pgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'pgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'unres_tau_xu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample' :tf.FixedLenFeature([],tf.float32),
                'zhloc_sample':tf.FixedLenFeature([],tf.float32)
            }

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        #parsed_features['ugradx_sample'] = _standardization(parsed_features['ugradx_sample'], means['ugradx'], stdevs['ugradx'])
        #parsed_features['ugrady_sample'] = _standardization(parsed_features['ugrady_sample'], means['ugrady'], stdevs['ugrady'])
        #parsed_features['ugradz_sample'] = _standardization(parsed_features['ugradz_sample'], means['ugradz'], stdevs['ugradz'])
        #parsed_features['vgradx_sample'] = _standardization(parsed_features['vgradx_sample'], means['vgradx'], stdevs['vgradx'])
        #parsed_features['vgrady_sample'] = _standardization(parsed_features['vgrady_sample'], means['vgrady'], stdevs['vgrady'])
        #parsed_features['vgradz_sample'] = _standardization(parsed_features['vgradz_sample'], means['vgradz'], stdevs['vgradz'])
        #parsed_features['wgradx_sample'] = _standardization(parsed_features['wgradx_sample'], means['wgradx'], stdevs['wgradx'])
        #parsed_features['wgrady_sample'] = _standardization(parsed_features['wgrady_sample'], means['wgrady'], stdevs['wgrady'])
        #parsed_features['wgradz_sample'] = _standardization(parsed_features['wgradz_sample'], means['wgradz'], stdevs['wgradz'])
        #parsed_features['pgradx_sample'] = _standardization(parsed_features['pgradx_sample'], means['pgradx'], stdevs['pgradx'])
        #parsed_features['pgrady_sample'] = _standardization(parsed_features['pgrady_sample'], means['pgrady'], stdevs['pgrady'])
        #parsed_features['pgradz_sample'] = _standardization(parsed_features['pgradz_sample'], means['pgradz'], stdevs['pgradz'])

    ##Extract labels from the features dictionary, store them in a new labels array, and standardize them
    #def _getlabel(parsed_features_array, label_name):
    #    single_label = parsed_features_array.pop(label_name)
    #    #single_label = _standardization(single_label, means_array[label_name], stdevs_array[label_name])
    #    return single_label

    labels = {}
    labels['unres_tau_xu'] =  parsed_features.pop('unres_tau_xu_sample')
    labels['unres_tau_yu'] =  parsed_features.pop('unres_tau_yu_sample')
    labels['unres_tau_zu'] =  parsed_features.pop('unres_tau_zu_sample')
    labels['unres_tau_xv'] =  parsed_features.pop('unres_tau_xv_sample')
    labels['unres_tau_yv'] =  parsed_features.pop('unres_tau_yv_sample')
    labels['unres_tau_zv'] =  parsed_features.pop('unres_tau_zv_sample')
    labels['unres_tau_xw'] =  parsed_features.pop('unres_tau_xw_sample')
    labels['unres_tau_yw'] =  parsed_features.pop('unres_tau_yw_sample')
    labels['unres_tau_zw'] =  parsed_features.pop('unres_tau_zw_sample')

    labels = tf.stack([ labels['unres_tau_xu'], labels['unres_tau_yu'], labels['unres_tau_zu'], labels['unres_tau_xv'],  labels['unres_tau_yv'], labels['unres_tau_zv'], labels['unres_tau_xw'], labels['unres_tau_yw'], labels['unres_tau_zw']], axis=0)

    return parsed_features,labels


#Define training input function
def train_input_fn(filenames, batch_size, means, stdevs):
    dataset = tf.data.TFRecordDataset(filenames)
    #dataset = dataset.shuffle(len(filenames)) #comment this line when cache() is done after map()
    dataset = dataset.map(lambda line:_parse_function(line, means, stdevs))
    dataset = dataset.cache() #NOTE: The unavoidable consequence of using cache() before shuffle is that during all epochs the order of the flow fields is approximately the same (which can be alleviated by choosing a large buffer size, but that costs quite some computational effort). However, using shuffle before cache() will strongly increase the computational effort since memory becomes saturated. 
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset.prefetch(1)

    return dataset

#Define evaluation function
def eval_input_fn(filenames, batch_size, means, stdevs):
    dataset = tf.data.TFRecordDataset(filenames)
    #dataset = dataset.shuffle(len(filenames)) #comment this line when cache() is done after map()
    dataset = dataset.map(lambda line:_parse_function(line, means, stdevs))
    dataset = dataset.cache() 
    dataset = dataset.batch(batch_size)
    dataset.prefetch(1)

    return dataset    

#Define function for splitting the training and validation set
#NOTE: this split is on purpose not random. Always the flow fields corresponding to the last time steps are selected (noting that each tfrecord file contains all the samples of one flow field), such that the flow fields used for validation are as independent as possible from the fields used for training.
def split_train_val(time_steps, val_ratio):
    shuffled_steps = np.flip(time_steps)
    val_set_size = max(int(len(time_steps) * val_ratio),1) #max(..) makes sure that always at least 1 file is selected for validation
    val_steps = shuffled_steps[:val_set_size]
    train_steps = shuffled_steps[val_set_size:]
    return train_steps,val_steps


#Define model function for MLP estimator
def MLP_model_fn(features, labels, mode, params):
    '''MLP model with 1 hidden layer. \\
            NOTE: this function accesses the global variables args.gradients, means_dict_avgt, stdevs_dict_avgt, and utau_ref.'''

    #Define tf.constants for storing the means and stdevs of the input variables & labels, which is needed for the normalisation and subsequent denormalisation in this graph

    if args.gradients is None:         
        
        means_inputs = tf.constant([[
            means_dict_avgt['uc'],
            means_dict_avgt['vc'],
            means_dict_avgt['wc'],
            means_dict_avgt['pc']]])
        
        stdevs_inputs = tf.constant([[
            stdevs_dict_avgt['uc'],
            stdevs_dict_avgt['vc'],
            stdevs_dict_avgt['wc'],
            stdevs_dict_avgt['pc']]])
    
    else:

        means_inputs = tf.constant([[
            means_dict_avgt['ugradx'],
            means_dict_avgt['ugrady'],
            means_dict_avgt['ugradz'],
            means_dict_avgt['vgradx'],
            means_dict_avgt['vgrady'],
            means_dict_avgt['vgradz'],
            means_dict_avgt['wgradx'],
            means_dict_avgt['wgrady'],
            means_dict_avgt['wgradz'],
            means_dict_avgt['pgradx'],
            means_dict_avgt['pgrady'],
            means_dict_avgt['pgradz']]])
        
        stdevs_inputs = tf.constant([[
            stdevs_dict_avgt['ugradx'],
            stdevs_dict_avgt['ugrady'],
            stdevs_dict_avgt['ugradz'],
            stdevs_dict_avgt['vgradx'],
            stdevs_dict_avgt['vgrady'],
            stdevs_dict_avgt['vgradz'],
            stdevs_dict_avgt['wgradx'],
            stdevs_dict_avgt['wgrady'],
            stdevs_dict_avgt['wgradz'],
            stdevs_dict_avgt['pgradx'],
            stdevs_dict_avgt['pgrady'],
            stdevs_dict_avgt['pgradz']]])
        
    means_labels = tf.constant([[ 
        means_dict_avgt['unres_tau_xu_sample'],
        means_dict_avgt['unres_tau_yu_sample'],
        means_dict_avgt['unres_tau_zu_sample'],
        means_dict_avgt['unres_tau_xv_sample'],
        means_dict_avgt['unres_tau_yv_sample'],
        means_dict_avgt['unres_tau_zv_sample'],
        means_dict_avgt['unres_tau_xw_sample'],
        means_dict_avgt['unres_tau_yw_sample'],
        means_dict_avgt['unres_tau_zw_sample']]])
    
    stdevs_labels = tf.constant([[ 
        stdevs_dict_avgt['unres_tau_xu_sample'],
        stdevs_dict_avgt['unres_tau_yu_sample'],
        stdevs_dict_avgt['unres_tau_zu_sample'],
        stdevs_dict_avgt['unres_tau_xv_sample'],
        stdevs_dict_avgt['unres_tau_yv_sample'],
        stdevs_dict_avgt['unres_tau_zv_sample'],
        stdevs_dict_avgt['unres_tau_xw_sample'],
        stdevs_dict_avgt['unres_tau_yw_sample'],
        stdevs_dict_avgt['unres_tau_zw_sample']]])

    #Define identity ops for input variables, which can be used to set-up a frozen graph for inference.
    if args.gradients is None:
        input_u      = tf.identity(features['uc_sample'], name = 'input_u')
        input_v      = tf.identity(features['vc_sample'], name = 'input_v')
        input_w      = tf.identity(features['wc_sample'], name = 'input_w')
        input_p      = tf.identity(features['pc_sample'], name = 'input_p')
        utau_ref     = tf.identity(utau_ref, name = 'utau_ref') #Allow to feed utau_ref during inference, which likely helps to achieve Re independent results.
        
    else:   
        input_ugradx = tf.identity(features['ugradx_sample'], name = 'input_ugradx')
        input_ugrady = tf.identity(features['ugrady_sample'], name = 'input_ugrady')
        input_ugradz = tf.identity(features['ugradz_sample'], name = 'input_ugradz')
        input_vgradx = tf.identity(features['vgradx_sample'], name = 'input_vgradx')
        input_vgrady = tf.identity(features['vgrady_sample'], name = 'input_vgrady')
        input_vgradz = tf.identity(features['vgradz_sample'], name = 'input_vgradz')
        input_wgradx = tf.identity(features['wgradx_sample'], name = 'input_wgradx')
        input_wgrady = tf.identity(features['wgrady_sample'], name = 'input_wgrady')
        input_wgradz = tf.identity(features['wgradz_sample'], name = 'input_wgradz')
        input_pgradx = tf.identity(features['pgradx_sample'], name = 'input_pgradx')
        input_pgrady = tf.identity(features['pgrady_sample'], name = 'input_pgrady')
        input_pgradz = tf.identity(features['pgradz_sample'], name = 'input_pgradz')

    #Define function to make input variables/labels non-dimensionless and standardize them
    def _standardization(input_variable, mean_variable, stdev_variable, scaling_factor):
        #a3 = tf.print("input_variable", input_variable[0,:5], output_stream=tf.logging.info, summarize=-1)
        #input_variable = tf.math.divide(input_variable, scaling_factor) ONLY COMMENT THIS LINE FOR OLD TFRECORD FILES!!!
        #a4 = tf.print("input_variable", input_variable[0,:5], output_stream=tf.logging.info, summarize=-1)
        input_variable = tf.math.subtract(input_variable, mean_variable)
        #a5 = tf.print("mean_variable",  mean_variable, output_stream=tf.logging.info, summarize=-1)
        #a6 = tf.print("input_variable_mean", input_variable[0,:5], output_stream=tf.logging.info, summarize=-1)
        input_variable = tf.math.divide(input_variable, stdev_variable)
        #a7 = tf.print("stdev_variable", stdev_variable, output_stream=tf.logging.info, summarize=-1)
        #a8 = tf.print("input_variable_final", input_variable[0,:5], output_stream=tf.logging.info, summarize=-1)
        return input_variable #, a3, a4, a5, a6, a7, a8

    #Standardize input variables
    if args.gradients is None:
        
        with tf.name_scope("standardization_inputs"): #Group nodes in name scope for easier visualisation in TensorBoard
            input_u_stand = _standardization(input_u, means_inputs[:,0], stdevs_inputs[:,0], utau_ref)
            input_v_stand = _standardization(input_v, means_inputs[:,1], stdevs_inputs[:,1], utau_ref)
            input_w_stand = _standardization(input_w, means_inputs[:,2], stdevs_inputs[:,2], utau_ref)
            input_p_stand = _standardization(input_p, means_inputs[:,3], stdevs_inputs[:,3], utau_ref)

    else:

        with tf.name_scope("standardization_inputs"): #Group nodes in name scope for easier visualisation in TensorBoard
            input_ugradx_stand = _standardization(input_ugradx, means_inputs[:,0],  stdevs_inputs[:,0], utau_ref)
            input_ugrady_stand = _standardization(input_ugrady, means_inputs[:,1],  stdevs_inputs[:,1], utau_ref)
            input_ugradz_stand = _standardization(input_ugradz, means_inputs[:,2],  stdevs_inputs[:,2], utau_ref)
            input_vgradx_stand = _standardization(input_vgradx, means_inputs[:,3],  stdevs_inputs[:,3], utau_ref)
            input_vgrady_stand = _standardization(input_vgrady, means_inputs[:,4],  stdevs_inputs[:,4], utau_ref)
            input_vgradz_stand = _standardization(input_vgradz, means_inputs[:,5],  stdevs_inputs[:,5], utau_ref)
            input_wgradx_stand = _standardization(input_wgradx, means_inputs[:,6],  stdevs_inputs[:,6], utau_ref)
            input_wgrady_stand = _standardization(input_wgrady, means_inputs[:,7],  stdevs_inputs[:,7], utau_ref)
            input_wgradz_stand = _standardization(input_wgradz, means_inputs[:,8],  stdevs_inputs[:,8], utau_ref)
            input_pgradx_stand = _standardization(input_pgradx, means_inputs[:,9],  stdevs_inputs[:,9], utau_ref)
            input_pgrady_stand = _standardization(input_pgrady, means_inputs[:,10], stdevs_inputs[:,10], utau_ref)
            input_pgradz_stand = _standardization(input_pgradz, means_inputs[:,11], stdevs_inputs[:,11], utau_ref)
    
    #Standardize labels
    #NOTE: the labels are already made dimensionless in the training data procedure, and thus in contrast to the inputs do not have to be multiplied by a scaling factor. 
    with tf.name_scope("standardization_labels"): #Group nodes in name scope for easier visualisation in TensorBoard
        #a1 = tf.print("labels", labels[0,:], output_stream=tf.logging.info, summarize=-1)
        labels_means = tf.math.subtract(labels, means_labels)
        #a2 = tf.print("means_labels", means_labels, output_stream=tf.logging.info, summarize=-1)
        #a3 = tf.print("labels_means", labels_means[0,:], output_stream=tf.logging.info, summarize=-1)
        labels_stand = tf.math.divide(labels_means, stdevs_labels, name = 'labels_stand')
        #a4 = tf.print("labels_stand", labels_stand[0,:], output_stream=tf.logging.info, summarize=-1)
        #a5 = tf.print("stdevs_labels", stdevs_labels, output_stream=tf.logging.info, summarize=-1)

    #Define input layer
    if args.gradients is None:
        
        input_layer = tf.concat([input_u_stand, input_v_stand, input_w_stand, input_p_stand], axis=1, name = 'input_layer')

    else:

        input_layer = tf.concat([input_ugradx_stand, input_ugrady_stand, input_ugradz_stand, \
                input_vgradx_stand, input_vgrady_stand, input_vgradz_stand, \
                input_wgradx_stand, input_wgrady_stand, input_wgradz_stand, \
                input_pgradx_stand, input_pgrady_stand, input_pgradz_stand], axis=1, name = 'input_layer')

    #Visualize non-dimensionless and standardized input values in TensorBoard
    tf.summary.histogram('input_u_stand', input_u_stand)
    tf.summary.histogram('input_v_stand', input_v_stand)
    tf.summary.histogram('input_w_stand', input_w_stand)
    tf.summary.histogram('input_p_stand', input_p_stand)
    tf.summary.histogram('input_layer', input_layer)

    #Define layers
    dense1_layerdef  = tf.layers.Dense(units=params["n_dense1"], name="dense1", \
            activation=params["activation_function"], kernel_initializer=params["kernel_initializer"])
    dense1 = dense1_layerdef.apply(input_layer)
    output_layerdef = tf.layers.Dense(units=num_labels, name="output", \
            activation=None, kernel_initializer=params["kernel_initializer"])
    output_stand = output_layerdef.apply(dense1)
    #Visualize activations hidden layer in TensorBoard
    tf.summary.histogram('activations_hidden_layer1', dense1)
    tf.summary.scalar('fraction_of_zeros_in_activations_hidden_layer1', tf.nn.zero_fraction(dense1))

    #Make output layer conditional on the height of the staggered vertical dimension. At zh=0, several components are by definition 0 in turbulent channel flow (i.e. the ones located at the bottom wall). Consequently, the corresponding output values are explicitly set to 0 by masking them. 
    #NOTE1: floating point comparison of zh to 0. is OK in this case because zh at the surface was defined as exactly 0. in the training data generation procedure (see grid_objects_trainining.py in Training data folder). Zero has an exact representation in floating point format. 
    input_height = tf.identity(features['zhloc_sample'], name = 'input_height')
    channel_bool = tf.expand_dims(tf.math.not_equal(input_height, 0.), axis=1) #Select all samples where components should not be set to 0.
    #a1 = tf.print("channel_bool: ", channel_bool, output_stream=tf.logging.info, summarize=-1)
    
    #Select all transport components that should not be set to 0.
    nonstaggered_components_bool = tf.constant(
            [[True,  True,  False,  #xu, yu, zu
              True,  True,  False,  #xv, yv, zv
              False, False, True]]) #xw, yw, zw
    #a2 = tf.print("nonstaggered_components_bool: ", nonstaggered_components_bool, output_stream=tf.logging.info, summarize=-1)
    mask = tf.cast(tf.math.logical_or(channel_bool, nonstaggered_components_bool), output_stand.dtype, name = 'mask_BC') #Cast boolean to float for multiplications below
    #a3 = tf.print("mask: ", mask, output_stream=tf.logging.info, summarize=-1)
    output_mask = tf.math.multiply(output_stand, mask, name = 'output_masked')
    #a4 = tf.print("output_mask: ", output_mask, output_stream=tf.logging.info, summarize=-1)
    labels_mask = tf.math.multiply(labels_stand, mask, name = 'labels_masked') #NOTE: the concerning labels should also be put to 0 because of the applied normalisation.
    #a5 = tf.print("labels_mask: ", labels_mask, output_stream=tf.logging.info, summarize=-1)
    
    ##Trick to execute tf.print ops defined in this script. For these ops, set output_stream to tf.logging.info and summarize to -1.
    #with tf.control_dependencies([a1,a2,a3,a4,a5]):
    #    output_mask = tf.identity(output_mask)
    
    ###Visualize outputs in TensorBoard
    tf.summary.histogram('output_norm', output_mask)

    #Denormalize the output fluxes for inference
    #NOTE1: As a last step, because of the denormalisation the mask defined above should again be applied to the output.
    #NOTE2: These calculations are only needed for inference, but in order to show up in the computation graph (and thus allowing to include it in the frozen graph) this should nonetheless be part of the main model_fn function).
    #NOTE3: In addition to undoing the standardization, the normalisation includes a multiplication with utau_ref. Earlier in the training data generation procedure, all data was made dimensionless by utau_ref. Therefore, the utau_ref is taken into account in the denormalisation below.
    #if mode == tf.estimator.ModeKeys.PREDICT:
    with tf.name_scope("denormalisation_output"): #Group nodes in name scope for easier visualisation in TensorBoard
        output_stdevs      = tf.math.multiply(output_mask, stdevs_labels)
        output_means       = tf.math.add(output_stdevs, means_labels)
        output_meansstdevs = tf.math.multiply(output_means, (utau_ref ** 2))
    output_denorm = tf.math.multiply(output_meansstdevs, mask, name = 'output_denorm')
    
    #Denormalize the labels for inference
    #NOTE1: in contrast to the code above, no mask needs to be applied as the concerning labels should already evaluate to 0 after denormalisation.
    #NOTE2: this does not have to be included in the frozen graph, and thus does not have to be included in the main code.
    #NOTE3: similar to the code above, utau_ref is included in the denormalisation.
    if mode == tf.estimator.ModeKeys.PREDICT:
        labels_stdevs = tf.math.multiply(labels_stand, stdevs_labels) #NOTE: on purpose labels_stand instead of labels_mask.
        labels_means  = tf.math.add(labels_stdevs, means_labels)
        labels_denorm = tf.math.multiply(labels_means, (utau_ref ** 2), name = 'labels_denorm')
        
        #Compute predictions
        if args.benchmark is None:
            return tf.estimator.EstimatorSpec(mode, predictions={
                'pred_tau_xu':output_denorm[:,0], 'label_tau_xu':labels_denorm[:,0],
                'pred_tau_yu':output_denorm[:,1], 'label_tau_yu':labels_denorm[:,1],
                'pred_tau_zu':output_denorm[:,2], 'label_tau_zu':labels_denorm[:,2],
                'pred_tau_xv':output_denorm[:,3], 'label_tau_xv':labels_denorm[:,3],
                'pred_tau_yv':output_denorm[:,4], 'label_tau_yv':labels_denorm[:,4],
                'pred_tau_zv':output_denorm[:,5], 'label_tau_zv':labels_denorm[:,5],
                'pred_tau_xw':output_denorm[:,6], 'label_tau_xw':labels_denorm[:,6],
                'pred_tau_yw':output_denorm[:,7], 'label_tau_yw':labels_denorm[:,7],
                'pred_tau_zw':output_denorm[:,8], 'label_tau_zw':labels_denorm[:,8],
                'tstep':features['tstep_sample'], 'zhloc':features['zhloc_sample'],
                'zloc':features['zloc_sample'], 'yhloc':features['yhloc_sample'],
                'yloc':features['yloc_sample'], 'xhloc':features['xhloc_sample'],
                'xloc':features['xloc_sample']})
 
        else:
            return tf.estimator.EstimatorSpec(mode, predictions={
                'pred_tau_xu':output_denorm[:,0],
                'pred_tau_yu':output_denorm[:,1], 
                'pred_tau_zu':output_denorm[:,2], 
                'pred_tau_xv':output_denorm[:,3], 
                'pred_tau_yv':output_denorm[:,4], 
                'pred_tau_zv':output_denorm[:,5], 
                'pred_tau_xw':output_denorm[:,6], 
                'pred_tau_yw':output_denorm[:,7], 
                'pred_tau_zw':output_denorm[:,8]}) 
    
    #Compute loss
    mse_tau_total = tf.losses.mean_squared_error(labels_mask, output_mask)
    loss = tf.reduce_mean(mse_tau_total)
        
    #Define function to calculate the logarithm
    def log10(values):
        numerator = tf.log(values)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    #Compute evaluation metrics.
    tf.summary.histogram('labels', labels_mask) #Visualize labels
    if mode == tf.estimator.ModeKeys.EVAL:
        mse_all, update_op = tf.metrics.mean_squared_error(labels_mask, output_mask)
        log_mse_all = log10(mse_all)
        log_mse_all_update_op = log10(update_op)
        rmse_all = tf.math.sqrt(mse_all)
        rmse_all_update_op = tf.math.sqrt(update_op)
        val_metrics = {'mse': (mse_all, update_op), 'rmse':(rmse_all, rmse_all_update_op),'log_loss':(log_mse_all, log_mse_all_update_op)} 
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=val_metrics)

    #Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    log_loss_training = log10(loss)
    tf.summary.scalar('log_loss', log_loss_training)

    optimizer = tf.train.AdamOptimizer(params['learning_rate'])

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    #Write all trainable variables to Tensorboard
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name,var)

    #Return tf.estimator.Estimatorspec for training mode
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

#Define filenames of tfrecords for training and validation
#NOTE: each tfrecords contains all the samples from a single 'snapshot' of the flow, and thus corresponds to a single time step.
nt_available = 30 #Amount of time steps that should be used for training/validation, assuming  that the number of the time step in the filenames ranges from 1 to nt_available without gaps.
nt_total = 100 #Amount of time steps INCLUDING all produced tfrecord files (also the ones not used for training/validation).
#nt_available = 2 #FOR TESTING PURPOSES ONLY!
#nt_total = 3 #FOR TESTING PURPOSES ONLY!
time_numbers = np.arange(nt_available)
train_stepnumbers, val_stepnumbers = split_train_val(time_numbers, 0.1) #Set aside 10% of files for validation.
train_filenames = np.zeros((len(train_stepnumbers),), dtype=object)
val_filenames   = np.zeros((len(val_stepnumbers),), dtype=object)

i=0
for train_stepnumber in train_stepnumbers: #Generate training filenames from selected step numbers and total steps
    if args.gradients is None:
        train_filenames[i] = args.input_dir + 'training_time_step_{0}_of_{1}.tfrecords'.format(train_stepnumber+1, nt_total)
    else:
        train_filenames[i] = args.input_dir + 'training_time_step_{0}_of_{1}_gradients.tfrecords'.format(train_stepnumber+1, nt_total)
    i+=1

j=0
for val_stepnumber in val_stepnumbers: #Generate validation filenames from selected step numbers and total steps
    if args.gradients is None:
        val_filenames[j] = args.input_dir + 'training_time_step_{0}_of_{1}.tfrecords'.format(val_stepnumber+1, nt_total)
    else:
        val_filenames[j] = args.input_dir + 'training_time_step_{0}_of_{1}_gradients.tfrecords'.format(val_stepnumber+1, nt_total)
    j+=1

#Extract friction velocity from training file (which is needed for the denormalisation implemented within the MLP)
training_file = nc.Dataset(args.training_filepath, 'r')
utau_ref = np.array(training_file['utau_ref'][:], dtype = 'f4')

#Calculate means and stdevs for input variables (which is needed for the normalisation).
#NOTE: in the code below, it is made sure that only the means and stdevs of the time steps used for training are taken into account.
means_stdevs_filepath = args.stored_means_stdevs_filepath
means_stdevs_file     = nc.Dataset(means_stdevs_filepath, 'r')

means_dict_t  = {}
stdevs_dict_t = {}
if args.gradients is None:
    means_dict_t['uc'] = np.array(means_stdevs_file['mean_uc'][:])
    means_dict_t['vc'] = np.array(means_stdevs_file['mean_vc'][:])
    means_dict_t['wc'] = np.array(means_stdevs_file['mean_wc'][:])
    means_dict_t['pc'] = np.array(means_stdevs_file['mean_pc'][:])
    
    stdevs_dict_t['uc'] = np.array(means_stdevs_file['stdev_uc'][:])
    stdevs_dict_t['vc'] = np.array(means_stdevs_file['stdev_vc'][:])
    stdevs_dict_t['wc'] = np.array(means_stdevs_file['stdev_wc'][:])
    stdevs_dict_t['pc'] = np.array(means_stdevs_file['stdev_pc'][:])

else:
    means_dict_t['ugradx'] = np.array(means_stdevs_file['mean_ugradx'][:])
    means_dict_t['ugrady'] = np.array(means_stdevs_file['mean_ugrady'][:])
    means_dict_t['ugradz'] = np.array(means_stdevs_file['mean_ugradz'][:])

    means_dict_t['vgradx'] = np.array(means_stdevs_file['mean_vgradx'][:])
    means_dict_t['vgrady'] = np.array(means_stdevs_file['mean_vgrady'][:])
    means_dict_t['vgradz'] = np.array(means_stdevs_file['mean_vgradz'][:])

    means_dict_t['wgradx'] = np.array(means_stdevs_file['mean_wgradx'][:])
    means_dict_t['wgrady'] = np.array(means_stdevs_file['mean_wgrady'][:])
    means_dict_t['wgradz'] = np.array(means_stdevs_file['mean_wgradz'][:])

    means_dict_t['pgradx'] = np.array(means_stdevs_file['mean_pgradx'][:])
    means_dict_t['pgrady'] = np.array(means_stdevs_file['mean_pgrady'][:])
    means_dict_t['pgradz'] = np.array(means_stdevs_file['mean_pgradz'][:])

    stdevs_dict_t['ugradx'] = np.array(means_stdevs_file['stdev_ugradx'][:])
    stdevs_dict_t['ugrady'] = np.array(means_stdevs_file['stdev_ugrady'][:])
    stdevs_dict_t['ugradz'] = np.array(means_stdevs_file['stdev_ugradz'][:])

    stdevs_dict_t['vgradx'] = np.array(means_stdevs_file['stdev_vgradx'][:])
    stdevs_dict_t['vgrady'] = np.array(means_stdevs_file['stdev_vgrady'][:])
    stdevs_dict_t['vgradz'] = np.array(means_stdevs_file['stdev_vgradz'][:])

    stdevs_dict_t['wgradx'] = np.array(means_stdevs_file['stdev_wgradx'][:])
    stdevs_dict_t['wgrady'] = np.array(means_stdevs_file['stdev_wgrady'][:])
    stdevs_dict_t['wgradz'] = np.array(means_stdevs_file['stdev_wgradz'][:])

    stdevs_dict_t['pgradx'] = np.array(means_stdevs_file['stdev_pgradx'][:])
    stdevs_dict_t['pgrady'] = np.array(means_stdevs_file['stdev_pgrady'][:])
    stdevs_dict_t['pgradz'] = np.array(means_stdevs_file['stdev_pgradz'][:])

#Extract mean & standard deviation labels
means_dict_t['unres_tau_xu_sample']  = np.array(means_stdevs_file['mean_unres_tau_xu_sample'][:])
stdevs_dict_t['unres_tau_xu_sample'] = np.array(means_stdevs_file['stdev_unres_tau_xu_sample'][:])
means_dict_t['unres_tau_yu_sample']  = np.array(means_stdevs_file['mean_unres_tau_yu_sample'][:])
stdevs_dict_t['unres_tau_yu_sample'] = np.array(means_stdevs_file['stdev_unres_tau_yu_sample'][:])
means_dict_t['unres_tau_zu_sample']  = np.array(means_stdevs_file['mean_unres_tau_zu_sample'][:])
stdevs_dict_t['unres_tau_zu_sample'] = np.array(means_stdevs_file['stdev_unres_tau_zu_sample'][:])
means_dict_t['unres_tau_xv_sample']  = np.array(means_stdevs_file['mean_unres_tau_xv_sample'][:])
stdevs_dict_t['unres_tau_xv_sample'] = np.array(means_stdevs_file['stdev_unres_tau_xv_sample'][:])
means_dict_t['unres_tau_yv_sample']  = np.array(means_stdevs_file['mean_unres_tau_yv_sample'][:])
stdevs_dict_t['unres_tau_yv_sample'] = np.array(means_stdevs_file['stdev_unres_tau_yv_sample'][:])
means_dict_t['unres_tau_zv_sample']  = np.array(means_stdevs_file['mean_unres_tau_zv_sample'][:])
stdevs_dict_t['unres_tau_zv_sample'] = np.array(means_stdevs_file['stdev_unres_tau_zv_sample'][:])
means_dict_t['unres_tau_xw_sample']  = np.array(means_stdevs_file['mean_unres_tau_xw_sample'][:])
stdevs_dict_t['unres_tau_xw_sample'] = np.array(means_stdevs_file['stdev_unres_tau_xw_sample'][:])
means_dict_t['unres_tau_yw_sample']  = np.array(means_stdevs_file['mean_unres_tau_yw_sample'][:])
stdevs_dict_t['unres_tau_yw_sample'] = np.array(means_stdevs_file['stdev_unres_tau_yw_sample'][:])
means_dict_t['unres_tau_zw_sample']  = np.array(means_stdevs_file['mean_unres_tau_zw_sample'][:])
stdevs_dict_t['unres_tau_zw_sample'] = np.array(means_stdevs_file['stdev_unres_tau_zw_sample'][:])

means_dict_avgt  = {}
stdevs_dict_avgt = {}

if args.gradients is None:
    means_dict_avgt['uc'] = np.mean(means_dict_t['uc'][train_stepnumbers])
    means_dict_avgt['vc'] = np.mean(means_dict_t['vc'][train_stepnumbers])
    means_dict_avgt['wc'] = np.mean(means_dict_t['wc'][train_stepnumbers])
    means_dict_avgt['pc'] = np.mean(means_dict_t['pc'][train_stepnumbers])
    
    stdevs_dict_avgt['uc'] = np.mean(stdevs_dict_t['uc'][train_stepnumbers])
    stdevs_dict_avgt['vc'] = np.mean(stdevs_dict_t['vc'][train_stepnumbers])
    stdevs_dict_avgt['wc'] = np.mean(stdevs_dict_t['wc'][train_stepnumbers])
    stdevs_dict_avgt['pc'] = np.mean(stdevs_dict_t['pc'][train_stepnumbers])

else:
    means_dict_avgt['ugradx'] = np.mean(means_dict_t['ugradx'][train_stepnumbers])
    means_dict_avgt['ugrady'] = np.mean(means_dict_t['ugrady'][train_stepnumbers])
    means_dict_avgt['ugradz'] = np.mean(means_dict_t['ugradz'][train_stepnumbers])

    means_dict_avgt['vgradx'] = np.mean(means_dict_t['vgradx'][train_stepnumbers])
    means_dict_avgt['vgrady'] = np.mean(means_dict_t['vgrady'][train_stepnumbers])
    means_dict_avgt['vgradz'] = np.mean(means_dict_t['vgradz'][train_stepnumbers])

    means_dict_avgt['wgradx'] = np.mean(means_dict_t['wgradx'][train_stepnumbers])
    means_dict_avgt['wgrady'] = np.mean(means_dict_t['wgrady'][train_stepnumbers])
    means_dict_avgt['wgradz'] = np.mean(means_dict_t['wgradz'][train_stepnumbers])

    means_dict_avgt['pgradx'] = np.mean(means_dict_t['pgradx'][train_stepnumbers])
    means_dict_avgt['pgrady'] = np.mean(means_dict_t['pgrady'][train_stepnumbers])
    means_dict_avgt['pgradz'] = np.mean(means_dict_t['pgradz'][train_stepnumbers])

    stdevs_dict_avgt['ugradx'] = np.mean(stdevs_dict_t['ugradx'][train_stepnumbers])
    stdevs_dict_avgt['ugrady'] = np.mean(stdevs_dict_t['ugrady'][train_stepnumbers])
    stdevs_dict_avgt['ugradz'] = np.mean(stdevs_dict_t['ugradz'][train_stepnumbers])

    stdevs_dict_avgt['vgradx'] = np.mean(stdevs_dict_t['vgradx'][train_stepnumbers])
    stdevs_dict_avgt['vgrady'] = np.mean(stdevs_dict_t['vgrady'][train_stepnumbers])
    stdevs_dict_avgt['vgradz'] = np.mean(stdevs_dict_t['vgradz'][train_stepnumbers])

    stdevs_dict_avgt['wgradx'] = np.mean(stdevs_dict_t['wgradx'][train_stepnumbers])
    stdevs_dict_avgt['wgrady'] = np.mean(stdevs_dict_t['wgrady'][train_stepnumbers])
    stdevs_dict_avgt['wgradz'] = np.mean(stdevs_dict_t['wgradz'][train_stepnumbers])

    stdevs_dict_avgt['pgradx'] = np.mean(stdevs_dict_t['pgradx'][train_stepnumbers])
    stdevs_dict_avgt['pgrady'] = np.mean(stdevs_dict_t['pgrady'][train_stepnumbers])
    stdevs_dict_avgt['pgradz'] = np.mean(stdevs_dict_t['pgradz'][train_stepnumbers])

#Extract temporally averaged (over the time steps used for training) mean & standard deviation labels
means_dict_avgt['unres_tau_xu_sample']  = np.mean(means_dict_t['unres_tau_xu_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_xu_sample'] = np.mean(stdevs_dict_t['unres_tau_xu_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_yu_sample']  = np.mean(means_dict_t['unres_tau_yu_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_yu_sample'] = np.mean(stdevs_dict_t['unres_tau_yu_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_zu_sample']  = np.mean(means_dict_t['unres_tau_zu_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_zu_sample'] = np.mean(stdevs_dict_t['unres_tau_zu_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_xv_sample']  = np.mean(means_dict_t['unres_tau_xv_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_xv_sample'] = np.mean(stdevs_dict_t['unres_tau_xv_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_yv_sample']  = np.mean(means_dict_t['unres_tau_yv_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_yv_sample'] = np.mean(stdevs_dict_t['unres_tau_yv_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_zv_sample']  = np.mean(means_dict_t['unres_tau_zv_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_zv_sample'] = np.mean(stdevs_dict_t['unres_tau_zv_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_xw_sample']  = np.mean(means_dict_t['unres_tau_xw_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_xw_sample'] = np.mean(stdevs_dict_t['unres_tau_xw_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_yw_sample']  = np.mean(means_dict_t['unres_tau_yw_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_yw_sample'] = np.mean(stdevs_dict_t['unres_tau_yw_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_zw_sample']  = np.mean(means_dict_t['unres_tau_zw_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_zw_sample'] = np.mean(stdevs_dict_t['unres_tau_zw_sample'][train_stepnumbers])

#Set configuration
config = tf.ConfigProto(log_device_placement=False)
# config.gpu_options.allow_growth = True
config.intra_op_parallelism_threads = args.intra_op_parallelism_threads
config.inter_op_parallelism_threads = args.inter_op_parallelism_threads
os.environ['KMP_BLOCKTIME'] = str(0)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['OMP_NUM_THREADS'] = str(args.intra_op_parallelism_threads)

#Set warmstart_dir to None to disable it
warmstart_dir = None

#Create RunConfig object to save check_point in the model_dir according to the specified schedule, and to define the session config
my_checkpointing_config = tf.estimator.RunConfig(model_dir=args.checkpoint_dir, tf_random_seed=random_seed, save_summary_steps=args.summary_steps, save_checkpoints_steps=args.checkpoint_steps, save_checkpoints_secs = None,session_config=config,keep_checkpoint_max=None, keep_checkpoint_every_n_hours=10000, log_step_count_steps=10, train_distribute=None) #Provide tf.contrib.distribute.DistributionStrategy instance to train_distribute parameter for distributed training

#Define hyperparameters
if args.gradients is None:
    kernelsize_conv1 = 5
else:
    kernelsize_conv1 = 3

hyperparams =  {
'n_dense1':80, #Equivalent to using 80 filters in CNN used before
'activation_function':tf.nn.leaky_relu, #NOTE: Define new activation function based on tf.nn.leaky_relu with lambda to adjust the default value for alpha (0.2)
'kernel_initializer':tf.initializers.he_uniform(),
'learning_rate':0.0001
}

#Instantiate an Estimator with model defined by model_fn
MLP = tf.estimator.Estimator(model_fn = MLP_model_fn, config=my_checkpointing_config, params = hyperparams, model_dir=args.checkpoint_dir, warm_start_from = warmstart_dir)

profiler_hook = tf.train.ProfilerHook(save_steps = args.profile_steps, output_dir = args.checkpoint_dir) #Hook designed for storing runtime statistics in Chrome trace JSON-format, which can be used in conjuction with the other summaries stored during training in Tensorboard.

if args.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    hooks = [profiler_hook, debug_hook]
#    hooks = [bcast_hook, debug_hook]
else:
    hooks = [profiler_hook]

#Train and evaluate MLP
train_spec = tf.estimator.TrainSpec(input_fn=lambda:train_input_fn(train_filenames, batch_size, means_dict_avgt, stdevs_dict_avgt), max_steps=num_steps, hooks=hooks)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda:eval_input_fn(val_filenames, batch_size, means_dict_avgt, stdevs_dict_avgt), steps=None, name='MLP1', start_delay_secs=30, throttle_secs=0)#NOTE: throttle_secs=0 implies that for every stored checkpoint the validation error is calculated
tf.estimator.train_and_evaluate(MLP, train_spec, eval_spec)

#NOTE: MLP.predict appeared to be unsuitable to compare the predictions from the MLP to the true labels stored in the TFRecords files: the labels are discarded by the tf.estimator.Estimator in predict mode. The alternative is the 'hacky' solution implemented in the code below.


######
#'Hacky' solution to compare the predictions of the MLP to the true labels stored in the TFRecords files. 
#NOTE1: the input and model function are called manually rather than using the tf.estimator.Estimator syntax.
#NOTE2: the resulting predictions and labels are automatically stored in a netCDF-file called MLP_predictions.nc, which is placed in the specified checkpoint_dir.
#NOTE3: this implementation of the inference is computationally not efficient, but does allow to inspect and visualize the predictions afterwards in detail using the produced netCDF-file and other scripts. Fast inference is currently being implemented by generating a frozen graph from the trained MLP.
if args.benchmark is None:

    print('Start making predictions for validation files.')
   
    #Loop over val files to prevent memory overflow issues
    
    create_file = True #Flag to make sure netCDF file is initialized
 
    #Initialize variables for keeping track of iterations
    tot_sample_end = 0
    tot_sample_begin = tot_sample_end

    for val_filename in val_filenames:

        tf.reset_default_graph() #Reset the graph for each tfrecord (i.e. each flow 'snapshot')
 
        #Generate iterator to extract features and labels from tfrecords
        iterator = eval_input_fn([val_filename], batch_size, means_dict_avgt, stdevs_dict_avgt).make_initializable_iterator() #All samples present in val_filenames are used for validation once.
 
        #Define operation to extract features and labels from iterator
        fes, lbls = iterator.get_next()

        #Define operation to generate predictions for extracted features and labels
        preds_op = MLP_model_fn(fes, lbls, \
                        tf.estimator.ModeKeys.PREDICT, hyperparams).predictions

        #Save CNN_model such that it can be restored in the tf.Session() below
        saver = tf.train.Saver()

        #Create/open netCDF-file to store predictions and labels
        if create_file:
            filepath = args.checkpoint_dir + '/MLP_predictions.nc'
            predictions_file = nc.Dataset(filepath, 'w')
            dim_ns = predictions_file.createDimension("ns",None)

            #Create variables for storage
            var_pred_tau_xu        = predictions_file.createVariable("preds_values_tau_xu","f8",("ns",))
            var_pred_random_tau_xu = predictions_file.createVariable("preds_values_random_tau_xu","f8",("ns",))
            var_lbl_tau_xu         = predictions_file.createVariable("lbls_values_tau_xu","f8",("ns",))
            var_res_tau_xu         = predictions_file.createVariable("residuals_tau_xu","f8",("ns",))
            var_res_random_tau_xu  = predictions_file.createVariable("residuals_random_tau_xu","f8",("ns",))
            #
            var_pred_tau_yu        = predictions_file.createVariable("preds_values_tau_yu","f8",("ns",))
            var_pred_random_tau_yu = predictions_file.createVariable("preds_values_random_tau_yu","f8",("ns",))
            var_lbl_tau_yu         = predictions_file.createVariable("lbls_values_tau_yu","f8",("ns",))
            var_res_tau_yu         = predictions_file.createVariable("residuals_tau_yu","f8",("ns",))
            var_res_random_tau_yu  = predictions_file.createVariable("residuals_random_tau_yu","f8",("ns",))
            #
            var_pred_tau_zu        = predictions_file.createVariable("preds_values_tau_zu","f8",("ns",))
            var_pred_random_tau_zu = predictions_file.createVariable("preds_values_random_tau_zu","f8",("ns",))
            var_lbl_tau_zu         = predictions_file.createVariable("lbls_values_tau_zu","f8",("ns",))
            var_res_tau_zu         = predictions_file.createVariable("residuals_tau_zu","f8",("ns",))
            var_res_random_tau_zu  = predictions_file.createVariable("residuals_random_tau_zu","f8",("ns",))
            #
            var_pred_tau_xv        = predictions_file.createVariable("preds_values_tau_xv","f8",("ns",))
            var_pred_random_tau_xv = predictions_file.createVariable("preds_values_random_tau_xv","f8",("ns",))
            var_lbl_tau_xv         = predictions_file.createVariable("lbls_values_tau_xv","f8",("ns",))
            var_res_tau_xv         = predictions_file.createVariable("residuals_tau_xv","f8",("ns",))
            var_res_random_tau_xv  = predictions_file.createVariable("residuals_random_tau_xv","f8",("ns",))
            #
            var_pred_tau_yv        = predictions_file.createVariable("preds_values_tau_yv","f8",("ns",))
            var_pred_random_tau_yv = predictions_file.createVariable("preds_values_random_tau_yv","f8",("ns",))
            var_lbl_tau_yv         = predictions_file.createVariable("lbls_values_tau_yv","f8",("ns",))
            var_res_tau_yv         = predictions_file.createVariable("residuals_tau_yv","f8",("ns",))
            var_res_random_tau_yv  = predictions_file.createVariable("residuals_random_tau_yv","f8",("ns",))
            #
            var_pred_tau_zv        = predictions_file.createVariable("preds_values_tau_zv","f8",("ns",))
            var_pred_random_tau_zv = predictions_file.createVariable("preds_values_random_tau_zv","f8",("ns",))
            var_lbl_tau_zv         = predictions_file.createVariable("lbls_values_tau_zv","f8",("ns",))
            var_res_tau_zv         = predictions_file.createVariable("residuals_tau_zv","f8",("ns",))
            var_res_random_tau_zv  = predictions_file.createVariable("residuals_random_tau_zv","f8",("ns",))
            #
            var_pred_tau_xw        = predictions_file.createVariable("preds_values_tau_xw","f8",("ns",))
            var_pred_random_tau_xw = predictions_file.createVariable("preds_values_random_tau_xw","f8",("ns",))
            var_lbl_tau_xw         = predictions_file.createVariable("lbls_values_tau_xw","f8",("ns",))
            var_res_tau_xw         = predictions_file.createVariable("residuals_tau_xw","f8",("ns",))
            var_res_random_tau_xw  = predictions_file.createVariable("residuals_random_tau_xw","f8",("ns",))
            #
            var_pred_tau_yw        = predictions_file.createVariable("preds_values_tau_yw","f8",("ns",))
            var_pred_random_tau_yw = predictions_file.createVariable("preds_values_random_tau_yw","f8",("ns",))
            var_lbl_tau_yw         = predictions_file.createVariable("lbls_values_tau_yw","f8",("ns",))
            var_res_tau_yw         = predictions_file.createVariable("residuals_tau_yw","f8",("ns",))
            var_res_random_tau_yw  = predictions_file.createVariable("residuals_random_tau_yw","f8",("ns",))
            #
            var_pred_tau_zw        = predictions_file.createVariable("preds_values_tau_zw","f8",("ns",))
            var_pred_random_tau_zw = predictions_file.createVariable("preds_values_random_tau_zw","f8",("ns",))
            var_lbl_tau_zw         = predictions_file.createVariable("lbls_values_tau_zw","f8",("ns",))
            var_res_tau_zw         = predictions_file.createVariable("residuals_tau_zw","f8",("ns",))
            var_res_random_tau_zw  = predictions_file.createVariable("residuals_random_tau_zw","f8",("ns",))
            #
            vartstep               = predictions_file.createVariable("tstep_samples","f8",("ns",))
            varzhloc               = predictions_file.createVariable("zhloc_samples","f8",("ns",))
            varzloc                = predictions_file.createVariable("zloc_samples","f8",("ns",))
            varyhloc               = predictions_file.createVariable("yhloc_samples","f8",("ns",))
            varyloc                = predictions_file.createVariable("yloc_samples","f8",("ns",))
            varxhloc               = predictions_file.createVariable("xhloc_samples","f8",("ns",))
            varxloc                = predictions_file.createVariable("xloc_samples","f8",("ns",))

            create_file=False #Make sure file is only created once

        else:
            predictions_file = nc.Dataset(filepath, 'r+')


        with tf.Session(config=config) as sess:

            #Restore CNN_model within tf.Session()
            #tf.reset_default_graph() #Make graph empty before restoring
            ckpt  = tf.train.get_checkpoint_state(args.checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)

            #Initialize iterator
            sess.run(iterator.initializer)

            while True:
                try:
                    #Execute computational graph to generate predictions
                    preds = sess.run(preds_op)

                    #Initialize variables for storage
                    preds_tau_xu               = []
                    preds_random_tau_xu        = []
                    lbls_tau_xu                = []
                    residuals_tau_xu           = []
                    residuals_random_tau_xu    = []
                    #
                    preds_tau_yu               = []
                    preds_random_tau_yu        = []
                    lbls_tau_yu                = []
                    residuals_tau_yu           = []
                    residuals_random_tau_yu    = []
                    #
                    preds_tau_zu               = []
                    preds_random_tau_zu        = []
                    lbls_tau_zu                = []
                    residuals_tau_zu           = []
                    residuals_random_tau_zu    = []
                    #
                    preds_tau_xv               = []
                    preds_random_tau_xv        = []
                    lbls_tau_xv                = []
                    residuals_tau_xv           = []
                    residuals_random_tau_xv    = []
                    #
                    preds_tau_yv               = []
                    preds_random_tau_yv        = []
                    lbls_tau_yv                = []
                    residuals_tau_yv           = []
                    residuals_random_tau_yv    = []
                    #
                    preds_tau_zv               = []
                    preds_random_tau_zv        = []
                    lbls_tau_zv                = []
                    residuals_tau_zv           = []
                    residuals_random_tau_zv    = []
                    #
                    preds_tau_xw               = []
                    preds_random_tau_xw        = []
                    lbls_tau_xw                = []
                    residuals_tau_xw           = []
                    residuals_random_tau_xw    = []
                    #
                    preds_tau_yw               = []
                    preds_random_tau_yw        = []
                    lbls_tau_yw                = []
                    residuals_tau_yw           = []
                    residuals_random_tau_yw    = []
                    #
                    preds_tau_zw               = []
                    preds_random_tau_zw        = []
                    lbls_tau_zw                = []
                    residuals_tau_zw           = []
                    residuals_random_tau_zw    = []
                    #
                    tstep_samples       = []
                    zhloc_samples       = []
                    zloc_samples        = []
                    yhloc_samples       = []
                    yloc_samples        = []
                    xhloc_samples       = []
                    xloc_samples        = []

                    for pred_tau_xu, lbl_tau_xu, pred_tau_yu, lbl_tau_yu, pred_tau_zu, lbl_tau_zu, \
                        pred_tau_xv, lbl_tau_xv, pred_tau_yv, lbl_tau_yv, pred_tau_zv, lbl_tau_zv, \
                        pred_tau_xw, lbl_tau_xw, pred_tau_yw, lbl_tau_yw, pred_tau_zw, lbl_tau_zw, \
                        tstep, zhloc, zloc, yhloc, yloc, xhloc, xloc in zip(
                                preds['pred_tau_xu'], preds['label_tau_xu'],
                                preds['pred_tau_yu'], preds['label_tau_yu'],
                                preds['pred_tau_zu'], preds['label_tau_zu'],
                                preds['pred_tau_xv'], preds['label_tau_xv'],
                                preds['pred_tau_yv'], preds['label_tau_yv'],
                                preds['pred_tau_zv'], preds['label_tau_zv'],
                                preds['pred_tau_xw'], preds['label_tau_xw'],
                                preds['pred_tau_yw'], preds['label_tau_yw'],
                                preds['pred_tau_zw'], preds['label_tau_zw'],
                                preds['tstep'], preds['zhloc'], preds['zloc'], 
                                preds['yhloc'], preds['yloc'], preds['xhloc'], preds['xloc']):
                        # 
                        preds_tau_xu               += [pred_tau_xu]
                        lbls_tau_xu                += [lbl_tau_xu]
                        residuals_tau_xu           += [abs(pred_tau_xu-lbl_tau_xu)]
                        pred_random_tau_xu          = random.choice(preds['label_tau_xu'][:][:]) #Generate random prediction
                        preds_random_tau_xu        += [pred_random_tau_xu]
                        residuals_random_tau_xu    += [abs(pred_random_tau_xu-lbl_tau_xu)]
                        #
                        preds_tau_yu               += [pred_tau_yu]
                        lbls_tau_yu                += [lbl_tau_yu]
                        residuals_tau_yu           += [abs(pred_tau_yu-lbl_tau_yu)]
                        pred_random_tau_yu          = random.choice(preds['label_tau_yu'][:][:]) #Generate random prediction
                        preds_random_tau_yu        += [pred_random_tau_yu]
                        residuals_random_tau_yu    += [abs(pred_random_tau_yu-lbl_tau_yu)]
                        #
                        preds_tau_zu               += [pred_tau_zu]
                        lbls_tau_zu                += [lbl_tau_zu]
                        residuals_tau_zu           += [abs(pred_tau_zu-lbl_tau_zu)]
                        pred_random_tau_zu          = random.choice(preds['label_tau_zu'][:][:]) #Generate random prediction
                        preds_random_tau_zu        += [pred_random_tau_zu]
                        residuals_random_tau_zu    += [abs(pred_random_tau_zu-lbl_tau_zu)]
                        #
                        preds_tau_xv               += [pred_tau_xv]
                        lbls_tau_xv                += [lbl_tau_xv]
                        residuals_tau_xv           += [abs(pred_tau_xv-lbl_tau_xv)]
                        pred_random_tau_xv          = random.choice(preds['label_tau_xv'][:][:]) #Generate random prediction
                        preds_random_tau_xv        += [pred_random_tau_xv]
                        residuals_random_tau_xv    += [abs(pred_random_tau_xv-lbl_tau_xv)]
                        #
                        preds_tau_yv               += [pred_tau_yv]
                        lbls_tau_yv                += [lbl_tau_yv]
                        residuals_tau_yv           += [abs(pred_tau_yv-lbl_tau_yv)]
                        pred_random_tau_yv          = random.choice(preds['label_tau_yv'][:][:]) #Generate random prediction
                        preds_random_tau_yv        += [pred_random_tau_yv]
                        residuals_random_tau_yv    += [abs(pred_random_tau_yv-lbl_tau_yv)]
                        #
                        preds_tau_zv               += [pred_tau_zv]
                        lbls_tau_zv                += [lbl_tau_zv]
                        residuals_tau_zv           += [abs(pred_tau_zv-lbl_tau_zv)]
                        pred_random_tau_zv          = random.choice(preds['label_tau_zv'][:][:]) #Generate random prediction
                        preds_random_tau_zv        += [pred_random_tau_zv]
                        residuals_random_tau_zv    += [abs(pred_random_tau_zv-lbl_tau_zv)]
                        #
                        preds_tau_xw               += [pred_tau_xw]
                        lbls_tau_xw                += [lbl_tau_xw]
                        residuals_tau_xw           += [abs(pred_tau_xw-lbl_tau_xw)]
                        pred_random_tau_xw          = random.choice(preds['label_tau_xw'][:][:]) #Generate random prediction
                        preds_random_tau_xw        += [pred_random_tau_xw]
                        residuals_random_tau_xw    += [abs(pred_random_tau_xw-lbl_tau_xw)]
                        #
                        preds_tau_yw               += [pred_tau_yw]
                        lbls_tau_yw                += [lbl_tau_yw]
                        residuals_tau_yw           += [abs(pred_tau_yw-lbl_tau_yw)]
                        pred_random_tau_yw          = random.choice(preds['label_tau_yw'][:][:]) #Generate random prediction
                        preds_random_tau_yw        += [pred_random_tau_yw]
                        residuals_random_tau_yw    += [abs(pred_random_tau_yw-lbl_tau_yw)]
                        #
                        preds_tau_zw               += [pred_tau_zw]
                        lbls_tau_zw                += [lbl_tau_zw]
                        residuals_tau_zw           += [abs(pred_tau_zw-lbl_tau_zw)]
                        pred_random_tau_zw          = random.choice(preds['label_tau_zw'][:][:]) #Generate random prediction
                        preds_random_tau_zw        += [pred_random_tau_zw]
                        residuals_random_tau_zw    += [abs(pred_random_tau_zw-lbl_tau_zw)]
                        #
                        tstep_samples += [tstep]
                        zhloc_samples += [zhloc]
                        zloc_samples  += [zloc]
                        yhloc_samples += [yhloc]
                        yloc_samples  += [yloc]
                        xhloc_samples += [xhloc]
                        xloc_samples  += [xloc]

                        tot_sample_end +=1
                        #print('next sample')
                    #print('next batch')
                    
                    #Store variables
                    #
                    var_pred_tau_xu[tot_sample_begin:tot_sample_end]        = preds_tau_xu[:]
                    var_pred_random_tau_xu[tot_sample_begin:tot_sample_end] = preds_random_tau_xu[:]
                    var_lbl_tau_xu[tot_sample_begin:tot_sample_end]         = lbls_tau_xu[:]
                    var_res_tau_xu[tot_sample_begin:tot_sample_end]         = residuals_tau_xu[:]
                    var_res_random_tau_xu[tot_sample_begin:tot_sample_end]  = residuals_random_tau_xu[:]
                    #
                    var_pred_tau_yu[tot_sample_begin:tot_sample_end]        = preds_tau_yu[:]
                    var_pred_random_tau_yu[tot_sample_begin:tot_sample_end] = preds_random_tau_yu[:]
                    var_lbl_tau_yu[tot_sample_begin:tot_sample_end]         = lbls_tau_yu[:]
                    var_res_tau_yu[tot_sample_begin:tot_sample_end]         = residuals_tau_yu[:]
                    var_res_random_tau_yu[tot_sample_begin:tot_sample_end]  = residuals_random_tau_yu[:]
                    #
                    var_pred_tau_zu[tot_sample_begin:tot_sample_end]        = preds_tau_zu[:]
                    var_pred_random_tau_zu[tot_sample_begin:tot_sample_end] = preds_random_tau_zu[:]
                    var_lbl_tau_zu[tot_sample_begin:tot_sample_end]         = lbls_tau_zu[:]
                    var_res_tau_zu[tot_sample_begin:tot_sample_end]         = residuals_tau_zu[:]
                    var_res_random_tau_zu[tot_sample_begin:tot_sample_end]  = residuals_random_tau_zu[:]
                    #
                    var_pred_tau_xv[tot_sample_begin:tot_sample_end]        = preds_tau_xv[:]
                    var_pred_random_tau_xv[tot_sample_begin:tot_sample_end] = preds_random_tau_xv[:]
                    var_lbl_tau_xv[tot_sample_begin:tot_sample_end]         = lbls_tau_xv[:]
                    var_res_tau_xv[tot_sample_begin:tot_sample_end]         = residuals_tau_xv[:]
                    var_res_random_tau_xv[tot_sample_begin:tot_sample_end]  = residuals_random_tau_xv[:]
                    #
                    var_pred_tau_yv[tot_sample_begin:tot_sample_end]        = preds_tau_yv[:]
                    var_pred_random_tau_yv[tot_sample_begin:tot_sample_end] = preds_random_tau_yv[:]
                    var_lbl_tau_yv[tot_sample_begin:tot_sample_end]         = lbls_tau_yv[:]
                    var_res_tau_yv[tot_sample_begin:tot_sample_end]         = residuals_tau_yv[:]
                    var_res_random_tau_yv[tot_sample_begin:tot_sample_end]  = residuals_random_tau_yv[:]
                    #
                    var_pred_tau_zv[tot_sample_begin:tot_sample_end]        = preds_tau_zv[:]
                    var_pred_random_tau_zv[tot_sample_begin:tot_sample_end] = preds_random_tau_zv[:]
                    var_lbl_tau_zv[tot_sample_begin:tot_sample_end]         = lbls_tau_zv[:]
                    var_res_tau_zv[tot_sample_begin:tot_sample_end]         = residuals_tau_zv[:]
                    var_res_random_tau_zv[tot_sample_begin:tot_sample_end]  = residuals_random_tau_zv[:]
                    #
                    var_pred_tau_xw[tot_sample_begin:tot_sample_end]        = preds_tau_xw[:]
                    var_pred_random_tau_xw[tot_sample_begin:tot_sample_end] = preds_random_tau_xw[:]
                    var_lbl_tau_xw[tot_sample_begin:tot_sample_end]         = lbls_tau_xw[:]
                    var_res_tau_xw[tot_sample_begin:tot_sample_end]         = residuals_tau_xw[:]
                    var_res_random_tau_xw[tot_sample_begin:tot_sample_end]  = residuals_random_tau_xw[:]
                    #
                    var_pred_tau_yw[tot_sample_begin:tot_sample_end]        = preds_tau_yw[:]
                    var_pred_random_tau_yw[tot_sample_begin:tot_sample_end] = preds_random_tau_yw[:]
                    var_lbl_tau_yw[tot_sample_begin:tot_sample_end]         = lbls_tau_yw[:]
                    var_res_tau_yw[tot_sample_begin:tot_sample_end]         = residuals_tau_yw[:]
                    var_res_random_tau_yw[tot_sample_begin:tot_sample_end]  = residuals_random_tau_yw[:]
                    #
                    var_pred_tau_zw[tot_sample_begin:tot_sample_end]        = preds_tau_zw[:]
                    var_pred_random_tau_zw[tot_sample_begin:tot_sample_end] = preds_random_tau_zw[:]
                    var_lbl_tau_zw[tot_sample_begin:tot_sample_end]         = lbls_tau_zw[:]
                    var_res_tau_zw[tot_sample_begin:tot_sample_end]         = residuals_tau_zw[:]
                    var_res_random_tau_zw[tot_sample_begin:tot_sample_end]  = residuals_random_tau_zw[:]
                    #
                    vartstep[tot_sample_begin:tot_sample_end]        = tstep_samples[:]
                    varzhloc[tot_sample_begin:tot_sample_end]        = zhloc_samples[:]
                    varzloc[tot_sample_begin:tot_sample_end]         = zloc_samples[:]
                    varyhloc[tot_sample_begin:tot_sample_end]        = yhloc_samples[:]
                    varyloc[tot_sample_begin:tot_sample_end]         = yloc_samples[:]
                    varxhloc[tot_sample_begin:tot_sample_end]        = xhloc_samples[:]
                    varxloc[tot_sample_begin:tot_sample_end]         = xloc_samples[:]

                    tot_sample_begin = tot_sample_end #Make sure stored variables are not overwritten.

                except tf.errors.OutOfRangeError:
                    break #Break out of while-loop after one validation file (i.e. one flow 'snapshot'). NOTE: for this part of the code it is important that the eval_input_fn do not implement the .repeat() method on the created tf.Dataset.
    
    predictions_file.close() #Close netCDF-file after each validation file
    print("Finished making predictions for each validation file.")
###
