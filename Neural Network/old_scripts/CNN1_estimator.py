from __future__ import print_function
from custom_hooks import MetadataHook
import numpy as np
import netCDF4 as nc
import tensorflow as tf
import horovod.tensorflow as hvd
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
                    help='Checkpoint directory (for rank 0)')
parser.add_argument('--input_dir', type=str, default='/projects/1/flowsim/simulation1/',
                    help='tfrecords filepaths')
parser.add_argument('--stored_means_stdevs_filepath', type=str, default='/projects/1/flowsim/simulation1/means_stdevs_allfields.nc', \
        help='filepath for stored means and standard deviations of input variables, which should refer to a nc-file created as part of the training data')
parser.add_argument('--gradients', default=None, \
        action='store_true', \
        help='Wind velocity gradients are used as input for the NN when this is true, otherwhise absolute wind velocities are used.')
parser.add_argument('--synthetic', default=None, \
        action='store_true', \
        help='Synthetic data is used as input when this is true, otherwhise real data from specified input_dir is used')
parser.add_argument('--benchmark', dest='benchmark', default=None, \
        action='store_true', \
        help='fullrun includes testing and storing preditions, otherwise it ends after validation loss to facilitate benchmark tests')
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
        help='Every nth step, a profile measurement is performed')
parser.add_argument('--summary_steps', type=int, default=100, \
        help='Every nth step, a summary is written for Tensorboard visualization')
parser.add_argument('--checkpoint_steps', type=int, default=10000, \
        help='Every nth step, a checkpoint of the model is written')
args = parser.parse_args()

#Initialize Horovod
hvd.init()

#Define settings
batch_size = int(args.batch_size / hvd.size()) #Compensate batch size for number of workers
num_steps = args.num_steps #Number of steps, i.e. number of batches times number of epochs
num_labels = 9
random_seed = 1234

#Define function for standardization
def _standardization(variable, mean, standard_dev):
    standardized_variable = (variable - mean) / standard_dev
    return standardized_variable

#Define parse function for tfrecord files, which gives for each component in the example_proto 
#the output in format (dict(features),tensor(labels)) and normalizes according to specified means and variances.
def _parse_function(example_proto,means,stdevs):

    if args.gradients is None: #NOTE: args.gradients is a global variable defined outside this function

        if args.benchmark is None:
            keys_to_features = {
                'uc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
                'vc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
                'wc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
                'pc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
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
                'uc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
                'vc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
                'wc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
                'pc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
                'unres_tau_xu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample' :tf.FixedLenFeature([],tf.float32),
            }

            
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        parsed_features['uc_sample'] = _standardization(parsed_features['uc_sample'], means['uc'], stdevs['uc'])
        parsed_features['vc_sample'] = _standardization(parsed_features['vc_sample'], means['vc'], stdevs['vc'])
        parsed_features['wc_sample'] = _standardization(parsed_features['wc_sample'], means['wc'], stdevs['wc'])
        parsed_features['pc_sample'] = _standardization(parsed_features['pc_sample'], means['pc'], stdevs['pc'])

    else:

        if args.benchmark is None:
            keys_to_features = {
                'ugradx_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'ugrady_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'ugradz_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'vgradx_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'vgrady_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'vgradz_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'wgradx_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'wgrady_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'wgradz_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'pgradx_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'pgrady_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'pgradz_sample':tf.FixedLenFeature([3,3,3],tf.float32),
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
                'ugradx_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'ugrady_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'ugradz_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'vgradx_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'vgrady_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'vgradz_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'wgradx_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'wgrady_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'wgradz_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'pgradx_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'pgrady_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'pgradz_sample':tf.FixedLenFeature([3,3,3],tf.float32),
                'unres_tau_xu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample' :tf.FixedLenFeature([],tf.float32),
            }

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        parsed_features['ugradx_sample'] = _standardization(parsed_features['ugradx_sample'], means['ugradx'], stdevs['ugradx'])
        parsed_features['ugrady_sample'] = _standardization(parsed_features['ugrady_sample'], means['ugrady'], stdevs['ugrady'])
        parsed_features['ugradz_sample'] = _standardization(parsed_features['ugradz_sample'], means['ugradz'], stdevs['ugradz'])
        parsed_features['vgradx_sample'] = _standardization(parsed_features['vgradx_sample'], means['vgradx'], stdevs['vgradx'])
        parsed_features['vgrady_sample'] = _standardization(parsed_features['vgrady_sample'], means['vgrady'], stdevs['vgrady'])
        parsed_features['vgradz_sample'] = _standardization(parsed_features['vgradz_sample'], means['vgradz'], stdevs['vgradz'])
        parsed_features['wgradx_sample'] = _standardization(parsed_features['wgradx_sample'], means['wgradx'], stdevs['wgradx'])
        parsed_features['wgrady_sample'] = _standardization(parsed_features['wgrady_sample'], means['wgrady'], stdevs['wgrady'])
        parsed_features['wgradz_sample'] = _standardization(parsed_features['wgradz_sample'], means['wgradz'], stdevs['wgradz'])
        parsed_features['pgradx_sample'] = _standardization(parsed_features['pgradx_sample'], means['pgradx'], stdevs['pgradx'])
        parsed_features['pgrady_sample'] = _standardization(parsed_features['pgrady_sample'], means['pgrady'], stdevs['pgrady'])
        parsed_features['pgradz_sample'] = _standardization(parsed_features['pgradz_sample'], means['pgradz'], stdevs['pgradz'])

    #Extract labels from the features dictionary, store them in a new labels array, and standardize them
    def _getlabel(parsed_features_array, label_name, means_array, stdevs_array):
        single_label = parsed_features_array.pop(label_name)
        single_label = _standardization(single_label, means_array[label_name], stdevs_array[label_name])
        return single_label

    labels = {}
    labels['unres_tau_xu'] = _getlabel(parsed_features, 'unres_tau_xu_sample', means, stdevs)
    labels['unres_tau_yu'] = _getlabel(parsed_features, 'unres_tau_yu_sample', means, stdevs)
    labels['unres_tau_zu'] = _getlabel(parsed_features, 'unres_tau_zu_sample', means, stdevs)
    labels['unres_tau_xv'] = _getlabel(parsed_features, 'unres_tau_xv_sample', means, stdevs)
    labels['unres_tau_yv'] = _getlabel(parsed_features, 'unres_tau_yv_sample', means, stdevs)
    labels['unres_tau_zv'] = _getlabel(parsed_features, 'unres_tau_zv_sample', means, stdevs)
    labels['unres_tau_xw'] = _getlabel(parsed_features, 'unres_tau_xw_sample', means, stdevs)
    labels['unres_tau_yw'] = _getlabel(parsed_features, 'unres_tau_yw_sample', means, stdevs)
    labels['unres_tau_zw'] = _getlabel(parsed_features, 'unres_tau_zw_sample', means, stdevs)

    labels = tf.convert_to_tensor([ labels['unres_tau_xu'], labels['unres_tau_yu'], labels['unres_tau_zu'], labels['unres_tau_xv'],  labels['unres_tau_yv'], labels['unres_tau_zv'], labels['unres_tau_xw'], labels['unres_tau_yw'], labels['unres_tau_zw']], dtype=tf.float32)

    return parsed_features,labels


#Define training input function
def train_input_fn(filenames, batch_size, means, stdevs):
    dataset = tf.data.TFRecordDataset(filenames)
    #dataset = dataset.shuffle(len(filenames)) #comment this line when cache() is done after map()
    dataset = dataset.map(lambda line:_parse_function(line, means, stdevs))
    dataset = dataset.cache() #NOTE: The unavoidable consequence of using cache() before shuffle is that during all epochs the order of the flow fields is approximately the same (which can be alleviated by choosing a large buffer size, but that costs quite some computational effort). However, using shuffle before cache() will strongly increase the computationel effort since memory becomes saturated. 
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset.prefetch(1)

    return dataset

def input_synthetic_fn(batch_size, num_steps_train, train_mode = True): #NOTE: used for both training and evaluation with synthetic data

    #For testing purpose num_steps is set equal to 3 million divided by batch_size, representing the case where 3 million examples are available for training (preventing memory issues when the number of training steps is high). 
    #NOTE1: in case of evaluation only 300.000 examples are being generated, representing the case where approximately 10% of the available examples is used for evaluation
    num_steps_train = int((3*10**6)/batch_size)
    if not train_mode:
        num_steps_train = int(0.1*num_steps_train)

    #Get features
    features = {}
    distribution = tf.distributions.Uniform(low=[-1.0], high=[1.0])
    if args.gradients is not None:
        raise RuntimeError("The usage of gradients in combination with synthetic data has not been implemented yet. Please adjust the settings accordingly.")
    features['uc_sample'] = tf.squeeze(distribution.sample(sample_shape=(batch_size*num_steps_train, 5, 5, 5)))
    features['vc_sample'] = tf.squeeze(distribution.sample(sample_shape=(batch_size*num_steps_train, 5, 5, 5)))
    features['wc_sample'] = tf.squeeze(distribution.sample(sample_shape=(batch_size*num_steps_train, 5, 5, 5)))
    features['pc_sample'] = tf.squeeze(distribution.sample(sample_shape=(batch_size*num_steps_train, 5, 5, 5)))
    
    #Get labels
    #linear
    #labels = tf.reduce_sum(features['uc_sample'], [1,2,3])/125 + \
    #         tf.reduce_sum(features['pc_sample'], [1,2,3])/125 + \
    #         tf.reduce_sum(features['vc_sample'], [1,2,3])/125 + \
    #         tf.reduce_sum(features['wc_sample'], [1,2,3])/125
    #constant
    #labels = [0.0] * batch_size * num_steps_train
    #constant with random Gaussian noise
    gaussian_noise = tf.distributions.Normal(loc=0., scale=0.01)
    labels = gaussian_noise.sample(sample_shape=(batch_size*num_steps_train, 9))

    #prepare the Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    if train_mode:
        dataset = dataset.batch(batch_size).repeat() #No shuffling needed because samples are randomly generated for all training steps.
    else:
        dataset = dataset.batch(batch_size) #No .repeat() method for evaluation to ensure all randomly generated samples are only evaluated once.
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
def split_train_val(time_steps, val_ratio, random_seed):
    shuffled_steps = np.flip(time_steps) #NOTE: this causes the split not to be random anymore! Always the last timesteps are selected for validation.
    val_set_size = max(int(len(time_steps) * val_ratio),1) #max(..) makes sure that always at least 1 file is selected for validation
    val_steps = shuffled_steps[:val_set_size]
    train_steps = shuffled_steps[val_set_size:]
    return train_steps,val_steps


#Define model function for CNN estimator
def CNN_model_fn(features,labels,mode,params):
    '''CNN model with 1 convolutional layer'''

    #Define input layer
    if args.gradients is None: #NOTE: args.gradients is a global variable defined outside this function
        input_layer = tf.stack([features['uc_sample'],features['vc_sample'], \
                features['wc_sample'],features['pc_sample']],axis=4) #According to channel_last data format, otherwhise change axis parameter
    
        #Visualize inputs
        tf.summary.histogram('input_u', input_layer[:,:,:,:,0])
        tf.summary.histogram('input_v', input_layer[:,:,:,:,1])
        tf.summary.histogram('input_w', input_layer[:,:,:,:,2])
        tf.summary.histogram('input_p', input_layer[:,:,:,:,3])

    else:
        input_layer = tf.stack([features['ugradx_sample'],features['ugrady_sample'],features['ugradz_sample'], \
                features['vgradx_sample'],features['vgrady_sample'],features['vgradz_sample'], \
                features['wgradx_sample'],features['wgrady_sample'],features['wgradz_sample'], \
                features['pgradx_sample'],features['pgrady_sample'],features['pgradz_sample']],axis=4)

        #Visualize inputs
        tf.summary.histogram('input_ugradx', input_layer[:,:,:,:,0])
        tf.summary.histogram('input_ugrady', input_layer[:,:,:,:,1])
        tf.summary.histogram('input_ugradz', input_layer[:,:,:,:,2])
        tf.summary.histogram('input_vgradx', input_layer[:,:,:,:,3])
        tf.summary.histogram('input_vgrady', input_layer[:,:,:,:,4])
        tf.summary.histogram('input_vgradz', input_layer[:,:,:,:,5])
        tf.summary.histogram('input_wgradx', input_layer[:,:,:,:,6])
        tf.summary.histogram('input_wgrady', input_layer[:,:,:,:,7])
        tf.summary.histogram('input_wgradz', input_layer[:,:,:,:,8])
        tf.summary.histogram('input_pgradx', input_layer[:,:,:,:,9])
        tf.summary.histogram('input_pgrady', input_layer[:,:,:,:,10])
        tf.summary.histogram('input_pgradz', input_layer[:,:,:,:,11])

    #Define layers
    conv1_layer = tf.layers.Conv3D(filters=params['n_conv1'], kernel_size=params['kernelsize_conv1'], \
         strides=params['stride_conv1'], activation=params['activation_function'], padding="valid", name='conv1', \
            kernel_initializer=params['kernel_initializer'], data_format = 'channels_last') 
    conv1 = conv1_layer.apply(input_layer)

    ###Visualize activations convolutional layer (NOTE: assuming that activation maps are 1*1*1, otherwhise visualization as an 2d-image may be relevant as well)
    tf.summary.histogram('activations_hidden_layer1', conv1)
    tf.summary.scalar('fraction_of_zeros_in_activations_hidden_layer1', tf.nn.zero_fraction(conv1))

    flatten = tf.layers.flatten(conv1, name='flatten')

    output = tf.layers.dense(flatten, units=num_labels, name="outputs", \
            activation=None, kernel_initializer=params['kernel_initializer'], reuse = tf.AUTO_REUSE) #reuse needed for second part of this script to work properly

    ###Visualize outputs
    tf.summary.histogram('output', output) 

    #Compute predictions
    if mode == tf.estimator.ModeKeys.PREDICT and args.benchmark is None:
        #Concantenate predictions of all threads together
        output   = hvd.allgather(output)
        labels   = hvd.allgather(labels)
        return tf.estimator.EstimatorSpec(mode, predictions={
            'pred_tau_xu':output[:,0], 'label_tau_xu':labels[:,0],
            'pred_tau_yu':output[:,1], 'label_tau_yu':labels[:,1],
            'pred_tau_zu':output[:,2], 'label_tau_zu':labels[:,2],
            'pred_tau_xv':output[:,3], 'label_tau_xv':labels[:,3],
            'pred_tau_yv':output[:,4], 'label_tau_yv':labels[:,4],
            'pred_tau_zv':output[:,5], 'label_tau_zv':labels[:,5],
            'pred_tau_xw':output[:,6], 'label_tau_xw':labels[:,6],
            'pred_tau_yw':output[:,7], 'label_tau_yw':labels[:,7],
            'pred_tau_zw':output[:,8], 'label_tau_zw':labels[:,8],
            'tstep':hvd.allgather(features['tstep_sample']), 'zhloc':hvd.allgather(features['zhloc_sample']),
            'zloc':hvd.allgather(features['zloc_sample']), 'yhloc':hvd.allgather(features['yhloc_sample']),
            'yloc':hvd.allgather(features['yloc_sample']), 'xhloc':hvd.allgather(features['xhloc_sample']),
            'xloc':hvd.allgather(features['xloc_sample'])})
 
    elif mode == tf.estimator.ModeKeys.PREDICT:
        #Concantenate predictions of all threads together
        output   = hvd.allgather(output)
        labels   = hvd.allgather(labels)
        return tf.estimator.EstimatorSpec(mode, predictions={
            'pred_tau_xu':output[:,0],
            'pred_tau_yu':output[:,1], 
            'pred_tau_zu':output[:,2], 
            'pred_tau_xv':output[:,3], 
            'pred_tau_yv':output[:,4], 
            'pred_tau_zv':output[:,5], 
            'pred_tau_xw':output[:,6], 
            'pred_tau_yw':output[:,7], 
            'pred_tau_zw':output[:,8]}) 
    
    #Compute loss
    mse_tau_total = tf.losses.mean_squared_error(labels, output)
    loss = tf.reduce_mean(mse_tau_total)
        
    #Define function to calculate the logarithm
    def log10(values):
        numerator = tf.log(values)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    #Compute evaluation metrics.
    tf.summary.histogram('labels', labels) #Visualize labels
    if mode == tf.estimator.ModeKeys.EVAL:
        mse, update_op = tf.metrics.mean_squared_error(labels,output)
        mse_all = hvd.allreduce(mse) #Average mse over all workers using allreduce, should be identical to the case where you calculate mse at once over all the samples the workers contain.
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

    optimizer = tf.train.AdamOptimizer(params['learning_rate'] * hvd.size()) #Should learning rate be scaled?
    
    #Add Horovod distributed optimizer
    optimizer = hvd.DistributedOptimizer(optimizer)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    #Write all trainable variables to Tensorboard
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name,var)

    #Return tf.estimator.Estimatorspec for training mode
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

#Define filenames for training and validation
nt_available = 90 #Amount of time steps available for training/validation, assuming 1) the test set is alread held separately (there are, including the test set, actually 100 time steps available), and 2) that the number of the time step in the filenames ranges from 1 to nt_available without gaps (and thus the time steps corresponding to the test set are located after nt_available).
nt_total = 100 #Amount of time steps INCLUDING the test set
#nt_available = 2 #FOR TESTING PURPOSES ONLY!
#nt_total = 3 #FOR TESTING PURPOSES ONLY!
time_numbers = np.arange(nt_available)
train_stepnumbers, val_stepnumbers = split_train_val(time_numbers, 0.1, random_seed=random_seed) #Set aside 10% of files for validation. Please note that a separate, independent test set should be created manually.
train_filenames = np.zeros((len(train_stepnumbers),), dtype=object)
val_filenames   = np.zeros((len(val_stepnumbers),), dtype=object)

i=0
for train_stepnumber in train_stepnumbers: #Generate training filenames from selected step numbers and total steps
    if args.gradients is None:
        train_filenames[i] = args.input_dir + 'training_time_step_{0}_of_{1}.tfrecords'.format(train_stepnumber+1, nt_total)
    else:
        train_filenames[i] = args.input_dir + 'training_time_step_{0}_of_{1}_gradients.tfrecords'.format(train_stepnumber+1,nt_total)
    i+=1

j=0
for val_stepnumber in val_stepnumbers: #Generate validation filenames from selected step numbers and total steps
    if args.gradients is None:
        val_filenames[j] = args.input_dir + 'training_time_step_{0}_of_{1}.tfrecords'.format(val_stepnumber+1, nt_total)
    else:
        val_filenames[j] = args.input_dir + 'training_time_step_{0}_of_{1}_gradients.tfrecords'.format(val_stepnumber+1, nt_total)
    j+=1

#Distribute training files over workers
try:
    train_filenames_worker = np.split(train_filenames, hvd.size())[int(hvd.rank())] #Get one subarray from whole array of training files for each worker, which have equal sizes.
    print('Train files of worker ' + str(hvd.rank()) + ': ' + str(train_filenames_worker))

    val_filenames_worker = np.split(val_filenames, hvd.size())[int(hvd.rank())] #Get one subarray from whole array of training files for each worker, which have equal sizes.
    print('Val files of worker ' + str(hvd.rank()) + ': ' + str(val_filenames_worker))
except ValueError:
    raise RuntimeError("If multiple workers/threads are used, each worker/thread receives an equal share of the training and validation files. This only works however if both the number of training and validation files are an exact multiple of the number of workers/threads. If for instance 3 workers/threads are used, both the number of training and validation files should be a multiple of 3.")

#Calculate means and stdevs for input variables
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

#Extract temporally averaged mean & standard deviation labels
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
os.environ['KMP_BLOCKTIME'] = str(1)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['OMP_NUM_THREADS'] = str(args.intra_op_parallelism_threads)

# Horovod: save checkpoints and variables for each worker in a different directory to prevent other workers from corrupting them. NOTE: if only the checkpoints and variables of worker 0 are stored, restarting the training from the last checkpoints results in errors being thrown.
if hvd.rank() == 0:
    checkpoint_dir       = args.checkpoint_dir
else:
    checkpoint_dir       = args.checkpoint_dir + '/worker' + str(hvd.rank())

#Set warmstart_dir to None to disable it
warmstart_dir = None

#Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from rank 0 to all other processes. \
        # This is necessary to ensure consistent initialisation of all workers when training is started with \
        # random weights or restored from a checkpoint.
bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

#Create RunConfig object to save check_point in the model_dir according to the specified schedule, and to define the session config
my_checkpointing_config = tf.estimator.RunConfig(model_dir=checkpoint_dir, tf_random_seed=random_seed, save_summary_steps=args.summary_steps, save_checkpoints_steps=args.checkpoint_steps, save_checkpoints_secs = None,session_config=config,keep_checkpoint_max=None, keep_checkpoint_every_n_hours=10000, log_step_count_steps=10, train_distribute=None) #Provide tf.contrib.distribute.DistributionStrategy instance to train_distribute parameter for distributed training

#Define hyperparameters
if args.gradients is None:
    kernelsize_conv1 = 5
else:
    kernelsize_conv1 = 3

hyperparams =  {
'n_conv1':80,
#'n_dense1':30,
#'n_dense2':30,
'kernelsize_conv1':kernelsize_conv1,
'stride_conv1':1,
'activation_function':tf.nn.leaky_relu, #NOTE: Define new activation function based on tf.nn.leaky_relu with lambda to adjust the default value for alpha (0.02)
'kernel_initializer':tf.initializers.he_uniform(),
'learning_rate':0.0001
}

#Instantiate an Estimator with model defined by model_fn
CNN = tf.estimator.Estimator(model_fn = CNN_model_fn, config=my_checkpointing_config, params = hyperparams, model_dir=checkpoint_dir, warm_start_from = warmstart_dir)

if hvd.rank() == 0:
    profiler_hook = tf.train.ProfilerHook(save_steps = args.profile_steps, output_dir = checkpoint_dir) #Hook designed for storing runtime statistics in Chrome trace format, can be used in conjuction with the other summaries stored during training in Tensorboard.

if hvd.rank() == 0 and args.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    hooks = [profiler_hook, bcast_hook, debug_hook]
#    hooks = [bcast_hook, debug_hook]
elif hvd.rank() == 0:
    hooks = [profiler_hook, bcast_hook]
else:
    hooks = [bcast_hook]

if args.synthetic is None:

    #Train and evaluate CNN
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:train_input_fn(train_filenames_worker, batch_size, means_dict_avgt, stdevs_dict_avgt), max_steps=num_steps, hooks=hooks) #Horovod:scaled batch size with number of workers
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:eval_input_fn(val_filenames_worker, batch_size, means_dict_avgt, stdevs_dict_avgt), steps=None, name='CNN1', start_delay_secs=30, throttle_secs=0)#NOTE: throttle_secs=0 implies that for every stored checkpoint the validation error is calculated
    tf.estimator.train_and_evaluate(CNN, train_spec, eval_spec)

#    NOTE: CNN.predict appeared to be unsuitable to compare the predictions from the CNN to the true labels stored in the TFRecords files: the labels are discarded by the tf.estimator.Estimator in predict mode. The alternative is the 'hacky' solution implemented in the code below.

else:
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_synthetic_fn(batch_size, num_steps, train_mode = True), max_steps=num_steps, hooks=hooks) #Horovod:scaled batch size with number of workers
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_synthetic_fn(batch_size, num_steps, train_mode = False), steps=None, name='CNN1', start_delay_secs=30, throttle_secs=0)#NOTE: throttle_secs=0 implies that for every stored checkpoint the validation error is calculated
    tf.estimator.train_and_evaluate(CNN, train_spec, eval_spec)

######
#'Hacky' solution to compare the predictions of the CNN to the true labels stored in the TFRecords files. NOTE: the input and model function are called manually rather than using the tf.estimator.Estimator syntax.
if args.benchmark is None and args.synthetic is None:

    print('Start making predictions for validation files.')
   
    #Loop over val files to prevent memory overflow issues
    if args.synthetic is not None:
        val_filenames_worker = ['dummy'] #Dummy value of length 1 to ensure loop is only done once for synthetic data
    
    create_file = True #Make sure netCDF file is initialized
 
    #Initialize variables for keeping track of iterations
    tot_sample_end = 0
    tot_sample_begin = tot_sample_end

    for val_filename in val_filenames_worker:

        tf.reset_default_graph() #Reset the graph for each iteration
 
        #Generate iterator to extract features and labels from input data
        if args.synthetic is None:
            iterator = eval_input_fn([val_filename], batch_size, means_dict_avgt, stdevs_dict_avgt).make_initializable_iterator() #All samples present in val_filenames are used for validation once (Note that no .repeat() method is included in eval_input_fn, which is in contrast to train_input_fn).
 
        else:
            iterator = input_synthetic_fn(batch_size, num_steps, train_mode = False).make_initializable_iterator()
 
        ##Run predictions node in computational graph and store both labels and predictions in netCDF file.

        #Define operation to extract features and labels from iterator
        fes, lbls = iterator.get_next()

        #Define operation to generate predictions for extracted features and labels
        preds_op = CNN_model_fn(fes, lbls, \
                        tf.estimator.ModeKeys.PREDICT, hyperparams).predictions

        #Save CNN_model such that it can be restored in the tf.Session() below
        saver = tf.train.Saver()

        #Create/open netCDF-file
        if create_file:
            filepath = checkpoint_dir + '/CNN_predictions_worker_' + str(hvd.rank()) + '.nc'
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
            ckpt  = tf.train.get_checkpoint_state(checkpoint_dir)
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
                    break #Break out of while-loop after one epoch. NOTE: for this part of the code it is important that the eval_input_fn and train_input_synthetic_fn do not implement the .repeat() method on the created tf.Dataset.
    
    predictions_file.close() #Close netCDF-file after each validation file
    print("Finished making predictions for each validation file.")
###
