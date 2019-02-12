from __future__ import print_function
from custom_hooks import MetadataHook
import numpy as np
import netCDF4 as nc
import tensorflow as tf
import random
import os
import subprocess
import glob
import argparse
import matplotlib
matplotlib.use('agg')
from tensorflow.python import debug as tf_debug
#import seaborn as sns
#sns.set(style="ticks")
#import pandas

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
        help='filepath for stored means and standard deviations of input variables, whihc should refer to a nc-file created as part of the training data')
parser.add_argument('--gradients', default=None, \
        action='store_true', \
        help='Wind velocity gradients are used as input for the NN when this is true, otherwhise absolute wind velocities are used.')
parser.add_argument('--synthetic', default=None, \
        action='store_true', \
        help='Synthetic data is used as input when this is true, otherwhise real data from specified input_dir is used')
parser.add_argument('--benchmark', dest='benchmark', default=None, \
        action='store_true', \
        help='fullrun includes testing and plotting, otherwise it ends after validation loss to facilitate benchmark tests')
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

#Define settings
batch_size = args.batch_size
num_steps = args.num_steps #Number of steps, i.e. number of batches times number of epochs
output_variable = 'unres_tau_zu_sample'

num_labels = 1
random_seed = 1234

#Define function for standardization
def _standardization(variable, mean, standard_dev):
    standardized_variable = (variable - mean)/ standard_dev
    return standardized_variable

#Define parse function for tfrecord files, which gives for each component in the example_proto 
#the output in format (dict(features),labels) and normalizes according to specified means and variances.
def _parse_function(example_proto,label_name,means,stdevs):

    if args.gradients is None: #NOTE: args.gradients is a global variable defined outside this function

        keys_to_features = {
            'uc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
            'vc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
            'wc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
            'pc_sample':tf.FixedLenFeature([5,5,5],tf.float32),
            label_name :tf.FixedLenFeature([],tf.float32),
            'x_sample_size':tf.FixedLenFeature([],tf.int64),
            'y_sample_size':tf.FixedLenFeature([],tf.int64),
            'z_sample_size':tf.FixedLenFeature([],tf.int64)
        }
    
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        parsed_features['uc_sample'] = _standardization(parsed_features['uc_sample'], means['uc'], stdevs['uc'])
        parsed_features['vc_sample'] = _standardization(parsed_features['vc_sample'], means['vc'], stdevs['vc'])
        parsed_features['wc_sample'] = _standardization(parsed_features['wc_sample'], means['wc'], stdevs['wc'])
        parsed_features['pc_sample'] = _standardization(parsed_features['pc_sample'], means['pc'], stdevs['pc'])

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
            label_name     :tf.FixedLenFeature([],tf.float32),
            'x_sample_size':tf.FixedLenFeature([],tf.int64),
            'y_sample_size':tf.FixedLenFeature([],tf.int64),
            'z_sample_size':tf.FixedLenFeature([],tf.int64)
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

    labels = parsed_features.pop(label_name)
    #labels = _standardization(labels, means[label_name], stdevs[label_name]) UNCOMMENT THIS LINE ONCE MEANS AND STDEVS LABELS ARE INDEED STORED!
    return parsed_features,labels


#Define training input function
def train_input_fn(filenames,batch_size,label_name,means,stdevs):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(batch_size).repeat()
    dataset = dataset.map(lambda line:_parse_function(line,label_name,means,stdevs))
    dataset = dataset.batch(batch_size)
    #dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(filenames), count=None))
    #dataset.apply(tf.data.experimental.map_and_batch(lambda line:_parse_function(line, label_name), batch_size))
    dataset.prefetch(1)
    return dataset

def input_synthetic_fn(batch_size, num_steps_train, train_mode = True): #NOTE: used for both training and evaluation with synthetic data

    #For testing purpose num_steps is set equal to 3 million divided by batch_size, representing the case where 3 million examples are available for training (preventing memory issues when the number of training steps is high). 
    #NOTE1: If num_steps is not explicitly set anymore (but rather determined via the input), remove .repeat() below when train_mode is set to True.
    #NOTE2: in case of evaluation only 300.000 examples are being generated, representing the case where approximately 10% of the available examples is used for evaluation
    num_steps_train = int((3*10**6)/batch_size)
    if not train_mode:
        num_steps_train = int(0.1*num_steps_train)

    #Get features
    features = {}
    distribution = tf.distributions.Uniform(low=[-1.0], high=[1.0])
    if args.gradients is not None:
        raise ValueError("The usage of gradients in combination with synthetic data has not been implemented yet. Please adjust the settings accordingly.")
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
    labels = gaussian_noise.sample(sample_shape=(batch_size*num_steps_train))

    #prepare the Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    if train_mode:
        dataset = dataset.batch(batch_size).repeat() #No shuffling needed because samples are randomly generated for all training steps.
    else:
        dataset = dataset.batch(batch_size) #No .repeat() method for evaluation to ensure all randomly generated samples are only evaluated once.
    dataset.prefetch(1)

    return dataset

#Define evaluation function
def eval_input_fn(filenames,batch_size,label_name,means,stdevs):
    dataset = tf.data.TFRecordDataset(filenames)
    #dataset = dataset.shuffle(len(filenames)) #Prevent for train_and_evaluate function applied below each time the same files are chosen NOTE: only needed when evaluation is not done on whole validation set (i.e. steps is not None)
    dataset = dataset.map(lambda line:_parse_function(line,label_name,means,stdevs))
    dataset = dataset.batch(batch_size)
    #dataset.apply(tf.data.experimental.map_and_batch(lambda line:_parse_function(line,label_name), batch_size))
    dataset.prefetch(1)

    return dataset    

#Define function for splitting the training and validation set
def split_train_val(time_steps, val_ratio, random_seed):
    np.random.seed(random_seed)
    shuffled_steps = np.random.permutation(time_steps)
    val_set_size = max(int(len(time_steps) * val_ratio),1) #max(..) makes sure that always at least 1 file is selected for validation
    val_steps = shuffled_steps[:val_set_size]
    train_steps = shuffled_steps[val_set_size:]
    return train_steps,val_steps


#Define model function for CNN estimator
def CNN_model_fn(features,labels,mode,params):
    '''CNN model with 1 convolutional layer'''

    #Define input layer
#    print(features)
    #input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    if args.gradients is None: #NOTE: args.gradients is a global variable defined outside this function
        input_layer = tf.stack([features['uc_sample'],features['vc_sample'], \
                features['wc_sample'],features['pc_sample']],axis=4) #According to channel_last data format, otherwhise change axis parameter
    
        #Visualize inputs
        tf.summary.histogram('input_u', input_layer[:,:,:,:,0])
        tf.summary.histogram('input_v', input_layer[:,:,:,:,1])
        tf.summary.histogram('input_w', input_layer[:,:,:,:,2])
        tf.summary.histogram('input_p', input_layer[:,:,:,:,3])

    else:
        input_layer = tf.stack([features['ugradx'],features['ugrady'],features['ugradz'], \
                features['vgradx'],features['vgrady'],features['vgradz'], \
                features['wgradx'],features['wgrady'],features['wgradz'], \
                features['pgradx'],features['pgrady'],features['pgradz']],axis=4)

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

#    print(input_layer[:,:,:,:,0])
    
    #Define layers
    conv1_layer = tf.layers.Conv3D(filters=params['n_conv1'], kernel_size=params['kernelsize_conv1'], \
         strides=params['stride_conv1'], activation=params['activation_function'], padding="valid", name='conv1', \
            kernel_initializer=params['kernel_initializer'], data_format = 'channels_last') 
#    x = tf.layers.batch_normalization(conv1, training=True, name='block4_sepconv1_bn')
    conv1 = conv1_layer.apply(input_layer)
#    print(conv1.shape)

    ###Visualize filters convolutional layer###
#    print(conv1_layer.weights[0])
    acts_filters = tf.unstack(conv1_layer.weights[0], axis=4)
    for i, acts_filter in enumerate(acts_filters):
        threedim_slices = tf.unstack(acts_filter, axis=0) #Each slice corresponds to one of the filters applied
#        print(acts_filter.shape)
        for j, threedim_slice in enumerate(threedim_slices):
            twodim_slices = tf.unstack(threedim_slice, axis=2) #Each slice correponds to one vertical level for all four variables
#            print(threedim_slice.shape)
            for k, twodim_slice in enumerate(twodim_slices):
#                print(twodim_slice.shape)
                tf.summary.image('filter'+str(i)+'_height'+str(j)+'_variable'+str(k), tf.expand_dims(tf.expand_dims(twodim_slice, axis=2), axis=0)) #Two times tf.expand_dims to construct a 4D Tensor from the resulting 2D Tensor, which is required by tf.summary.image.
    ###

    ###Visualize activations convolutional layer (NOTE: assuming that activation maps are 1*1*1, otherwhise visualization as an 2d-image may be relevant as well)
    tf.summary.histogram('activations_hidden_layer1', conv1)
    tf.summary.scalar('fraction_of_zeros_in_activations_hidden_layer1', tf.nn.zero_fraction(conv1))

    flatten = tf.layers.flatten(conv1, name='flatten')
#    print(flatten.shape)
    output = tf.layers.dense(flatten, units=num_labels, name="outputs", \
            activation=None, kernel_initializer=params['kernel_initializer'], reuse = tf.AUTO_REUSE) #reuse needed for second part of this script to work properly
#    print(output.shape)

    ###Visualize outputs (NOTE: consider other visualization when producing more than 1 output)
    tf.summary.histogram('output', output) 

    #Compute predictions 
    labels = tf.reshape(labels,[-1,num_labels])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'value':output,'label':labels})

    #Compute loss
    mse_tau_total = tf.losses.mean_squared_error(labels, output)
    loss = tf.reduce_mean(mse_tau_total)

    #Define custom metric to store the logarithm of the los both during training and evaluation
    def log_loss_metric(labels, output):
        result, update_op = tf.metrics.mean_squared_error(labels, output)
        return tf.math.log(result), update_op

    #Compute evaluation metrics.
    rmse_tau_total,update_op_rmse = tf.metrics.root_mean_squared_error(labels, output)
    tf.summary.histogram('labels', labels) #Visualize labels
    log_loss_eval, update_op_loss = log_loss_metric(labels, output)
    metrics = {'rmse':(rmse_tau_total,update_op_rmse),'log_loss':(log_loss_eval, update_op_loss)}
    #tf.summary.scalar('rmse',rmse_tau_total)
    #tf.summary.scalar('log_loss',log_loss)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        #return tf.estimator.EstimatorSpec(mode, loss=loss)

    #Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    log_loss_training = tf.math.log(loss)
    tf.summary.scalar('log_loss', log_loss_training)

    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    #Write all trainable variables to Tensorboard
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name,var)

    #Return tf.estimator.Estimatorspec for training mode
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

#Define filenames for training and validation
nt_available = 90 #Amount of time steps available for training/validation, assuming 1) the test set is alread held separately (there are, including the test set, actually 100 time steps available), and 2) that the number of the time step in the filenames ranges from 0 to nt-1 without gaps (and thus the time steps corresponding to the test set are located after nt-1).
nt_total = 100 #Amount of time steps INCLUDING the test set
#nt_available = 2 #FOR TESTING PURPOSES ONLY!
#nt_total = 3 #FOR TESTING PURPOSES ONLY!
time_numbers = np.arange(nt_available)
#files = glob.glob(args.input_dir)
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

print('Train files: ' + str(train_filenames))
print('Validation files: ' + str(val_filenames))

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
#means_dict_t[output_variable]  = np.array(means_stdevs_file['mean_'+output_variable][:])UNCOMMENT THIS LINE ONCE MEANS AND STDEVS LABELS ARE INDEED STORED!
#stdevs_dict_t[output_variable] = np.array(means_stdevs_file['stdev_'+output_variable][:])UNCOMMENT THIS LINE ONCE MEANS AND STDEVS LABELS ARE INDEED STORED!

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
#means_dict_avgt[output_variable]  = np.mean(means_dict_t[output_variable][train_stepnumbers])UNCOMMENT THIS LINE ONCE MEANS AND STDEVS LABELS ARE INDEED STORED!
#stdevs_dict_avgt[output_variable] = np.mean(stdevs_dict_t[output_variable][train_stepnumbers])UNCOMMENT THIS LINE ONCE MEANS AND STDEVS LABELS ARE INDEED STORED!

#Set configuration
config = tf.ConfigProto(log_device_placement=False)
# config.gpu_options.allow_growth = True
config.intra_op_parallelism_threads = args.intra_op_parallelism_threads
config.inter_op_parallelism_threads = args.inter_op_parallelism_threads
os.environ['KMP_BLOCKTIME'] = str(1)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['OMP_NUM_THREADS'] = str(args.intra_op_parallelism_threads)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
# if hvd.rank()==0:
checkpoint_dir = args.checkpoint_dir

#Create RunConfig object to save check_point in the model_dir according to the specified schedule, and to define the session config
my_checkpointing_config = tf.estimator.RunConfig(model_dir=checkpoint_dir,tf_random_seed=random_seed,save_summary_steps=args.summary_steps,save_checkpoints_steps=args.checkpoint_steps, session_config=config,keep_checkpoint_max=None,keep_checkpoint_every_n_hours=10000,log_step_count_steps=10,train_distribute=None) #Provide tf.contrib.distribute.DistributionStrategy instance to train_distribute parameter for distributed training

#Instantiate an Estimator with model defined by model_fn
hyperparams =  {
#'feature_columns':feature_columns,
#'n_conv1':10,
'n_conv1':40,
'kernelsize_conv1':5,
'stride_conv1':1,
'activation_function':tf.nn.leaky_relu, #NOTE: Define new activation function based on tf.nn.leaky_relu with lambda to adjust the default value for alpha (0.02)
#'activation_function':tf.nn.relu,
#'kernel_initializer':tf.initializers.random_uniform(0,0.0001), #NOTE: small range weights ensures that the output is already from the beginning in the correct order of magnitude
'kernel_initializer':tf.initializers.he_uniform(),
#'kernel_initializer':tf.glorot_uniform_initializer(),
'learning_rate':0.0001
}

##val whether tfrecords files are correctly read.
#dataset_samples = train_input_fn(train_filenames, batch_size, output_variable)

CNN = tf.estimator.Estimator(model_fn = CNN_model_fn,config=my_checkpointing_config, params = hyperparams, model_dir=checkpoint_dir)

#custom_hook = MetadataHook(save_steps = 1000, output_dir = checkpoint_dir) #Initialize custom hook designed for storing runtime statistics that can be read using TensorBoard in combination with tf.Estimator.NOTE:an unavoidable consequence is unfortunately that the other summaries are not stored anymore. The only option to store all summaries and the runtime statistics in Tensorboard is to use low-level Tensorflow API.

profiler_hook = tf.train.ProfilerHook(save_steps = args.profile_steps, output_dir = checkpoint_dir) #Hook designed for storing runtime statistics in Chrome trace format, can be used in conjuction with the other summaries stored during training in Tensorboard.

if args.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    hooks = [profiler_hook, debug_hook]
else:
    hooks = [profiler_hook]

if args.synthetic is None:
    #Train and evaluate CNN
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:train_input_fn(train_filenames,batch_size,output_variable,means_dict_avgt,stdevs_dict_avgt), max_steps=num_steps, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:eval_input_fn(val_filenames,batch_size,output_variable,means_dict_avgt,stdevs_dict_avgt), steps=None, name='CNN1', start_delay_secs=120, throttle_secs=0)#NOTE: throttle_secs=0 implies that for every stored checkpoint the validation error is calculated for 1000 training steps
    tf.estimator.train_and_evaluate(CNN, train_spec, eval_spec)

#    #Train the CNN
#    CNN.train(input_fn=lambda:train_input_fn(train_filenames,batch_size,output_variable),steps=num_steps, hooks=[profiler_hook])

    #Evaluate the CNN on all validation data (no imposed limit on training steps)
    eval_results = CNN.evaluate(input_fn=lambda:eval_input_fn(val_filenames,batch_size,output_variable,means_dict_avgt,stdevs_dict_avgt), steps=None)
    print('\nValidation set RMSE: {rmse:.10e}\n'.format(**eval_results))
    print('Used real data')
#    predictions = CNN.predict(input_fn = lambda:eval_input_fn(val_filenames, batch_size, output_variable))
#    NOTE: CNN.predict appeared to be unsuitable to compare the predictions from the CNN to the true labels stored in the TFRecords files: the labels are discarded by the tf.estimator.Estimator in predict mode. The alternative is the 'hacky' solution implemented in the code below.

else:
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_synthetic_fn(batch_size, num_steps, train_mode = True), max_steps=num_steps, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_synthetic_fn(batch_size, num_steps, train_mode = False), steps=None, name='CNN1', start_delay_secs=120, throttle_secs=0)#NOTE: throttle_secs=0 implies that for every stored checkpoint the validation error is calculated for 1000 training steps
    tf.estimator.train_and_evaluate(CNN, train_spec, eval_spec)

#    #Train the CNN
#    CNN.train(input_fn=lambda:train_input_synthetic_fn(batch_size, num_steps), steps=num_steps, hooks=[profiler_hook])
    #Evaluate the CNN afterwards
    eval_results = CNN.evaluate(input_fn=lambda:input_synthetic_fn(batch_size, num_steps, train_mode = False), steps=None) #NOTE: putting steps at None or 1000 is equivalent.
    print('\nValidation set RMSE: {rmse:.10e}\n'.format(**eval_results))  
    print('Used synthetic data')


#'Hacky' solution to compare the predictions of the CNN to the true labels stored in the TFRecords files. NOTE: the input and model function are called manually rather than using the tf.estimator.Estimator syntax.
if args.benchmark is None:
   
    #val_filenames = train_filenames #NOTE: this line only implemented to test the script. REMOVE it later on!!!
 
    #Loop over val files to prevent memory overflow issues
    if args.synthetic is not None:
        val_filenames = ['dummy'] #Dummy value of length 1 to ensure loop is only done once for synthetic data
    
    create_file = True #Make sure netCDF file is initialized
 
    #Print used data
    #if args.synthetic is None:
        #print('Validation filenames:')
        #print(val_filenames)
    #else:
        #print('Used synthetic data')
 
    create_file = True #Make sure netCDF file is initialized
 
    #Initialize variables for keeping track of iterations
    tot_sample_end = 0
    tot_sample_begin = tot_sample_end

    for val_filename in val_filenames:

        tf.reset_default_graph() #Reset the graph for each iteration
 
        #Generate iterator to extra features and labels from input data
        if args.synthetic is None:
            iterator = eval_input_fn([val_filename], batch_size, output_variable, means_dict_avgt,stdevs_dict_avgt).make_initializable_iterator() #All samples present in val_filenames are used for validation once (Note that no .repeat() method is included in eval_input_fn, which is in contrast to train_input_fn).
 
        else:
#           iterator = train_input_synthetic_fn(batch_size, 1000).make_one_shot_iterator() #1000 samples are generated and subsequently used for validation, NOTE: this line raises error message because the one_shot_iterator cannot capture statefull nodes contained in train_input_synthetic_fn.
            iterator = input_synthetic_fn(batch_size, num_steps, train_mode = False).make_initializable_iterator()
 
#            #needed without eager execution
#            tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
#            sess.run(iterator.initializer) #Initialize iterator
 
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
            filepath = checkpoint_dir + '/CNN_predictions.nc'
            predictions_file = nc.Dataset(filepath, 'w')
            dim_ns = predictions_file.createDimension("ns",None)

            #Create variables for storage
            var_pred = predictions_file.createVariable("preds_values","f8",("ns",))
            var_pred_random = predictions_file.createVariable("preds_values_random","f8",("ns",))
            var_lbl = predictions_file.createVariable("lbls_values","f8",("ns",))
            var_res = predictions_file.createVariable("residuals","f8",("ns",))
            var_res_random = predictions_file.createVariable("residuals_random","f8",("ns",))

            create_file=False #Make sure file is only created once

        else:
            predictions_file = nc.Dataset(filepath, 'r+')


        with tf.Session(config=config) as sess:

#            for n in tf.get_default_graph().as_graph_def().node:
#                if 'Variable' in n.op:
#                    print(n)

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
                    preds_values = []
                    preds_values_random = []
                    lbls_values = []
                    residuals = []
                    residuals_random = []

                    #print(preds['label'])
                    for pred,lbl in zip(preds['value'],preds['label']):
                        #print('\nPrediction is "{:.10e}", expectation is "{:.10e}".'.format(pred[0],lbl)) #Index 0 needed to index single value in ndarray
                        preds_values += [pred[0]]
                        #print('prediction :' + str(pred[0]))
                        lbls_values += [lbl[0]]
                        #print('label :' + str(lbl[0]))
                        residuals += [abs(pred[0]-lbl[0])]
                        pred_random = random.choice(preds['label'][:][:]) #Generate random prediction
                        #print('random prediction:' + str(pred_random[0]))
                        preds_values_random += [pred_random]
                        residuals_random += [abs(pred_random-lbl[0])]
                        tot_sample_end +=1
                        #print('next sample')
                    #print('next batch')
                    
                    #Store variables
                    var_pred[tot_sample_begin:tot_sample_end] = preds_values[:]
                    var_pred_random[tot_sample_begin:tot_sample_end] = preds_values_random[:]
                    var_lbl[tot_sample_begin:tot_sample_end] = lbls_values[:]
                    var_res[tot_sample_begin:tot_sample_end] = residuals[:]
                    var_res_random[tot_sample_begin:tot_sample_end] = residuals_random[:]
                    tot_sample_begin = tot_sample_end #Make sure stored variables are not overwritten.

                except tf.errors.OutOfRangeError:
                    break #Break out of while-loop after one epoch. NOTE: for this part of the code it is important that the eval_input_fn and train_input_synthetic_fn do not implement the .repeat() method on the created tf.Dataset.
        #print('next_file')
    predictions_file.close() #Close netCDF-file after each validation file
