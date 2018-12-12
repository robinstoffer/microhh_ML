from __future__ import print_function
import numpy as np
import netCDF4 as nc
import tensorflow as tf
import os
import glob
import argparse
import matplotlib
matplotlib.use('agg')
#import seaborn as sns
#sns.set(style="ticks")
#import pandas

##Enable eager execution
#tf.enable_eager_execution()

# Instantiate the parser
parser = argparse.ArgumentParser(description='microhh_ML')
parser.add_argument('--checkpoint_dir', type=str, default='/projects/1/flowsim/simulation1/CNN_checkpoints',
                    help='Checkpoint directory (for rank 0)')
parser.add_argument('--input_dir', type=str, default='/projects/1/flowsim/simulation1/training_time_step*[0-9]_of*[0-9].tfrecords',
                    help='tfrecords filepaths')
parser.add_argument('--synthetic', default=None, \
        action='store_true', \
        help='Synthetic data is used as input when this is true, otherwhise real data from specified input_dir is used')
parser.add_argument('--benchmark', dest='benchmark', default=None, \
        action='store_true', \
        help='fullrun includes testing and plotting, otherwise it ends after validation loss to facilitate benchmark tests')
parser.add_argument('--intra_op_parallelism_threads', type=int, default=31, \
        help='intra_op_parallelism_threads')
parser.add_argument('--inter_op_parallelism_threads', type=int, default=1, \
        help='inter_op_parallelism_threads')
parser.add_argument('--num_steps', type=int, default=10000, \
        help='Number of steps, i.e. number of batches times number of epochs')
parser.add_argument('--batch_size', type=int, default=100, \
        help='Number of samples selected in each batch')
args = parser.parse_args()

#Define settings
batch_size = args.batch_size
num_steps = args.num_steps #Number of steps, i.e. number of batches times number of epochs
output_variable = 'unres_tau_zu_sample'

num_labels = 1
random_seed = 1234

#Define parse function for tfrecord files, which gives for each component in the example_proto 
#the output in format (dict(features),labels)
def _parse_function(example_proto,label_name):
    keys_to_features = {
#        'uc_sample':tf.VarLenFeature(tf.float64),
#        'vc_sample':tf.VarLenFeature(tf.float64),
#        'wc_sample':tf.VarLenFeature(tf.float64),
#        'pc_sample':tf.VarLenFeature(tf.float64),
#        label_name :tf.VarLenFeature(tf.float64),
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

    #Reshape input data
#    xsize = parsed_features['x_sample_size']
#    ysize = parsed_features['y_sample_size']
#    zsize = parsed_features['z_sample_size']
#    xsize = 5
#    ysize = 5
#    zsize = 5
#    parsed_features['uc_sample'] = tf.reshape(parsed_features['uc_sample'], [zsize,ysize,xsize])
#    parsed_features['vc_sample'] = tf.reshape(parsed_features['vc_sample'], [zsize,ysize,xsize])
#    parsed_features['wc_sample'] = tf.reshape(parsed_features['wc_sample'], [zsize,ysize,xsize])
#    parsed_features['pc_sample'] = tf.reshape(parsed_features['pc_sample'], [zsize,ysize,xsize])
    
#    #Convert sparse tensors to dense tensors
#    parsed_features['uc_sample'] = tf.sparse_tensor_to_dense(parsed_features['uc_sample'],default_value=-9999)
#    parsed_features['vc_sample'] = tf.sparse_tensor_to_dense(parsed_features['vc_sample'],default_value=-9999)
#    parsed_features['wc_sample'] = tf.sparse_tensor_to_dense(parsed_features['wc_sample'],default_value=-9999)
#    parsed_features['pc_sample'] = tf.sparse_tensor_to_dense(parsed_features['pc_sample'],default_value=-9999)
#    parsed_features[label_name] = tf.sparse_tensor_to_dense(parsed_features[label_name],default_value=-9999)
    labels = parsed_features.pop(label_name)
    return parsed_features,labels


#Define training input function
def train_input_fn(filenames,batch_num,label_name):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda line:_parse_function(line,label_name))
#    dataset = dataset.batch(batch_num)
    dataset = dataset.shuffle(1000000).repeat().batch(batch_num)
#    features, labels = dataset.make_one_shot_iterator().get_next()
#    return features,labels
    return dataset

def train_input_synthetic_fn(batch_num, num_steps=num_steps):
    #Get features    
    features = {}
    distribution = tf.distributions.Uniform(low=[-1.0], high=[1.0])
    features['uc_sample'] = tf.squeeze(distribution.sample(sample_shape=(batch_num*num_steps, 5, 5, 5)))
    features['vc_sample'] = tf.squeeze(distribution.sample(sample_shape=(batch_num*num_steps, 5, 5, 5)))
    features['wc_sample'] = tf.squeeze(distribution.sample(sample_shape=(batch_num*num_steps, 5, 5, 5)))
    features['pc_sample'] = tf.squeeze(distribution.sample(sample_shape=(batch_num*num_steps, 5, 5, 5)))
    
    #Get labels
    #linear
    # labels = tf.reduce_sum(features['uc_sample'], [1,2,3])/125 + \
    #         tf.reduce_sum(features['pc_sample'], [1,2,3])/125 + \
    #         tf.reduce_sum(features['vc_sample'], [1,2,3])/125 + \
    #         tf.reduce_sum(features['wc_sample'], [1,2,3])/125
    #constant
    labels = [0.0] * batch_num * num_steps

    #prepare the Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(batch_num*num_steps*10).batch(batch_num)
#    data = dataset.make_initializable_iterator()
#
#    #needed without eager execution
#    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, data.initializer)
#    features, labels = data.get_next()
#    features, labels = dataset.make_one_shot_iterator().get_next()
#    return features, labels
    return dataset

#Define evaluation function
def eval_input_fn(filenames,batch_num,label_name):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda line:_parse_function(line,label_name))
    dataset = dataset.batch(batch_num)
#    features, labels = dataset.make_one_shot_iterator().get_next()
#    return features, labels
    return dataset    

#Define function for splitting the training and test set
def split_train_test(files,test_ratio):
    np.random.seed(random_seed)
    shuffled_files = np.random.permutation(files)
    test_set_size = int(len(files) * test_ratio)
    test_files = shuffled_files[:test_set_size]
    train_files = shuffled_files[test_set_size:]
    return train_files,test_files


#Define model function for CNN estimator
def CNN_model_fn(features,labels,mode,params):
    '''CNN model with 1 convolutional layer'''

    #Define input layer
    print(features)
    #input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    input_layer = tf.stack([features['uc_sample'],features['vc_sample'],features['wc_sample'],features['pc_sample']],axis=4) #According to channel_last data format, otherwhise change axis parameter
    print(input_layer.shape)

    #Define layers
    conv1 = tf.layers.conv3d(input_layer, filters=params['n_conv1'], kernel_size=params['kernelsize_conv1'], \
            strides=params['stride_conv1'], activation=params['activation_function'], padding="valid", name='conv1', \
            kernel_initializer=params['kernel_initializer'], data_format = 'channels_last', reuse = tf.AUTO_REUSE) 
    # x = tf.layers.batch_normalization(conv1, training=True, name='block4_sepconv1_bn')
    print(conv1.shape)
    flatten = tf.layers.flatten(conv1, name='flatten')
    print(flatten.shape)
    output = tf.layers.dense(flatten, units=num_labels, name="outputs", \
            activation=None, kernel_initializer=params['kernel_initializer'], reuse = tf.AUTO_REUSE)
    print(output.shape)

    #Compute predictions 
    labels = tf.reshape(labels,[-1,num_labels])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'value':output,'label':labels})

    #Compute loss
    mse_tau_total = tf.losses.mean_squared_error(labels, output)
    loss = tf.reduce_mean(mse_tau_total)

    #Compute evaluation metrics.
    rmse_tau_total,update_op = tf.metrics.root_mean_squared_error(labels, output)
    metrics = {'rmse':(rmse_tau_total,update_op)}
    tf.summary.scalar('rmse',rmse_tau_total)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        #return tf.estimator.EstimatorSpec(mode, loss=loss)

    #Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


#Define filenames for training and testing
files = glob.glob(args.input_dir)
train_filenames, test_filenames = split_train_test(files,0.1) #Set aside 10% of files for testing

##Define feature columns
#feature_columns = [tf.feature_column.numeric_column(key  = 'uc_sample',shape = [5,5,5],dtype=tf.float64),tf.feature_column.numeric_column('vc_sample',shape = [5,5,5],dtype=tf.float64),tf.feature_column.numeric_column('wc_sample',shape = [5,5,5],dtype=tf.float64),tf.feature_column.numeric_column('pc_sample',shape = [5,5,5],dtype=tf.float64)]

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
my_checkpointing_config = tf.estimator.RunConfig(model_dir=checkpoint_dir,tf_random_seed=random_seed,save_summary_steps=100,save_checkpoints_steps=9999,session_config=config,keep_checkpoint_max=None,keep_checkpoint_every_n_hours=10000,log_step_count_steps=10,train_distribute=None) #Provide tf.contrib.distribute.DistributionStrategy instance to train_distribute parameter for distributed training

#Instantiate an Estimator with model defined by model_fn
hyperparams =  {
#'feature_columns':feature_columns,
'n_conv1':10,
'kernelsize_conv1':5,
'stride_conv1':1,
'activation_function':tf.nn.relu,
'kernel_initializer':tf.glorot_uniform_initializer(),
'learning_rate':0.0001
}

##Test whether tfrecords files are correctly read.
#dataset_samples = train_input_fn(train_filenames, batch_size, output_variable)

CNN = tf.estimator.Estimator(model_fn = CNN_model_fn,config=my_checkpointing_config, params = hyperparams, model_dir=checkpoint_dir)

if args.synthetic is None:
    #Train the CNN
    CNN.train(input_fn=lambda:train_input_fn(train_filenames,batch_size,output_variable),steps=num_steps)
    #Evaluate the CNN
    eval_results = CNN.evaluate(input_fn=lambda:eval_input_fn(test_filenames,batch_size,output_variable))
    print('\nTest set RMSE: {rmse:.10e}\n'.format(**eval_results))  
    print('Used real data')
#    predictions = CNN.predict(input_fn = lambda:eval_input_fn(test_filenames, batch_size, output_variable))
#    NOTE: CNN.predict appeared to be unsuitable to compare the predictions from the CNN to the true labels stored in the TFRecords files: the labels are discarded by the tf.estimator.Estimator in predict mode. The alternative is the 'hacky' solution implemented in the code below.
else:
    #Train the CNN
    CNN.train(input_fn=lambda:train_input_synthetic_fn(batch_size, num_steps), steps=num_steps)
    #Evaluate the CNN
    eval_results = CNN.evaluate(input_fn=lambda:train_input_synthetic_fn(batch_size, num_steps), steps=num_steps)
    print('\nTest set RMSE: {rmse:.10e}\n'.format(**eval_results))  
    print('Used synthetic data')

#'Hacky' solution to compare the predictions of the CNN to the true labels stored in the TFRecords files. Note that the input and model function are called manually rather than using the tf.estimator.Estimator syntax.
if args.benchmark is None:
   
    test_filenames = train_filenames #NOTE: this line only implemented to test the script. REMOVE it later on!!!
 
    #Loop over test files to prevent memory overflow issues
    if args.synthetic is None:
        test_filenames = ['dummy'] #Dummy value of length 1 to ensure loop is only done once for synthetic data
    
    create_file = True #Make sure netCDF file is initialized
 
    #Print used data
    if args.synthetic is None:
        print('Test filename:')
        print(test_filenames)
    else:
        print('Used synthetic data')
 
    create_file = True #Make sure netCDF file is initialized
 
    #Initialize variables for keeping track of iterations
    tot_sample_end = 0
    tot_sample_begin = tot_sample_end

    for test_filename in test_filenames:
 
        #Generate iterator to extra features and labels from input data
        if args.synthetic is None:
            iterator = eval_input_fn(test_filename, batch_size, output_variable).make_initializable_iterator() #All samples present in test_filenames are used for testing once (Note that no .repeat() method is included in eval_input_fn, which is in contrast to train_input_fn).
 
        else:
#           iterator = train_input_synthetic_fn(batch_size, 1000).make_one_shot_iterator() #1000 samples are generated and subsequently used for testing, NOTE: this line raises error message because the one_shot_iterator cannot capture statefull nodes contained in train_input_synthetic_fn.
            iterator = train_input_synthetic_fn(batch_size, 1000).make_initializable_iterator()
 
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

        with tf.Session(config=config) as sess:

            #Restore CNN_model within tf.Session()
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

                    for pred,lbl in zip(preds['value'],preds['label']):
                        #print('\nPrediction is "{:.10e}", expectation is "{:.10e}".'.format(pred[0],lbl)) #Index 0 needed to index single value in ndarray
                        preds_values += [pred[0]]
                        lbls_values += [lbl[0]]
                        residuals += [abs(pred[0]-lbl[0])]
                        pred_random = np.random.choice(preds['label'][:][0]) #Generate random prediction
                        preds_values_random += [pred_random]
                        residuals_random += [abs(pred_random-lbl[0])]
                        tot_sample_end +=1
                    
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
        
                    #Store variables
                    var_pred[tot_sample_begin:tot_sample_end] = preds_values[:]
                    var_pred_random[tot_sample_begin:tot_sample_end] = preds_values_random[:]
                    var_lbl[tot_sample_begin:tot_sample_end] = lbls_values[:]
                    var_res[tot_sample_begin:tot_sample_end] = residuals[:]
                    var_res_random[tot_sample_begin:tot_sample_end] = residuals_random[:]
                    tot_sample_begin = tot_sample_end #Make sure stored variables are not overwritten.

                except tf.errors.OutOfRangeError:
                    break #Break out of while-loop after one epoch. NOTE: for this part of the code it is important that the eval_input_fn and train_input_synthetic_fn do not implement the .repeat() method on the created tf.Dataset.
    
    predictions_file.close() #Close netCDF-file after loop over test files
#        results = pandas.DataFrame.from_records({\
#            'index':range(len(preds_values)),
#            'var_pred':preds_values, \
#            'var_pred_random':preds_values_random, \
#            'var_lbl':lbls_values, \
#            'var_res':residuals, \
#            'var_res_random':residuals_random \
#            })
#
#        import ipdb; ipdb.set_trace()
#        plt = sns.pairplot(results)
#        plt.savefig("diagonal_allresults.png")
#        plt = sns.pairplot(results[['var_pred','var_lbl']])
#        plt.savefig("diagonal_predlbl.png")
#        plt = sns.pairplot(results[['var_pred_random','var_lbl']])
#        plt.savefig("diagonal_randompredlbl.png")
#        plt = sns.tsplot(results[['var_pred_random','var_lbl']])
#        plt.savefig("diagonal_randompredlbl.png")
