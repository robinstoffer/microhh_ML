from __future__ import print_function
import numpy as np
import netCDF4 as nc
import tensorflow as tf
import os
import glob
import argparse
import matplotlib
matplotlib.use('agg')
import seaborn as sns
sns.set(style="ticks")
import pandas

# Instantiate the parser
parser = argparse.ArgumentParser(description='microhh_ML')
parser.add_argument('--checkpoint_dir', type=str, default='"/home/robins/microhh/cases/moser600/simulation2/CNN_checkpoints"',
                    help='Checkpoint directory (for rank 0)')
parser.add_argument('--input_dir', type=str, default='"/home/robins/microhh/cases/moser600/simulation2/training_time_step*[0-9]_of*[0-9].tfrecords"',
                    help='TFRECORDS input directory')
parser.add_argument('--synthetic', default=None, \
        action='store_true', \
        help='Input type, it can needs to be set to TFRecordDataset for the real data run')
parser.add_argument('--benchmark', dest='benchmark', default=None, \
        action='store_true', \
        help='fullrun includes testing and plotting, otherwise it ends after validation loss')
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
#batch_size = 10000 #Number of samples selected in each batch
num_steps = args.num_steps #Number of steps, i.e. number of batches times number of epochs
output_variable = 'unres_tau_xu_sample'
num_labels = 1

#Define parse function for tfrecord files, which gives for each component in the example_proto 
#the output in format (dict(features),labels)
def _parse_function(example_proto,label_name):
    keys_to_features = {
        'uc_sample':tf.VarLenFeature(tf.float32),
        'vc_sample':tf.VarLenFeature(tf.float32),
        'wc_sample':tf.VarLenFeature(tf.float32),
        'pc_sample':tf.VarLenFeature(tf.float32),
        label_name:tf.VarLenFeature(tf.float32),
        'x_sample_size':tf.FixedLenFeature([],tf.int64),
        'y_sample_size':tf.FixedLenFeature([],tf.int64),
        'z_sample_size':tf.FixedLenFeature([],tf.int64)
    }

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    
    #Convert sparse tensors to dense tensors
    parsed_features['uc_sample'] = tf.sparse_tensor_to_dense(parsed_features['uc_sample'],default_value=-9999)
    parsed_features['vc_sample'] = tf.sparse_tensor_to_dense(parsed_features['vc_sample'],default_value=-9999)
    parsed_features['wc_sample'] = tf.sparse_tensor_to_dense(parsed_features['wc_sample'],default_value=-9999)
    parsed_features['pc_sample'] = tf.sparse_tensor_to_dense(parsed_features['pc_sample'],default_value=-9999)
    parsed_features[label_name] = tf.sparse_tensor_to_dense(parsed_features[label_name],default_value=-9999)
    labels = parsed_features.pop(label_name)
    return parsed_features,labels


#Define training input function
def train_input_fn(filenames,batch_num,label_name,num_steps=num_steps):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda line:_parse_function(line,label_name))
    dataset = dataset.shuffle(batch_num).batch(batch_num)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features,labels


def train_input_synthetic_fn(batch_size=batch_size, num_steps=num_steps):
    #Get features    
    features = {}
    distribution = tf.distributions.Uniform(low=[-1.0], high=[1.0])
    features['uc_sample'] = distribution.sample(sample_shape=(batch_size*num_steps, 5, 5, 5))
    features['vc_sample'] = distribution.sample(sample_shape=(batch_size*num_steps, 5, 5, 5))
    features['wc_sample'] = distribution.sample(sample_shape=(batch_size*num_steps, 5, 5, 5))
    features['pc_sample'] = distribution.sample(sample_shape=(batch_size*num_steps, 5, 5, 5))
    
    #Get labels
    #liniar
    # labels = tf.reduce_sum(features['uc_sample'], [1,2,3])/125 + \
    #         tf.reduce_sum(features['pc_sample'], [1,2,3])/125 + \
    #         tf.reduce_sum(features['vc_sample'], [1,2,3])/125 + \
    #         tf.reduce_sum(features['wc_sample'], [1,2,3])/125
    #constant
    labels = [0.0] * batch_size * num_steps

    #prepare the Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(batch_size).batch(batch_size)
    data = dataset.make_initializable_iterator()

    #needed without eager execution
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, data.initializer)
    features, labels = data.get_next()
    return features, labels


#Define evaluation function
def eval_input_fn(filenames,batch_num,label_name,num_steps=num_steps):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda line:_parse_function(line,label_name))
    dataset = dataset.batch(batch_num)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels
    

#Define function for splitting the training and test set
def split_train_test(files,test_ratio):
    np.random.seed(1234)
    shuffled_files = np.random.permutation(files)
    test_set_size = int(len(files) * test_ratio)
    test_files = shuffled_files[:test_set_size]
    train_files = shuffled_files[test_set_size:]
    return train_files,test_files


#Define model function for CNN estimator
def CNN_model_fn(features,labels,mode,params):
    '''CNN model with 1 convolutional layer'''
    #Reshape input data
    uc_sample = tf.reshape(features['uc_sample'], [-1,5,5,5])
    vc_sample = tf.reshape(features['vc_sample'], [-1,5,5,5])
    wc_sample = tf.reshape(features['wc_sample'], [-1,5,5,5])
    pc_sample = tf.reshape(features['pc_sample'], [-1,5,5,5])
    input_layer = tf.stack([uc_sample,vc_sample,wc_sample,pc_sample],axis=4) #According to channel_last data format, otherwhise change axis parameter

    #Define layers
    conv1 = tf.layers.conv3d(input_layer, filters=params['n_conv1'], kernel_size=params['kernelsize_conv1'], \
            strides=params['stride_conv1'], activation=params['activation_function'], padding="valid", name='conv1', \
            kernel_initializer=params['kernel_initializer'], data_format = 'channels_last')
    # x = tf.layers.batch_normalization(conv1, training=True, name='block4_sepconv1_bn')
    flatten = tf.layers.flatten(conv1, name='flatten')
    output = tf.layers.dense(flatten, units=num_labels, name="outputs", \
            activation=None, kernel_initializer=params['kernel_initializer'])
    #Compute predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'value':output})
    # Compute loss
    labels = tf.reshape(labels,[-1,num_labels])
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
#feature_columns = [tf.feature_column.numeric_column('uc_sample',shape = [17523,5,5,5],dtype=tf.float64),tf.feature_column.numeric_column('vc_sample',shape = [17523,5,5,5],dtype=tf.float64),tf.feature_column.numeric_column('wc_sample',shape = [17523,5,5,5],dtype=tf.float64),tf.feature_column.numeric_column('pc_sample',shape = [17523,5,5,5],dtype=tf.float64)]

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
my_checkpointing_config = tf.estimator.RunConfig(model_dir=checkpoint_dir,tf_random_seed=1234,save_summary_steps=100,save_checkpoints_steps=9999,session_config=config,keep_checkpoint_max=None,keep_checkpoint_every_n_hours=10000,log_step_count_steps=10,train_distribute=None) #Provide tf.contrib.distribute.DistributionStrategy instance to train_distribute parameter for distributed training

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

CNN = tf.estimator.Estimator(model_fn = CNN_model_fn,config=my_checkpointing_config, params = hyperparams, model_dir=checkpoint_dir)

if args.synthetic is None:
    #Train the CNN for unres_tau_xu_samples
    CNN.train(input_fn=lambda:train_input_fn(train_filenames,batch_size,output_variable,hyperparams),steps=num_steps)
    #Evaluate the CNN for unres_tau_xu_samples
    eval_results = CNN.evaluate(input_fn=lambda:eval_input_fn(test_filenames,batch_size,output_variable,hyperparams),steps=1)
    print('\nTest set RMSE unres_tau_xu: {rmse:.10e}\n'.format(**eval_results))  
    print('Used real data')

else:
    #Train the CNN for unres_tau_xu_samples
    CNN.train(input_fn=lambda:train_input_synthetic_fn(batch_size, num_steps), steps=num_steps)
    #Evaluate the CNN for unres_tau_xu_samples
    eval_results = CNN.evaluate(input_fn=lambda:train_input_synthetic_fn(batch_size, num_steps), steps=1)
    print('\nTest set RMSE unres_tau_xu: {rmse:.10e}\n'.format(**eval_results))  
    print('Used synthetic data')

if args.benchmark is None:
    #Show the predicted transports of the CNN for the test set with the corresponding labels
    if args.synthetic is None:
        features_samples, label_samples = eval_input_fn(test_filenames, 1000, output_variable, hyperparams)
    else:
        features_samples, label_samples = train_input_synthetic_fn(batch_size, 1000)

    predictions = CNN_model_fn(features_samples, label_samples, \
                    tf.estimator.ModeKeys.PREDICT, hyperparams).predictions

    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        if args.synthetic is None:
            print('Train filenames:')
            print(train_filenames)
            print('Test filename:')
            print(test_filenames)
        else:
            print('Used synthetic data')

        tot_sample_end = 0
        create_file = True
        sample=0
        while True:
            sample += 1
            np.random.seed(1234)
            try:
                tot_sample_begin = tot_sample_end
                preds_values = []
                preds_values_random = []
                lbls_values = []
                residuals = []
                residuals_random = []
                lbls = sess.run(label_samples)
                preds = sess.run(predictions)
                for pred,lbl in zip(preds['value'],lbls):
                    print('\nPrediction is "{:.10e}", expectation is "{:.10e}".'.format(pred[0],lbl[0])) #Index 0 needed to index single value in ndarray
                    preds_values += [pred[0]]
                    lbls_values += [lbl[0]]
                    residuals += [abs(pred[0]-lbl[0])]
                    pred_random = np.random.choice(lbls[:,0]) #Generate random prediction
                    preds_values_random += [pred_random]
                    residuals_random += [abs(pred_random-lbl[0])]
                    tot_sample_end +=1
                
                #Create/openb netCDF-file
                if create_file:
                    predictions_file = nc.Dataset('CNN_predictions.nc', 'w')
                    dim_ns = predictions_file.createDimension("ns",None)

                    #Create variables for storage
                    var_pred = predictions_file.createVariable("preds_values","f8",("ns",))
                    var_pred_random = predictions_file.createVariable("preds_values_random","f8",("ns",))
                    var_lbl = predictions_file.createVariable("lbls_values","f8",("ns",))
                    var_res = predictions_file.createVariable("residuals","f8",("ns",))
                    var_res_random = predictions_file.createVariable("residuals_random","f8",("ns",))

                    create_file=False #Make sure file is only created once

                else:
                    predictions_file = nc.Dataset('CNN_predictions.nc', 'r+')

                #Store variables
                var_pred[tot_sample_begin:tot_sample_end] = preds_values[:]
                var_pred_random[tot_sample_begin:tot_sample_end] = preds_values_random[:]
                var_lbl[tot_sample_begin:tot_sample_end] = lbls_values[:]
                var_res[tot_sample_begin:tot_sample_end] = residuals[:]
                var_res_random[tot_sample_begin:tot_sample_end] = residuals_random[:]
                predictions_file.close()
                results = pandas.DataFrame.from_records({\
                    'index':range(len(preds_values)),
                    'var_pred':preds_values, \
                    'var_pred_random':preds_values_random, \
                    'var_lbl':lbls_values, \
                    'var_res':residuals, \
                    'vr_res_random':residuals_random \
                    })

                import ipdb; ipdb.set_trace()
                plt = sns.pairplot(results)
                plt.savefig("diagonal_allresults_{}.png".format(sample))
                plt = sns.pairplot(results[['var_pred','var_lbl']])
                plt.savefig("diagonal_predlbl_{}.png".format(sample))
                plt = sns.pairplot(results[['var_pred_random','var_lbl']])
                plt.savefig("diagonal_randompredlbl_{}.png".format(sample))
                plt = sns.tsplot(results[['var_pred_random','var_lbl']])
                plt.savefig("diagonal_randompredlbl_{}.png".format(sample))
            except tf.errors.OutOfRangeError:
                break