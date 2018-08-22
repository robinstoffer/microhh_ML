from __future__ import print_function
import numpy as np
import netCDF4 as nc
import tensorflow as tf
import horovod.tensorflow as hvd
import os
import glob
#import mpi4py.rc
#mpi4py.rc.initialize = False #Make sure mpi4py is not re-initialized
#from mpi4py import MPI
#from sklearn.preprocessing import StandardScaler


#tf.reset_default_graph()

#Initialize Horovod
hvd.init()

#Define settingsi
batch_size = 100
#batch_size = 10000 #Number of samples selected in each batch
num_steps = 10000 #Number of steps, i.e. number of batches times number of epochs
output_variable = 'unres_tau_xu_sample'
num_labels = 1

#Define parse function for tfrecord files, which gives for each component in the example_proto the output in format (dict(features),labels)
def _parse_function(example_proto,label_name):
   # keys_to_features = {
   #     'uc_sample':tf.FixedLenFeature([],tf.float32),
   #     'vc_sample':tf.FixedLenFeature([],tf.float32),
   #     'wc_sample':tf.FixedLenFeature([],tf.float32),
   #     'pc_sample':tf.FixedLenFeature([],tf.float32),
   #     label_name:tf.FixedLenFeature([],tf.float32),
   #     'x_sample_size':tf.FixedLenFeature([],tf.int64),
   #     'y_sample_size':tf.FixedLenFeature([],tf.int64),
   #     'z_sample_size':tf.FixedLenFeature([],tf.int64)
   # }
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

    #Reshape data from tfrecords
  #  print(parsed_features['uc_sample'])
    #parsed_features['uc_sample'] = tf.reshape(parsed_features['uc_sample'],tf.stack([parsed_sample_sizes['z_sample_size'],parsed_sample_sizes['y_sample_size'],parsed_sample_sizes['x_sample_size']]))
    #parsed_features['vc_sample'] = tf.reshape(parsed_features['vc_sample'],tf.stack([parsed_sample_sizes['z_sample_size'],parsed_sample_sizes['y_sample_size'],parsed_sample_sizes['x_sample_size']]))
    #parsed_features['wc_sample'] = tf.reshape(parsed_features['wc_sample'],tf.stack([parsed_sample_sizes['z_sample_size'],parsed_sample_sizes['y_sample_size'],parsed_sample_sizes['x_sample_size']]))
    #parsed_features['pc_sample'] = tf.reshape(parsed_features['pc_sample'],tf.stack([parsed_sample_sizes['z_sample_size'],parsed_sample_sizes['y_sample_size'],parsed_sample_sizes['x_sample_size']]))
   # parsed_features['uc_sample'] = tf.reshape(parsed_features['uc_sample'],tf.stack([parsed_features['z_sample_size'],parsed_features['y_sample_size'],parsed_features['x_sample_size']]))
   # parsed_features['vc_sample'] = tf.reshape(parsed_features['vc_sample'],tf.stack([parsed_features['z_sample_size'],parsed_features['y_sample_size'],parsed_features['x_sample_size']]))
   # parsed_features['wc_sample'] = tf.reshape(parsed_features['wc_sample'],tf.stack([parsed_features['z_sample_size'],parsed_features['y_sample_size'],parsed_features['x_sample_size']]))
   # parsed_features['pc_sample'] = tf.reshape(parsed_features['pc_sample'],tf.stack([parsed_features['z_sample_size'],parsed_features['y_sample_size'],parsed_features['x_sample_size']]))
    labels = parsed_features.pop(label_name)
    return parsed_features,labels

#Define training input function
def train_input_fn(filenames,batch_num,label_name):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda line:_parse_function(line,label_name))
    dataset = dataset.shuffle(batch_num*2).repeat().batch(batch_num)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features,labels

#Define evaluation function
def eval_input_fn(filenames,batch_num,label_name):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda line:_parse_function(line,label_name))
    dataset = dataset.batch(batch_num)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features,labels

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
    uc_sample = tf.reshape(features['uc_sample'],[-1,5,5,5])
    vc_sample = tf.reshape(features['vc_sample'],[-1,5,5,5])
    wc_sample = tf.reshape(features['wc_sample'],[-1,5,5,5])
    pc_sample = tf.reshape(features['pc_sample'],[-1,5,5,5])
    input_layer = tf.stack([uc_sample,vc_sample,wc_sample,pc_sample],axis=4) #According to channel_last data format, otherwhise change axis parameter
    #print(features['uc_sample'])
    #print(input_layer.shape)
    #input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    #print(input_layer.shape)

    #Define layers
    conv1 = tf.layers.conv3d(input_layer, filters=params['n_conv1'], kernel_size=params['kernelsize_conv1'],strides=params['stride_conv1'],
                             activation=params['activation_function'],padding="valid",name='conv1',kernel_initializer=params['kernel_initializer'],data_format = 'channels_last')

    flatten = tf.layers.flatten(conv1,name='flatten')
    output = tf.layers.dense(flatten, units=num_labels, name="outputs",
                              activation=None, kernel_initializer=params['kernel_initializer'])
    #Compute predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,predictions={'value':output})

    # Compute loss
    labels = tf.reshape(labels,[-1,num_labels])
    mse_tau_total = tf.losses.mean_squared_error(labels,output)
    loss = tf.reduce_mean(mse_tau_total)

    #Compute evaluation metrics.
    rmse_tau_total,update_op = tf.metrics.root_mean_squared_error(labels,output)
    metrics = {'rmse':(rmse_tau_total,update_op)}
    tf.summary.scalar('rmse',rmse_tau_total)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        #return tf.estimator.EstimatorSpec(mode, loss=loss)

    #Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(params['learning_rate']*hvd.size()) #Scale learning rate with workers
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.001* hvd.size())#Scale learning rate with workers
    optimizer = hvd.DistributedOptimizer(optimizer) #Add Horovod Distributed Optimizer
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


#Define filenames for training and testing
files = glob.glob('/home/robins/microhh/cases/moser600/simulation2/training_time_step*[0-9]_of*[0-9].tfrecords')
train_filenames, test_filenames = split_train_test(files,0.1) #Set aside 10% of files for testing

##Define feature columns
#feature_columns = [tf.feature_column.numeric_column('uc_sample',shape = [17523,5,5,5],dtype=tf.float64),tf.feature_column.numeric_column('vc_sample',shape = [17523,5,5,5],dtype=tf.float64),tf.feature_column.numeric_column('wc_sample',shape = [17523,5,5,5],dtype=tf.float64),tf.feature_column.numeric_column('pc_sample',shape = [17523,5,5,5],dtype=tf.float64)]

#Set configuration
config = tf.ConfigProto(log_device_placement=False)
#config.gpu_options.visible_device_list = str(hvd.local_rank())#divides the processes over the devices, local rank: within server. A server can have multiple devices (in this case 2). Not needed for CPUs, does the operating system (os). os-settings can be found below
config.intra_op_parallelism_threads = 11
config.inter_op_parallelism_threads = 1
os.environ['KMP_BLOCKTIME'] = str(1)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0' #Specific for most Intel CPUs, except KNL
os.environ['OMP_NUM_THREADS'] = str(11)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank()==0:
    checkpoint_dir = '/home/robins/microhh/cases/moser600/simulation2/CNN_checkpoints'
else: 
    checkpoint_dir = '/home/robins/microhh/cases/moser600/simulation2/temp_checkpoints'

#Create RunConfig object to save check_point in the model_dir according to the specified schedule, and to define the session config
my_checkpointing_config = tf.estimator.RunConfig(model_dir=checkpoint_dir,tf_random_seed=1234,save_summary_steps=10,save_checkpoints_steps=10,session_config=config,keep_checkpoint_max=None,keep_checkpoint_every_n_hours=10000,log_step_count_steps=10,train_distribute=None) #Provide tf.contrib.distribute.DistributionStrategy instance to train_distribute parameter for distributed training

#Instantiate an Estimator with model defined by model_fn
hyperparams =  {
#'feature_columns':feature_columns,
'n_conv1':10,
'kernelsize_conv1':5,
'stride_conv1':1,
'activation_function':tf.nn.relu,
'kernel_initializer':tf.glorot_uniform_initializer(),
'learning_rate':0.0001*hvd.size()}
CNN = tf.estimator.Estimator(model_fn = CNN_model_fn,config=my_checkpointing_config, params = hyperparams, model_dir=checkpoint_dir)

#Ensure consistent initialization over all threads
bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

#Train the CNN for unres_tau_xu_samples
CNN.train(input_fn=lambda:train_input_fn(train_filenames,batch_size,output_variable),steps=num_steps // hvd.size(),hooks=[bcast_hook])

#Make sure evaluation and prediction is only done on main thread    
if hvd.rank() == 0:
    #Evaluate the CNN for unres_tau_xu_samples
    eval_results = CNN.evaluate(input_fn=lambda:eval_input_fn(test_filenames,batch_size,output_variable),steps=1)
    print('\nTest set RMSE unres_tau_xu: {rmse:.10e}\n'.format(**eval_results))    

    #Show the predicted transports of the CNN for the test set with the corresponding labels
    features_samples,label_samples = eval_input_fn(test_filenames,batch_size,output_variable)
    predictions = CNN_model_fn(features_samples,label_samples,tf.estimator.ModeKeys.PREDICT,hyperparams).predictions
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state('/home/robins/microhh/cases/moser600/simulation2/CNN_checkpoints')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Train filenames:')
        print(train_filenames)
        print('Test filename:')
        print(test_filenames)
        tot_sample_end = 0
        create_file = True
        while True:
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

                ##Convert to numpy arrays
                #preds_values = np.array(preds_values)
                #preds_values_random = np.array(preds_values_random)
                #lbls_values = np.array(lbls_values)
                #residuals = np.array(residuals)
                #residuals_random = np.array(residuals_random)

                #Store variables
                var_pred[tot_sample_begin:tot_sample_end] = preds_values[:]
                var_pred_random[tot_sample_begin:tot_sample_end] = preds_values_random[:]
                var_lbl[tot_sample_begin:tot_sample_end] = lbls_values[:]
                var_res[tot_sample_begin:tot_sample_end] = residuals[:]
                var_res_random[tot_sample_begin:tot_sample_end] = residuals_random[:]
                predictions_file.close() 
    
            except tf.errors.OutOfRangeError:
                break
    
