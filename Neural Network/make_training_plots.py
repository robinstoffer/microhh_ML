#Read Tensorflow log file and make training plots
#Taken from: https://gist.github.com/tomrunia/1e1d383fb21841e8f144
#Adapted by: Robin Stoffer (robin.stoffer@wur.nl)
#Date of creation: 11-6-2019
import warnings
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
mpl.use('Agg') #Use 'Agg' backend, Tk backend not supported on cartesius
mpl.rcParams.update({'figure.autolayout':True})
import matplotlib.pyplot as plt

def plot_tensorflow_log(log_file_training, log_file_validation, save_path):
    ''' Function to load Tensorflow Event files and make plots to visualize the training.
    log_file_training:   string specifying location of Tensorflow Event file for training.
    log_file_validation: string specifying location of Tensorflow Event file for validation.
    save_path: string specifying path to store generated plots.'''

    # Loading too much data is slow...
    tf_size_guidance = {
            'COMPRESSED_HISTOGRAMS':1,
            'IMAGES':1,
            'AUDIO':1,
            'SCALARS':0, #0 means all values are loaded
            'HISTOGRAMS':1,
            'TENSORS':0,
            }

    event_acc_training   = EventAccumulator(log_file_training, tf_size_guidance)
    event_acc_training.Reload()
    event_acc_validation = EventAccumulator(log_file_validation, tf_size_guidance)
    event_acc_validation.Reload()

    # Show all tags in the log file
    print("Tags present in training file: ")
    print(event_acc_training.Tags())
    print("Tags present in validation file: ")
    print(event_acc_validation.Tags())

    training_loss     = event_acc_training.Scalars('log_loss')
    validation_loss   = event_acc_validation.Scalars('log_loss')
    validation_rmse   = event_acc_validation.Scalars('rmse')
    train_steps       = len(training_loss)
    validation_steps  = len(validation_loss)
    print("Training steps: ", train_steps)
    print("Validation steps: ",validation_steps)
    x_train           = np.arange(train_steps, dtype='f') #NOTE: specified as floats to prevent rounding off issues
    x_validation      = np.arange(validation_steps, dtype='f') 
    y_train           = np.arange(train_steps, dtype='f')
    y_validation_loss = np.arange(validation_steps, dtype='f')
    y_validation_rmse = np.arange(validation_steps, dtype='f')
    #
    try:
        for i in range(train_steps):
            x_train[i]    = training_loss[i][1]   # get training iteration
            y_train[i]    = training_loss[i][2]   # get value
    except IndexError as error:
       warnings.warn("The amount of training steps specified is larger than what is present in the log file. By default, the script will continue to execute while selecting only the steps that are available. Check whether this is intended behaviour.", RuntimeWarning)

    try:
        for i in range(validation_steps):
            x_validation[i]      = validation_loss[i][1] # get training iteration
            y_validation_loss[i] = validation_loss[i][2] # get value
            y_validation_rmse[i] = validation_rmse[i][2] # get value
    except IndexError as error:
       warnings.warn("The amount of validation steps specified is larger than what is present in the log file. By default, the script will continue to execute while selecting only the steps that are available. Check whether this is intended behaviour.", RuntimeWarning) 

    #Make plot for training/validation loss
    plt.figure()
    plt.plot(x_train*(10**(-5)), y_train, 'r-', label='Training',   linewidth=0.2)
    plt.plot(x_validation*(10**(-5)), y_validation_loss, 'b-', label='Validation', linewidth=2.)
    
    plt.xlabel(r'$\rm Training\ iterations\ *\ 10^5\ [-]$', fontsize=20)
    plt.ylabel(r'$\rm log(loss)\ [-]$', fontsize=20)
    plt.xticks(fontsize = 16, rotation = 0)
    plt.yticks(fontsize = 16, rotation = 0)
    leg = plt.legend(loc='upper right', fontsize=20, frameon=True)
    #Set linewidth in legend (different than in plot)
    for line in leg.get_lines():
        line.set_linewidth(2.)
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_edgecolor('k')
    plt.savefig(save_path + "/loss.png")

    #Make plot for RMSE
    plt.figure()
    plt.plot(x_validation*(10**(-5)), y_validation_rmse, 'b-', label='Validation', linewidth=2.)
    plt.xlabel(r'$\rm Training\ iterations\ *\ 10^5\ [-]$', fontsize=20)
    plt.ylabel(r'$RMSE [-]$', fontsize=20)
    plt.xticks(fontsize = 16, rotation = 0)
    plt.yticks(fontsize = 16, rotation = 0)
    leg = plt.legend(loc='upper right', fontsize=20, frameon=True)
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_edgecolor('k')
    plt.savefig(save_path + "/rmse.png")

    #Close figures
    plt.close()

if __name__ == '__main__':
    log_file_training  = "/home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP13/events.out.tfevents.1569488124.tcn91.bullx"
    log_file_validation = "/home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP13/eval_MLP1/events.out.tfevents.1569489322.tcn91.bullx"
    save_path = "/home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP13"
    plot_tensorflow_log(log_file_training, log_file_validation, save_path)
