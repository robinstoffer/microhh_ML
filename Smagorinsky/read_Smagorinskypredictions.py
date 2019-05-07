import numpy as np
import netCDF4 as nc
#import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
#from matplotlib import rcParams
mpl.rcParams.update({'figure.autolayout':True})
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='microhh_ML')
parser.add_argument('--smagorinsky_file', default=None, \
        help='NetCDF file that contains the calculated sub-grid scale transports according to the Smagorinsky-Lilly sub-grid model')

parser.add_argument('--training_file', default=None, \
        help='NetCDF file that contains the training data, which includes the actual unresolved transports.')

args = parser.parse_args()

#Fetch Smagorinsky fluxes and training data
a=nc.Dataset(args.smagorinsky_file,'r')
b=nc.Dataset(args.training_file,'r')


#Loop over heights
zc = np.array(b['zc'][:])
for k in range(len(zc)):
    t=1 #NOTE: FOR TESTING PURPOSES
    smag_tau_xu  = np.array(a['smag_tau_xu'][t,k,:,:])
    unres_tau_xu = np.array(b['unres_tau_xu_tot'][t,k,:,:])
    #ntzu, nzzu, nyzu, nxzu = smag_tau_xu[np.newaxis,:,:,:].shape 
    nzzu, nyzu, nxzu = smag_tau_xu[np.newaxis,:,:].shape #NOTE: FOR TESTING PURPOSES ONLY!!!
    smag_tau_xu  = smag_tau_xu.flatten()
    unres_tau_xu = unres_tau_xu.flatten()
    
    ##Read height of samples
    #zhc  = np.array(b['zhc'][2:-2]) #Discard bottom/top two vertical levels as this is also done in the samples for the CNN
    ##half_channel_width = b['half_channel_width'][:]
    #half_channel_width = 1 #NOTE: FOR TESTING PURPOSES ONLY!!!
    #dist_wall_zh = abs(zhc - half_channel_width)
    #
    ##Broadcast dist_wall to all three dimensions and subsequently flatten the resulting array
    #dist_wall_zh_zu = np.broadcast_to(dist_wall_zh[np.newaxis,:,np.newaxis,np.newaxis], (ntzu, nzzu, nyzu, nxzu))
    #dist_wall_zh_zu = dist_wall_zh_zu.flatten()
    
    ###QUICK AND DIRTY: STANDARDIZE OUTPUT FOR BETTER COMPARISION WITH SCATTERPLOTS CNN  
    #unres_tau_zu = (unres_tau_zu - np.mean(unres_tau_zu))/np.std(unres_tau_zu)
    
    ###QUICK AND DIRTY: MULTIPLY VARIABLES  WITH U*2 TO DENORMALIZE THEM
    utau_ref     = np.array(b['utau_ref'][:])
    unres_tau_xu = unres_tau_xu * (utau_ref)**2
    #smag_tau_xu  = smag_tau_xu  * (utau_ref)**2
    #smag_tau_xu  = smag_tau_xu / utau_ref

    #Make scatterplots of Smagorinsky fluxes versus labels
    corrcoef = np.round(np.corrcoef(unres_tau_xu, smag_tau_xu)[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    plt.figure()
    #plt.scatter(unres_tau_zu, smag_tau_zu, c=dist_wall_zh_zu, cmap='jet', s=6, marker='o', alpha=0.2)
    plt.scatter(unres_tau_xu, smag_tau_xu, s=6, marker='o', alpha=0.2) 
    axes = plt.gca()
    #axes.axis('Equal')
    #plt.xlim(min(lbls_values)*0.8,max(lbls_values)*1.2)
    #plt.ylim(min(lbls_values)*0.8,max(lbls_values)*1.2)
    axes.set_xlim(-0.0010,0.0010)
    axes.set_ylim(-0.000010,0.000010)
    #axes.set_ylim(-0.0010,0.0010)
    #axes.set_xlim([-25.0,25.0])
    #axes.set_ylim([-8.0,8.0])
    plt.plot(axes.get_xlim(),axes.get_ylim(),'b--')
    #plt.gca().set_aspect('equal',adjustable='box')
    plt.xlabel("Labels",fontsize = 20)
    plt.ylabel("Smagorinsky",fontsize = 20)
    plt.title("Corrcoef = " + str(corrcoef),fontsize = 20)
    plt.axhline(c='black')
    plt.axvline(c='black')
    plt.xticks(fontsize = 16, rotation = 90)
    plt.yticks(fontsize = 16, rotation = 0)
    #cbar = plt.colorbar()
    #cbar.ax.set_ylabel("Distance to center channel [-]", labelpad=15, fontsize = 20, rotation = 90)
    #cbar.solids.set(alpha=1.0)
    plt.savefig("Scatter_Smagorinsky_vs_labels_height" + str(zc[k]) + ".png")
    plt.close()

#Close files
a.close()
b.close()
