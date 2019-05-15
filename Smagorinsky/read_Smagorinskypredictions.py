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

###Fetch Smagorinsky fluxes, training data and heights. Next, calculate isotropic part subgrid-scale stress.###
a = nc.Dataset(args.smagorinsky_file,'r')
b = nc.Dataset(args.training_file,'r')
utau_ref = np.array(b['utau_ref'][:])

#Specify time steps NOTE: FOR TESTING PURPOSES, OTHERWHISE SHOULD BE 81 TO 90!
tstart = 81
tend   = 90

#Extract smagorinsky and training fluxes, and make Smagorinsky fluxes dimensionless
smag_tau_xu  = np.array(a['smag_tau_xu'][tstart:tend,:,:,:]) / (utau_ref ** 2)
smag_tau_yu  = np.array(a['smag_tau_yu'][tstart:tend,:,:,:]) / (utau_ref ** 2)
smag_tau_zu  = np.array(a['smag_tau_zu'][tstart:tend,:,:,:]) / (utau_ref ** 2)
smag_tau_xv  = np.array(a['smag_tau_xv'][tstart:tend,:,:,:]) / (utau_ref ** 2)
smag_tau_yv  = np.array(a['smag_tau_yv'][tstart:tend,:,:,:]) / (utau_ref ** 2)
smag_tau_zv  = np.array(a['smag_tau_zv'][tstart:tend,:,:,:]) / (utau_ref ** 2)
smag_tau_xw  = np.array(a['smag_tau_xw'][tstart:tend,:,:,:]) / (utau_ref ** 2)
smag_tau_yw  = np.array(a['smag_tau_yw'][tstart:tend,:,:,:]) / (utau_ref ** 2)
smag_tau_zw  = np.array(a['smag_tau_zw'][tstart:tend,:,:,:]) / (utau_ref ** 2)
#
unres_tau_xu = np.array(b['unres_tau_xu_tot'][tstart:tend,:,:,:])# * (utau_ref ** 2)
unres_tau_yu = np.array(b['unres_tau_yu_tot'][tstart:tend,:,:,:])# * (utau_ref ** 2)
unres_tau_zu = np.array(b['unres_tau_zu_tot'][tstart:tend,:,:,:])# * (utau_ref ** 2)
unres_tau_xv = np.array(b['unres_tau_xv_tot'][tstart:tend,:,:,:])# * (utau_ref ** 2)
unres_tau_yv = np.array(b['unres_tau_yv_tot'][tstart:tend,:,:,:])# * (utau_ref ** 2)
unres_tau_zv = np.array(b['unres_tau_zv_tot'][tstart:tend,:,:,:])# * (utau_ref ** 2)
unres_tau_xw = np.array(b['unres_tau_xw_tot'][tstart:tend,:,:,:])# * (utau_ref ** 2)
unres_tau_yw = np.array(b['unres_tau_yw_tot'][tstart:tend,:,:,:])# * (utau_ref ** 2)
unres_tau_zw = np.array(b['unres_tau_zw_tot'][tstart:tend,:,:,:])# * (utau_ref ** 2)

#Extract heights
zc  = np.array(b['zc'][:])
zhc = np.array(b['zhc'][:])

#Calculate trace part of subgrid-stress, and substract this from the diagonal components of the labels
trace = (unres_tau_xu + unres_tau_yv + unres_tau_zw) * (1./3.)
print(trace[:,:,5,6])
print(trace.shape)
unres_tau_xu = unres_tau_xu - trace
unres_tau_yv = unres_tau_yv - trace
unres_tau_zw = unres_tau_zw - trace

###Loop over heights for all components considering the time steps specified below, and make scatterplots of labels vs fluxes at each height for all specified time steps combined###
def make_scatterplot_heights(smag_tau_all, unres_tau_all, heights, component):
    #for k in range(len(heights)+1):
        k = len(heights) #Choose this to plot spatial averages
        if k == len(heights):
            smag_tau  = np.mean(smag_tau_all[:,:,:,:], axis=(2,3), keepdims=False)
            unres_tau = np.mean(unres_tau_all[:,:,:,:], axis=(2,3), keepdims=False)
        else:
            smag_tau  = smag_tau_all[:,k,:,:]
            unres_tau = unres_tau_all[:,k,:,:]
        smag_tau  = smag_tau.flatten()
        unres_tau = unres_tau.flatten()
        
        #Make scatterplots of Smagorinsky fluxes versus labels
        corrcoef = np.round(np.corrcoef(smag_tau, unres_tau)[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
        plt.figure()
        plt.scatter(smag_tau, unres_tau, s=6, marker='o', alpha=0.2) 
        #plt.xlim([-0.001, 0.001])
        #plt.ylim([-0.001, 0.001])
        #plt.xlim([-40.0, 40.0])
        #plt.ylim([-40.0, 40.0])
        plt.xlim([-0.25, 0.25])
        plt.ylim([-0.25, 0.25])
        axes = plt.gca()
        plt.plot(axes.get_xlim(),axes.get_ylim(),'b--')
        #plt.gca().set_aspect('equal',adjustable='box')
        plt.ylabel("Labels",fontsize = 20)
        plt.xlabel("Smagorinsky",fontsize = 20)
        plt.title("Corrcoef = " + str(corrcoef),fontsize = 20)
        plt.axhline(c='black')
        plt.axvline(c='black')
        plt.xticks(fontsize = 16, rotation = 90)
        plt.yticks(fontsize = 16, rotation = 0)
        if k == len(heights):
            plt.savefig("Scatter_" + str(component) + "_Smagorinsky_vs_labels_allheights.png")
        else:
            plt.savefig("Scatter_" + str(component) + "_Smagorinsky_vs_labels_height" + str(heights[k]) + ".png")
        plt.close()

#Call function nine times to make all plots
make_scatterplot_heights(smag_tau_xu, unres_tau_xu, zc, 'xu')
make_scatterplot_heights(smag_tau_yu, unres_tau_yu, zc, 'yu')
make_scatterplot_heights(smag_tau_zu, unres_tau_zu, zhc, 'zu')
make_scatterplot_heights(smag_tau_xv, unres_tau_xv, zc, 'xv')
make_scatterplot_heights(smag_tau_yv, unres_tau_yv, zc, 'yv')
make_scatterplot_heights(smag_tau_zv, unres_tau_zv, zhc, 'zv')
make_scatterplot_heights(smag_tau_xw, unres_tau_xw, zhc, 'xw')
make_scatterplot_heights(smag_tau_yw, unres_tau_yw, zhc, 'yw')
make_scatterplot_heights(smag_tau_zw, unres_tau_zw, zc, 'zw')

#Close files
a.close()
b.close()
