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
parser.add_argument('--prediction_file', default=None, \
        help='NetCDF file that contains the predictions')
args = parser.parse_args()

#Fetch predictions made by CNN
a=nc.Dataset(args.prediction_file,'r')

#Read variables
preds_values_xu = np.array(a['preds_values_tau_xu'][:])
lbls_values_xu  = np.array(a['lbls_values_tau_xu'][:])
preds_values_yu = np.array(a['preds_values_tau_yu'][:])
lbls_values_yu  = np.array(a['lbls_values_tau_yu'][:])
preds_values_zu = np.array(a['preds_values_tau_zu'][:])
lbls_values_zu  = np.array(a['lbls_values_tau_zu'][:])
preds_values_xv = np.array(a['preds_values_tau_xv'][:])
lbls_values_xv  = np.array(a['lbls_values_tau_xv'][:])
preds_values_yv = np.array(a['preds_values_tau_yv'][:])
lbls_values_yv  = np.array(a['lbls_values_tau_yv'][:])
preds_values_zv = np.array(a['preds_values_tau_zv'][:])
lbls_values_zv  = np.array(a['lbls_values_tau_zv'][:])
preds_values_xw = np.array(a['preds_values_tau_xw'][:])
lbls_values_xw  = np.array(a['lbls_values_tau_xw'][:])
preds_values_yw = np.array(a['preds_values_tau_yw'][:])
lbls_values_yw  = np.array(a['lbls_values_tau_yw'][:])
preds_values_zw = np.array(a['preds_values_tau_zw'][:])
lbls_values_zw  = np.array(a['lbls_values_tau_zw'][:])
zhloc_values    = np.array(a['zhloc_samples'][:])
zloc_values     = np.array(a['zloc_samples'][:])

####Make scatterplots of predictions versus labels###
#
##Define function for this purpose
#def _make_scatterplot(x, y, component):
#    plt.figure()
#    plt.scatter(x, y, s=6, marker='o')
#    plt.gca().axis('Equal')
#    #plt.xlim(min(lbls_values)*0.8,max(lbls_values)*1.2)
#    #plt.ylim(min(lbls_values)*0.8,max(lbls_values)*1.2)
#    #plt.xlim(-0.0010,0.0010)
#    #plt.ylim(-0.0010,0.0010)
#    plt.xlim(-15.0,15.0)
#    plt.ylim(-15.0,15.0)
#    plt.plot(plt.gca().get_xlim(), plt.gca().get_ylim(), 'b--')
#    plt.gca().set_aspect('equal',adjustable='box')
#    plt.ylabel("Labels " + component, fontsize = 20)
#    plt.xlabel("Predictions CNN " + component, fontsize = 20) 
#    plt.axhline(c='black')
#    plt.axvline(c='black')
#    plt.xticks(fontsize = 16, rotation = 90)
#    plt.yticks(fontsize = 16, rotation = 0)
#    plt.savefig("Scatter_tau_" + component + ".png")
#    plt.close()
#
##Call function to make plots
#_make_scatterplot(preds_values_xu, lbls_values_xu, 'xu')
#_make_scatterplot(preds_values_yu, lbls_values_xu, 'yu')
#_make_scatterplot(preds_values_zu, lbls_values_xu, 'zu')
#_make_scatterplot(preds_values_xv, lbls_values_xu, 'xv')
#_make_scatterplot(preds_values_yv, lbls_values_xu, 'yv')
#_make_scatterplot(preds_values_zv, lbls_values_xu, 'zv')
#_make_scatterplot(preds_values_xw, lbls_values_xu, 'xw')
#_make_scatterplot(preds_values_yw, lbls_values_xu, 'yw')
#_make_scatterplot(preds_values_zw, lbls_values_xu, 'zw')

###Make scatterplots of predictions versus labels for each height###

#Define function for this purpose
def _make_scatterplot_single_height(x, y, heights, heights_unique, component):
    for height in heights_unique: #Loop over all unique heights contained in samples
        indices = np.where(heights == height)
        x_height = x[indices]
        y_height = y[indices]

        #Make plot for selected height
        plt.figure()
        plt.scatter(x_height, y_height, s=6, marker='o')
        plt.gca().axis('Equal')
        #plt.xlim(min(lbls_values)*0.8,max(lbls_values)*1.2)
        #plt.ylim(min(lbls_values)*0.8,max(lbls_values)*1.2)
        #plt.xlim(-0.0010,0.0010)
        #plt.ylim(-0.0010,0.0010)
        plt.xlim(-15.0,15.0)
        plt.ylim(-15.0,15.0)
        plt.plot(plt.gca().get_xlim(), plt.gca().get_ylim(), 'b--')
        plt.gca().set_aspect('equal',adjustable='box')
        plt.ylabel("Labels " + component, fontsize = 20)
        plt.xlabel("Predictions CNN " + component, fontsize = 20) 
        plt.axhline(c='black')
        plt.axvline(c='black')
        plt.xticks(fontsize = 16, rotation = 90)
        plt.yticks(fontsize = 16, rotation = 0)
        plt.savefig("Scatter_tau_" + component + "_" + str(height) + ".png")
        plt.close()

#Call function to make plots
zloc_unique  = np.unique(zloc_values)
zhloc_unique = np.unique(zhloc_values)
_make_scatterplot_single_height(preds_values_xu, lbls_values_xu, zloc_values, zloc_unique, 'xu')
_make_scatterplot_single_height(preds_values_yu, lbls_values_yu, zloc_values, zloc_unique, 'yu')
_make_scatterplot_single_height(preds_values_zu, lbls_values_zu, zhloc_values, zhloc_unique, 'zu')
_make_scatterplot_single_height(preds_values_xv, lbls_values_xv, zloc_values, zloc_unique, 'xv')
_make_scatterplot_single_height(preds_values_yv, lbls_values_yv, zloc_values, zloc_unique, 'yv')
_make_scatterplot_single_height(preds_values_zv, lbls_values_zv, zhloc_values, zhloc_unique, 'zv')
_make_scatterplot_single_height(preds_values_xw, lbls_values_xw, zhloc_values, zhloc_unique, 'xw')
_make_scatterplot_single_height(preds_values_yw, lbls_values_yw, zhloc_values, zhloc_unique, 'yw')
_make_scatterplot_single_height(preds_values_zw, lbls_values_zw, zloc_values, zloc_unique, 'zw')

#plt.figure()
#plt.scatter(lbls_values,preds_values_random,s=6,marker='o')
#plt.gca().axis('Equal')
##plt.xlim(min(lbls_values)*0.8,max(lbls_values)*1.2)
##plt.ylim(min(lbls_values)*0.8,max(lbls_values)*1.2)
##plt.xlim(-0.0010,0.0010)
##plt.ylim(-0.0010,0.0010)
#plt.xlim(-15.0,15.0)
#plt.ylim(-15.0,15.0)
#plt.plot(plt.gca().get_xlim(),plt.gca().get_ylim(),'b--')
##plt.gca().set_aspect('equal',adjustable='box')
#plt.xlabel("Labels",fontsize = 20)
#plt.ylabel("Predictions random",fontsize = 20)
#plt.axhline(c='black')
#plt.axvline(c='black')
#plt.xticks(fontsize = 16, rotation = 90)
#plt.yticks(fontsize = 16, rotation = 0)
#plt.savefig("Scatter_randompredictions_vs_labels.png")
#plt.close()
#
#plt.figure()
#plt.hist(residuals,bins=20)
##plt.xlim(0,0.001)
#plt.xlim(0,15.0)
##plt.ylim(-0.01,0.01)
#plt.xlabel("Magnitude",fontsize = 20)
#plt.ylabel("Count",fontsize = 20)
#plt.xticks(fontsize = 16, rotation = 90)
#plt.yticks(fontsize = 16, rotation = 0)
#plt.savefig("Hist_CNNpredictions_vs_labels.png")
#plt.close()
#
#plt.figure()
#plt.hist(residuals_random,bins=20)
##plt.xlim(0,0.001)
#plt.xlim(0,15.0)
##plt.ylim(-0.01,0.01)
#plt.xlabel("Magnitude",fontsize = 20)
#plt.ylabel("Count",fontsize = 20)
#plt.xticks(fontsize = 16, rotation = 90)
#plt.yticks(fontsize = 16, rotation = 0)
#plt.savefig("Hist_randompredictions_vs_labels.png")
#plt.close()
#
#plt.figure()
#plt.hist(preds_values,bins=20)
##plt.xlim(-0.001,0.001)
#plt.xlim(-15.0,15.0)
##plt.ylim(-0.01,0.01)
#plt.xlabel("Magnitude",fontsize = 20)
#plt.ylabel("Count",fontsize = 20)
#plt.xticks(fontsize = 16, rotation = 90)
#plt.yticks(fontsize = 16, rotation = 0)
#plt.savefig("Hist_CNNpredictions.png")
#plt.close()
#
#plt.figure()
#plt.hist(lbls_values,bins=20)
##plt.xlim(-0.001,0.001)
#plt.xlim(-15.0,15.0)
##plt.ylim(-0.01,0.01)
#plt.xlabel("Magnitude",fontsize = 20)
#plt.ylabel("Count",fontsize = 20)
#plt.xticks(fontsize = 16, rotation = 90)
#plt.yticks(fontsize = 16, rotation = 0)
#plt.savefig("Hist_labels.png")
#plt.close()
#
#plt.figure()
#plt.hist(preds_values_random,bins=20)
##plt.xlim(-0.001,0.001)
#plt.xlim(-15.0,15.0)
##plt.ylim(-0.01,0.01)
#plt.xlabel("Magnitude",fontsize = 20)
#plt.ylabel("Count",fontsize = 20)
#plt.xticks(fontsize = 16, rotation = 90)
#plt.yticks(fontsize = 16, rotation = 0)
#plt.savefig("Hist_randompredictions.png")
#plt.close()
