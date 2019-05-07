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
parser.add_argument('--make_plots', dest='make_plots', default=None, \
        action='store_true', \
        help='Make plots at each height for the predictions of the CNN')
parser.add_argument('--reconstruct_fields', dest='reconstruct_fields', default=None, \
        action='store_true', \
        help='reconstruct the corresponding transport fields for the predictions of the CNN')
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
yhloc_values    = np.array(a['yhloc_samples'][:])
yloc_values     = np.array(a['yloc_samples'][:])
xhloc_values    = np.array(a['xhloc_samples'][:])
xloc_values     = np.array(a['xloc_samples'][:])
tstep_values    = np.array(a['tstep_samples'][:])


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
def _make_scatterplot_heights(x, y, heights, heights_unique, component):
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
if args.make_plots:
    zloc_unique  = np.unique(zloc_values)
    zhloc_unique = np.unique(zhloc_values)
    _make_scatterplot_heights(preds_values_xu, lbls_values_xu, zloc_values, zloc_unique, 'xu')
    _make_scatterplot_heights(preds_values_yu, lbls_values_yu, zloc_values, zloc_unique, 'yu')
    _make_scatterplot_heights(preds_values_zu, lbls_values_zu, zhloc_values, zhloc_unique, 'zu')
    _make_scatterplot_heights(preds_values_xv, lbls_values_xv, zloc_values, zloc_unique, 'xv')
    _make_scatterplot_heights(preds_values_yv, lbls_values_yv, zloc_values, zloc_unique, 'yv')
    _make_scatterplot_heights(preds_values_zv, lbls_values_zv, zhloc_values, zhloc_unique, 'zv')
    _make_scatterplot_heights(preds_values_xw, lbls_values_xw, zhloc_values, zhloc_unique, 'xw')
    _make_scatterplot_heights(preds_values_yw, lbls_values_yw, zhloc_values, zhloc_unique, 'yw')
    _make_scatterplot_heights(preds_values_zw, lbls_values_zw, zloc_values, zloc_unique, 'zw')

###Reconstruct flow fields###
def reconstruct_field(preds, x, xs_unique, y, ys_unique, z, zs_unique, tstep, tsteps_unique):
    
    #Initialize empty array for storage
    preds_rec = np.empty((len(tsteps_unique), len(zs_unique), len(ys_unique), len(xs_unique)))

    #For each unique combination of x, y, tstep find the corresponding value and store it
    t = 0
    for tstep_unique in tsteps_unique:
        tstep_indices = (tstep == tstep_unique)
        k = 0

        for z_unique in zs_unique:
            z_indices = (z == z_unique)
            j = 0
            for y_unique in ys_unique:
                y_indices = (y == y_unique)
                i = 0
                for x_unique in xs_unique:
                    x_indices = (x == x_unique)
                    tot_index = np.all([tstep_indices, z_indices, y_indices, x_indices], axis=0)
                    preds_rec[t,k,j,i] = preds[tot_index]
                    i += 1
                j += 1
            k += 1
            print('Finished height k = ' + str(k))
        t += 1
        print('Finished time step t = ' + str(t))

    return preds_rec

#Reconstruct flow fields if specified to do so
if args.reconstruct_fields:

    #Create netCDF-file to store reconstructed fields
    b=nc.Dataset('reconstructed_fields.nc','w')

    #Extract unique coordinates and time steps
    zloc_unique  = np.unique(zloc_values)
    zhloc_unique = np.unique(zhloc_values)
    yloc_unique  = np.unique(yloc_values)
    yhloc_unique = np.unique(yhloc_values)
    xloc_unique  = np.unique(xloc_values)
    xhloc_unique = np.unique(xhloc_values)
    tstep_unique = np.unique(tstep_values)

    #Create dimensions for storage in nc-file
    dim_zloc_unique  = b.createDimension("zloc_unique", len(zloc_unique))
    dim_zhloc_unique = b.createDimension("zhloc_unique",len(zhloc_unique))
    dim_yloc_unique  = b.createDimension("yloc_unique", len(yloc_unique))
    dim_yhloc_unique = b.createDimension("yhloc_unique",len(yhloc_unique))
    dim_xloc_unique  = b.createDimension("xloc_unique", len(xloc_unique))
    dim_xhloc_unique = b.createDimension("xhloc_unique",len(xhloc_unique))
    dim_tstep_unique = b.createDimension("tstep_unique",len(tstep_unique))

    #Create variables for dimensions and store them
    var_zloc_unique  = b.createVariable("zloc_unique" ,"f8",("zloc_unique",))
    var_zhloc_unique = b.createVariable("zhloc_unique","f8",("zhloc_unique",))
    var_yloc_unique  = b.createVariable("yloc_unique" ,"f8",("yloc_unique",))
    var_yhloc_unique = b.createVariable("yhloc_unique","f8",("yhloc_unique",))
    var_xloc_unique  = b.createVariable("xloc_unique" ,"f8",("xloc_unique",))
    var_xhloc_unique = b.createVariable("xhloc_unique","f8",("xhloc_unique",))
    var_tstep_unique = b.createVariable("tstep_unique","f8",("tstep_unique",))

    var_zloc_unique[:]  = zloc_unique[:]
    var_zhloc_unique[:] = zhloc_unique[:]
    var_yloc_unique[:]  = yloc_unique[:]
    var_yhloc_unique[:] = yhloc_unique[:]
    var_xloc_unique[:]  = xloc_unique[:]
    var_xhloc_unique[:] = xhloc_unique[:]
    var_tstep_unique[:] = tstep_unique[:]

    #Create variables for storage reconstructed fields
    var_unres_tau_xu_tot = b.createVariable("unres_tau_xu_tot","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_xv_tot = b.createVariable("unres_tau_xv_tot","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_xw_tot = b.createVariable("unres_tau_xw_tot","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_yu_tot = b.createVariable("unres_tau_yu_tot","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_yv_tot = b.createVariable("unres_tau_yv_tot","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_yw_tot = b.createVariable("unres_tau_yw_tot","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zu_tot = b.createVariable("unres_tau_zu_tot","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_zv_tot = b.createVariable("unres_tau_zv_tot","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zw_tot = b.createVariable("unres_tau_zw_tot","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))

    #Call function to recontruct fields for all nine components
    var_unres_tau_xu_tot[:,:,:,:] = reconstruct_field(preds_values_xu, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('first component done')
    var_unres_tau_xv_tot[:,:,:,:] = reconstruct_field(preds_values_xv, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    #print('second component done')
    var_unres_tau_xw_tot[:,:,:,:] = reconstruct_field(preds_values_xw, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    #print('third component done')
    var_unres_tau_yu_tot[:,:,:,:] = reconstruct_field(preds_values_yu, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    #print('fourth component done')
    var_unres_tau_yv_tot[:,:,:,:] = reconstruct_field(preds_values_yv, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    #print('fifth component done')
    var_unres_tau_yw_tot[:,:,:,:] = reconstruct_field(preds_values_yw, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    #print('sixth component done')
    var_unres_tau_zu_tot[:,:,:,:] = reconstruct_field(preds_values_zu, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    #print('seventh component done')
    var_unres_tau_zv_tot[:,:,:,:] = reconstruct_field(preds_values_zv, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    #print('eighth component done')
    var_unres_tau_zw_tot[:,:,:,:] = reconstruct_field(preds_values_zw, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    #print('nineth component done')
    
    #Close netCDF-file
    b.close()

#Close netCDF-file
a.close()

