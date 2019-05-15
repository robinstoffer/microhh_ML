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
parser.add_argument('--training_file', default=None, \
        help='NetCDF file that contains the total fluxes')
parser.add_argument('--stats_file', default=None, \
        help='NetCDF file that contains the means and stdevs of the labels')
parser.add_argument('--make_plots', dest='make_plots', default=None, \
        action='store_true', \
        help='Make plots at each height for the predictions of the CNN')
parser.add_argument('--reconstruct_fields', dest='reconstruct_fields', default=None, \
        action='store_true', \
        help='reconstruct the corresponding transport fields for the predictions of the CNN')
parser.add_argument('--undo_normalisation', dest='undo_normalisation', default=None, \
        action='store_true', \
        help='Undo the normalisation of the labels and the predictions of the CNN')
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
tstep_values    = np.array(a['tstep_samples'][:]).astype('int') #make sure time steps are stored as integers, not as floats

if args.undo_normalisation:
    #Fetch means and stdevs to undo normalisation labels and predictions
    stats = nc.Dataset(args.stats_file,'r')
    mean_xu = np.array(stats['mean_unres_tau_xu_sample'][:])
    mean_yu = np.array(stats['mean_unres_tau_yu_sample'][:])
    mean_zu = np.array(stats['mean_unres_tau_zu_sample'][:])
    mean_xv = np.array(stats['mean_unres_tau_xv_sample'][:])
    mean_yv = np.array(stats['mean_unres_tau_yv_sample'][:])
    mean_zv = np.array(stats['mean_unres_tau_zv_sample'][:])
    mean_xw = np.array(stats['mean_unres_tau_xw_sample'][:])
    mean_yw = np.array(stats['mean_unres_tau_yw_sample'][:])
    mean_zw = np.array(stats['mean_unres_tau_zw_sample'][:])
    #
    stdev_xu = np.array(stats['stdev_unres_tau_xu_sample'][:])
    stdev_yu = np.array(stats['stdev_unres_tau_yu_sample'][:])
    stdev_zu = np.array(stats['stdev_unres_tau_zu_sample'][:])
    stdev_xv = np.array(stats['stdev_unres_tau_xv_sample'][:])
    stdev_yv = np.array(stats['stdev_unres_tau_yv_sample'][:])
    stdev_zv = np.array(stats['stdev_unres_tau_zv_sample'][:])
    stdev_xw = np.array(stats['stdev_unres_tau_xw_sample'][:])
    stdev_yw = np.array(stats['stdev_unres_tau_yw_sample'][:])
    stdev_zw = np.array(stats['stdev_unres_tau_zw_sample'][:])
    
    #Average means over time steps used for training (steps 0 up to and including 80) since only these were used to normalize the data
    meant_xu = np.mean(mean_xu[0:81])
    meant_yu = np.mean(mean_yu[0:81])
    meant_zu = np.mean(mean_zu[0:81])
    meant_xv = np.mean(mean_xv[0:81])
    meant_yv = np.mean(mean_yv[0:81])
    meant_zv = np.mean(mean_zv[0:81])
    meant_xw = np.mean(mean_xw[0:81])
    meant_yw = np.mean(mean_yw[0:81])
    meant_zw = np.mean(mean_zw[0:81])
    #
    stdevt_xu = np.mean(stdev_xu[0:81])
    stdevt_yu = np.mean(stdev_yu[0:81])
    stdevt_zu = np.mean(stdev_zu[0:81])
    stdevt_xv = np.mean(stdev_xv[0:81])
    stdevt_yv = np.mean(stdev_yv[0:81])
    stdevt_zv = np.mean(stdev_zv[0:81])
    stdevt_xw = np.mean(stdev_xw[0:81])
    stdevt_yw = np.mean(stdev_yw[0:81])
    stdevt_zw = np.mean(stdev_zw[0:81])
    


    #Undo normalisation
    print('begin to undo normalisation')
    def undo_normalisation(lbls, means, stdevs, time_steps):
        lbls  = (lbls * stdevs) + means
        return lbls
    
    preds_values_xu = undo_normalisation(preds_values_xu, meant_xu, stdevt_xu, tstep_values) 
    lbls_values_xu  = undo_normalisation(lbls_values_xu , meant_xu, stdevt_xu, tstep_values)
    preds_values_yu = undo_normalisation(preds_values_yu, meant_yu, stdevt_yu, tstep_values)
    lbls_values_yu  = undo_normalisation(lbls_values_yu , meant_yu, stdevt_yu, tstep_values) 
    preds_values_zu = undo_normalisation(preds_values_zu, meant_zu, stdevt_zu, tstep_values)
    lbls_values_zu  = undo_normalisation(lbls_values_zu , meant_zu, stdevt_zu, tstep_values) 
    preds_values_xv = undo_normalisation(preds_values_xv, meant_xv, stdevt_xv, tstep_values)
    lbls_values_xv  = undo_normalisation(lbls_values_xv , meant_xv, stdevt_xv, tstep_values)
    preds_values_yv = undo_normalisation(preds_values_yv, meant_yv, stdevt_yv, tstep_values)
    lbls_values_yv  = undo_normalisation(lbls_values_yv , meant_yv, stdevt_yv, tstep_values)
    preds_values_zv = undo_normalisation(preds_values_zv, meant_zv, stdevt_zv, tstep_values)
    lbls_values_zv  = undo_normalisation(lbls_values_zv , meant_zv, stdevt_zv, tstep_values)
    preds_values_xw = undo_normalisation(preds_values_xw, meant_xw, stdevt_xw, tstep_values)
    lbls_values_xw  = undo_normalisation(lbls_values_xw , meant_xw, stdevt_xw, tstep_values)
    preds_values_yw = undo_normalisation(preds_values_yw, meant_yw, stdevt_yw, tstep_values)
    lbls_values_yw  = undo_normalisation(lbls_values_yw , meant_yw, stdevt_yw, tstep_values)
    preds_values_zw = undo_normalisation(preds_values_zw, meant_zw, stdevt_zw, tstep_values)
    lbls_values_zw  = undo_normalisation(lbls_values_zw , meant_zw, stdevt_zw, tstep_values)
    print('finished undoing normalisation')

    #Close netCDF-file
    stats.close()


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

        #Calculate correlation coefficient
        corrcoef = np.round(np.corrcoef(x_height, y_height)[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix 

        #Make plot for selected height
        plt.figure()
        plt.title("Corrcoef = " + str(corrcoef),fontsize = 20)
        plt.scatter(x_height, y_height, s=6, marker='o')
        #plt.gca().axis('Equal')
        #plt.xlim(min(lbls_values)*0.8,max(lbls_values)*1.2)
        #plt.ylim(min(lbls_values)*0.8,max(lbls_values)*1.2)
        #plt.xlim(-0.2,0.2)
        #plt.ylim(-0.2,0.2)
        plt.xlim(-40.0,40.0)
        plt.ylim(-40.0,40.0)
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
        preds1 = preds[tstep_indices]
        z1 = z[tstep_indices]
        y1 = y[tstep_indices]
        x1 = x[tstep_indices]
        k = 0
        for z_unique in zs_unique:
            z_indices = (z1 == z_unique)
            preds2 = preds1[z_indices]
            y2 = y1[z_indices]
            x2 = x1[z_indices]
            j = 0
            for y_unique in ys_unique:
                y_indices = (y2 == y_unique)
                preds3 = preds2[y_indices]
                x3 = x2[y_indices]
                i = 0
                for x_unique in xs_unique:
                    x_index = (x3 == x_unique)
                    preds_rec[t,k,j,i] = preds3[x_index]
                    i += 1
                j += 1
            k += 1
            #print('Finished height k = ' + str(k))
        t += 1
        #print('Finished time step t = ' + str(t))

    return preds_rec

#Reconstruct flow fields if specified to do so
if args.reconstruct_fields:

    #Create netCDF-file to store reconstructed fields
    b = nc.Dataset('reconstructed_fields.nc','w')

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

    #Create variables for storage labels
    var_unres_tau_xu_tot_lbls = b.createVariable("unres_tau_xu_tot_lbls","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_xv_tot_lbls = b.createVariable("unres_tau_xv_tot_lbls","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_xw_tot_lbls = b.createVariable("unres_tau_xw_tot_lbls","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_yu_tot_lbls = b.createVariable("unres_tau_yu_tot_lbls","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_yv_tot_lbls = b.createVariable("unres_tau_yv_tot_lbls","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_yw_tot_lbls = b.createVariable("unres_tau_yw_tot_lbls","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zu_tot_lbls = b.createVariable("unres_tau_zu_tot_lbls","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_zv_tot_lbls = b.createVariable("unres_tau_zv_tot_lbls","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zw_tot_lbls = b.createVariable("unres_tau_zw_tot_lbls","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))

    #Call function to recontruct fields of labels for all nine components
    print('start reconstructing labels')
    var_unres_tau_xu_tot_lbls[:,:,:,:] = reconstruct_field(lbls_values_xu, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('first component done')
    var_unres_tau_xv_tot_lbls[:,:,:,:] = reconstruct_field(lbls_values_xv, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('second component done')
    var_unres_tau_xw_tot_lbls[:,:,:,:] = reconstruct_field(lbls_values_xw, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('third component done')
    var_unres_tau_yu_tot_lbls[:,:,:,:] = reconstruct_field(lbls_values_yu, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fourth component done')
    var_unres_tau_yv_tot_lbls[:,:,:,:] = reconstruct_field(lbls_values_yv, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fifth component done')
    var_unres_tau_yw_tot_lbls[:,:,:,:] = reconstruct_field(lbls_values_yw, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('sixth component done')
    var_unres_tau_zu_tot_lbls[:,:,:,:] = reconstruct_field(lbls_values_zu, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('seventh component done')
    var_unres_tau_zv_tot_lbls[:,:,:,:] = reconstruct_field(lbls_values_zv, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('eighth component done')
    var_unres_tau_zw_tot_lbls[:,:,:,:] = reconstruct_field(lbls_values_zw, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('nineth component done')

    #Create variables for storage reconstructed fields of predictions
    var_unres_tau_xu_tot_CNN = b.createVariable("unres_tau_xu_tot_CNN","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_xv_tot_CNN = b.createVariable("unres_tau_xv_tot_CNN","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_xw_tot_CNN = b.createVariable("unres_tau_xw_tot_CNN","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_yu_tot_CNN = b.createVariable("unres_tau_yu_tot_CNN","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_yv_tot_CNN = b.createVariable("unres_tau_yv_tot_CNN","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_yw_tot_CNN = b.createVariable("unres_tau_yw_tot_CNN","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zu_tot_CNN = b.createVariable("unres_tau_zu_tot_CNN","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_zv_tot_CNN = b.createVariable("unres_tau_zv_tot_CNN","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zw_tot_CNN = b.createVariable("unres_tau_zw_tot_CNN","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))

    #Call function to recontruct fields of predictions for all nine components
    print('start reconstructing predictions')
    var_unres_tau_xu_tot_CNN[:,:,:,:] = reconstruct_field(preds_values_xu, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('first component done')
    var_unres_tau_xv_tot_CNN[:,:,:,:] = reconstruct_field(preds_values_xv, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('second component done')
    var_unres_tau_xw_tot_CNN[:,:,:,:] = reconstruct_field(preds_values_xw, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('third component done')
    var_unres_tau_yu_tot_CNN[:,:,:,:] = reconstruct_field(preds_values_yu, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fourth component done')
    var_unres_tau_yv_tot_CNN[:,:,:,:] = reconstruct_field(preds_values_yv, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fifth component done')
    var_unres_tau_yw_tot_CNN[:,:,:,:] = reconstruct_field(preds_values_yw, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('sixth component done')
    var_unres_tau_zu_tot_CNN[:,:,:,:] = reconstruct_field(preds_values_zu, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('seventh component done')
    var_unres_tau_zv_tot_CNN[:,:,:,:] = reconstruct_field(preds_values_zv, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('eighth component done')
    var_unres_tau_zw_tot_CNN[:,:,:,:] = reconstruct_field(preds_values_zw, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('nineth component done')
    
    #Create variables for storage fractions sub-grid fluxes
    #var_frac_unres_tau_xu = b.createVariable("frac_unres_tau_xu","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    #var_frac_unres_tau_xv = b.createVariable("frac_unres_tau_xv","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    #var_frac_unres_tau_xw = b.createVariable("frac_unres_tau_xw","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    #var_frac_unres_tau_yu = b.createVariable("frac_unres_tau_yu","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    #var_frac_unres_tau_yv = b.createVariable("frac_unres_tau_yv","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    #var_frac_unres_tau_yw = b.createVariable("frac_unres_tau_yw","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    #var_frac_unres_tau_zu = b.createVariable("frac_unres_tau_zu","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    #var_frac_unres_tau_zv = b.createVariable("frac_unres_tau_zv","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    #var_frac_unres_tau_zw = b.createVariable("frac_unres_tau_zw","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    #NOTE: uncomment lines below and comment lines above when calculating spatial averages
    var_frac_unres_tau_xu = b.createVariable("frac_unres_tau_xu","f8",("tstep_unique","zloc_unique"))
    var_frac_unres_tau_xv = b.createVariable("frac_unres_tau_xv","f8",("tstep_unique","zloc_unique"))
    var_frac_unres_tau_xw = b.createVariable("frac_unres_tau_xw","f8",("tstep_unique","zhloc_unique"))
    var_frac_unres_tau_yu = b.createVariable("frac_unres_tau_yu","f8",("tstep_unique","zloc_unique"))
    var_frac_unres_tau_yv = b.createVariable("frac_unres_tau_yv","f8",("tstep_unique","zloc_unique"))
    var_frac_unres_tau_yw = b.createVariable("frac_unres_tau_yw","f8",("tstep_unique","zhloc_unique"))
    var_frac_unres_tau_zu = b.createVariable("frac_unres_tau_zu","f8",("tstep_unique","zhloc_unique"))
    var_frac_unres_tau_zv = b.createVariable("frac_unres_tau_zv","f8",("tstep_unique","zhloc_unique"))
    var_frac_unres_tau_zw = b.createVariable("frac_unres_tau_zw","f8",("tstep_unique","zloc_unique"))


    #Fetch total and unresolved fluxes from training data to calculate contribution sub-grid flux
    c = nc.Dataset(args.training_file,'r')
    count_index = 0
    for t in tstep_unique:
        #NOTE: For some indices the last value is removed; labels did not contain the values at the downstream/top edges in contrast to training data.
        #NOTE: Some extreme outliers in the fractions occur when the total momentum transport reachec 0. To preven this from happening, the fractions are confined to the range -1 to 1.
        print('Time step: ' + str(t))
        #total_tau_xu_turb = np.array(c['total_tau_xu_turb'][t,:,:,:])
        #total_tau_xu_visc = np.array(c['total_tau_xu_visc'][t,:,:,:])
        #unres_tau_xu_tot  = np.array(c['unres_tau_xu_tot' ][t,:,:,:])
        #total_tau_xu_tot  = total_tau_xu_turb + total_tau_xu_visc
        #var_frac_unres_tau_xu[count_index,:,:,:] = np.maximum(np.minimum(1,unres_tau_xu_tot / np.mean(total_tau_xu_tot, axis=(1,2), keepdims=True)),-1)
        ##
        #total_tau_xv_turb = np.array(c['total_tau_xv_turb'][t,:,:-1,:-1])
        #total_tau_xv_visc = np.array(c['total_tau_xv_visc'][t,:,:-1,:-1])
        #unres_tau_xv_tot  = np.array(c['unres_tau_xv_tot' ][t,:,:-1,:-1])
        #total_tau_xv_tot  = total_tau_xv_turb + total_tau_xv_visc
        #var_frac_unres_tau_xv[count_index,:,:,:] = np.maximum(np.minimum(1,unres_tau_xv_tot / np.mean(total_tau_xv_tot, axis=(1,2), keepdims=True)),-1)
        ##
        #total_tau_xw_turb = np.array(c['total_tau_xw_turb'][t,:-1,:,:-1])
        #total_tau_xw_visc = np.array(c['total_tau_xw_visc'][t,:-1,:,:-1])
        #unres_tau_xw_tot  = np.array(c['unres_tau_xw_tot' ][t,:-1,:,:-1])
        #total_tau_xw_tot  = total_tau_xw_turb + total_tau_xw_visc
        #var_frac_unres_tau_xw[count_index,:,:,:] = np.maximum(np.minimum(1,unres_tau_xw_tot / np.mean(total_tau_xw_tot, axis=(1,2), keepdims=True)),-1)
        ##
        #total_tau_yu_turb = np.array(c['total_tau_yu_turb'][t,:,:-1,:-1])
        #total_tau_yu_visc = np.array(c['total_tau_yu_visc'][t,:,:-1,:-1])
        #unres_tau_yu_tot  = np.array(c['unres_tau_yu_tot' ][t,:,:-1,:-1])
        #total_tau_yu_tot  = total_tau_yu_turb + total_tau_yu_visc
        #var_frac_unres_tau_yu[count_index,:,:,:] = np.maximum(np.minimum(1,unres_tau_yu_tot / np.mean(total_tau_yu_tot, axis=(1,2), keepdims=True)),-1)
        ##
        #total_tau_yv_turb = np.array(c['total_tau_yv_turb'][t,:,:,:])
        #total_tau_yv_visc = np.array(c['total_tau_yv_visc'][t,:,:,:])
        #unres_tau_yv_tot  = np.array(c['unres_tau_yv_tot' ][t,:,:,:])
        #total_tau_yv_tot  = total_tau_yv_turb + total_tau_yv_visc
        #var_frac_unres_tau_yv[count_index,:,:,:] = np.maximum(np.minimum(1,unres_tau_yv_tot / np.mean(total_tau_yv_tot, axis=(1,2), keepdims=True)),-1)
        ##
        #total_tau_yw_turb = np.array(c['total_tau_yw_turb'][t,:-1,:-1,:])
        #total_tau_yw_visc = np.array(c['total_tau_yw_visc'][t,:-1,:-1,:])
        #unres_tau_yw_tot  = np.array(c['unres_tau_yw_tot' ][t,:-1,:-1,:])
        #total_tau_yw_tot  = total_tau_yw_turb + total_tau_yw_visc
        #var_frac_unres_tau_yw[count_index,:,:,:] = np.maximum(np.minimum(1,unres_tau_yw_tot / np.mean(total_tau_yw_tot, axis=(1,2), keepdims=True)),-1)
        ##
        #total_tau_zu_turb = np.array(c['total_tau_zu_turb'][t,:-1,:,:-1])
        #total_tau_zu_visc = np.array(c['total_tau_zu_visc'][t,:-1,:,:-1])
        #unres_tau_zu_tot  = np.array(c['unres_tau_zu_tot' ][t,:-1,:,:-1])
        #total_tau_zu_tot  = total_tau_zu_turb + total_tau_zu_visc
        #var_frac_unres_tau_zu[count_index,:,:,:] = np.maximum(np.minimum(1,unres_tau_zu_tot / np.mean(total_tau_zu_tot, axis=(1,2), keepdims=True)),-1)
        ##
        #total_tau_zv_turb = np.array(c['total_tau_zv_turb'][t,:-1,:-1,:])
        #total_tau_zv_visc = np.array(c['total_tau_zv_visc'][t,:-1,:-1,:])
        #unres_tau_zv_tot  = np.array(c['unres_tau_zv_tot' ][t,:-1,:-1,:])
        #total_tau_zv_tot  = total_tau_zv_turb + total_tau_zv_visc
        #var_frac_unres_tau_zv[count_index,:,:,:] = np.maximum(np.minimum(1,unres_tau_zv_tot / np.mean(total_tau_zv_tot, axis=(1,2), keepdims=True)),-1)
        ##
        #total_tau_zw_turb = np.array(c['total_tau_zw_turb'][t,:,:,:])
        #total_tau_zw_visc = np.array(c['total_tau_zw_visc'][t,:,:,:])
        #unres_tau_zw_tot  = np.array(c['unres_tau_zw_tot' ][t,:,:,:])
        #total_tau_zw_tot  = total_tau_zw_turb + total_tau_zw_visc
        #var_frac_unres_tau_zw[count_index,:,:,:] = np.maximum(np.minimum(1,unres_tau_zw_tot / np.mean(total_tau_zw_tot, axis=(1,2), keepdims=True)),-1)
        ##
        #count_index += 1
#################
        #res_tau_xu_turb = np.array(c['res_tau_xu_turb'][t,:,:,:])
        #res_tau_xu_visc = np.array(c['res_tau_xu_visc'][t,:,:,:])
        #unres_tau_xu_tot  = np.array(c['unres_tau_xu_tot' ][t,:,:,:])
        #res_tau_xu_tot  = res_tau_xu_turb + res_tau_xu_visc
        #var_frac_unres_tau_xu[count_index,:,:,:] = np.maximum(np.minimum(10,unres_tau_xu_tot / res_tau_xu_tot),-10)
        ##
        #res_tau_xv_turb = np.array(c['res_tau_xv_turb'][t,:,:-1,:-1])
        #res_tau_xv_visc = np.array(c['res_tau_xv_visc'][t,:,:-1,:-1])
        #unres_tau_xv_tot  = np.array(c['unres_tau_xv_tot' ][t,:,:-1,:-1])
        #res_tau_xv_tot  = res_tau_xv_turb + res_tau_xv_visc
        #var_frac_unres_tau_xv[count_index,:,:,:] = np.maximum(np.minimum(10,unres_tau_xv_tot / res_tau_xv_tot),-10)
        ##
        #res_tau_xw_turb = np.array(c['res_tau_xw_turb'][t,:-1,:,:-1])
        #res_tau_xw_visc = np.array(c['res_tau_xw_visc'][t,:-1,:,:-1])
        #unres_tau_xw_tot  = np.array(c['unres_tau_xw_tot' ][t,:-1,:,:-1])
        #res_tau_xw_tot  = res_tau_xw_turb + res_tau_xw_visc
        #var_frac_unres_tau_xw[count_index,:,:,:] = np.maximum(np.minimum(10,unres_tau_xw_tot / res_tau_xw_tot),-10)
        ##
        #res_tau_yu_turb = np.array(c['res_tau_yu_turb'][t,:,:-1,:-1])
        #res_tau_yu_visc = np.array(c['res_tau_yu_visc'][t,:,:-1,:-1])
        #unres_tau_yu_tot  = np.array(c['unres_tau_yu_tot' ][t,:,:-1,:-1])
        #res_tau_yu_tot  = res_tau_yu_turb + res_tau_yu_visc
        #var_frac_unres_tau_yu[count_index,:,:,:] = np.maximum(np.minimum(10,unres_tau_yu_tot / res_tau_yu_tot),-10)
        ##
        #res_tau_yv_turb = np.array(c['res_tau_yv_turb'][t,:,:,:])
        #res_tau_yv_visc = np.array(c['res_tau_yv_visc'][t,:,:,:])
        #unres_tau_yv_tot  = np.array(c['unres_tau_yv_tot' ][t,:,:,:])
        #res_tau_yv_tot  = res_tau_yv_turb + res_tau_yv_visc
        #var_frac_unres_tau_yv[count_index,:,:,:] = np.maximum(np.minimum(10,unres_tau_yv_tot / res_tau_yv_tot),-10)
        ##
        #res_tau_yw_turb = np.array(c['res_tau_yw_turb'][t,:-1,:-1,:])
        #res_tau_yw_visc = np.array(c['res_tau_yw_visc'][t,:-1,:-1,:])
        #unres_tau_yw_tot  = np.array(c['unres_tau_yw_tot' ][t,:-1,:-1,:])
        #res_tau_yw_tot  = res_tau_yw_turb + res_tau_yw_visc
        #var_frac_unres_tau_yw[count_index,:,:,:] = np.maximum(np.minimum(10,unres_tau_yw_tot / res_tau_yw_tot),-10)
        ##
        #res_tau_zu_turb = np.array(c['res_tau_zu_turb'][t,:-1,:,:-1])
        #res_tau_zu_visc = np.array(c['res_tau_zu_visc'][t,:-1,:,:-1])
        #unres_tau_zu_tot  = np.array(c['unres_tau_zu_tot' ][t,:-1,:,:-1])
        #res_tau_zu_tot  = res_tau_zu_turb + res_tau_zu_visc
        #var_frac_unres_tau_zu[count_index,:,:,:] = np.maximum(np.minimum(10,unres_tau_zu_tot / res_tau_zu_tot),-10)
        ##
        #res_tau_zv_turb = np.array(c['res_tau_zv_turb'][t,:-1,:-1,:])
        #res_tau_zv_visc = np.array(c['res_tau_zv_visc'][t,:-1,:-1,:])
        #unres_tau_zv_tot  = np.array(c['unres_tau_zv_tot' ][t,:-1,:-1,:])
        #res_tau_zv_tot  = res_tau_zv_turb + res_tau_zv_visc
        #var_frac_unres_tau_zv[count_index,:,:,:] = np.maximum(np.minimum(10,unres_tau_zv_tot / res_tau_zv_tot),-10)
        ##
        #res_tau_zw_turb = np.array(c['res_tau_zw_turb'][t,:,:,:])
        #res_tau_zw_visc = np.array(c['res_tau_zw_visc'][t,:,:,:])
        #unres_tau_zw_tot  = np.array(c['unres_tau_zw_tot' ][t,:,:,:])
        #res_tau_zw_tot  = res_tau_zw_turb + res_tau_zw_visc
        #var_frac_unres_tau_zw[count_index,:,:,:] = np.maximum(np.minimum(10,unres_tau_zw_tot / res_tau_zw_tot),-10)
        ##
        #count_index += 1
#################
        res_tau_xu_turb   = np.mean(np.array(c['res_tau_xu_turb'][t,:,:,:]), axis=(1,2), keepdims=False)
        res_tau_xu_visc   = np.mean(np.array(c['res_tau_xu_visc'][t,:,:,:]), axis=(1,2), keepdims=False)
        unres_tau_xu_tot  = np.mean(np.array(c['unres_tau_xu_tot' ][t,:,:,:]), axis=(1,2), keepdims=False)
        res_tau_xu_tot  = res_tau_xu_turb + res_tau_xu_visc
        print(res_tau_xu_tot.shape)
        print(unres_tau_xu_tot.shape)
        print(res_tau_xu_turb.shape)
        print(res_tau_xu_visc.shape)
        var_frac_unres_tau_xu[count_index,:] = np.maximum(np.minimum(10,unres_tau_xu_tot / res_tau_xu_tot),-10)
        #
        res_tau_xv_turb   = np.mean(np.array(c['res_tau_xv_turb'][t,:,:-1,:-1]), axis=(1,2), keepdims=False)
        res_tau_xv_visc   = np.mean(np.array(c['res_tau_xv_visc'][t,:,:-1,:-1]), axis=(1,2), keepdims=False)
        unres_tau_xv_tot  = np.mean(np.array(c['unres_tau_xv_tot' ][t,:,:-1,:-1]), axis=(1,2), keepdims=False)
        res_tau_xv_tot  = res_tau_xv_turb + res_tau_xv_visc
        var_frac_unres_tau_xv[count_index,:] = np.maximum(np.minimum(10,unres_tau_xv_tot / res_tau_xv_tot),-10)
        #
        res_tau_xw_turb   = np.mean(np.array(c['res_tau_xw_turb'][t,:-1,:,:-1]), axis=(1,2), keepdims=False)
        res_tau_xw_visc   = np.mean(np.array(c['res_tau_xw_visc'][t,:-1,:,:-1]), axis=(1,2), keepdims=False)
        unres_tau_xw_tot  = np.mean(np.array(c['unres_tau_xw_tot' ][t,:-1,:,:-1]), axis=(1,2), keepdims=False)
        res_tau_xw_tot  = res_tau_xw_turb + res_tau_xw_visc
        var_frac_unres_tau_xw[count_index,:] = np.maximum(np.minimum(10,unres_tau_xw_tot / res_tau_xw_tot),-10)
        #
        res_tau_yu_turb   = np.mean(np.array(c['res_tau_yu_turb'][t,:,:-1,:-1]), axis=(1,2), keepdims=False)
        res_tau_yu_visc   = np.mean(np.array(c['res_tau_yu_visc'][t,:,:-1,:-1]), axis=(1,2), keepdims=False)
        unres_tau_yu_tot  = np.mean(np.array(c['unres_tau_yu_tot' ][t,:,:-1,:-1]), axis=(1,2), keepdims=False)
        res_tau_yu_tot  = res_tau_yu_turb + res_tau_yu_visc
        var_frac_unres_tau_yu[count_index,:] = np.maximum(np.minimum(10,unres_tau_yu_tot / res_tau_yu_tot),-10)
        #
        res_tau_yv_turb   = np.mean(np.array(c['res_tau_yv_turb'][t,:,:,:]), axis=(1,2), keepdims=False)
        res_tau_yv_visc   = np.mean(np.array(c['res_tau_yv_visc'][t,:,:,:]), axis=(1,2), keepdims=False)
        unres_tau_yv_tot  = np.mean(np.array(c['unres_tau_yv_tot' ][t,:,:,:]), axis=(1,2), keepdims=False)
        res_tau_yv_tot  = res_tau_yv_turb + res_tau_yv_visc
        var_frac_unres_tau_yv[count_index,:] = np.maximum(np.minimum(10,unres_tau_yv_tot / res_tau_yv_tot),-10)
        #
        res_tau_yw_turb   = np.mean(np.array(c['res_tau_yw_turb'][t,:-1,:-1,:]), axis=(1,2), keepdims=False)
        res_tau_yw_visc   = np.mean(np.array(c['res_tau_yw_visc'][t,:-1,:-1,:]), axis=(1,2), keepdims=False)
        unres_tau_yw_tot  = np.mean(np.array(c['unres_tau_yw_tot' ][t,:-1,:-1,:]), axis=(1,2), keepdims=False)
        res_tau_yw_tot  = res_tau_yw_turb + res_tau_yw_visc
        var_frac_unres_tau_yw[count_index,:] = np.maximum(np.minimum(10,unres_tau_yw_tot / res_tau_yw_tot),-10)
        #
        res_tau_zu_turb   = np.mean(np.array(c['res_tau_zu_turb'][t,:-1,:,:-1]), axis=(1,2), keepdims=False)
        res_tau_zu_visc   = np.mean(np.array(c['res_tau_zu_visc'][t,:-1,:,:-1]), axis=(1,2), keepdims=False)
        unres_tau_zu_tot  = np.mean(np.array(c['unres_tau_zu_tot' ][t,:-1,:,:-1]), axis=(1,2), keepdims=False)
        res_tau_zu_tot  = res_tau_zu_turb + res_tau_zu_visc
        var_frac_unres_tau_zu[count_index,:] = np.maximum(np.minimum(10,unres_tau_zu_tot / res_tau_zu_tot),-10)
        #
        res_tau_zv_turb   = np.mean(np.array(c['res_tau_zv_turb'][t,:-1,:-1,:]), axis=(1,2), keepdims=False)
        res_tau_zv_visc   = np.mean(np.array(c['res_tau_zv_visc'][t,:-1,:-1,:]), axis=(1,2), keepdims=False)
        unres_tau_zv_tot  = np.mean(np.array(c['unres_tau_zv_tot' ][t,:-1,:-1,:]), axis=(1,2), keepdims=False)
        res_tau_zv_tot  = res_tau_zv_turb + res_tau_zv_visc
        var_frac_unres_tau_zv[count_index,:] = np.maximum(np.minimum(10,unres_tau_zv_tot / res_tau_zv_tot),-10)
        #
        res_tau_zw_turb   = np.mean(np.array(c['res_tau_zw_turb'][t,:,:,:]), axis=(1,2), keepdims=False)
        res_tau_zw_visc   = np.mean(np.array(c['res_tau_zw_visc'][t,:,:,:]), axis=(1,2), keepdims=False)
        unres_tau_zw_tot  = np.mean(np.array(c['unres_tau_zw_tot' ][t,:,:,:]), axis=(1,2), keepdims=False)
        res_tau_zw_tot  = res_tau_zw_turb + res_tau_zw_visc
        var_frac_unres_tau_zw[count_index,:] = np.maximum(np.minimum(10,unres_tau_zw_tot / res_tau_zw_tot),-10)
        #
        count_index += 1


    #Close netCDF-files
    b.close()
    c.close()

#Close netCDF-file
a.close()

