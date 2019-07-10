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
parser.add_argument('--smagorinsky_file', default=None, \
        help='NetCDF file that contains the calculated sub-grid scale transports according to the Smagorinsky-Lilly sub-grid model')
parser.add_argument('--training_file', default=None, \
        help='NetCDF file that contains the training data, including the actual unresolved transports.')
parser.add_argument('--make_plots', dest='make_plots', default=None, \
        action='store_true', \
        help='Make plots at each height for the predictions of the CNN, Smagorinsky, and the training data')
parser.add_argument('--reconstruct_fields', dest='reconstruct_fields', default=None, \
        action='store_true', \
        help="Reconstruct the corresponding transport fields for the predictions of the CNN, which includes denormalisation. If not specified, a netCDF file called 'reconstructed_fields.nc' should be present in the current directory.")
parser.add_argument('--stats_file', default=None, \
        help='NetCDF file that contains the means and stdevs of the labels. Only needs to be specified when undo_normalisation is switched on.')
args = parser.parse_args()

###Fetch Smagorinsky fluxes, training fluxes, CNN predictions, and heights. Next, calculate isotropic part subgrid-scale stress and subtract it.###
a = nc.Dataset(args.prediction_file,'r')
b = nc.Dataset(args.smagorinsky_file,'r')
c = nc.Dataset(args.training_file,'r')

#Define reference mid-channel height and representative friction velocity
#NOTE: this is done to rescale the transports and grid domain to realistic values
delta_height = 1250 # in [m]
#delta_height = 1 #NOTE: uncomment when the height of the channel flow should be used.
utau_ref_channel = np.array(c['utau_ref'][:]) #NOTE: used friction velocity in [m/s] for the channel flow, needed for rescaling below.
utau_ref = 0.2 #NOTE: representative friction velocity in [m/s] for a realistic atmospheric flow, needed for rescaling below. 

#Specify time steps NOTE: SHOULD BE 27 TO 30 for validation, and all time steps ahead should be the used training steps. The CNN predictions should all originate from these time steps as well!
tstart = 27
tend   = 30

#Extract smagorinsky fluxes, training fluxes (including resolved and total fluxes), CNN fluxes.
#NOTE1:rescale Smagorinsky and training fluxes with a representative friction velocity.
#NOTE2:for some Smagorinsky and training fluxes the downstream/top cells are removed to make the dimensions consistent with the labels and predictions.
smag_tau_xu  = np.array(b['smag_tau_xu'][tstart:tend,:,:,:])     * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_yu  = np.array(b['smag_tau_yu'][tstart:tend,:,:-1,:-1]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_zu  = np.array(b['smag_tau_zu'][tstart:tend,:-1,:,:-1]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_xv  = np.array(b['smag_tau_xv'][tstart:tend,:,:-1,:-1]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_yv  = np.array(b['smag_tau_yv'][tstart:tend,:,:,:])     * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_zv  = np.array(b['smag_tau_zv'][tstart:tend,:-1,:-1,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_xw  = np.array(b['smag_tau_xw'][tstart:tend,:-1,:,:-1]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_yw  = np.array(b['smag_tau_yw'][tstart:tend,:-1,:-1,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_zw  = np.array(b['smag_tau_zw'][tstart:tend,:,:,:])     * ((utau_ref ** 2) / (utau_ref_channel ** 2))
#
unres_tau_xu = np.array(c['unres_tau_xu_turb'] [tstart:tend,:,:,:])     * ((utau_ref ** 2) / (utau_ref_channel ** 2))
unres_tau_yu = np.array(c['unres_tau_yu_turb'] [tstart:tend,:,:-1,:-1]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
unres_tau_zu = np.array(c['unres_tau_zu_turb'] [tstart:tend,:-1,:,:-1]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
unres_tau_xv = np.array(c['unres_tau_xv_turb'] [tstart:tend,:,:-1,:-1]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
unres_tau_yv = np.array(c['unres_tau_yv_turb'] [tstart:tend,:,:,:])     * ((utau_ref ** 2) / (utau_ref_channel ** 2))
unres_tau_zv = np.array(c['unres_tau_zv_turb'] [tstart:tend,:-1,:-1,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
unres_tau_xw = np.array(c['unres_tau_xw_turb'] [tstart:tend,:-1,:,:-1]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
unres_tau_yw = np.array(c['unres_tau_yw_turb'] [tstart:tend,:-1,:-1,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
unres_tau_zw = np.array(c['unres_tau_zw_turb'] [tstart:tend,:,:,:])     * ((utau_ref ** 2) / (utau_ref_channel ** 2))
res_tau_xu   = np.array(c['res_tau_xu_turb']  [tstart:tend,:,:,:])      * ((utau_ref ** 2) / (utau_ref_channel ** 2))
res_tau_yu   = np.array(c['res_tau_yu_turb']  [tstart:tend,:,:-1,:-1])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))   
res_tau_zu   = np.array(c['res_tau_zu_turb']  [tstart:tend,:-1,:,:-1])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))   
res_tau_xv   = np.array(c['res_tau_xv_turb']  [tstart:tend,:,:-1,:-1])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))   
res_tau_yv   = np.array(c['res_tau_yv_turb']  [tstart:tend,:,:,:])      * ((utau_ref ** 2) / (utau_ref_channel ** 2))   
res_tau_zv   = np.array(c['res_tau_zv_turb']  [tstart:tend,:-1,:-1,:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))   
res_tau_xw   = np.array(c['res_tau_xw_turb']  [tstart:tend,:-1,:,:-1])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))   
res_tau_yw   = np.array(c['res_tau_yw_turb']  [tstart:tend,:-1,:-1,:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))   
res_tau_zw   = np.array(c['res_tau_zw_turb']  [tstart:tend,:,:,:])      * ((utau_ref ** 2) / (utau_ref_channel ** 2))   
tot_tau_xu   = np.array(c['total_tau_xu_turb'][tstart:tend,:,:,:])      *  ((utau_ref ** 2) / (utau_ref_channel ** 2))   
tot_tau_yu   = np.array(c['total_tau_yu_turb'][tstart:tend,:,:-1,:-1])  *  ((utau_ref ** 2) / (utau_ref_channel ** 2))
tot_tau_zu   = np.array(c['total_tau_zu_turb'][tstart:tend,:-1,:,:-1])  *  ((utau_ref ** 2) / (utau_ref_channel ** 2))
tot_tau_xv   = np.array(c['total_tau_xv_turb'][tstart:tend,:,:-1,:-1])  *  ((utau_ref ** 2) / (utau_ref_channel ** 2))
tot_tau_yv   = np.array(c['total_tau_yv_turb'][tstart:tend,:,:,:])      *  ((utau_ref ** 2) / (utau_ref_channel ** 2))
tot_tau_zv   = np.array(c['total_tau_zv_turb'][tstart:tend,:-1,:-1,:])  *  ((utau_ref ** 2) / (utau_ref_channel ** 2))
tot_tau_xw   = np.array(c['total_tau_xw_turb'][tstart:tend,:-1,:,:-1])  *  ((utau_ref ** 2) / (utau_ref_channel ** 2))
tot_tau_yw   = np.array(c['total_tau_yw_turb'][tstart:tend,:-1,:-1,:])  *  ((utau_ref ** 2) / (utau_ref_channel ** 2))
tot_tau_zw   = np.array(c['total_tau_zw_turb'][tstart:tend,:,:,:])      *  ((utau_ref ** 2) / (utau_ref_channel ** 2))
#
if args.reconstruct_fields:
    preds_values_xu = np.array(a['preds_values_tau_xu'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_xu  = np.array(a['lbls_values_tau_xu'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_yu = np.array(a['preds_values_tau_yu'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_yu  = np.array(a['lbls_values_tau_yu'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_zu = np.array(a['preds_values_tau_zu'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_zu  = np.array(a['lbls_values_tau_zu'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_xv = np.array(a['preds_values_tau_xv'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_xv  = np.array(a['lbls_values_tau_xv'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_yv = np.array(a['preds_values_tau_yv'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_yv  = np.array(a['lbls_values_tau_yv'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_zv = np.array(a['preds_values_tau_zv'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_zv  = np.array(a['lbls_values_tau_zv'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_xw = np.array(a['preds_values_tau_xw'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_xw  = np.array(a['lbls_values_tau_xw'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_yw = np.array(a['preds_values_tau_yw'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_yw  = np.array(a['lbls_values_tau_yw'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_zw = np.array(a['preds_values_tau_zw'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_zw  = np.array(a['lbls_values_tau_zw'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    zhloc_values    = np.array(a['zhloc_samples'][:])
    zloc_values     = np.array(a['zloc_samples'][:])
    yhloc_values    = np.array(a['yhloc_samples'][:])
    yloc_values     = np.array(a['yloc_samples'][:])
    xhloc_values    = np.array(a['xhloc_samples'][:])
    xloc_values     = np.array(a['xloc_samples'][:])
    tstep_values    = np.array(a['tstep_samples'][:]).astype('int') #make sure time steps are stored as integers, not as floats

#Extract heights
zc  = np.array(c['zc'][:])
zhc = np.array(c['zhc'][:-1])

##Calculate trace part of subgrid-stress, and substract this from the diagonal components (except for Smagorinsky for which it is not included)
#trace_train = (unres_tau_xu + unres_tau_yv + unres_tau_zw) * (1./3.)
#print(trace_train[:,:,5,6])
#print(trace_train.shape)
#unres_tau_xu = unres_tau_xu - trace_train
#unres_tau_yv = unres_tau_yv - trace_train
#unres_tau_zw = unres_tau_zw - trace_train
##
#if args.reconstruct_fields:
#    trace_CNN = (preds_values_xu + preds_values_yv + preds_values_zw) * (1./3.)
#    preds_values_xu = preds_values_xu - trace_CNN
#    preds_values_yv = preds_values_yv - trace_CNN
#    preds_values_zw = preds_values_zw - trace_CNN

#Calculate trace part of subgrid-stress, and subtract this from labels for fair comparison with Smagorinsky fluxes
trace_train = (unres_tau_xu + unres_tau_yv + unres_tau_zw) * (1./3.)
print(trace_train[:,:,5,6])
print(trace_train.shape)
unres_tau_xu_traceless = unres_tau_xu - trace_train
unres_tau_yv_traceless = unres_tau_yv - trace_train
unres_tau_zw_traceless = unres_tau_zw - trace_train

#Close files
a.close()
b.close()
c.close()


#if args.reconstruct_fields:
#    
#    #Fetch means and stdevs to undo normalisation labels and predictions
#    stats = nc.Dataset(args.stats_file,'r')
#    mean_xu = np.array(stats['mean_unres_tau_xu_sample'][:])
#    mean_yu = np.array(stats['mean_unres_tau_yu_sample'][:])
#    mean_zu = np.array(stats['mean_unres_tau_zu_sample'][:])
#    mean_xv = np.array(stats['mean_unres_tau_xv_sample'][:])
#    mean_yv = np.array(stats['mean_unres_tau_yv_sample'][:])
#    mean_zv = np.array(stats['mean_unres_tau_zv_sample'][:])
#    mean_xw = np.array(stats['mean_unres_tau_xw_sample'][:])
#    mean_yw = np.array(stats['mean_unres_tau_yw_sample'][:])
#    mean_zw = np.array(stats['mean_unres_tau_zw_sample'][:])
#    #
#    stdev_xu = np.array(stats['stdev_unres_tau_xu_sample'][:])
#    stdev_yu = np.array(stats['stdev_unres_tau_yu_sample'][:])
#    stdev_zu = np.array(stats['stdev_unres_tau_zu_sample'][:])
#    stdev_xv = np.array(stats['stdev_unres_tau_xv_sample'][:])
#    stdev_yv = np.array(stats['stdev_unres_tau_yv_sample'][:])
#    stdev_zv = np.array(stats['stdev_unres_tau_zv_sample'][:])
#    stdev_xw = np.array(stats['stdev_unres_tau_xw_sample'][:])
#    stdev_yw = np.array(stats['stdev_unres_tau_yw_sample'][:])
#    stdev_zw = np.array(stats['stdev_unres_tau_zw_sample'][:])
#    
#    #Average means over time steps used for training (steps 0 up to and including 27) since only these were used to normalize the data
#    meant_xu = np.mean(mean_xu[0:tstart])
#    meant_yu = np.mean(mean_yu[0:tstart])
#    meant_zu = np.mean(mean_zu[0:tstart])
#    meant_xv = np.mean(mean_xv[0:tstart])
#    meant_yv = np.mean(mean_yv[0:tstart])
#    meant_zv = np.mean(mean_zv[0:tstart])
#    meant_xw = np.mean(mean_xw[0:tstart])
#    meant_yw = np.mean(mean_yw[0:tstart])
#    meant_zw = np.mean(mean_zw[0:tstart])
#    #
#    stdevt_xu = np.mean(stdev_xu[0:tstart])
#    stdevt_yu = np.mean(stdev_yu[0:tstart])
#    stdevt_zu = np.mean(stdev_zu[0:tstart])
#    stdevt_xv = np.mean(stdev_xv[0:tstart])
#    stdevt_yv = np.mean(stdev_yv[0:tstart])
#    stdevt_zv = np.mean(stdev_zv[0:tstart])
#    stdevt_xw = np.mean(stdev_xw[0:tstart])
#    stdevt_yw = np.mean(stdev_yw[0:tstart])
#    stdevt_zw = np.mean(stdev_zw[0:tstart])    
#
#    #Undo normalisation, including the one done earlier with the friction velocity
#    print('begin to undo normalisation')
#    def undo_normalisation(lbls, means, stdevs, time_steps):
#        lbls  = (lbls * stdevs) + means
#        return lbls
#    
#    preds_values_xu = undo_normalisation(preds_values_xu, meant_xu, stdevt_xu, tstep_values) * (utau_ref ** 2) 
#    lbls_values_xu  = undo_normalisation(lbls_values_xu , meant_xu, stdevt_xu, tstep_values) * (utau_ref ** 2)
#    preds_values_yu = undo_normalisation(preds_values_yu, meant_yu, stdevt_yu, tstep_values) * (utau_ref ** 2)
#    lbls_values_yu  = undo_normalisation(lbls_values_yu , meant_yu, stdevt_yu, tstep_values) * (utau_ref ** 2)
#    preds_values_zu = undo_normalisation(preds_values_zu, meant_zu, stdevt_zu, tstep_values) * (utau_ref ** 2)
#    lbls_values_zu  = undo_normalisation(lbls_values_zu , meant_zu, stdevt_zu, tstep_values) * (utau_ref ** 2)
#    preds_values_xv = undo_normalisation(preds_values_xv, meant_xv, stdevt_xv, tstep_values) * (utau_ref ** 2)
#    lbls_values_xv  = undo_normalisation(lbls_values_xv , meant_xv, stdevt_xv, tstep_values) * (utau_ref ** 2)
#    preds_values_yv = undo_normalisation(preds_values_yv, meant_yv, stdevt_yv, tstep_values) * (utau_ref ** 2)
#    lbls_values_yv  = undo_normalisation(lbls_values_yv , meant_yv, stdevt_yv, tstep_values) * (utau_ref ** 2)
#    preds_values_zv = undo_normalisation(preds_values_zv, meant_zv, stdevt_zv, tstep_values) * (utau_ref ** 2)
#    lbls_values_zv  = undo_normalisation(lbls_values_zv , meant_zv, stdevt_zv, tstep_values) * (utau_ref ** 2)
#    preds_values_xw = undo_normalisation(preds_values_xw, meant_xw, stdevt_xw, tstep_values) * (utau_ref ** 2)
#    lbls_values_xw  = undo_normalisation(lbls_values_xw , meant_xw, stdevt_xw, tstep_values) * (utau_ref ** 2)
#    preds_values_yw = undo_normalisation(preds_values_yw, meant_yw, stdevt_yw, tstep_values) * (utau_ref ** 2)
#    lbls_values_yw  = undo_normalisation(lbls_values_yw , meant_yw, stdevt_yw, tstep_values) * (utau_ref ** 2)
#    preds_values_zw = undo_normalisation(preds_values_zw, meant_zw, stdevt_zw, tstep_values) * (utau_ref ** 2)
#    lbls_values_zw  = undo_normalisation(lbls_values_zw , meant_zw, stdevt_zw, tstep_values) * (utau_ref ** 2)
#    print('finished undoing normalisation')
#
#    #Close netCDF-file
#    stats.close()
#
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
    d = nc.Dataset('reconstructed_fields.nc','w')

    #Extract unique coordinates and time steps
    zloc_unique  = np.unique(zloc_values)
    zhloc_unique = np.unique(zhloc_values)
    yloc_unique  = np.unique(yloc_values)
    yhloc_unique = np.unique(yhloc_values)
    xloc_unique  = np.unique(xloc_values)
    xhloc_unique = np.unique(xhloc_values)
    tstep_unique = np.unique(tstep_values)

    #Create dimensions for storage in nc-file
    dim_zloc_unique  = d.createDimension("zloc_unique", len(zloc_unique))
    dim_zhloc_unique = d.createDimension("zhloc_unique",len(zhloc_unique))
    dim_yloc_unique  = d.createDimension("yloc_unique", len(yloc_unique))
    dim_yhloc_unique = d.createDimension("yhloc_unique",len(yhloc_unique))
    dim_xloc_unique  = d.createDimension("xloc_unique", len(xloc_unique))
    dim_xhloc_unique = d.createDimension("xhloc_unique",len(xhloc_unique))
    dim_tstep_unique = d.createDimension("tstep_unique",len(tstep_unique))

    #Create variables for dimensions and store them
    var_zloc_unique  = d.createVariable("zloc_unique" ,"f8",("zloc_unique",))
    var_zhloc_unique = d.createVariable("zhloc_unique","f8",("zhloc_unique",))
    var_yloc_unique  = d.createVariable("yloc_unique" ,"f8",("yloc_unique",))
    var_yhloc_unique = d.createVariable("yhloc_unique","f8",("yhloc_unique",))
    var_xloc_unique  = d.createVariable("xloc_unique" ,"f8",("xloc_unique",))
    var_xhloc_unique = d.createVariable("xhloc_unique","f8",("xhloc_unique",))
    var_tstep_unique = d.createVariable("tstep_unique","f8",("tstep_unique",))

    var_zloc_unique[:]  = zloc_unique[:]
    var_zhloc_unique[:] = zhloc_unique[:]
    var_yloc_unique[:]  = yloc_unique[:]
    var_yhloc_unique[:] = yhloc_unique[:]
    var_xloc_unique[:]  = xloc_unique[:]
    var_xhloc_unique[:] = xhloc_unique[:]
    var_tstep_unique[:] = tstep_unique[:]

    #Create variables for storage labels
    var_unres_tau_xu_lbls = d.createVariable("unres_tau_xu_lbls","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_xv_lbls = d.createVariable("unres_tau_xv_lbls","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_xw_lbls = d.createVariable("unres_tau_xw_lbls","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_yu_lbls = d.createVariable("unres_tau_yu_lbls","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_yv_lbls = d.createVariable("unres_tau_yv_lbls","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_yw_lbls = d.createVariable("unres_tau_yw_lbls","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zu_lbls = d.createVariable("unres_tau_zu_lbls","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_zv_lbls = d.createVariable("unres_tau_zv_lbls","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zw_lbls = d.createVariable("unres_tau_zw_lbls","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))

    #Call function to recontruct fields of labels for all nine components
    print('start reconstructing labels')
    var_unres_tau_xu_lbls[:,:,:,:] = reconstruct_field(lbls_values_xu, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('first component done')
    var_unres_tau_xv_lbls[:,:,:,:] = reconstruct_field(lbls_values_xv, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('second component done')
    var_unres_tau_xw_lbls[:,:,:,:] = reconstruct_field(lbls_values_xw, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('third component done')
    var_unres_tau_yu_lbls[:,:,:,:] = reconstruct_field(lbls_values_yu, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fourth component done')
    var_unres_tau_yv_lbls[:,:,:,:] = reconstruct_field(lbls_values_yv, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fifth component done')
    var_unres_tau_yw_lbls[:,:,:,:] = reconstruct_field(lbls_values_yw, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('sixth component done')
    var_unres_tau_zu_lbls[:,:,:,:] = reconstruct_field(lbls_values_zu, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('seventh component done')
    var_unres_tau_zv_lbls[:,:,:,:] = reconstruct_field(lbls_values_zv, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('eighth component done')
    var_unres_tau_zw_lbls[:,:,:,:] = reconstruct_field(lbls_values_zw, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('nineth component done')

    #Create variables for storage reconstructed fields of predictions
    var_unres_tau_xu_CNN = d.createVariable("unres_tau_xu_CNN","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_xv_CNN = d.createVariable("unres_tau_xv_CNN","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_xw_CNN = d.createVariable("unres_tau_xw_CNN","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_yu_CNN = d.createVariable("unres_tau_yu_CNN","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_yv_CNN = d.createVariable("unres_tau_yv_CNN","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_yw_CNN = d.createVariable("unres_tau_yw_CNN","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zu_CNN = d.createVariable("unres_tau_zu_CNN","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_zv_CNN = d.createVariable("unres_tau_zv_CNN","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zw_CNN = d.createVariable("unres_tau_zw_CNN","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))

    #Call function to recontruct fields of predictions for all nine components
    print('start reconstructing predictions')
    preds_values_xu = reconstruct_field(preds_values_xu, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    var_unres_tau_xu_CNN[:,:,:,:] = preds_values_xu
    print('first component done')
    preds_values_xv = reconstruct_field(preds_values_xv, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    var_unres_tau_xv_CNN[:,:,:,:] = preds_values_xv
    print('second component done')
    preds_values_xw = reconstruct_field(preds_values_xw, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique)
    var_unres_tau_xw_CNN[:,:,:,:] = preds_values_xw
    print('third component done')
    preds_values_yu = reconstruct_field(preds_values_yu, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    var_unres_tau_yu_CNN[:,:,:,:] = preds_values_yu
    print('fourth component done')
    preds_values_yv = reconstruct_field(preds_values_yv, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    var_unres_tau_yv_CNN[:,:,:,:] = preds_values_yv
    print('fifth component done')
    preds_values_yw = reconstruct_field(preds_values_yw, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    var_unres_tau_yw_CNN[:,:,:,:] = preds_values_yw
    print('sixth component done')
    preds_values_zu = reconstruct_field(preds_values_zu, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    var_unres_tau_zu_CNN[:,:,:,:] = preds_values_zu
    print('seventh component done')
    preds_values_zv = reconstruct_field(preds_values_zv, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    var_unres_tau_zv_CNN[:,:,:,:] = preds_values_zv
    print('eighth component done')
    preds_values_zw = reconstruct_field(preds_values_zw, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    var_unres_tau_zw_CNN[:,:,:,:] = preds_values_zw
    print('nineth component done')
    
    #Create variables for storage unresolved, resolved, and total transports
    var_unres_tau_xu_traceless = d.createVariable("unres_tau_xu_traceless","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_yv_traceless = d.createVariable("unres_tau_yv_traceless","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_zw_traceless = d.createVariable("unres_tau_zw_traceless","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    #
    var_unres_tau_xu = d.createVariable("unres_tau_xu","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_xv = d.createVariable("unres_tau_xv","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_xw = d.createVariable("unres_tau_xw","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_yu = d.createVariable("unres_tau_yu","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_yv = d.createVariable("unres_tau_yv","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_yw = d.createVariable("unres_tau_yw","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zu = d.createVariable("unres_tau_zu","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_zv = d.createVariable("unres_tau_zv","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zw = d.createVariable("unres_tau_zw","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_res_tau_xu   = d.createVariable("res_tau_xu"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_res_tau_xv   = d.createVariable("res_tau_xv"  ,"f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_res_tau_xw   = d.createVariable("res_tau_xw"  ,"f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_res_tau_yu   = d.createVariable("res_tau_yu"  ,"f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_res_tau_yv   = d.createVariable("res_tau_yv"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_res_tau_yw   = d.createVariable("res_tau_yw"  ,"f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_res_tau_zu   = d.createVariable("res_tau_zu"  ,"f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_res_tau_zv   = d.createVariable("res_tau_zv"  ,"f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_res_tau_zw   = d.createVariable("res_tau_zw"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_tot_tau_xu   = d.createVariable("tot_tau_xu"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_tot_tau_xv   = d.createVariable("tot_tau_xv"  ,"f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_tot_tau_xw   = d.createVariable("tot_tau_xw"  ,"f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_tot_tau_yu   = d.createVariable("tot_tau_yu"  ,"f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_tot_tau_yv   = d.createVariable("tot_tau_yv"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_tot_tau_yw   = d.createVariable("tot_tau_yw"  ,"f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_tot_tau_zu   = d.createVariable("tot_tau_zu"  ,"f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_tot_tau_zv   = d.createVariable("tot_tau_zv"  ,"f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_tot_tau_zw   = d.createVariable("tot_tau_zw"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    #
    var_unres_tau_xu_smag = d.createVariable("unres_tau_xu_smag","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_xv_smag = d.createVariable("unres_tau_xv_smag","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_xw_smag = d.createVariable("unres_tau_xw_smag","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_yu_smag = d.createVariable("unres_tau_yu_smag","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_unres_tau_yv_smag = d.createVariable("unres_tau_yv_smag","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_unres_tau_yw_smag = d.createVariable("unres_tau_yw_smag","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zu_smag = d.createVariable("unres_tau_zu_smag","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_unres_tau_zv_smag = d.createVariable("unres_tau_zv_smag","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_unres_tau_zw_smag = d.createVariable("unres_tau_zw_smag","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_tot_tau_xu_smag   = d.createVariable("tot_tau_xu_smag"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_tot_tau_xv_smag   = d.createVariable("tot_tau_xv_smag"  ,"f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_tot_tau_xw_smag   = d.createVariable("tot_tau_xw_smag"  ,"f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_tot_tau_yu_smag   = d.createVariable("tot_tau_yu_smag"  ,"f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_tot_tau_yv_smag   = d.createVariable("tot_tau_yv_smag"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_tot_tau_yw_smag   = d.createVariable("tot_tau_yw_smag"  ,"f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_tot_tau_zu_smag   = d.createVariable("tot_tau_zu_smag"  ,"f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_tot_tau_zv_smag   = d.createVariable("tot_tau_zv_smag"  ,"f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_tot_tau_zw_smag   = d.createVariable("tot_tau_zw_smag"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    #
    var_tot_tau_xu_CNN   = d.createVariable("tot_tau_xu_CNN"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_tot_tau_xv_CNN   = d.createVariable("tot_tau_xv_CNN"  ,"f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_tot_tau_xw_CNN   = d.createVariable("tot_tau_xw_CNN"  ,"f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_tot_tau_yu_CNN   = d.createVariable("tot_tau_yu_CNN"  ,"f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_tot_tau_yv_CNN   = d.createVariable("tot_tau_yv_CNN"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_tot_tau_yw_CNN   = d.createVariable("tot_tau_yw_CNN"  ,"f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_tot_tau_zu_CNN   = d.createVariable("tot_tau_zu_CNN"  ,"f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_tot_tau_zv_CNN   = d.createVariable("tot_tau_zv_CNN"  ,"f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_tot_tau_zw_CNN   = d.createVariable("tot_tau_zw_CNN"  ,"f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))

    #Store values for unresolved, resolved, and total fluxes training data, Smagorinsky, and CNN
    var_unres_tau_xu_traceless[:,:,:,:] = unres_tau_xu_traceless
    var_unres_tau_yv_traceless[:,:,:,:] = unres_tau_yv_traceless
    var_unres_tau_zw_traceless[:,:,:,:] = unres_tau_zw_traceless
    #
    var_unres_tau_xu[:,:,:,:] = unres_tau_xu
    var_unres_tau_xv[:,:,:,:] = unres_tau_xv
    var_unres_tau_xw[:,:,:,:] = unres_tau_xw
    var_unres_tau_yu[:,:,:,:] = unres_tau_yu
    var_unres_tau_yv[:,:,:,:] = unres_tau_yv
    var_unres_tau_yw[:,:,:,:] = unres_tau_yw
    var_unres_tau_zu[:,:,:,:] = unres_tau_zu
    var_unres_tau_zv[:,:,:,:] = unres_tau_zv
    var_unres_tau_zw[:,:,:,:] = unres_tau_zw
    var_res_tau_xu[:,:,:,:]   = res_tau_xu
    var_res_tau_xv[:,:,:,:]   = res_tau_xv
    var_res_tau_xw[:,:,:,:]   = res_tau_xw
    var_res_tau_yu[:,:,:,:]   = res_tau_yu
    var_res_tau_yv[:,:,:,:]   = res_tau_yv
    var_res_tau_yw[:,:,:,:]   = res_tau_yw
    var_res_tau_zu[:,:,:,:]   = res_tau_zu
    var_res_tau_zv[:,:,:,:]   = res_tau_zv
    var_res_tau_zw[:,:,:,:]   = res_tau_zw
    var_tot_tau_xu[:,:,:,:]   = tot_tau_xu
    var_tot_tau_xu[:,:,:,:]   = tot_tau_xu
    var_tot_tau_xv[:,:,:,:]   = tot_tau_xv
    var_tot_tau_xw[:,:,:,:]   = tot_tau_xw
    var_tot_tau_yu[:,:,:,:]   = tot_tau_yu
    var_tot_tau_yv[:,:,:,:]   = tot_tau_yv
    var_tot_tau_yw[:,:,:,:]   = tot_tau_yw
    var_tot_tau_zu[:,:,:,:]   = tot_tau_zu
    var_tot_tau_zv[:,:,:,:]   = tot_tau_zv
    var_tot_tau_zw[:,:,:,:]   = tot_tau_zw
    #
    var_unres_tau_xu_smag[:,:,:,:] = smag_tau_xu
    var_unres_tau_xv_smag[:,:,:,:] = smag_tau_xv
    var_unres_tau_xw_smag[:,:,:,:] = smag_tau_xw
    var_unres_tau_yu_smag[:,:,:,:] = smag_tau_yu
    var_unres_tau_yv_smag[:,:,:,:] = smag_tau_yv
    var_unres_tau_yw_smag[:,:,:,:] = smag_tau_yw
    var_unres_tau_zu_smag[:,:,:,:] = smag_tau_zu
    var_unres_tau_zv_smag[:,:,:,:] = smag_tau_zv
    var_unres_tau_zw_smag[:,:,:,:] = smag_tau_zw
    var_tot_tau_xu_smag[:,:,:,:]   = smag_tau_xu + res_tau_xu
    var_tot_tau_xv_smag[:,:,:,:]   = smag_tau_xv + res_tau_xv
    var_tot_tau_xw_smag[:,:,:,:]   = smag_tau_xw + res_tau_xw
    var_tot_tau_yu_smag[:,:,:,:]   = smag_tau_yu + res_tau_yu
    var_tot_tau_yv_smag[:,:,:,:]   = smag_tau_yv + res_tau_yv
    var_tot_tau_yw_smag[:,:,:,:]   = smag_tau_yw + res_tau_yw
    var_tot_tau_zu_smag[:,:,:,:]   = smag_tau_zu + res_tau_zu
    var_tot_tau_zv_smag[:,:,:,:]   = smag_tau_zv + res_tau_zv
    var_tot_tau_zw_smag[:,:,:,:]   = smag_tau_zw + res_tau_zw
    #
    var_tot_tau_xu_CNN[:,:,:,:]    = preds_values_xu  + res_tau_xu
    var_tot_tau_xv_CNN[:,:,:,:]    = preds_values_xv  + res_tau_xv
    var_tot_tau_xw_CNN[:,:,:,:]    = preds_values_xw  + res_tau_xw
    var_tot_tau_yu_CNN[:,:,:,:]    = preds_values_yu  + res_tau_yu
    var_tot_tau_yv_CNN[:,:,:,:]    = preds_values_yv  + res_tau_yv
    var_tot_tau_yw_CNN[:,:,:,:]    = preds_values_yw  + res_tau_yw
    var_tot_tau_zu_CNN[:,:,:,:]    = preds_values_zu  + res_tau_zu
    var_tot_tau_zv_CNN[:,:,:,:]    = preds_values_zv  + res_tau_zv
    var_tot_tau_zw_CNN[:,:,:,:]    = preds_values_zw  + res_tau_zw

    #Create variables for storage horizontal averages
    var_unres_tau_xu_horavg = d.createVariable("unres_tau_xu_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_xv_horavg = d.createVariable("unres_tau_xv_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_xw_horavg = d.createVariable("unres_tau_xw_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_yu_horavg = d.createVariable("unres_tau_yu_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_yv_horavg = d.createVariable("unres_tau_yv_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_yw_horavg = d.createVariable("unres_tau_yw_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_zu_horavg = d.createVariable("unres_tau_zu_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_zv_horavg = d.createVariable("unres_tau_zv_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_zw_horavg = d.createVariable("unres_tau_zw_horavg","f8",("tstep_unique","zloc_unique"))
    var_res_tau_xu_horavg   = d.createVariable("res_tau_xu_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_res_tau_xv_horavg   = d.createVariable("res_tau_xv_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_res_tau_xw_horavg   = d.createVariable("res_tau_xw_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_res_tau_yu_horavg   = d.createVariable("res_tau_yu_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_res_tau_yv_horavg   = d.createVariable("res_tau_yv_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_res_tau_yw_horavg   = d.createVariable("res_tau_yw_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_res_tau_zu_horavg   = d.createVariable("res_tau_zu_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_res_tau_zv_horavg   = d.createVariable("res_tau_zv_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_res_tau_zw_horavg   = d.createVariable("res_tau_zw_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_xu_horavg   = d.createVariable("tot_tau_xu_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_xv_horavg   = d.createVariable("tot_tau_xv_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_xw_horavg   = d.createVariable("tot_tau_xw_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_yu_horavg   = d.createVariable("tot_tau_yu_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_yv_horavg   = d.createVariable("tot_tau_yv_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_yw_horavg   = d.createVariable("tot_tau_yw_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_zu_horavg   = d.createVariable("tot_tau_zu_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_zv_horavg   = d.createVariable("tot_tau_zv_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_zw_horavg   = d.createVariable("tot_tau_zw_horavg","f8",  ("tstep_unique","zloc_unique"))
    #
    var_unres_tau_xu_smag_horavg = d.createVariable("unres_tau_xu_smag_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_xv_smag_horavg = d.createVariable("unres_tau_xv_smag_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_xw_smag_horavg = d.createVariable("unres_tau_xw_smag_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_yu_smag_horavg = d.createVariable("unres_tau_yu_smag_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_yv_smag_horavg = d.createVariable("unres_tau_yv_smag_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_yw_smag_horavg = d.createVariable("unres_tau_yw_smag_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_zu_smag_horavg = d.createVariable("unres_tau_zu_smag_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_zv_smag_horavg = d.createVariable("unres_tau_zv_smag_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_zw_smag_horavg = d.createVariable("unres_tau_zw_smag_horavg","f8",("tstep_unique","zloc_unique"))
    var_tot_tau_xu_smag_horavg   = d.createVariable("tot_tau_xu_smag_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_xv_smag_horavg   = d.createVariable("tot_tau_xv_smag_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_xw_smag_horavg   = d.createVariable("tot_tau_xw_smag_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_yu_smag_horavg   = d.createVariable("tot_tau_yu_smag_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_yv_smag_horavg   = d.createVariable("tot_tau_yv_smag_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_yw_smag_horavg   = d.createVariable("tot_tau_yw_smag_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_zu_smag_horavg   = d.createVariable("tot_tau_zu_smag_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_zv_smag_horavg   = d.createVariable("tot_tau_zv_smag_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_zw_smag_horavg   = d.createVariable("tot_tau_zw_smag_horavg","f8",  ("tstep_unique","zloc_unique"))
    #
    var_unres_tau_xu_CNN_horavg = d.createVariable("unres_tau_xu_CNN_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_xv_CNN_horavg = d.createVariable("unres_tau_xv_CNN_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_xw_CNN_horavg = d.createVariable("unres_tau_xw_CNN_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_yu_CNN_horavg = d.createVariable("unres_tau_yu_CNN_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_yv_CNN_horavg = d.createVariable("unres_tau_yv_CNN_horavg","f8",("tstep_unique","zloc_unique"))
    var_unres_tau_yw_CNN_horavg = d.createVariable("unres_tau_yw_CNN_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_zu_CNN_horavg = d.createVariable("unres_tau_zu_CNN_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_zv_CNN_horavg = d.createVariable("unres_tau_zv_CNN_horavg","f8",("tstep_unique","zhloc_unique"))
    var_unres_tau_zw_CNN_horavg = d.createVariable("unres_tau_zw_CNN_horavg","f8",("tstep_unique","zloc_unique"))
    var_tot_tau_xu_CNN_horavg   = d.createVariable("tot_tau_xu_CNN_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_xv_CNN_horavg   = d.createVariable("tot_tau_xv_CNN_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_xw_CNN_horavg   = d.createVariable("tot_tau_xw_CNN_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_yu_CNN_horavg   = d.createVariable("tot_tau_yu_CNN_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_yv_CNN_horavg   = d.createVariable("tot_tau_yv_CNN_horavg","f8",  ("tstep_unique","zloc_unique"))
    var_tot_tau_yw_CNN_horavg   = d.createVariable("tot_tau_yw_CNN_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_zu_CNN_horavg   = d.createVariable("tot_tau_zu_CNN_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_zv_CNN_horavg   = d.createVariable("tot_tau_zv_CNN_horavg","f8",  ("tstep_unique","zhloc_unique"))
    var_tot_tau_zw_CNN_horavg   = d.createVariable("tot_tau_zw_CNN_horavg","f8",  ("tstep_unique","zloc_unique"))
 
    #Create variables for storage fractions sub-grid fluxes
    var_frac_unres_tau_xu = d.createVariable("frac_unres_tau_xu","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_frac_unres_tau_xv = d.createVariable("frac_unres_tau_xv","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_frac_unres_tau_xw = d.createVariable("frac_unres_tau_xw","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_frac_unres_tau_yu = d.createVariable("frac_unres_tau_yu","f8",("tstep_unique","zloc_unique","yhloc_unique","xhloc_unique"))
    var_frac_unres_tau_yv = d.createVariable("frac_unres_tau_yv","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_frac_unres_tau_yw = d.createVariable("frac_unres_tau_yw","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_frac_unres_tau_zu = d.createVariable("frac_unres_tau_zu","f8",("tstep_unique","zhloc_unique","yloc_unique","xhloc_unique"))
    var_frac_unres_tau_zv = d.createVariable("frac_unres_tau_zv","f8",("tstep_unique","zhloc_unique","yhloc_unique","xloc_unique"))
    var_frac_unres_tau_zw = d.createVariable("frac_unres_tau_zw","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    var_frac_unres_tau_xu_horavg = d.createVariable("frac_unres_tau_xu_horavg","f8",("tstep_unique","zloc_unique"))
    var_frac_unres_tau_xv_horavg = d.createVariable("frac_unres_tau_xv_horavg","f8",("tstep_unique","zloc_unique"))
    var_frac_unres_tau_xw_horavg = d.createVariable("frac_unres_tau_xw_horavg","f8",("tstep_unique","zhloc_unique"))
    var_frac_unres_tau_yu_horavg = d.createVariable("frac_unres_tau_yu_horavg","f8",("tstep_unique","zloc_unique"))
    var_frac_unres_tau_yv_horavg = d.createVariable("frac_unres_tau_yv_horavg","f8",("tstep_unique","zloc_unique"))
    var_frac_unres_tau_yw_horavg = d.createVariable("frac_unres_tau_yw_horavg","f8",("tstep_unique","zhloc_unique"))
    var_frac_unres_tau_zu_horavg = d.createVariable("frac_unres_tau_zu_horavg","f8",("tstep_unique","zhloc_unique"))
    var_frac_unres_tau_zv_horavg = d.createVariable("frac_unres_tau_zv_horavg","f8",("tstep_unique","zhloc_unique"))
    var_frac_unres_tau_zw_horavg = d.createVariable("frac_unres_tau_zw_horavg","f8",("tstep_unique","zloc_unique"))

    #Calculate fraction unresolved compared to resolved fluxes, both point-by-point and horizontally averaged
    #NOTE: Some extreme outliers in the fractions occur when the total momentum transport reaches 0. To preven this from happening, the fractions are confined to the range -10 to 10.
    var_unres_tau_xu_horavg[:,:]          = np.mean(unres_tau_xu,                 axis=(2,3), keepdims=False)
    var_res_tau_xu_horavg[:,:]            = np.mean(res_tau_xu,                   axis=(2,3), keepdims=False)
    var_tot_tau_xu_horavg[:,:]            = np.mean(tot_tau_xu,                   axis=(2,3), keepdims=False)
    var_unres_tau_xu_smag_horavg[:,:]     = np.mean(smag_tau_xu,                  axis=(2,3), keepdims=False)
    var_unres_tau_xu_CNN_horavg[:,:]      = np.mean(preds_values_xu,              axis=(2,3), keepdims=False)
    var_tot_tau_xu_smag_horavg[:,:]       = np.mean(smag_tau_xu + res_tau_xu,     axis=(2,3), keepdims=False)
    var_tot_tau_xu_CNN_horavg[:,:]        = np.mean(preds_values_xu + res_tau_xu, axis=(2,3), keepdims=False)
    var_frac_unres_tau_xu[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_xu[:,:,:,:])        / np.array(var_res_tau_xu[:,:,:,:]))       ,-10)
    var_frac_unres_tau_xu_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_xu_horavg[:,:]) / np.array(var_res_tau_xu_horavg[:,:])),-10)
    #
    var_unres_tau_xv_horavg[:,:]          = np.mean(unres_tau_xv,                 axis=(2,3), keepdims=False)
    var_res_tau_xv_horavg[:,:]            = np.mean(res_tau_xv,                   axis=(2,3), keepdims=False)
    var_tot_tau_xv_horavg[:,:]            = np.mean(tot_tau_xv,                   axis=(2,3), keepdims=False)
    var_unres_tau_xv_smag_horavg[:,:]     = np.mean(smag_tau_xv,                  axis=(2,3), keepdims=False)
    var_unres_tau_xv_CNN_horavg[:,:]      = np.mean(preds_values_xv,              axis=(2,3), keepdims=False)
    var_tot_tau_xv_smag_horavg[:,:]       = np.mean(smag_tau_xv + res_tau_xv,     axis=(2,3), keepdims=False)
    var_tot_tau_xv_CNN_horavg[:,:]        = np.mean(preds_values_xv + res_tau_xv, axis=(2,3), keepdims=False)
    var_frac_unres_tau_xv[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_xv[:,:,:,:])        / np.array(var_res_tau_xv[:,:,:,:]))       ,-10)
    var_frac_unres_tau_xv_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_xv_horavg[:,:]) / np.array(var_res_tau_xv_horavg[:,:])),-10)
    #
    var_unres_tau_xw_horavg[:,:]          = np.mean(unres_tau_xw,                 axis=(2,3), keepdims=False)
    var_res_tau_xw_horavg[:,:]            = np.mean(res_tau_xw,                   axis=(2,3), keepdims=False)
    var_tot_tau_xw_horavg[:,:]            = np.mean(tot_tau_xw,                   axis=(2,3), keepdims=False)
    var_unres_tau_xw_smag_horavg[:,:]     = np.mean(smag_tau_xw,                  axis=(2,3), keepdims=False)
    var_unres_tau_xw_CNN_horavg[:,:]      = np.mean(preds_values_xw,              axis=(2,3), keepdims=False)
    var_tot_tau_xw_smag_horavg[:,:]       = np.mean(smag_tau_xw + res_tau_xw,     axis=(2,3), keepdims=False)
    var_tot_tau_xw_CNN_horavg[:,:]        = np.mean(preds_values_xw + res_tau_xw, axis=(2,3), keepdims=False)
    var_frac_unres_tau_xw[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_xw[:,:,:,:])        / np.array(var_res_tau_xw[:,:,:,:]))       ,-10)
    var_frac_unres_tau_xw_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_xw_horavg[:,:]) / np.array(var_res_tau_xw_horavg[:,:])),-10)
    #
    var_unres_tau_yu_horavg[:,:]          = np.mean(unres_tau_yu,                 axis=(2,3), keepdims=False)
    var_res_tau_yu_horavg[:,:]            = np.mean(res_tau_yu,                   axis=(2,3), keepdims=False)
    var_tot_tau_yu_horavg[:,:]            = np.mean(tot_tau_yu,                   axis=(2,3), keepdims=False)
    var_unres_tau_yu_smag_horavg[:,:]     = np.mean(smag_tau_yu,                  axis=(2,3), keepdims=False)
    var_unres_tau_yu_CNN_horavg[:,:]      = np.mean(preds_values_yu,              axis=(2,3), keepdims=False)
    var_tot_tau_yu_smag_horavg[:,:]       = np.mean(smag_tau_yu + res_tau_yu,     axis=(2,3), keepdims=False)
    var_tot_tau_yu_CNN_horavg[:,:]        = np.mean(preds_values_yu + res_tau_yu, axis=(2,3), keepdims=False)
    var_frac_unres_tau_yu[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_yu[:,:,:,:])        / np.array(var_res_tau_yu[:,:,:,:]))       ,-10)
    var_frac_unres_tau_yu_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_yu_horavg[:,:]) / np.array(var_res_tau_yu_horavg[:,:])),-10)
    #
    var_unres_tau_yv_horavg[:,:]          = np.mean(unres_tau_yv,                 axis=(2,3), keepdims=False)
    var_res_tau_yv_horavg[:,:]            = np.mean(res_tau_yv,                   axis=(2,3), keepdims=False)
    var_tot_tau_yv_horavg[:,:]            = np.mean(tot_tau_yv,                   axis=(2,3), keepdims=False)
    var_unres_tau_yv_smag_horavg[:,:]     = np.mean(smag_tau_yv,                  axis=(2,3), keepdims=False)
    var_unres_tau_yv_CNN_horavg[:,:]      = np.mean(preds_values_yv,              axis=(2,3), keepdims=False)
    var_tot_tau_yv_smag_horavg[:,:]       = np.mean(smag_tau_yv + res_tau_yv,     axis=(2,3), keepdims=False)
    var_tot_tau_yv_CNN_horavg[:,:]        = np.mean(preds_values_yv + res_tau_yv, axis=(2,3), keepdims=False)
    var_frac_unres_tau_yv[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_yv[:,:,:,:])        / np.array(var_res_tau_yv[:,:,:,:]))       ,-10)
    var_frac_unres_tau_yv_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_yv_horavg[:,:]) / np.array(var_res_tau_yv_horavg[:,:])),-10)
    #
    var_unres_tau_yw_horavg[:,:]          = np.mean(unres_tau_yw,                 axis=(2,3), keepdims=False)
    var_res_tau_yw_horavg[:,:]            = np.mean(res_tau_yw,                   axis=(2,3), keepdims=False)
    var_tot_tau_yw_horavg[:,:]            = np.mean(tot_tau_yw,                   axis=(2,3), keepdims=False)
    var_unres_tau_yw_smag_horavg[:,:]     = np.mean(smag_tau_yw,                  axis=(2,3), keepdims=False)
    var_unres_tau_yw_CNN_horavg[:,:]      = np.mean(preds_values_yw,              axis=(2,3), keepdims=False)
    var_tot_tau_yw_smag_horavg[:,:]       = np.mean(smag_tau_yw + res_tau_yw,     axis=(2,3), keepdims=False)
    var_tot_tau_yw_CNN_horavg[:,:]        = np.mean(preds_values_yw + res_tau_yw, axis=(2,3), keepdims=False)
    var_frac_unres_tau_yw[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_yw[:,:,:,:])        / np.array(var_res_tau_yw[:,:,:,:]))       ,-10)
    var_frac_unres_tau_yw_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_yw_horavg[:,:]) / np.array(var_res_tau_yw_horavg[:,:])),-10)
    #
    var_unres_tau_zu_horavg[:,:]          = np.mean(unres_tau_zu,                 axis=(2,3), keepdims=False)
    var_res_tau_zu_horavg[:,:]            = np.mean(res_tau_zu,                   axis=(2,3), keepdims=False)
    var_tot_tau_zu_horavg[:,:]            = np.mean(tot_tau_zu,                   axis=(2,3), keepdims=False)
    var_unres_tau_zu_smag_horavg[:,:]     = np.mean(smag_tau_zu,                  axis=(2,3), keepdims=False)
    var_unres_tau_zu_CNN_horavg[:,:]      = np.mean(preds_values_zu,              axis=(2,3), keepdims=False)
    var_tot_tau_zu_smag_horavg[:,:]       = np.mean(smag_tau_zu + res_tau_zu,     axis=(2,3), keepdims=False)
    var_tot_tau_zu_CNN_horavg[:,:]        = np.mean(preds_values_zu + res_tau_zu, axis=(2,3), keepdims=False)
    var_frac_unres_tau_zu[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_zu[:,:,:,:])        / np.array(var_res_tau_zu[:,:,:,:]))       ,-10)
    var_frac_unres_tau_zu_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_zu_horavg[:,:]) / np.array(var_res_tau_zu_horavg[:,:])),-10)
    #
    var_unres_tau_zv_horavg[:,:]          = np.mean(unres_tau_zv,                 axis=(2,3), keepdims=False)
    var_res_tau_zv_horavg[:,:]            = np.mean(res_tau_zv,                   axis=(2,3), keepdims=False)
    var_tot_tau_zv_horavg[:,:]            = np.mean(tot_tau_zv,                   axis=(2,3), keepdims=False)
    var_unres_tau_zv_smag_horavg[:,:]     = np.mean(smag_tau_zv,                  axis=(2,3), keepdims=False)
    var_unres_tau_zv_CNN_horavg[:,:]      = np.mean(preds_values_zv,              axis=(2,3), keepdims=False)
    var_tot_tau_zv_smag_horavg[:,:]       = np.mean(smag_tau_zv + res_tau_zv,     axis=(2,3), keepdims=False)
    var_tot_tau_zv_CNN_horavg[:,:]        = np.mean(preds_values_zv + res_tau_zv, axis=(2,3), keepdims=False)
    var_frac_unres_tau_zv[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_zv[:,:,:,:])        / np.array(var_res_tau_zv[:,:,:,:]))       ,-10)
    var_frac_unres_tau_zv_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_zv_horavg[:,:]) / np.array(var_res_tau_zv_horavg[:,:])),-10)
    #
    var_unres_tau_zw_horavg[:,:]          = np.mean(unres_tau_zw,                 axis=(2,3), keepdims=False)
    var_res_tau_zw_horavg[:,:]            = np.mean(res_tau_zw,                   axis=(2,3), keepdims=False)
    var_tot_tau_zw_horavg[:,:]            = np.mean(tot_tau_zw,                   axis=(2,3), keepdims=False)
    var_unres_tau_zw_smag_horavg[:,:]     = np.mean(smag_tau_zw,                  axis=(2,3), keepdims=False)
    var_unres_tau_zw_CNN_horavg[:,:]      = np.mean(preds_values_zw,              axis=(2,3), keepdims=False)
    var_tot_tau_zw_smag_horavg[:,:]       = np.mean(smag_tau_zw + res_tau_zw,     axis=(2,3), keepdims=False)
    var_tot_tau_zw_CNN_horavg[:,:]        = np.mean(preds_values_zw + res_tau_zw, axis=(2,3), keepdims=False)
    var_frac_unres_tau_zw[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_zw[:,:,:,:])        / np.array(var_res_tau_zw[:,:,:,:]))       ,-10)
    var_frac_unres_tau_zw_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_zw_horavg[:,:]) / np.array(var_res_tau_zw_horavg[:,:])),-10)

    #Close netCDF-files
    d.close()

###Loop over heights for all components considering the time steps specified below, and make scatterplots of labels vs fluxes (CNN and Smagorinsky) at each height for all specified time steps combined###
print('Start making plots')
#Fetch unresolved fluxes from netCDF-file
fields = nc.Dataset("reconstructed_fields.nc", "r")

#Extract CNN fluxes
unres_tau_xu_CNN = np.array(fields['unres_tau_xu_CNN'][:,:,:,:])
unres_tau_xv_CNN = np.array(fields['unres_tau_xv_CNN'][:,:,:,:])
unres_tau_xw_CNN = np.array(fields['unres_tau_xw_CNN'][:,:,:,:])
unres_tau_yu_CNN = np.array(fields['unres_tau_yu_CNN'][:,:,:,:])
unres_tau_yv_CNN = np.array(fields['unres_tau_yv_CNN'][:,:,:,:])
unres_tau_yw_CNN = np.array(fields['unres_tau_yw_CNN'][:,:,:,:])
unres_tau_zu_CNN = np.array(fields['unres_tau_zu_CNN'][:,:,:,:])
unres_tau_zv_CNN = np.array(fields['unres_tau_zv_CNN'][:,:,:,:])
unres_tau_zw_CNN = np.array(fields['unres_tau_zw_CNN'][:,:,:,:])
#
unres_tau_xu_CNN_horavg = np.array(fields['unres_tau_xu_CNN_horavg'][:,:])
unres_tau_xv_CNN_horavg = np.array(fields['unres_tau_xv_CNN_horavg'][:,:])
unres_tau_xw_CNN_horavg = np.array(fields['unres_tau_xw_CNN_horavg'][:,:])
unres_tau_yu_CNN_horavg = np.array(fields['unres_tau_yu_CNN_horavg'][:,:])
unres_tau_yv_CNN_horavg = np.array(fields['unres_tau_yv_CNN_horavg'][:,:])
unres_tau_yw_CNN_horavg = np.array(fields['unres_tau_yw_CNN_horavg'][:,:])
unres_tau_zu_CNN_horavg = np.array(fields['unres_tau_zu_CNN_horavg'][:,:])
unres_tau_zv_CNN_horavg = np.array(fields['unres_tau_zv_CNN_horavg'][:,:])
unres_tau_zw_CNN_horavg = np.array(fields['unres_tau_zw_CNN_horavg'][:,:])

#Extract Smagorinsky fluxes
unres_tau_xu_smag = np.array(fields['unres_tau_xu_smag'][:,:,:,:])
unres_tau_xv_smag = np.array(fields['unres_tau_xv_smag'][:,:,:,:])
unres_tau_xw_smag = np.array(fields['unres_tau_xw_smag'][:,:,:,:])
unres_tau_yu_smag = np.array(fields['unres_tau_yu_smag'][:,:,:,:])
unres_tau_yv_smag = np.array(fields['unres_tau_yv_smag'][:,:,:,:])
unres_tau_yw_smag = np.array(fields['unres_tau_yw_smag'][:,:,:,:])
unres_tau_zu_smag = np.array(fields['unres_tau_zu_smag'][:,:,:,:])
unres_tau_zv_smag = np.array(fields['unres_tau_zv_smag'][:,:,:,:])
unres_tau_zw_smag = np.array(fields['unres_tau_zw_smag'][:,:,:,:])
#
unres_tau_xu_smag_horavg = np.array(fields['unres_tau_xu_smag_horavg'][:,:])
unres_tau_xv_smag_horavg = np.array(fields['unres_tau_xv_smag_horavg'][:,:])
unres_tau_xw_smag_horavg = np.array(fields['unres_tau_xw_smag_horavg'][:,:])
unres_tau_yu_smag_horavg = np.array(fields['unres_tau_yu_smag_horavg'][:,:])
unres_tau_yv_smag_horavg = np.array(fields['unres_tau_yv_smag_horavg'][:,:])
unres_tau_yw_smag_horavg = np.array(fields['unres_tau_yw_smag_horavg'][:,:])
unres_tau_zu_smag_horavg = np.array(fields['unres_tau_zu_smag_horavg'][:,:])
unres_tau_zv_smag_horavg = np.array(fields['unres_tau_zv_smag_horavg'][:,:])
unres_tau_zw_smag_horavg = np.array(fields['unres_tau_zw_smag_horavg'][:,:])

#Extract training fluxes
unres_tau_xu_traceless = np.array(fields['unres_tau_xu_traceless'][:,:,:,:])
unres_tau_yv_traceless = np.array(fields['unres_tau_yv_traceless'][:,:,:,:])
unres_tau_zw_traceless = np.array(fields['unres_tau_zw_traceless'][:,:,:,:])
#
unres_tau_xu = np.array(fields['unres_tau_xu'][:,:,:,:])
unres_tau_xv = np.array(fields['unres_tau_xv'][:,:,:,:])
unres_tau_xw = np.array(fields['unres_tau_xw'][:,:,:,:])
unres_tau_yu = np.array(fields['unres_tau_yu'][:,:,:,:])
unres_tau_yv = np.array(fields['unres_tau_yv'][:,:,:,:])
unres_tau_yw = np.array(fields['unres_tau_yw'][:,:,:,:])
unres_tau_zu = np.array(fields['unres_tau_zu'][:,:,:,:])
unres_tau_zv = np.array(fields['unres_tau_zv'][:,:,:,:])
unres_tau_zw = np.array(fields['unres_tau_zw'][:,:,:,:])
#
unres_tau_xu_horavg = np.array(fields['unres_tau_xu_horavg'][:,:])
unres_tau_xv_horavg = np.array(fields['unres_tau_xv_horavg'][:,:])
unres_tau_xw_horavg = np.array(fields['unres_tau_xw_horavg'][:,:])
unres_tau_yu_horavg = np.array(fields['unres_tau_yu_horavg'][:,:])
unres_tau_yv_horavg = np.array(fields['unres_tau_yv_horavg'][:,:])
unres_tau_yw_horavg = np.array(fields['unres_tau_yw_horavg'][:,:])
unres_tau_zu_horavg = np.array(fields['unres_tau_zu_horavg'][:,:])
unres_tau_zv_horavg = np.array(fields['unres_tau_zv_horavg'][:,:])
unres_tau_zw_horavg = np.array(fields['unres_tau_zw_horavg'][:,:])

#Extract coordinates
xc = np.array(fields['xloc_unique'][:])
xhc = np.array(fields['xhloc_unique'][:])
yc = np.array(fields['yloc_unique'][:])
yhc = np.array(fields['yhloc_unique'][:])
zc = np.array(fields['zloc_unique'][:]) #NOTE: Already defined earlier in a different way, but both ways should be identical.
zhc = np.array(fields['zhloc_unique'][:])

#Close netCDF-file
fields.close()

#Define function for making horizontal cross-sections
def make_horcross_heights(values, z, y, x, component, is_lbl, time_step = 0, delta = 500):
    #NOTE1: fourth last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: third last input of this function is a boolean that specifies whether the labels (True) or the NN predictions are being plotted.
    #NOTE3: the second last input of this function is an integer specifying which validation time step stored in the nc-file is plotted (by default the first one, which now corresponds to time step 28 used for validation).
    #NOTE4: the last input of this function is an integer specifying the channel half with [in meter] used to rescale the horizontal dimensions (by default 500m). 
    for k in range(len(z)):
        values_height = values[time_step,k,:,:]

        #Make horizontal cross-sections of the values
        plt.figure()
        plt.pcolormesh(x * delta, y * delta, values_height, vmin=-0.15, vmax=0.15)
        #plt.pcolormesh(x * delta, y * delta, values_height, vmin=-0.00015, vmax=0.00015)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(r'$\rm {[m^{2}\ s^{-2}}]$',rotation=270,fontsize=20,labelpad=30)
        plt.xlabel(r'$\rm x\ [m]$',fontsize=20)
        plt.ylabel(r'$\rm y\ [m]$',fontsize=20)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16, rotation=0)
        plt.tight_layout()
        if not is_lbl:
            plt.savefig("Horcross_tau_" + component + "_" + str(z[k]) + ".png", dpi = 200)
        else:
            plt.savefig("Horcross_label_tau_" + component + "_" + str(z[k]) + ".png", dpi = 200)
        plt.close()

#Define function for making scatterplots
def make_scatterplot_heights(preds, lbls, preds_horavg, lbls_horavg, heights, component, is_smag):
    #NOTE1: second last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: last input of this function is a boolean that specifies wether the Smagorinsky fluxes are being plotted (True) or the CNN fluxes (False).
    for k in range(len(heights)+1):
        if k == len(heights):
            preds_height = preds_horavg
            lbls_height  = lbls_horavg
        else:
            preds_height = preds[:,k,:,:]
            lbls_height  = lbls[:,k,:,:]
        preds_height = preds_height.flatten()
        lbls_height  = lbls_height.flatten()
        
        #Make scatterplots of Smagorinsky/CNN fluxes versus labels
        corrcoef = np.round(np.corrcoef(preds_height, lbls_height)[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
        plt.figure()
        plt.scatter(lbls_height, preds_height, s=6, marker='o', alpha=0.2)
        if k == len(heights):
            plt.xlim([-0.004, 0.004])
            plt.ylim([-0.004, 0.004])
            #plt.xlim([-0.000006, 0.000006])
            #plt.ylim([-0.000006, 0.000006])
            #plt.xlim([-0.2, 0.2])
            #plt.ylim([-0.2, 0.2])
        else:
            plt.xlim([-0.5, 0.5])
            plt.ylim([-0.5, 0.5])
            #plt.xlim([-40.0, 40.0])
            #plt.ylim([-40.0, 40.0])
            #plt.xlim([-0.0005, 0.0005])
            #plt.ylim([-0.0005, 0.0005])
        axes = plt.gca()
        plt.plot(axes.get_xlim(),axes.get_ylim(),'b--')
        #plt.gca().set_aspect('equal',adjustable='box')
        plt.xlabel(r'$\rm labels\ {[m^{2}\ s^{-2}}]$',fontsize = 20)
        if is_smag:
            plt.ylabel(r'$\rm Smagorinsky\ {[m^{2}\ s^{-2}}]$',fontsize = 20)
        else:
            plt.ylabel(r'$\rm NN\ {[m^{2}\ s^{-2}}]$',fontsize = 20)
        plt.title("Corrcoef = " + str(corrcoef),fontsize = 20)
        plt.axhline(c='black')
        plt.axvline(c='black')
        plt.xticks(fontsize = 16, rotation = 90)
        plt.yticks(fontsize = 16, rotation = 0)
        if is_smag:
            if k == len(heights):
                plt.savefig("Scatter_Smagorinsky_tau_" + component + "_horavg.png", dpi = 200)
            else:
                plt.savefig("Scatter_Smagorinsky_tau_" + component + "_" + str(heights[k]) + ".png", dpi = 200)
        else:
            if k == len(heights):
                plt.savefig("Scatter_tau_" + component + "__horavg.png", dpi = 200)
            else:
                plt.savefig("Scatter_tau_" + component + "_" + str(heights[k]) + ".png", dpi = 200)
        plt.close()

#Call function multiple times to make all plots for smagorinsky and CNN
if args.make_plots:
    make_horcross_heights(unres_tau_xu, zc, yc, xc, 'xu',   True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yu, zc, yhc, xhc, 'yu', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zu, zhc, yc, xhc, 'zu', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_xv, zc, yhc, xhc, 'xv', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yv, zc, yc, xc, 'yv',   True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zv, zhc, yhc, xc, 'zv', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_xw, zhc, yc, xhc, 'xw', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yw, zhc, yhc, xc, 'yw', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zw, zhc, yc, xc, 'zw',  True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_xu_CNN, zc, yc, xc, 'xu',   False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yu_CNN, zc, yhc, xhc, 'yu', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zu_CNN, zhc, yc, xhc, 'zu', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_xv_CNN, zc, yhc, xhc, 'xv', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yv_CNN, zc, yc, xc, 'yv',   False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zv_CNN, zhc, yhc, xc, 'zv', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_xw_CNN, zhc, yc, xhc, 'xw', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yw_CNN, zhc, yhc, xc, 'yw', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zw_CNN, zhc, yc, xc, 'zw',  False, time_step = 0, delta = delta_height)
    #
    make_scatterplot_heights(unres_tau_xu_smag, unres_tau_xu_traceless, unres_tau_xu_smag_horavg, unres_tau_xu_horavg, zc,  'xu', True)
    make_scatterplot_heights(unres_tau_yu_smag, unres_tau_yu, unres_tau_yu_smag_horavg, unres_tau_yu_horavg, zc,  'yu', True)
    make_scatterplot_heights(unres_tau_zu_smag, unres_tau_zu, unres_tau_zu_smag_horavg, unres_tau_zu_horavg, zhc, 'zu', True)
    make_scatterplot_heights(unres_tau_xv_smag, unres_tau_xv, unres_tau_xv_smag_horavg, unres_tau_xv_horavg, zc,  'xv', True)
    make_scatterplot_heights(unres_tau_yv_smag, unres_tau_yv_traceless, unres_tau_yv_smag_horavg, unres_tau_yv_horavg, zc,  'yv', True)
    make_scatterplot_heights(unres_tau_zv_smag, unres_tau_zv, unres_tau_zv_smag_horavg, unres_tau_zv_horavg, zhc, 'zv', True)
    make_scatterplot_heights(unres_tau_xw_smag, unres_tau_xw, unres_tau_xw_smag_horavg, unres_tau_xw_horavg, zhc, 'xw', True)
    make_scatterplot_heights(unres_tau_yw_smag, unres_tau_yw, unres_tau_yw_smag_horavg, unres_tau_yw_horavg, zhc, 'yw', True)
    make_scatterplot_heights(unres_tau_zw_smag, unres_tau_zw_traceless, unres_tau_zw_smag_horavg, unres_tau_zw_horavg, zc,  'zw', True)
    #
    make_scatterplot_heights(unres_tau_xu_CNN, unres_tau_xu, unres_tau_xu_CNN_horavg, unres_tau_xu_horavg, zc,  'xu', False)
    make_scatterplot_heights(unres_tau_yu_CNN, unres_tau_yu, unres_tau_yu_CNN_horavg, unres_tau_yu_horavg, zc,  'yu', False)
    make_scatterplot_heights(unres_tau_zu_CNN, unres_tau_zu, unres_tau_zu_CNN_horavg, unres_tau_zu_horavg, zhc, 'zu', False)
    make_scatterplot_heights(unres_tau_xv_CNN, unres_tau_xv, unres_tau_xv_CNN_horavg, unres_tau_xv_horavg, zc,  'xv', False)
    make_scatterplot_heights(unres_tau_yv_CNN, unres_tau_yv, unres_tau_yv_CNN_horavg, unres_tau_yv_horavg, zc,  'yv', False)
    make_scatterplot_heights(unres_tau_zv_CNN, unres_tau_zv, unres_tau_zv_CNN_horavg, unres_tau_zv_horavg, zhc, 'zv', False)
    make_scatterplot_heights(unres_tau_xw_CNN, unres_tau_xw, unres_tau_xw_CNN_horavg, unres_tau_xw_horavg, zhc, 'xw', False)
    make_scatterplot_heights(unres_tau_yw_CNN, unres_tau_yw, unres_tau_yw_CNN_horavg, unres_tau_yw_horavg, zhc, 'yw', False)
    make_scatterplot_heights(unres_tau_zw_CNN, unres_tau_zw, unres_tau_zw_CNN_horavg, unres_tau_zw_horavg, zc,  'zw', False)
