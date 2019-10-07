import numpy as np
import netCDF4 as nc
#import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
#from matplotlib import rcParams
mpl.rcParams.update({'figure.autolayout':True})
import matplotlib.pyplot as plt
import argparse

from matplotlib.ticker import FormatStrFormatter

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
#utau_ref = utau_ref_channel #FOR TESTING PURPOSES ONLY!
#Specify time steps NOTE: SHOULD BE 27 TO 30 for validation, and all time steps ahead should be the used training steps. The CNN predictions should all originate from these time steps as well!
tstart = 27
tend   = 30

#Extract smagorinsky fluxes, training fluxes (including resolved and total fluxes), CNN fluxes.
#NOTE1:rescale Smagorinsky, training fluxes, and CNN with a representative friction velocity.
smag_tau_xu  = np.array(b['smag_tau_xu'][tstart:tend,:,:,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_yu  = np.array(b['smag_tau_yu'][tstart:tend,:,:,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_zu  = np.array(b['smag_tau_zu'][tstart:tend,:,:,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_xv  = np.array(b['smag_tau_xv'][tstart:tend,:,:,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_yv  = np.array(b['smag_tau_yv'][tstart:tend,:,:,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_zv  = np.array(b['smag_tau_zv'][tstart:tend,:,:,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_xw  = np.array(b['smag_tau_xw'][tstart:tend,:,:,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_yw  = np.array(b['smag_tau_yw'][tstart:tend,:,:,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
smag_tau_zw  = np.array(b['smag_tau_zw'][tstart:tend,:,:,:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
#
unres_tau_xu = np.array(c['unres_tau_xu_turb'] [tstart:tend,:,:,:]) * (utau_ref ** 2) #NOTE: data from training file is dimensionless in contrast to the data from the MLP and Smagorinsky scripts, and consquently the rescaling factor is different!
#NOTE: size of array zw transport component is reduced by 1 in z-direction compared to training file, therefore this compensated by taking slices
unres_tau_yu = np.array(c['unres_tau_yu_turb'] [tstart:tend,:,:,:]) * (utau_ref ** 2) 
unres_tau_zu = np.array(c['unres_tau_zu_turb'] [tstart:tend,:,:,:]) * (utau_ref ** 2) 
unres_tau_xv = np.array(c['unres_tau_xv_turb'] [tstart:tend,:,:,:]) * (utau_ref ** 2) 
unres_tau_yv = np.array(c['unres_tau_yv_turb'] [tstart:tend,:,:,:]) * (utau_ref ** 2) 
unres_tau_zv = np.array(c['unres_tau_zv_turb'] [tstart:tend,:,:,:]) * (utau_ref ** 2) 
unres_tau_xw = np.array(c['unres_tau_xw_turb'] [tstart:tend,:,:,:]) * (utau_ref ** 2) 
unres_tau_yw = np.array(c['unres_tau_yw_turb'] [tstart:tend,:,:,:]) * (utau_ref ** 2) 
unres_tau_zw = np.array(c['unres_tau_zw_turb'] [tstart:tend,1:,:,:])* (utau_ref ** 2) 
res_tau_xu   = np.array(c['res_tau_xu_turb']  [tstart:tend,:,:,:])  * (utau_ref ** 2) 
res_tau_yu   = np.array(c['res_tau_yu_turb']  [tstart:tend,:,:,:])  * (utau_ref ** 2)   
res_tau_zu   = np.array(c['res_tau_zu_turb']  [tstart:tend,:,:,:])  * (utau_ref ** 2)   
res_tau_xv   = np.array(c['res_tau_xv_turb']  [tstart:tend,:,:,:])  * (utau_ref ** 2)   
res_tau_yv   = np.array(c['res_tau_yv_turb']  [tstart:tend,:,:,:])  * (utau_ref ** 2)   
res_tau_zv   = np.array(c['res_tau_zv_turb']  [tstart:tend,:,:,:])  * (utau_ref ** 2)   
res_tau_xw   = np.array(c['res_tau_xw_turb']  [tstart:tend,:,:,:])  * (utau_ref ** 2)   
res_tau_yw   = np.array(c['res_tau_yw_turb']  [tstart:tend,:,:,:])  * (utau_ref ** 2)   
res_tau_zw   = np.array(c['res_tau_zw_turb']  [tstart:tend,1:,:,:]) * (utau_ref ** 2)   
tot_tau_xu   = np.array(c['total_tau_xu_turb'][tstart:tend,:,:,:])  * (utau_ref ** 2)  
tot_tau_yu   = np.array(c['total_tau_yu_turb'][tstart:tend,:,:,:])  * (utau_ref ** 2)
tot_tau_zu   = np.array(c['total_tau_zu_turb'][tstart:tend,:,:,:])  * (utau_ref ** 2)
tot_tau_xv   = np.array(c['total_tau_xv_turb'][tstart:tend,:,:,:])  * (utau_ref ** 2)
tot_tau_yv   = np.array(c['total_tau_yv_turb'][tstart:tend,:,:,:])  * (utau_ref ** 2)
tot_tau_zv   = np.array(c['total_tau_zv_turb'][tstart:tend,:,:,:])  * (utau_ref ** 2)
tot_tau_xw   = np.array(c['total_tau_xw_turb'][tstart:tend,:,:,:])  * (utau_ref ** 2)
tot_tau_yw   = np.array(c['total_tau_yw_turb'][tstart:tend,:,:,:])  * (utau_ref ** 2)
tot_tau_zw   = np.array(c['total_tau_zw_turb'][tstart:tend,1:,:,:]) * (utau_ref ** 2)
#
if args.reconstruct_fields:
    preds_values_xu_upstream   = np.array(a['preds_values_tau_xu_upstream'][:])   * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_xu_upstream    = np.array(a['lbls_values_tau_xu_upstream'][:])    * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_xu_downstream = np.array(a['preds_values_tau_xu_downstream'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_xu_downstream  = np.array(a['lbls_values_tau_xu_downstream'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_yu_upstream   = np.array(a['preds_values_tau_yu_upstream'][:])   * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_yu_upstream    = np.array(a['lbls_values_tau_yu_upstream'][:])    * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_yu_downstream = np.array(a['preds_values_tau_yu_downstream'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_yu_downstream  = np.array(a['lbls_values_tau_yu_downstream'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_zu_upstream   = np.array(a['preds_values_tau_zu_upstream'][:])   * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_zu_upstream    = np.array(a['lbls_values_tau_zu_upstream'][:])    * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_zu_downstream = np.array(a['preds_values_tau_zu_downstream'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_zu_downstream  = np.array(a['lbls_values_tau_zu_downstream'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_xv_upstream   = np.array(a['preds_values_tau_xv_upstream'][:])   * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_xv_upstream    = np.array(a['lbls_values_tau_xv_upstream'][:])    * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_xv_downstream = np.array(a['preds_values_tau_xv_downstream'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_xv_downstream  = np.array(a['lbls_values_tau_xv_downstream'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_yv_upstream   = np.array(a['preds_values_tau_yv_upstream'][:])   * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_yv_upstream    = np.array(a['lbls_values_tau_yv_upstream'][:])    * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_yv_downstream = np.array(a['preds_values_tau_yv_downstream'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_yv_downstream  = np.array(a['lbls_values_tau_yv_downstream'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_zv_upstream   = np.array(a['preds_values_tau_zv_upstream'][:])   * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_zv_upstream    = np.array(a['lbls_values_tau_zv_upstream'][:])    * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_zv_downstream = np.array(a['preds_values_tau_zv_downstream'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_zv_downstream  = np.array(a['lbls_values_tau_zv_downstream'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_xw_upstream   = np.array(a['preds_values_tau_xw_upstream'][:])   * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_xw_upstream    = np.array(a['lbls_values_tau_xw_upstream'][:])    * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_xw_downstream = np.array(a['preds_values_tau_xw_downstream'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_xw_downstream  = np.array(a['lbls_values_tau_xw_downstream'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_yw_upstream = np.array(a['preds_values_tau_yw_upstream'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_yw_upstream  = np.array(a['lbls_values_tau_yw_upstream'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_yw_downstream = np.array(a['preds_values_tau_yw_downstream'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_yw_downstream  = np.array(a['lbls_values_tau_yw_downstream'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_zw_upstream = np.array(a['preds_values_tau_zw_upstream'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_zw_upstream  = np.array(a['lbls_values_tau_zw_upstream'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    preds_values_zw_downstream = np.array(a['preds_values_tau_zw_downstream'][:]) * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    lbls_values_zw_downstream  = np.array(a['lbls_values_tau_zw_downstream'][:])  * ((utau_ref ** 2) / (utau_ref_channel ** 2))
    zhloc_values    = np.array(a['zhloc_samples'][:])
    zloc_values     = np.array(a['zloc_samples'][:])
    yhloc_values    = np.array(a['yhloc_samples'][:])
    yloc_values     = np.array(a['yloc_samples'][:])
    xhloc_values    = np.array(a['xhloc_samples'][:])
    xloc_values     = np.array(a['xloc_samples'][:])
    tstep_values    = np.array(a['tstep_samples'][:]).astype('int') #make sure time steps are stored as integers, not as floats

#Extract coordinates
nt, _, _, _ = unres_tau_xu.shape
zc  = np.array(c['zc'][:])
nz = len(zc)
zhc = np.array(c['zhc'][:])
zgcextra = np.array(c['zgcextra'][:])
yc  = np.array(c['yc'][:])
ny = len(yc)
yhc = np.array(c['yhc'][:])
ygcextra = np.array(c['ygcextra'][:])
xc  = np.array(c['xc'][:])
nx = len(xc)
xhc = np.array(c['xhc'][:])
xgcextra = np.array(c['xgcextra'][:])

#NOTE: commented out code below does not work anymore because additional ghost cells have been added to the used transport components in a non-symmetric way.
##Calculate trace part of subgrid-stress, and subtract this from labels for fair comparison with Smagorinsky fluxes
#trace_train = (unres_tau_xu + unres_tau_yv + unres_tau_zw) * (1./3.)
#unres_tau_xu_traceless = unres_tau_xu - trace_train
#unres_tau_yv_traceless = unres_tau_yv - trace_train
#unres_tau_zw_traceless = unres_tau_zw - trace_train

#Close files
a.close()
b.close()
c.close()

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

    #Extract additional coordinates and time steps
    zhcless = zhc[:-1]
    yhcless = yhc[:-1]
    xhcless = xhc[:-1]
    tstep_unique = np.unique(tstep_values)
    zloc_unique  = np.unique(zloc_values)
    zhloc_unique = np.unique(zhloc_values)
    yloc_unique  = np.unique(yloc_values)
    yhloc_unique = np.unique(yhloc_values)
    xloc_unique  = np.unique(xloc_values)
    xhloc_unique = np.unique(xhloc_values)


    #Create dimensions for storage in nc-file
    d.createDimension("zc", len(zc))
    d.createDimension("zgcextra", len(zgcextra))
    d.createDimension("zhc",len(zhc))
    d.createDimension("zhcless",len(zhcless))
    d.createDimension("yc", len(yc))
    d.createDimension("ygcextra", len(ygcextra))
    d.createDimension("yhc",len(yhc))
    d.createDimension("yhcless",len(yhcless))
    d.createDimension("xc", len(xc))
    d.createDimension("xgcextra", len(xgcextra))
    d.createDimension("xhc",len(xhc))
    d.createDimension("xhcless",len(xhcless))
    d.createDimension("tstep_unique",len(tstep_unique))

    #Create variables for dimensions and store them
    var_zc           = d.createVariable("zc",           "f8", ("zc",))
    var_zgcextra     = d.createVariable("zgcextra",     "f8", ("zgcextra",))
    var_zhc          = d.createVariable("zhc",          "f8", ("zhc",))
    var_zhcless      = d.createVariable("zhcless",      "f8", ("zhcless",))
    var_yc           = d.createVariable("yc",           "f8", ("yc",))
    var_ygcextra     = d.createVariable("ygcextra",     "f8", ("ygcextra",))
    var_yhc          = d.createVariable("yhc",          "f8", ("yhc",))
    var_yhcless      = d.createVariable("yhcless",      "f8", ("yhcless",))
    var_xc           = d.createVariable("xc",           "f8", ("xc",))
    var_xgcextra     = d.createVariable("xgcextra",     "f8", ("xgcextra",))
    var_xhc          = d.createVariable("xhc",          "f8", ("xhc",))
    var_xhcless      = d.createVariable("xhcless",      "f8", ("xhcless",))
    var_tstep_unique = d.createVariable("tstep_unique", "f8", ("tstep_unique",))

    var_zc[:]            = zc
    var_zgcextra[:]      = zgcextra
    var_zhc[:]           = zhc
    var_zhcless[:]       = zhcless
    var_yc[:]            = yc
    var_ygcextra[:]      = ygcextra
    var_yhc[:]           = yhc
    var_yhcless[:]       = yhcless
    var_xc[:]            = xc
    var_xgcextra[:]      = xgcextra
    var_xhc[:]           = xhc
    var_xhcless[:]       = xhcless
    var_tstep_unique[:]  = tstep_unique

    #Create variables for storage labels
    var_unres_tau_xu_lbls = d.createVariable("unres_tau_xu_lbls","f8",("tstep_unique","zc","yc","xgcextra"))
    var_unres_tau_xv_lbls = d.createVariable("unres_tau_xv_lbls","f8",("tstep_unique","zc","yhcless","xhc"))
    var_unres_tau_xw_lbls = d.createVariable("unres_tau_xw_lbls","f8",("tstep_unique","zhcless","yc","xhc"))
    var_unres_tau_yu_lbls = d.createVariable("unres_tau_yu_lbls","f8",("tstep_unique","zc","yhc","xhcless"))
    var_unres_tau_yv_lbls = d.createVariable("unres_tau_yv_lbls","f8",("tstep_unique","zc","ygcextra","xc"))
    var_unres_tau_yw_lbls = d.createVariable("unres_tau_yw_lbls","f8",("tstep_unique","zhcless","yhc","xc"))
    var_unres_tau_zu_lbls = d.createVariable("unres_tau_zu_lbls","f8",("tstep_unique","zhc","yc","xhcless"))
    var_unres_tau_zv_lbls = d.createVariable("unres_tau_zv_lbls","f8",("tstep_unique","zhc","yhcless","xc"))
    var_unres_tau_zw_lbls = d.createVariable("unres_tau_zw_lbls","f8",("tstep_unique","zc","yc","xc"))
    #
    var_unres_tau_xu_lbls_upstream = d.createVariable("unres_tau_xu_lbls_upstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv_lbls_upstream = d.createVariable("unres_tau_xv_lbls_upstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_xw_lbls_upstream = d.createVariable("unres_tau_xw_lbls_upstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_yu_lbls_upstream = d.createVariable("unres_tau_yu_lbls_upstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv_lbls_upstream = d.createVariable("unres_tau_yv_lbls_upstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_lbls_upstream = d.createVariable("unres_tau_yw_lbls_upstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu_lbls_upstream = d.createVariable("unres_tau_zu_lbls_upstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_zv_lbls_upstream = d.createVariable("unres_tau_zv_lbls_upstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zw_lbls_upstream = d.createVariable("unres_tau_zw_lbls_upstream","f8",("tstep_unique","zc","yc","xc"))
    #
    var_unres_tau_xu_lbls_downstream = d.createVariable("unres_tau_xu_lbls_downstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv_lbls_downstream = d.createVariable("unres_tau_xv_lbls_downstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_xw_lbls_downstream = d.createVariable("unres_tau_xw_lbls_downstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_yu_lbls_downstream = d.createVariable("unres_tau_yu_lbls_downstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv_lbls_downstream = d.createVariable("unres_tau_yv_lbls_downstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_lbls_downstream = d.createVariable("unres_tau_yw_lbls_downstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu_lbls_downstream = d.createVariable("unres_tau_zu_lbls_downstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_zv_lbls_downstream = d.createVariable("unres_tau_zv_lbls_downstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zw_lbls_downstream = d.createVariable("unres_tau_zw_lbls_downstream","f8",("tstep_unique","zc","yc","xc"))
    #

    #Call function to recontruct fields of labels for all nine components and both upstream/downstream components
    print('start reconstructing labels')
    unres_tau_xu_lbls_upstream = reconstruct_field(lbls_values_xu_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    unres_tau_xu_lbls_downstream = reconstruct_field(lbls_values_xu_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('first component done')
    unres_tau_xv_lbls_upstream = reconstruct_field(lbls_values_xv_upstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    unres_tau_xv_lbls_downstream = reconstruct_field(lbls_values_xv_downstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('second component done')
    unres_tau_xw_lbls_upstream = reconstruct_field(lbls_values_xw_upstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique)
    unres_tau_xw_lbls_downstream = reconstruct_field(lbls_values_xw_downstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique)
    print('third component done')
    unres_tau_yu_lbls_upstream = reconstruct_field(lbls_values_yu_upstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    unres_tau_yu_lbls_downstream = reconstruct_field(lbls_values_yu_downstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fourth component done')
    unres_tau_yv_lbls_upstream = reconstruct_field(lbls_values_yv_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    unres_tau_yv_lbls_downstream = reconstruct_field(lbls_values_yv_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fifth component done')
    unres_tau_yw_lbls_upstream = reconstruct_field(lbls_values_yw_upstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    unres_tau_yw_lbls_downstream = reconstruct_field(lbls_values_yw_downstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('sixth component done')
    unres_tau_zu_lbls_upstream = reconstruct_field(lbls_values_zu_upstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    unres_tau_zu_lbls_downstream = reconstruct_field(lbls_values_zu_downstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('seventh component done')
    unres_tau_zv_lbls_upstream = reconstruct_field(lbls_values_zv_upstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    unres_tau_zv_lbls_downstream = reconstruct_field(lbls_values_zv_downstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('eighth component done')
    unres_tau_zw_lbls_upstream = reconstruct_field(lbls_values_zw_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    unres_tau_zw_lbls_downstream = reconstruct_field(lbls_values_zw_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('nineth component done')

    #Create variables for storage reconstructed fields of predictions
    var_unres_tau_xu_CNN = d.createVariable("unres_tau_xu_CNN","f8",("tstep_unique","zc","yc","xgcextra"))      
    var_unres_tau_xv_CNN = d.createVariable("unres_tau_xv_CNN","f8",("tstep_unique","zc","yhcless","xhc"))     
    var_unres_tau_xw_CNN = d.createVariable("unres_tau_xw_CNN","f8",("tstep_unique","zhcless","yc","xhc"))     
    var_unres_tau_yu_CNN = d.createVariable("unres_tau_yu_CNN","f8",("tstep_unique","zc","yhc","xhcless"))
    var_unres_tau_yv_CNN = d.createVariable("unres_tau_yv_CNN","f8",("tstep_unique","zc","ygcextra","xc"))
    var_unres_tau_yw_CNN = d.createVariable("unres_tau_yw_CNN","f8",("tstep_unique","zhcless","yhc","xc"))
    var_unres_tau_zu_CNN = d.createVariable("unres_tau_zu_CNN","f8",("tstep_unique","zhc","yc","xhcless"))
    var_unres_tau_zv_CNN = d.createVariable("unres_tau_zv_CNN","f8",("tstep_unique","zhc","yhcless","xc"))
    var_unres_tau_zw_CNN = d.createVariable("unres_tau_zw_CNN","f8",("tstep_unique","zc","yc","xc")) 
    #
    var_unres_tau_xu_CNN_upstream = d.createVariable("unres_tau_xu_CNN_upstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv_CNN_upstream = d.createVariable("unres_tau_xv_CNN_upstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_xw_CNN_upstream = d.createVariable("unres_tau_xw_CNN_upstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_yu_CNN_upstream = d.createVariable("unres_tau_yu_CNN_upstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv_CNN_upstream = d.createVariable("unres_tau_yv_CNN_upstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_CNN_upstream = d.createVariable("unres_tau_yw_CNN_upstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu_CNN_upstream = d.createVariable("unres_tau_zu_CNN_upstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_zv_CNN_upstream = d.createVariable("unres_tau_zv_CNN_upstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zw_CNN_upstream = d.createVariable("unres_tau_zw_CNN_upstream","f8",("tstep_unique","zc","yc","xc"))
    #
    var_unres_tau_xu_CNN_downstream = d.createVariable("unres_tau_xu_CNN_downstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv_CNN_downstream = d.createVariable("unres_tau_xv_CNN_downstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_xw_CNN_downstream = d.createVariable("unres_tau_xw_CNN_downstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_yu_CNN_downstream = d.createVariable("unres_tau_yu_CNN_downstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv_CNN_downstream = d.createVariable("unres_tau_yv_CNN_downstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_CNN_downstream = d.createVariable("unres_tau_yw_CNN_downstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu_CNN_downstream = d.createVariable("unres_tau_zu_CNN_downstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_zv_CNN_downstream = d.createVariable("unres_tau_zv_CNN_downstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zw_CNN_downstream = d.createVariable("unres_tau_zw_CNN_downstream","f8",("tstep_unique","zc","yc","xc"))
    #

    #Call function to recontruct fields of predictions for all nine components
    print('start reconstructing predictions')
    preds_values_xu_upstream = reconstruct_field(preds_values_xu_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    preds_values_xu_downstream = reconstruct_field(preds_values_xu_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('first component done')
    preds_values_xv_upstream = reconstruct_field(preds_values_xv_upstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    preds_values_xv_downstream = reconstruct_field(preds_values_xv_downstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('second component done')
    preds_values_xw_upstream = reconstruct_field(preds_values_xw_upstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique)
    preds_values_xw_downstream = reconstruct_field(preds_values_xw_downstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique)
    print('third component done')
    preds_values_yu_upstream = reconstruct_field(preds_values_yu_upstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    preds_values_yu_downstream = reconstruct_field(preds_values_yu_downstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fourth component done')
    preds_values_yv_upstream = reconstruct_field(preds_values_yv_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    preds_values_yv_downstream = reconstruct_field(preds_values_yv_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('fifth component done')
    preds_values_yw_upstream = reconstruct_field(preds_values_yw_upstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    preds_values_yw_downstream = reconstruct_field(preds_values_yw_downstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('sixth component done')
    preds_values_zu_upstream = reconstruct_field(preds_values_zu_upstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    preds_values_zu_downstream = reconstruct_field(preds_values_zu_downstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('seventh component done')
    preds_values_zv_upstream = reconstruct_field(preds_values_zv_upstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    preds_values_zv_downstream = reconstruct_field(preds_values_zv_downstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('eighth component done')
    preds_values_zw_upstream = reconstruct_field(preds_values_zw_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    preds_values_zw_downstream = reconstruct_field(preds_values_zw_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('nineth component done')
    
    #Combine upstream and downstream components together, where at each height the upstream and downstream components are selected alternately (as should be done during inference).
    #NOTE: last input value is a boolean flag for the zw turbulent transport component, because this component needs special treatment to allow for symmetric inference.
    print('Start combining upstream and downstream components.')
    def _combine_upstream_downstream(upstream_field, downstream_field, sample_dir, dims, zw_flag = False):
        
        #Choose sample indices and output shape based on specified sample_dir
        if not (sample_dir == 'x' or sample_dir == 'y' or sample_dir == 'z'):
            raise RuntimeError("The specified sampling direction should be 'x', 'y', or 'z'.")
        
        if sample_dir   == 'x':
            dims[3] = dims[3] + 1
            sample_dim = dims[3]
            dim1_ind = np.arange(dims[1])
            dim2_ind = np.arange(dims[2])
        elif sample_dir == 'y':
            dims[2] = dims[2] + 1
            sample_dim = dims[2]
            dim1_ind = np.arange(dims[1])
            dim2_ind = np.arange(dims[3])
        else:
            if not zw_flag:
                dims[1] = dims[1] + 1 #zw transport component does not have one additional grid cell in sample_dir
            sample_dim = dims[1]
            dim1_ind = np.arange(dims[2])
            dim2_ind = np.arange(dims[3])

        combined_field = np.zeros((dims[0],dims[1],dims[2],dims[3]))
        #Define where the sum of the indices for x- and y-direction are odd, such that the fields can be alternately stored in specified sample_dir.
        dim1cor, dim2cor  = np.meshgrid(dim2_ind, dim1_ind)
        twodimcor         = dim1cor + dim2cor
        twodimcor_odd     = twodimcor % 2 != 0
        twodimcor_odd     = np.tile(twodimcor_odd, (nt,1,1))
        twodimcor_even    = twodimcor % 2 == 0
        twodimcor_even    = np.tile(twodimcor_even, (nt,1,1))

        for i in range(sample_dim):
            #Determine index slices for upstream and downstream fields
            if sample_dir == 'x':
                indices_up   = np.s_[:,:,:,i]
                indices_down = np.s_[:,:,:,i-1]
            elif sample_dir == 'y':
                indices_up   = np.s_[:,:,i,:]
                indices_down = np.s_[:,:,i-1,:]
            elif zw_flag: #Compensate for reduction of sample_dir in combined array: the stored upstream and downstream components are not reduced
                indices_up   = np.s_[:,i+1,:,:]
                indices_down = np.s_[:,i,:,:]
            else:
                indices_up   = np.s_[:,i,:,:]
                indices_down = np.s_[:,i-1,:,:]

            #Determine index slice for combined field (depends only on zw_flag), and compensate for shifted indices in the alternate sampling
            if zw_flag:
                indices_combined = indices_down
                twodimcor_odd2   = twodimcor_even
                twodimcor_even2  = twodimcor_odd
            else:
                twodimcor_odd2   = twodimcor_odd
                twodimcor_even2  = twodimcor_even
                indices_combined = indices_up
            
            #Assign upstream and downstream values to combined field
            if i == 0:
                combined_field[indices_combined] = upstream_field[indices_up] 
            elif i == (sample_dim-1):
                combined_field[indices_combined] = downstream_field[indices_down]
            
            #Make distinction between i=even and i=odd to alternate storage
            elif i % 2 == 0: #i is even
                combined_field[indices_combined][twodimcor_odd2]  = upstream_field[indices_up][twodimcor_odd2]
                combined_field[indices_combined][twodimcor_even2] = downstream_field[indices_down][twodimcor_even2]
            elif i % 2 != 0: #i is odd
                combined_field[indices_combined][twodimcor_even2] = upstream_field[indices_up][twodimcor_even2]
                combined_field[indices_combined][twodimcor_odd2]  = downstream_field[indices_down][twodimcor_odd2]
            else:
                raise RuntimeError("Error occured in script as this line should not have been reached. Check carefully for bugs.")
        return combined_field

    #Call function above for all preds and labels
    preds_values_xu =  _combine_upstream_downstream(preds_values_xu_upstream, preds_values_xu_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x')
    preds_values_yu =  _combine_upstream_downstream(preds_values_yu_upstream, preds_values_yu_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y')
    preds_values_zu =  _combine_upstream_downstream(preds_values_zu_upstream, preds_values_zu_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z')
    preds_values_xv =  _combine_upstream_downstream(preds_values_xv_upstream, preds_values_xv_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x')
    preds_values_yv =  _combine_upstream_downstream(preds_values_yv_upstream, preds_values_yv_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y')
    preds_values_zv =  _combine_upstream_downstream(preds_values_zv_upstream, preds_values_zv_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z')
    preds_values_xw =  _combine_upstream_downstream(preds_values_xw_upstream, preds_values_xw_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x')
    preds_values_yw =  _combine_upstream_downstream(preds_values_yw_upstream, preds_values_yw_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y')
    preds_values_zw =  _combine_upstream_downstream(preds_values_zw_upstream, preds_values_zw_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z', zw_flag = True)
    #
    unres_tau_xu_lbls =  _combine_upstream_downstream(unres_tau_xu_lbls_upstream, unres_tau_xu_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x')
    unres_tau_yu_lbls =  _combine_upstream_downstream(unres_tau_yu_lbls_upstream, unres_tau_yu_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y')
    unres_tau_zu_lbls =  _combine_upstream_downstream(unres_tau_zu_lbls_upstream, unres_tau_zu_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z')
    unres_tau_xv_lbls =  _combine_upstream_downstream(unres_tau_xv_lbls_upstream, unres_tau_xv_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x')
    unres_tau_yv_lbls =  _combine_upstream_downstream(unres_tau_yv_lbls_upstream, unres_tau_yv_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y')
    unres_tau_zv_lbls =  _combine_upstream_downstream(unres_tau_zv_lbls_upstream, unres_tau_zv_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z')
    unres_tau_xw_lbls =  _combine_upstream_downstream(unres_tau_xw_lbls_upstream, unres_tau_xw_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x')
    unres_tau_yw_lbls =  _combine_upstream_downstream(unres_tau_yw_lbls_upstream, unres_tau_yw_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y')
    unres_tau_zw_lbls =  _combine_upstream_downstream(unres_tau_zw_lbls_upstream, unres_tau_zw_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z', zw_flag = True)

    #Store variables in netCDF file
    #Labels
    var_unres_tau_xu_lbls[:,:,:,:] = unres_tau_xu_lbls[:,:,:,:]
    var_unres_tau_xv_lbls[:,:,:,:] = unres_tau_xv_lbls[:,:,:,:]
    var_unres_tau_xw_lbls[:,:,:,:] = unres_tau_xw_lbls[:,:,:,:]
    var_unres_tau_yu_lbls[:,:,:,:] = unres_tau_yu_lbls[:,:,:,:]
    var_unres_tau_yv_lbls[:,:,:,:] = unres_tau_yv_lbls[:,:,:,:]
    var_unres_tau_yw_lbls[:,:,:,:] = unres_tau_yw_lbls[:,:,:,:]
    var_unres_tau_zu_lbls[:,:,:,:] = unres_tau_zu_lbls[:,:,:,:]
    var_unres_tau_zv_lbls[:,:,:,:] = unres_tau_zv_lbls[:,:,:,:]
    var_unres_tau_zw_lbls[:,:,:,:] = unres_tau_zw_lbls[:,:,:,:]
    #
    var_unres_tau_xu_lbls_upstream[:,:,:,:] = unres_tau_xu_lbls_upstream[:,:,:,:]
    var_unres_tau_xv_lbls_upstream[:,:,:,:] = unres_tau_xv_lbls_upstream[:,:,:,:]
    var_unres_tau_xw_lbls_upstream[:,:,:,:] = unres_tau_xw_lbls_upstream[:,:,:,:]
    var_unres_tau_yu_lbls_upstream[:,:,:,:] = unres_tau_yu_lbls_upstream[:,:,:,:]
    var_unres_tau_yv_lbls_upstream[:,:,:,:] = unres_tau_yv_lbls_upstream[:,:,:,:]
    var_unres_tau_yw_lbls_upstream[:,:,:,:] = unres_tau_yw_lbls_upstream[:,:,:,:]
    var_unres_tau_zu_lbls_upstream[:,:,:,:] = unres_tau_zu_lbls_upstream[:,:,:,:]
    var_unres_tau_zv_lbls_upstream[:,:,:,:] = unres_tau_zv_lbls_upstream[:,:,:,:]
    var_unres_tau_zw_lbls_upstream[:,:,:,:] = unres_tau_zw_lbls_upstream[:,:,:,:]
    #
    var_unres_tau_xu_lbls_downstream[:,:,:,:] = unres_tau_xu_lbls_downstream[:,:,:,:]
    var_unres_tau_xv_lbls_downstream[:,:,:,:] = unres_tau_xv_lbls_downstream[:,:,:,:]
    var_unres_tau_xw_lbls_downstream[:,:,:,:] = unres_tau_xw_lbls_downstream[:,:,:,:]
    var_unres_tau_yu_lbls_downstream[:,:,:,:] = unres_tau_yu_lbls_downstream[:,:,:,:]
    var_unres_tau_yv_lbls_downstream[:,:,:,:] = unres_tau_yv_lbls_downstream[:,:,:,:]
    var_unres_tau_yw_lbls_downstream[:,:,:,:] = unres_tau_yw_lbls_downstream[:,:,:,:]
    var_unres_tau_zu_lbls_downstream[:,:,:,:] = unres_tau_zu_lbls_downstream[:,:,:,:]
    var_unres_tau_zv_lbls_downstream[:,:,:,:] = unres_tau_zv_lbls_downstream[:,:,:,:]
    var_unres_tau_zw_lbls_downstream[:,:,:,:] = unres_tau_zw_lbls_downstream[:,:,:,:]
    #Predictions
    var_unres_tau_xu_CNN[:,:,:,:] = preds_values_xu[:,:,:,:]
    var_unres_tau_xv_CNN[:,:,:,:] = preds_values_xv[:,:,:,:]
    var_unres_tau_xw_CNN[:,:,:,:] = preds_values_xw[:,:,:,:]
    var_unres_tau_yu_CNN[:,:,:,:] = preds_values_yu[:,:,:,:]
    var_unres_tau_yv_CNN[:,:,:,:] = preds_values_yv[:,:,:,:]
    var_unres_tau_yw_CNN[:,:,:,:] = preds_values_yw[:,:,:,:]
    var_unres_tau_zu_CNN[:,:,:,:] = preds_values_zu[:,:,:,:]
    var_unres_tau_zv_CNN[:,:,:,:] = preds_values_zv[:,:,:,:]
    var_unres_tau_zw_CNN[:,:,:,:] = preds_values_zw[:,:,:,:]
    #
    var_unres_tau_xu_CNN_upstream[:,:,:,:] = preds_values_xu_upstream[:,:,:,:]
    var_unres_tau_xv_CNN_upstream[:,:,:,:] = preds_values_xv_upstream[:,:,:,:]
    var_unres_tau_xw_CNN_upstream[:,:,:,:] = preds_values_xw_upstream[:,:,:,:]
    var_unres_tau_yu_CNN_upstream[:,:,:,:] = preds_values_yu_upstream[:,:,:,:]
    var_unres_tau_yv_CNN_upstream[:,:,:,:] = preds_values_yv_upstream[:,:,:,:]
    var_unres_tau_yw_CNN_upstream[:,:,:,:] = preds_values_yw_upstream[:,:,:,:]
    var_unres_tau_zu_CNN_upstream[:,:,:,:] = preds_values_zu_upstream[:,:,:,:]
    var_unres_tau_zv_CNN_upstream[:,:,:,:] = preds_values_zv_upstream[:,:,:,:]
    var_unres_tau_zw_CNN_upstream[:,:,:,:] = preds_values_zw_upstream[:,:,:,:]
    #
    var_unres_tau_xu_CNN_downstream[:,:,:,:] = preds_values_xu_downstream[:,:,:,:]
    var_unres_tau_xv_CNN_downstream[:,:,:,:] = preds_values_xv_downstream[:,:,:,:]
    var_unres_tau_xw_CNN_downstream[:,:,:,:] = preds_values_xw_downstream[:,:,:,:]
    var_unres_tau_yu_CNN_downstream[:,:,:,:] = preds_values_yu_downstream[:,:,:,:]
    var_unres_tau_yv_CNN_downstream[:,:,:,:] = preds_values_yv_downstream[:,:,:,:]
    var_unres_tau_yw_CNN_downstream[:,:,:,:] = preds_values_yw_downstream[:,:,:,:]
    var_unres_tau_zu_CNN_downstream[:,:,:,:] = preds_values_zu_downstream[:,:,:,:]
    var_unres_tau_zv_CNN_downstream[:,:,:,:] = preds_values_zv_downstream[:,:,:,:]
    var_unres_tau_zw_CNN_downstream[:,:,:,:] = preds_values_zw_downstream[:,:,:,:]
    var_unres_tau_zw_CNN_downstream[:,:,:,:] = preds_values_zw_downstream[:,:,:,:]
    
    
    #Create variables for storage unresolved, resolved, and total transports
    #var_unres_tau_xu_traceless = d.createVariable("unres_tau_xu_traceless","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    #var_unres_tau_yv_traceless = d.createVariable("unres_tau_yv_traceless","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    #var_unres_tau_zw_traceless = d.createVariable("unres_tau_zw_traceless","f8",("tstep_unique","zloc_unique","yloc_unique","xloc_unique"))
    #
    var_unres_tau_xu = d.createVariable("unres_tau_xu","f8",("tstep_unique","zc","yc","xgcextra"))
    var_unres_tau_xv = d.createVariable("unres_tau_xv","f8",("tstep_unique","zc","yhc","xhc"))
    var_unres_tau_xw = d.createVariable("unres_tau_xw","f8",("tstep_unique","zhc","yc","xhc"))
    var_unres_tau_yu = d.createVariable("unres_tau_yu","f8",("tstep_unique","zc","yhc","xhc"))
    var_unres_tau_yv = d.createVariable("unres_tau_yv","f8",("tstep_unique","zc","ygcextra","xc"))
    var_unres_tau_yw = d.createVariable("unres_tau_yw","f8",("tstep_unique","zhc","yhc","xc"))
    var_unres_tau_zu = d.createVariable("unres_tau_zu","f8",("tstep_unique","zhc","yc","xhc"))
    var_unres_tau_zv = d.createVariable("unres_tau_zv","f8",("tstep_unique","zhc","yhc","xc"))
    var_unres_tau_zw = d.createVariable("unres_tau_zw","f8",("tstep_unique","zc","yc","xc"))
    var_res_tau_xu   = d.createVariable("res_tau_xu"  ,"f8",("tstep_unique","zc","yc","xgcextra"))
    var_res_tau_xv   = d.createVariable("res_tau_xv"  ,"f8",("tstep_unique","zc","yhc","xhc"))
    var_res_tau_xw   = d.createVariable("res_tau_xw"  ,"f8",("tstep_unique","zhc","yc","xhc"))
    var_res_tau_yu   = d.createVariable("res_tau_yu"  ,"f8",("tstep_unique","zc","yhc","xhc"))
    var_res_tau_yv   = d.createVariable("res_tau_yv"  ,"f8",("tstep_unique","zc","ygcextra","xc"))
    var_res_tau_yw   = d.createVariable("res_tau_yw"  ,"f8",("tstep_unique","zhc","yhc","xc"))
    var_res_tau_zu   = d.createVariable("res_tau_zu"  ,"f8",("tstep_unique","zhc","yc","xhc"))
    var_res_tau_zv   = d.createVariable("res_tau_zv"  ,"f8",("tstep_unique","zhc","yhc","xc"))
    var_res_tau_zw   = d.createVariable("res_tau_zw"  ,"f8",("tstep_unique","zc","yc","xc"))
    var_tot_tau_xu   = d.createVariable("tot_tau_xu"  ,"f8",("tstep_unique","zc","yc","xgcextra"))
    var_tot_tau_xv   = d.createVariable("tot_tau_xv"  ,"f8",("tstep_unique","zc","yhc","xhc"))
    var_tot_tau_xw   = d.createVariable("tot_tau_xw"  ,"f8",("tstep_unique","zhc","yc","xhc"))
    var_tot_tau_yu   = d.createVariable("tot_tau_yu"  ,"f8",("tstep_unique","zc","yhc","xhc"))
    var_tot_tau_yv   = d.createVariable("tot_tau_yv"  ,"f8",("tstep_unique","zc","ygcextra","xc"))
    var_tot_tau_yw   = d.createVariable("tot_tau_yw"  ,"f8",("tstep_unique","zhc","yhc","xc"))
    var_tot_tau_zu   = d.createVariable("tot_tau_zu"  ,"f8",("tstep_unique","zhc","yc","xhc"))
    var_tot_tau_zv   = d.createVariable("tot_tau_zv"  ,"f8",("tstep_unique","zhc","yhc","xc"))
    var_tot_tau_zw   = d.createVariable("tot_tau_zw"  ,"f8",("tstep_unique","zc","yc","xc"))
    #
    var_unres_tau_xu_smag = d.createVariable("unres_tau_xu_smag","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv_smag = d.createVariable("unres_tau_xv_smag","f8",("tstep_unique","zc","yhc","xhc"))
    var_unres_tau_xw_smag = d.createVariable("unres_tau_xw_smag","f8",("tstep_unique","zhc","yc","xhc"))
    var_unres_tau_yu_smag = d.createVariable("unres_tau_yu_smag","f8",("tstep_unique","zc","yhc","xhc"))
    var_unres_tau_yv_smag = d.createVariable("unres_tau_yv_smag","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_smag = d.createVariable("unres_tau_yw_smag","f8",("tstep_unique","zhc","yhc","xc"))
    var_unres_tau_zu_smag = d.createVariable("unres_tau_zu_smag","f8",("tstep_unique","zhc","yc","xhc"))
    var_unres_tau_zv_smag = d.createVariable("unres_tau_zv_smag","f8",("tstep_unique","zhc","yhc","xc"))
    var_unres_tau_zw_smag = d.createVariable("unres_tau_zw_smag","f8",("tstep_unique","zc","yc","xc"))
    var_tot_tau_xu_smag   = d.createVariable("tot_tau_xu_smag"  ,"f8",("tstep_unique","zc","yc","xc"))
    var_tot_tau_xv_smag   = d.createVariable("tot_tau_xv_smag"  ,"f8",("tstep_unique","zc","yhc","xhc"))
    var_tot_tau_xw_smag   = d.createVariable("tot_tau_xw_smag"  ,"f8",("tstep_unique","zhc","yc","xhc"))
    var_tot_tau_yu_smag   = d.createVariable("tot_tau_yu_smag"  ,"f8",("tstep_unique","zc","yhc","xhc"))
    var_tot_tau_yv_smag   = d.createVariable("tot_tau_yv_smag"  ,"f8",("tstep_unique","zc","yc","xc"))
    var_tot_tau_yw_smag   = d.createVariable("tot_tau_yw_smag"  ,"f8",("tstep_unique","zhc","yhc","xc"))
    var_tot_tau_zu_smag   = d.createVariable("tot_tau_zu_smag"  ,"f8",("tstep_unique","zhc","yc","xhc"))
    var_tot_tau_zv_smag   = d.createVariable("tot_tau_zv_smag"  ,"f8",("tstep_unique","zhc","yhc","xc"))
    var_tot_tau_zw_smag   = d.createVariable("tot_tau_zw_smag"  ,"f8",("tstep_unique","zc","yc","xc"))
    #
    var_tot_tau_xu_CNN   = d.createVariable("tot_tau_xu_CNN"  ,"f8",("tstep_unique","zc","yc","xgcextra"))
    var_tot_tau_xv_CNN   = d.createVariable("tot_tau_xv_CNN"  ,"f8",("tstep_unique","zc","yhcless","xhc"))
    var_tot_tau_xw_CNN   = d.createVariable("tot_tau_xw_CNN"  ,"f8",("tstep_unique","zhcless","yc","xhc"))
    var_tot_tau_yu_CNN   = d.createVariable("tot_tau_yu_CNN"  ,"f8",("tstep_unique","zc","yhc","xhcless"))
    var_tot_tau_yv_CNN   = d.createVariable("tot_tau_yv_CNN"  ,"f8",("tstep_unique","zc","ygcextra","xc"))
    var_tot_tau_yw_CNN   = d.createVariable("tot_tau_yw_CNN"  ,"f8",("tstep_unique","zhcless","yhc","xc"))
    var_tot_tau_zu_CNN   = d.createVariable("tot_tau_zu_CNN"  ,"f8",("tstep_unique","zhc","yc","xhcless"))
    var_tot_tau_zv_CNN   = d.createVariable("tot_tau_zv_CNN"  ,"f8",("tstep_unique","zhc","yhcless","xc"))
    var_tot_tau_zw_CNN   = d.createVariable("tot_tau_zw_CNN"  ,"f8",("tstep_unique","zc","yc","xc"))

    #Store values for unresolved, resolved, and total fluxes training data, Smagorinsky, and CNN
    #var_unres_tau_xu_traceless[:,:,:,:] = unres_tau_xu_traceless
    #var_unres_tau_yv_traceless[:,:,:,:] = unres_tau_yv_traceless
    #var_unres_tau_zw_traceless[:,:,:,:] = unres_tau_zw_traceless
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
    var_tot_tau_xu_smag[:,:,:,:]   = smag_tau_xu + res_tau_xu[:,:,:,1:]
    var_tot_tau_xv_smag[:,:,:,:]   = smag_tau_xv + res_tau_xv
    var_tot_tau_xw_smag[:,:,:,:]   = smag_tau_xw + res_tau_xw
    var_tot_tau_yu_smag[:,:,:,:]   = smag_tau_yu + res_tau_yu
    var_tot_tau_yv_smag[:,:,:,:]   = smag_tau_yv + res_tau_yv[:,:,1:,:]
    var_tot_tau_yw_smag[:,:,:,:]   = smag_tau_yw + res_tau_yw
    var_tot_tau_zu_smag[:,:,:,:]   = smag_tau_zu + res_tau_zu
    var_tot_tau_zv_smag[:,:,:,:]   = smag_tau_zv + res_tau_zv
    var_tot_tau_zw_smag[:,:,:,:]   = smag_tau_zw + res_tau_zw #res_tau_zw[:,1:,:,:] Uncomment this when zgcextra is included in res_tau_zw
    #NOTE: compensate for missing ghost cells in CNN preds
    var_tot_tau_xu_CNN[:,:,:,:]    = preds_values_xu  + res_tau_xu
    var_tot_tau_xv_CNN[:,:,:,:]    = preds_values_xv  + res_tau_xv[:,:,:-1,:]
    var_tot_tau_xw_CNN[:,:,:,:]    = preds_values_xw  + res_tau_xw[:,:-1,:,:]
    var_tot_tau_yu_CNN[:,:,:,:]    = preds_values_yu  + res_tau_yu[:,:,:,:-1]
    var_tot_tau_yv_CNN[:,:,:,:]    = preds_values_yv  + res_tau_yv
    var_tot_tau_yw_CNN[:,:,:,:]    = preds_values_yw  + res_tau_yw[:,:-1,:,:]
    var_tot_tau_zu_CNN[:,:,:,:]    = preds_values_zu  + res_tau_zu[:,:,:,:-1]
    var_tot_tau_zv_CNN[:,:,:,:]    = preds_values_zv  + res_tau_zv[:,:,:-1,:]
    var_tot_tau_zw_CNN[:,:,:,:]    = preds_values_zw  + res_tau_zw

    #Create variables for storage horizontal averages
    var_unres_tau_xu_horavg = d.createVariable("unres_tau_xu_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xv_horavg = d.createVariable("unres_tau_xv_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xw_horavg = d.createVariable("unres_tau_xw_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_yu_horavg = d.createVariable("unres_tau_yu_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yv_horavg = d.createVariable("unres_tau_yv_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yw_horavg = d.createVariable("unres_tau_yw_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zu_horavg = d.createVariable("unres_tau_zu_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zv_horavg = d.createVariable("unres_tau_zv_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zw_horavg = d.createVariable("unres_tau_zw_horavg","f8",("tstep_unique","zc"))
    var_res_tau_xu_horavg   = d.createVariable("res_tau_xu_horavg","f8",  ("tstep_unique","zc"))
    var_res_tau_xv_horavg   = d.createVariable("res_tau_xv_horavg","f8",  ("tstep_unique","zc"))
    var_res_tau_xw_horavg   = d.createVariable("res_tau_xw_horavg","f8",  ("tstep_unique","zhc"))
    var_res_tau_yu_horavg   = d.createVariable("res_tau_yu_horavg","f8",  ("tstep_unique","zc"))
    var_res_tau_yv_horavg   = d.createVariable("res_tau_yv_horavg","f8",  ("tstep_unique","zc"))
    var_res_tau_yw_horavg   = d.createVariable("res_tau_yw_horavg","f8",  ("tstep_unique","zhc"))
    var_res_tau_zu_horavg   = d.createVariable("res_tau_zu_horavg","f8",  ("tstep_unique","zhc"))
    var_res_tau_zv_horavg   = d.createVariable("res_tau_zv_horavg","f8",  ("tstep_unique","zhc"))
    var_res_tau_zw_horavg   = d.createVariable("res_tau_zw_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_xu_horavg   = d.createVariable("tot_tau_xu_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_xv_horavg   = d.createVariable("tot_tau_xv_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_xw_horavg   = d.createVariable("tot_tau_xw_horavg","f8",  ("tstep_unique","zhc"))
    var_tot_tau_yu_horavg   = d.createVariable("tot_tau_yu_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_yv_horavg   = d.createVariable("tot_tau_yv_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_yw_horavg   = d.createVariable("tot_tau_yw_horavg","f8",  ("tstep_unique","zhc"))
    var_tot_tau_zu_horavg   = d.createVariable("tot_tau_zu_horavg","f8",  ("tstep_unique","zhc"))
    var_tot_tau_zv_horavg   = d.createVariable("tot_tau_zv_horavg","f8",  ("tstep_unique","zhc"))
    var_tot_tau_zw_horavg   = d.createVariable("tot_tau_zw_horavg","f8",  ("tstep_unique","zc"))
    #
    var_unres_tau_xu_smag_horavg = d.createVariable("unres_tau_xu_smag_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xv_smag_horavg = d.createVariable("unres_tau_xv_smag_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xw_smag_horavg = d.createVariable("unres_tau_xw_smag_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_yu_smag_horavg = d.createVariable("unres_tau_yu_smag_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yv_smag_horavg = d.createVariable("unres_tau_yv_smag_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yw_smag_horavg = d.createVariable("unres_tau_yw_smag_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zu_smag_horavg = d.createVariable("unres_tau_zu_smag_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zv_smag_horavg = d.createVariable("unres_tau_zv_smag_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zw_smag_horavg = d.createVariable("unres_tau_zw_smag_horavg","f8",("tstep_unique","zc"))
    var_tot_tau_xu_smag_horavg   = d.createVariable("tot_tau_xu_smag_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_xv_smag_horavg   = d.createVariable("tot_tau_xv_smag_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_xw_smag_horavg   = d.createVariable("tot_tau_xw_smag_horavg","f8",  ("tstep_unique","zhc"))
    var_tot_tau_yu_smag_horavg   = d.createVariable("tot_tau_yu_smag_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_yv_smag_horavg   = d.createVariable("tot_tau_yv_smag_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_yw_smag_horavg   = d.createVariable("tot_tau_yw_smag_horavg","f8",  ("tstep_unique","zhc"))
    var_tot_tau_zu_smag_horavg   = d.createVariable("tot_tau_zu_smag_horavg","f8",  ("tstep_unique","zhc"))
    var_tot_tau_zv_smag_horavg   = d.createVariable("tot_tau_zv_smag_horavg","f8",  ("tstep_unique","zhc"))
    var_tot_tau_zw_smag_horavg   = d.createVariable("tot_tau_zw_smag_horavg","f8",  ("tstep_unique","zc"))
    #
    var_unres_tau_xu_CNN_horavg = d.createVariable("unres_tau_xu_CNN_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xv_CNN_horavg = d.createVariable("unres_tau_xv_CNN_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xw_CNN_horavg = d.createVariable("unres_tau_xw_CNN_horavg","f8",("tstep_unique","zhcless"))
    var_unres_tau_yu_CNN_horavg = d.createVariable("unres_tau_yu_CNN_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yv_CNN_horavg = d.createVariable("unres_tau_yv_CNN_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yw_CNN_horavg = d.createVariable("unres_tau_yw_CNN_horavg","f8",("tstep_unique","zhcless"))
    var_unres_tau_zu_CNN_horavg = d.createVariable("unres_tau_zu_CNN_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zv_CNN_horavg = d.createVariable("unres_tau_zv_CNN_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zw_CNN_horavg = d.createVariable("unres_tau_zw_CNN_horavg","f8",("tstep_unique","zc"))
    var_tot_tau_xu_CNN_horavg   = d.createVariable("tot_tau_xu_CNN_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_xv_CNN_horavg   = d.createVariable("tot_tau_xv_CNN_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_xw_CNN_horavg   = d.createVariable("tot_tau_xw_CNN_horavg","f8",  ("tstep_unique","zhcless"))
    var_tot_tau_yu_CNN_horavg   = d.createVariable("tot_tau_yu_CNN_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_yv_CNN_horavg   = d.createVariable("tot_tau_yv_CNN_horavg","f8",  ("tstep_unique","zc"))
    var_tot_tau_yw_CNN_horavg   = d.createVariable("tot_tau_yw_CNN_horavg","f8",  ("tstep_unique","zhcless"))
    var_tot_tau_zu_CNN_horavg   = d.createVariable("tot_tau_zu_CNN_horavg","f8",  ("tstep_unique","zhc"))
    var_tot_tau_zv_CNN_horavg   = d.createVariable("tot_tau_zv_CNN_horavg","f8",  ("tstep_unique","zhc"))
    var_tot_tau_zw_CNN_horavg   = d.createVariable("tot_tau_zw_CNN_horavg","f8",  ("tstep_unique","zc"))
 
    #Create variables for storage fractions sub-grid fluxes
    var_frac_unres_tau_xu = d.createVariable("frac_unres_tau_xu","f8",("tstep_unique","zc","yc","xgcextra"))
    var_frac_unres_tau_xv = d.createVariable("frac_unres_tau_xv","f8",("tstep_unique","zc","yhc","xhc"))
    var_frac_unres_tau_xw = d.createVariable("frac_unres_tau_xw","f8",("tstep_unique","zhc","yc","xhc"))
    var_frac_unres_tau_yu = d.createVariable("frac_unres_tau_yu","f8",("tstep_unique","zc","yhc","xhc"))
    var_frac_unres_tau_yv = d.createVariable("frac_unres_tau_yv","f8",("tstep_unique","zc","ygcextra","xc"))
    var_frac_unres_tau_yw = d.createVariable("frac_unres_tau_yw","f8",("tstep_unique","zhc","yhc","xc"))
    var_frac_unres_tau_zu = d.createVariable("frac_unres_tau_zu","f8",("tstep_unique","zhc","yc","xhc"))
    var_frac_unres_tau_zv = d.createVariable("frac_unres_tau_zv","f8",("tstep_unique","zhc","yhc","xc"))
    var_frac_unres_tau_zw = d.createVariable("frac_unres_tau_zw","f8",("tstep_unique","zc","yc","xc"))
    var_frac_unres_tau_xu_horavg = d.createVariable("frac_unres_tau_xu_horavg","f8",("tstep_unique","zc"))
    var_frac_unres_tau_xv_horavg = d.createVariable("frac_unres_tau_xv_horavg","f8",("tstep_unique","zc"))
    var_frac_unres_tau_xw_horavg = d.createVariable("frac_unres_tau_xw_horavg","f8",("tstep_unique","zhc"))
    var_frac_unres_tau_yu_horavg = d.createVariable("frac_unres_tau_yu_horavg","f8",("tstep_unique","zc"))
    var_frac_unres_tau_yv_horavg = d.createVariable("frac_unres_tau_yv_horavg","f8",("tstep_unique","zc"))
    var_frac_unres_tau_yw_horavg = d.createVariable("frac_unres_tau_yw_horavg","f8",("tstep_unique","zhc"))
    var_frac_unres_tau_zu_horavg = d.createVariable("frac_unres_tau_zu_horavg","f8",("tstep_unique","zhc"))
    var_frac_unres_tau_zv_horavg = d.createVariable("frac_unres_tau_zv_horavg","f8",("tstep_unique","zhc"))
    var_frac_unres_tau_zw_horavg = d.createVariable("frac_unres_tau_zw_horavg","f8",("tstep_unique","zc"))

    #Calculate fraction unresolved compared to resolved fluxes, both point-by-point and horizontally averaged
    #NOTE1: Some extreme outliers in the fractions occur when the total momentum transport reaches 0. To preven this from happening, the fractions are confined to the range -10 to 10.
    #NOTE2: The Smagorinsky fluxes and CNN fluxes do not have some additional ghost cells as opposed to the other variables. Therefore, these are removed below when required.
    var_unres_tau_xu_horavg[:,:]          = np.mean(unres_tau_xu,                 axis=(2,3), keepdims=False)
    var_res_tau_xu_horavg[:,:]            = np.mean(res_tau_xu,                   axis=(2,3), keepdims=False)
    var_tot_tau_xu_horavg[:,:]            = np.mean(tot_tau_xu,                   axis=(2,3), keepdims=False)
    var_unres_tau_xu_smag_horavg[:,:]     = np.mean(smag_tau_xu,                  axis=(2,3), keepdims=False)
    var_unres_tau_xu_CNN_horavg[:,:]      = np.mean(preds_values_xu,              axis=(2,3), keepdims=False)
    var_tot_tau_xu_smag_horavg[:,:]       = np.mean(smag_tau_xu + res_tau_xu[:,:,:,1:],     axis=(2,3), keepdims=False)
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
    var_tot_tau_xv_CNN_horavg[:,:]        = np.mean(preds_values_xv + res_tau_xv[:,:,:-1,:], axis=(2,3), keepdims=False)
    var_frac_unres_tau_xv[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_xv[:,:,:,:])        / np.array(var_res_tau_xv[:,:,:,:]))       ,-10)
    var_frac_unres_tau_xv_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_xv_horavg[:,:]) / np.array(var_res_tau_xv_horavg[:,:])),-10)
    #
    var_unres_tau_xw_horavg[:,:]          = np.mean(unres_tau_xw,                 axis=(2,3), keepdims=False)
    var_res_tau_xw_horavg[:,:]            = np.mean(res_tau_xw,                   axis=(2,3), keepdims=False)
    var_tot_tau_xw_horavg[:,:]            = np.mean(tot_tau_xw,                   axis=(2,3), keepdims=False)
    var_unres_tau_xw_smag_horavg[:,:]     = np.mean(smag_tau_xw,                  axis=(2,3), keepdims=False)
    var_unres_tau_xw_CNN_horavg[:,:]      = np.mean(preds_values_xw,              axis=(2,3), keepdims=False)
    var_tot_tau_xw_smag_horavg[:,:]       = np.mean(smag_tau_xw + res_tau_xw,     axis=(2,3), keepdims=False)
    var_tot_tau_xw_CNN_horavg[:,:]        = np.mean(preds_values_xw + res_tau_xw[:,:-1,:,:], axis=(2,3), keepdims=False)
    var_frac_unres_tau_xw[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_xw[:,:,:,:])        / np.array(var_res_tau_xw[:,:,:,:]))       ,-10)
    var_frac_unres_tau_xw_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_xw_horavg[:,:]) / np.array(var_res_tau_xw_horavg[:,:])),-10)
    #
    var_unres_tau_yu_horavg[:,:]          = np.mean(unres_tau_yu,                 axis=(2,3), keepdims=False)
    var_res_tau_yu_horavg[:,:]            = np.mean(res_tau_yu,                   axis=(2,3), keepdims=False)
    var_tot_tau_yu_horavg[:,:]            = np.mean(tot_tau_yu,                   axis=(2,3), keepdims=False)
    var_unres_tau_yu_smag_horavg[:,:]     = np.mean(smag_tau_yu,                  axis=(2,3), keepdims=False)
    var_unres_tau_yu_CNN_horavg[:,:]      = np.mean(preds_values_yu,              axis=(2,3), keepdims=False)
    var_tot_tau_yu_smag_horavg[:,:]       = np.mean(smag_tau_yu + res_tau_yu,     axis=(2,3), keepdims=False)
    var_tot_tau_yu_CNN_horavg[:,:]        = np.mean(preds_values_yu + res_tau_yu[:,:,:,:-1], axis=(2,3), keepdims=False)
    var_frac_unres_tau_yu[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_yu[:,:,:,:])        / np.array(var_res_tau_yu[:,:,:,:]))       ,-10)
    var_frac_unres_tau_yu_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_yu_horavg[:,:]) / np.array(var_res_tau_yu_horavg[:,:])),-10)
    #
    var_unres_tau_yv_horavg[:,:]          = np.mean(unres_tau_yv,                 axis=(2,3), keepdims=False)
    var_res_tau_yv_horavg[:,:]            = np.mean(res_tau_yv,                   axis=(2,3), keepdims=False)
    var_tot_tau_yv_horavg[:,:]            = np.mean(tot_tau_yv,                   axis=(2,3), keepdims=False)
    var_unres_tau_yv_smag_horavg[:,:]     = np.mean(smag_tau_yv,                  axis=(2,3), keepdims=False)
    var_unres_tau_yv_CNN_horavg[:,:]      = np.mean(preds_values_yv,              axis=(2,3), keepdims=False)
    var_tot_tau_yv_smag_horavg[:,:]       = np.mean(smag_tau_yv + res_tau_yv[:,:,1:,:],     axis=(2,3), keepdims=False)
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
    var_tot_tau_yw_CNN_horavg[:,:]        = np.mean(preds_values_yw + res_tau_yw[:,:-1,:,:], axis=(2,3), keepdims=False)
    var_frac_unres_tau_yw[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_yw[:,:,:,:])        / np.array(var_res_tau_yw[:,:,:,:]))       ,-10)
    var_frac_unres_tau_yw_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_yw_horavg[:,:]) / np.array(var_res_tau_yw_horavg[:,:])),-10)
    #
    var_unres_tau_zu_horavg[:,:]          = np.mean(unres_tau_zu,                 axis=(2,3), keepdims=False)
    var_res_tau_zu_horavg[:,:]            = np.mean(res_tau_zu,                   axis=(2,3), keepdims=False)
    var_tot_tau_zu_horavg[:,:]            = np.mean(tot_tau_zu,                   axis=(2,3), keepdims=False)
    var_unres_tau_zu_smag_horavg[:,:]     = np.mean(smag_tau_zu,                  axis=(2,3), keepdims=False)
    var_unres_tau_zu_CNN_horavg[:,:]      = np.mean(preds_values_zu,              axis=(2,3), keepdims=False)
    var_tot_tau_zu_smag_horavg[:,:]       = np.mean(smag_tau_zu + res_tau_zu,     axis=(2,3), keepdims=False)
    var_tot_tau_zu_CNN_horavg[:,:]        = np.mean(preds_values_zu + res_tau_zu[:,:,:,:-1], axis=(2,3), keepdims=False)
    var_frac_unres_tau_zu[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_zu[:,:,:,:])        / np.array(var_res_tau_zu[:,:,:,:]))       ,-10)
    var_frac_unres_tau_zu_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_zu_horavg[:,:]) / np.array(var_res_tau_zu_horavg[:,:])),-10)
    #
    var_unres_tau_zv_horavg[:,:]          = np.mean(unres_tau_zv,                 axis=(2,3), keepdims=False)
    var_res_tau_zv_horavg[:,:]            = np.mean(res_tau_zv,                   axis=(2,3), keepdims=False)
    var_tot_tau_zv_horavg[:,:]            = np.mean(tot_tau_zv,                   axis=(2,3), keepdims=False)
    var_unres_tau_zv_smag_horavg[:,:]     = np.mean(smag_tau_zv,                  axis=(2,3), keepdims=False)
    var_unres_tau_zv_CNN_horavg[:,:]      = np.mean(preds_values_zv,              axis=(2,3), keepdims=False)
    var_tot_tau_zv_smag_horavg[:,:]       = np.mean(smag_tau_zv + res_tau_zv,     axis=(2,3), keepdims=False)
    var_tot_tau_zv_CNN_horavg[:,:]        = np.mean(preds_values_zv + res_tau_zv[:,:,:-1,:], axis=(2,3), keepdims=False)
    var_frac_unres_tau_zv[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_zv[:,:,:,:])        / np.array(var_res_tau_zv[:,:,:,:]))       ,-10)
    var_frac_unres_tau_zv_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_zv_horavg[:,:]) / np.array(var_res_tau_zv_horavg[:,:])),-10)
    #
    var_unres_tau_zw_horavg[:,:]          = np.mean(unres_tau_zw,                 axis=(2,3), keepdims=False)
    var_res_tau_zw_horavg[:,:]            = np.mean(res_tau_zw,                   axis=(2,3), keepdims=False)
    var_tot_tau_zw_horavg[:,:]            = np.mean(tot_tau_zw,                   axis=(2,3), keepdims=False)
    var_unres_tau_zw_smag_horavg[:,:]     = np.mean(smag_tau_zw,                  axis=(2,3), keepdims=False)
    var_unres_tau_zw_CNN_horavg[:,:]      = np.mean(preds_values_zw,              axis=(2,3), keepdims=False)
    var_tot_tau_zw_smag_horavg[:,:]       = np.mean(smag_tau_zw + res_tau_zw[:,:,:,:],     axis=(2,3), keepdims=False)
    var_tot_tau_zw_CNN_horavg[:,:]        = np.mean(preds_values_zw + res_tau_zw, axis=(2,3), keepdims=False)
    var_frac_unres_tau_zw[:,:,:,:]    = np.maximum(np.minimum(10,np.array(var_unres_tau_zw[:,:,:,:])        / np.array(var_res_tau_zw[:,:,:,:]))       ,-10)
    var_frac_unres_tau_zw_horavg[:,:] = np.maximum(np.minimum(10,np.array(var_unres_tau_zw_horavg[:,:]) / np.array(var_res_tau_zw_horavg[:,:])),-10)

    #Close netCDF-files
    d.close()

###Loop over heights for all components considering the time steps specified below, and make scatterplots of labels vs fluxes (CNN and Smagorinsky) at each height for all specified time steps combined###
if args.make_plots:
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
    #unres_tau_xu_traceless = np.array(fields['unres_tau_xu_traceless'][:,:,:,:])
    #unres_tau_yv_traceless = np.array(fields['unres_tau_yv_traceless'][:,:,:,:])
    #unres_tau_zw_traceless = np.array(fields['unres_tau_zw_traceless'][:,:,:,:])
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
    xc       = np.array(fields['xc'][:])
    xhc      = np.array(fields['xhc'][:])
    xhcless  = np.array(fields['xhcless'][:])
    xgcextra = np.array(fields['xgcextra'][:])
    yc       = np.array(fields['yc'][:])
    yhc      = np.array(fields['yhc'][:])
    yhcless  = np.array(fields['yhcless'][:])
    ygcextra = np.array(fields['ygcextra'][:])
    zc       = np.array(fields['zc'][:])
    zhc      = np.array(fields['zhc'][:])
    zhcless  = np.array(fields['zhcless'][:])
    zgcextra = np.array(fields['zgcextra'][:])
    
    #Close netCDF-file
    fields.close()

#Define function for making horizontal cross-sections
def make_horcross_heights(values, z, y, x, component, is_lbl, time_step = 0, delta = 500):
    #NOTE1: fourth last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: third last input of this function is a boolean that specifies whether the labels (True) or the NN predictions are being plotted.
    #NOTE3: the second last input of this function is an integer specifying which validation time step stored in the nc-file is plotted (by default the first one, which now corresponds to time step 28 used for validation).
    #NOTE4: the last input of this function is an integer specifying the channel half with [in meter] used to rescale the horizontal dimensions (by default 500m). 
    for k in range(len(z)-1):
        values_height = values[time_step,k,:,:]

        #Make horizontal cross-sections of the values
        plt.figure()
        plt.pcolormesh(x * delta/1000., y * delta/1000., values_height, vmin=-0.15, vmax=0.15)
        #plt.pcolormesh(x * delta, y * delta, values_height, vmin=-0.00015, vmax=0.00015)
        ax = plt.gca() 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(r'$\rm {[m^{2}\ s^{-2}}]$',rotation=270,fontsize=20,labelpad=30)
        plt.xlabel(r'$\rm x\ [km]$',fontsize=20)
        plt.ylabel(r'$\rm y\ [km]$',fontsize=20)
        #plt.xticks(fontsize=16, rotation=90)
        plt.xticks(fontsize=16, rotation=0)
        plt.yticks(fontsize=16, rotation=0)
        plt.tight_layout()
        if not is_lbl:
            plt.savefig("Horcross_tau_" + component + "_" + str((z[k]+z[k+1])/2.) + ".png", dpi = 200)
        else:
            plt.savefig("Horcross_label_tau_" + component + "_" + str((z[k]+z[k+1])/2.) + ".png", dpi = 200)
        plt.close()

#Define function for making pdfs
def make_pdfs_heights(values, labels, z, component, time_step = 0, delta = 500):
    #NOTE1: third last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: the second last input of this function is an integer specifying which validation time step stored in the nc-file is plotted (by default the first one, which now corresponds to time step 28 used for validation).
    #NOTE3: the last input of this function is an integer specifying the channel half with [in meter] used to rescale the horizontal dimensions (by default 500m). 
    for k in range(len(z)+1):
        if k == len(z):
            values_height = values[time_step,:,:,:].flatten()
            labels_height = labels[time_step,:,:,:].flatten()
            #range_bins = (0.6,0.6)
        else:
            values_height = values[time_step,k,:,:].flatten()
            labels_height = labels[time_step,k,:,:].flatten()
            #range_bins = (-2.0,2.0)

        #Determine bins
        num_bins = 100
        min_val = min(values_height.min(), labels_height.min())
        max_val = max(values_height.max(), labels_height.max())
        bin_edges = np.linspace(min_val, max_val, num_bins)

        #Make pdfs of the values and labels
        plt.figure()
        plt.hist(values_height, bins = bin_edges, density = True, histtype = 'step', label = 'MLP')
        plt.hist(labels_height, bins = bin_edges, density = True, histtype = 'step', label = 'labels')
        plt.ylabel(r'$\rm Normalized\ density\ [-]$',fontsize=20)
        plt.xlabel(r'$\rm Transport\ {[m^{2}\ s^{-2}]}$',fontsize=20)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16, rotation=0)
        plt.legend(loc='upper right')
        plt.tight_layout()
        if k == len(z):
            plt.savefig("PDF_tau_" + component + ".png", dpi = 200)
        else:
            plt.savefig("PDF_tau_" + component + "_" + str(z[k]) + ".png", dpi = 200)
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
            plt.ylabel(r'$\rm Smagorinsky\ {[m^{2}\ s^{-2}]}$',fontsize = 20)
        else:
            plt.ylabel(r'$\rm NN\ {[m^{2}\ s^{-2}]}$',fontsize = 20)
        plt.title("ρ = " + str(corrcoef),fontsize = 20)
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
    
    ##Make PDFs of labels and MLP predictions
    #make_pdfs_heights(unres_tau_xu_CNN, unres_tau_xu, zc,       'xu', time_step = 0, delta = delta_height)
    #make_pdfs_heights(unres_tau_yu_CNN, unres_tau_yu, zc,       'yu', time_step = 0, delta = delta_height)
    #make_pdfs_heights(unres_tau_zu_CNN, unres_tau_zu, zhc,      'zu', time_step = 0, delta = delta_height)
    #make_pdfs_heights(unres_tau_xv_CNN, unres_tau_xv, zc,       'xv', time_step = 0, delta = delta_height)
    #make_pdfs_heights(unres_tau_yv_CNN, unres_tau_yv, zc,       'yv', time_step = 0, delta = delta_height)
    #make_pdfs_heights(unres_tau_zv_CNN, unres_tau_zv, zhc,      'zv', time_step = 0, delta = delta_height)
    #make_pdfs_heights(unres_tau_xw_CNN, unres_tau_xw, zhcless,  'xw', time_step = 0, delta = delta_height)
    #make_pdfs_heights(unres_tau_yw_CNN, unres_tau_yw, zhcless,  'yw', time_step = 0, delta = delta_height)
    #make_pdfs_heights(unres_tau_zw_CNN, unres_tau_zw, zc,       'zw', time_step = 0, delta = delta_height)
    
    #Make horizontal cross-sections
    #NOTE1: some transport components are adjusted to convert them in a consistent way to equal shapes.
    #NOTE2: pcolormesh expects grid edges rather than grid centers, not an issue in z-direction!
    make_horcross_heights(unres_tau_xu,            zhc, yhc, xhc,           'xu', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yu[:,:,:,:-1], zhc, ygcextra, xgcextra, 'yu', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zu[:,:,:,:-1], zgcextra, yhc, xgcextra, 'zu', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_xv[:,:,:-1,:], zhc, ygcextra, xgcextra, 'xv', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yv,            zhc, yhc, xhc,           'yv', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zv[:,:,:-1,:], zgcextra, ygcextra, xhc, 'zv', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_xw[:,:-1,:,:], zgcextra, yhc, xgcextra, 'xw', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yw[:,:-1,:,:], zgcextra, ygcextra, xhc, 'yw', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zw,            zhc, yhc, xhc,           'zw', True, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_xu_CNN, zhc, yhc, xhc,           'xu', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yu_CNN, zhc, ygcextra, xgcextra, 'yu', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zu_CNN, zgcextra, yhc, xgcextra, 'zu', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_xv_CNN, zhc, ygcextra, xgcextra, 'xv', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yv_CNN, zhc, yhc, xhc,           'yv', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zv_CNN, zgcextra, ygcextra, xhc, 'zv', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_xw_CNN, zgcextra, yhc, xgcextra, 'xw', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_yw_CNN, zgcextra, ygcextra, xhc, 'yw', False, time_step = 0, delta = delta_height)
    make_horcross_heights(unres_tau_zw_CNN, zhc, yhc, xhc,           'zw', False, time_step = 0, delta = delta_height)
    ##Make scatterplots
    ##NOTE: some transport components are adjusted to convert them in a consistent way to equal shapes.
    #make_scatterplot_heights(unres_tau_xu_smag, unres_tau_xu[:,:,:,1:], unres_tau_xu_smag_horavg, unres_tau_xu_horavg, zc,  'xu', True)
    #make_scatterplot_heights(unres_tau_yu_smag, unres_tau_yu, unres_tau_yu_smag_horavg, unres_tau_yu_horavg, zc,  'yu', True)
    #make_scatterplot_heights(unres_tau_zu_smag, unres_tau_zu, unres_tau_zu_smag_horavg, unres_tau_zu_horavg, zhc, 'zu', True)
    #make_scatterplot_heights(unres_tau_xv_smag, unres_tau_xv, unres_tau_xv_smag_horavg, unres_tau_xv_horavg, zc,  'xv', True)
    #make_scatterplot_heights(unres_tau_yv_smag, unres_tau_yv[:,:,1:,:], unres_tau_yv_smag_horavg, unres_tau_yv_horavg, zc,  'yv', True)
    #make_scatterplot_heights(unres_tau_zv_smag, unres_tau_zv, unres_tau_zv_smag_horavg, unres_tau_zv_horavg, zhc, 'zv', True)
    #make_scatterplot_heights(unres_tau_xw_smag, unres_tau_xw, unres_tau_xw_smag_horavg, unres_tau_xw_horavg, zhc, 'xw', True)
    #make_scatterplot_heights(unres_tau_yw_smag, unres_tau_yw, unres_tau_yw_smag_horavg, unres_tau_yw_horavg, zhc, 'yw', True)
    #make_scatterplot_heights(unres_tau_zw_smag, unres_tau_zw, unres_tau_zw_smag_horavg, unres_tau_zw_horavg, zc,  'zw', True)
    ##
    #make_scatterplot_heights(unres_tau_xu_CNN, unres_tau_xu, unres_tau_xu_CNN_horavg, unres_tau_xu_horavg, zc,  'xu', False)
    #make_scatterplot_heights(unres_tau_yu_CNN, unres_tau_yu, unres_tau_yu_CNN_horavg, unres_tau_yu_horavg, zc,  'yu', False)
    #make_scatterplot_heights(unres_tau_zu_CNN, unres_tau_zu, unres_tau_zu_CNN_horavg, unres_tau_zu_horavg, zhc, 'zu', False)
    #make_scatterplot_heights(unres_tau_xv_CNN, unres_tau_xv, unres_tau_xv_CNN_horavg, unres_tau_xv_horavg, zc,  'xv', False)
    #make_scatterplot_heights(unres_tau_yv_CNN, unres_tau_yv, unres_tau_yv_CNN_horavg, unres_tau_yv_horavg, zc,  'yv', False)
    #make_scatterplot_heights(unres_tau_zv_CNN, unres_tau_zv, unres_tau_zv_CNN_horavg, unres_tau_zv_horavg, zhc, 'zv', False)
    #make_scatterplot_heights(unres_tau_xw_CNN, unres_tau_xw, unres_tau_xw_CNN_horavg, unres_tau_xw_horavg[:,:-1], zhcless, 'xw', False)
    #make_scatterplot_heights(unres_tau_yw_CNN, unres_tau_yw, unres_tau_yw_CNN_horavg, unres_tau_yw_horavg[:,:-1], zhcless, 'yw', False)
    #make_scatterplot_heights(unres_tau_zw_CNN, unres_tau_zw, unres_tau_zw_CNN_horavg, unres_tau_zw_horavg, zc,  'zw', False)
