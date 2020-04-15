#Script that calculates the time averaged vertical profiles for the training data
import pdb
import sys
import numpy
import struct
import netCDF4 as nc
#import pdb
#import tkinter
import matplotlib as mpl
mpl.use('agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
from matplotlib.pyplot import *

#Fetch training data
training_filepath = "/projects/1/flowsim/simulation1/lesscoarse/training_data.nc"
a = nc.Dataset(training_filepath, 'r')

#Read friction velocity
ustar   = np.array(a['utau_ref'][:], dtype = 'f8')

#Read coordinates from the training file
zhc        = np.array(a['zhc'][:])
zc         = np.array(a['zc'][:])
yhc        = np.array(a['yhc'][:])
yc         = np.array(a['yc'][:])
xhc        = np.array(a['xhc'][:])
xc         = np.array(a['xc'][:])
zhgc        = np.array(a['zhgc'][:])
zgc         = np.array(a['zgc'][:])
yhgc        = np.array(a['yhgc'][:])
ygc         = np.array(a['ygc'][:])
xhgc        = np.array(a['xhgc'][:])
xgc         = np.array(a['xgc'][:])

nzc = zc.shape[0]
nyc = yc.shape[0]
nxc = xc.shape[0]

#Read variables from netCDF-files for all training time steps (t=0 untill t=26)
#NOTE: undo normalisation with friction velocity!
tstart=0
tend=27
tsteps=np.arange(tstart,tend,step=1)
uc_tavgfields = np.mean(np.array(a['uc'][tstart:tend,:,:,:]) * ustar, axis=(0,2,3))
vc_tavgfields = np.mean(np.array(a['vc'][tstart:tend,:,:,:]) * ustar, axis=(0,2,3))
wc_tavgfields = np.mean(np.array(a['wc'][tstart:tend,:,:,:]) * ustar, axis=(0,2,3))
uc_tstdfields = np.std(np.array(a['uc'][tstart:tend,:,:,:]) * ustar, axis=(0,2,3))
vc_tstdfields = np.std(np.array(a['vc'][tstart:tend,:,:,:]) * ustar, axis=(0,2,3))
wc_tstdfields = np.std(np.array(a['wc'][tstart:tend,:,:,:]) * ustar, axis=(0,2,3))
#
unres_tau_xu_tavgfields = np.mean(np.array(a['unres_tau_xu_tot'][tstart:tend,:,:,:-1]) * (ustar ** 2.), axis=(0,2,3))#Remove extra grid cell in training data
unres_tau_yu_tavgfields = np.mean(np.array(a['unres_tau_yu_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_zu_tavgfields = np.mean(np.array(a['unres_tau_zu_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_xv_tavgfields = np.mean(np.array(a['unres_tau_xv_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_yv_tavgfields = np.mean(np.array(a['unres_tau_yv_tot'][tstart:tend,:,:-1,:]) * (ustar ** 2.), axis=(0,2,3))#Remove extra grid cell in training data
unres_tau_zv_tavgfields = np.mean(np.array(a['unres_tau_zv_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_xw_tavgfields = np.mean(np.array(a['unres_tau_xw_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_yw_tavgfields = np.mean(np.array(a['unres_tau_yw_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_zw_tavgfields = np.mean(np.array(a['unres_tau_zw_tot'][tstart:tend,1:,:,:]) * (ustar ** 2.), axis=(0,2,3))#Remove extra grid cell in training data, which should be the first one to keep the fluxes consistent with the defined heights
#
unres_tau_xu_tstdfields = np.std(np.array(a['unres_tau_xu_tot'][tstart:tend,:,:,:-1]) * (ustar ** 2.), axis=(0,2,3))#Remove extra grid cell in training data
unres_tau_yu_tstdfields = np.std(np.array(a['unres_tau_yu_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_zu_tstdfields = np.std(np.array(a['unres_tau_zu_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_xv_tstdfields = np.std(np.array(a['unres_tau_xv_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_yv_tstdfields = np.std(np.array(a['unres_tau_yv_tot'][tstart:tend,:,:-1,:]) * (ustar ** 2.), axis=(0,2,3))#Remove extra grid cell in training data
unres_tau_zv_tstdfields = np.std(np.array(a['unres_tau_zv_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_xw_tstdfields = np.std(np.array(a['unres_tau_xw_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_yw_tstdfields = np.std(np.array(a['unres_tau_yw_tot'][tstart:tend,:,:,:]) * (ustar ** 2.), axis=(0,2,3))
unres_tau_zw_tstdfields = np.std(np.array(a['unres_tau_zw_tot'][tstart:tend,1:,:,:]) * (ustar ** 2.), axis=(0,2,3))#Remove extra grid cell in training data, which should be the first one to keep the fluxes consistent with the defined heights

#Write averaged fields to nc-file
tavg_vert_prof_filepath = "/projects/1/flowsim/simulation1/lesscoarse/tavg_vert_prof.nc"
b = nc.Dataset(tavg_vert_prof_filepath,'w')

#Create dimension
#b.createDimension("nt",len(tsteps))
b.createDimension("zhc",len(zhc))
b.createDimension("zc",len(zc))
b.createDimension("zhgc",len(zhgc))
b.createDimension("zgc",len(zgc))

#Create variables for dimensions and store them
#var_nt      = b.createVariable("nt","f8",("nt",))
var_zhc  = b.createVariable("zhc","f8",("zhc",))
var_zc   = b.createVariable("zc","f8",("zc",))
var_zhgc = b.createVariable("zhgc","f8",("zhgc",))
var_zgc  = b.createVariable("zgc","f8",("zgc",))
#
#var_nt[:]      = tsteps
var_zhc[:]  = zhc
var_zc[:]   = zc
var_zhgc[:] = zhgc
var_zgc[:]  = zgc

#Create variables for storage, and store them in nc-file
var_ucavgfields = b.createVariable("ucavgfields","f8",("zgc",))
var_vcavgfields = b.createVariable("vcavgfields","f8",("zgc",))
var_wcavgfields = b.createVariable("wcavgfields","f8",("zhgc",))
var_ucstdfields = b.createVariable("ucstdfields","f8",("zgc",))
var_vcstdfields = b.createVariable("vcstdfields","f8",("zgc",))
var_wcstdfields = b.createVariable("wcstdfields","f8",("zhgc",))
#
var_ucavgfields[:] = uc_tavgfields
var_vcavgfields[:] = vc_tavgfields
var_wcavgfields[:] = wc_tavgfields
var_ucstdfields[:] = uc_tstdfields
var_vcstdfields[:] = vc_tstdfields
var_wcstdfields[:] = wc_tstdfields
#
var_unresxutavgfields = b.createVariable("unresxuavgfields","f8",("zc",))
var_unresyutavgfields = b.createVariable("unresyuavgfields","f8",("zc",))
var_unreszutavgfields = b.createVariable("unreszuavgfields","f8",("zhc",))
var_unresxvtavgfields = b.createVariable("unresxvavgfields","f8",("zc",))
var_unresyvtavgfields = b.createVariable("unresyvavgfields","f8",("zc",))
var_unreszvtavgfields = b.createVariable("unreszvavgfields","f8",("zhc",))
var_unresxwtavgfields = b.createVariable("unresxwavgfields","f8",("zhc",))
var_unresywtavgfields = b.createVariable("unresywavgfields","f8",("zhc",))
var_unreszwtavgfields = b.createVariable("unreszwavgfields","f8",("zc",))
#
var_unresxutavgfields[:] = unres_tau_xu_tavgfields
var_unresyutavgfields[:] = unres_tau_yu_tavgfields
var_unreszutavgfields[:] = unres_tau_zu_tavgfields
var_unresxvtavgfields[:] = unres_tau_xv_tavgfields
var_unresyvtavgfields[:] = unres_tau_yv_tavgfields
var_unreszvtavgfields[:] = unres_tau_zv_tavgfields
var_unresxwtavgfields[:] = unres_tau_xw_tavgfields
var_unresywtavgfields[:] = unres_tau_yw_tavgfields
var_unreszwtavgfields[:] = unres_tau_zw_tavgfields
#
var_unresxutstdfields = b.createVariable("unresxustdfields","f8",("zc",))
var_unresyutstdfields = b.createVariable("unresyustdfields","f8",("zc",))
var_unreszutstdfields = b.createVariable("unreszustdfields","f8",("zhc",))
var_unresxvtstdfields = b.createVariable("unresxvstdfields","f8",("zc",))
var_unresyvtstdfields = b.createVariable("unresyvstdfields","f8",("zc",))
var_unreszvtstdfields = b.createVariable("unreszvstdfields","f8",("zhc",))
var_unresxwtstdfields = b.createVariable("unresxwstdfields","f8",("zhc",))
var_unresywtstdfields = b.createVariable("unresywstdfields","f8",("zhc",))
var_unreszwtstdfields = b.createVariable("unreszwstdfields","f8",("zc",))
#
var_unresxutstdfields[:] = unres_tau_xu_tstdfields
var_unresyutstdfields[:] = unres_tau_yu_tstdfields
var_unreszutstdfields[:] = unres_tau_zu_tstdfields
var_unresxvtstdfields[:] = unres_tau_xv_tstdfields
var_unresyvtstdfields[:] = unres_tau_yv_tstdfields
var_unreszvtstdfields[:] = unres_tau_zv_tstdfields
var_unresxwtstdfields[:] = unres_tau_xw_tstdfields
var_unresywtstdfields[:] = unres_tau_yw_tstdfields
var_unreszwtstdfields[:] = unres_tau_zw_tstdfields

#Close netCDF-file
b.close()
