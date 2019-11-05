import sys
import numpy
import struct
import netCDF4 as nc
#import pdb
#import tkinter
import matplotlib as mpl
mpl.use('agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
from matplotlib.pyplot import *

#Define height indices to consider the xy-crosssection. When comparing the plots from this script to the plots of the equivalent script for the horizontal crosssections produced by MicroHH, ensure that these height indices are consistent!!!
indexes_local = [0,1,2,3,16,32,47,60,61,62,63]


#Fetch training data
training_filepath = "/projects/1/flowsim/simulation1/lesscoarse/training_data.nc"
a = nc.Dataset(training_filepath, 'r')

#Read friction velocity
ustar   = np.array(a['utau_ref'][:], dtype = 'f4')

#Read ghost cells, indices, and coordinates from the training file
igc        = int(a['igc'][:])
jgc        = int(a['jgc'][:])
kgc_center = int(a['kgc_center'][:])
kgc_edge   = int(a['kgc_edge'][:])
iend       = int(a['iend'][:])
jend       = int(a['jend'][:])
kend       = int(a['kend'][:])
zhc        = np.array(a['zhc'][:])
zc         = np.array(a['zc'][:])
yhc        = np.array(a['yhc'][:])
yc         = np.array(a['yc'][:])
xhc        = np.array(a['xhc'][:])
xc         = np.array(a['xc'][:])

nz = zc.shape[0]
ny = yc.shape[0]
nx = xc.shape[0]

#Read variables from netCDF-file for an arbitrarty time step t=0
#NOTE: undo normalisation with friction velocity!
t=0 #Can be any time step, results should not change substantially
uc_singlefield = np.array(a['uc'][t,kgc_center:kend,jgc:jend,igc:iend]) * ustar
vc_singlefield = np.array(a['vc'][t,kgc_center:kend,jgc:jend,igc:iend]) * ustar
wc_singlefield = np.array(a['wc'][t,kgc_center:kend,jgc:jend,igc:iend]) * ustar

#Loop over heights to calculate spectra
num_var = 3
nwave_modes_x = int(nx * 0.5)
nwave_modes_y = int(ny * 0.5)
spectra_x  = numpy.zeros((3,nz,nwave_modes_x))
spectra_y  = numpy.zeros((3,nz,nwave_modes_y))
pdf_fields = numpy.zeros((3,nz,ny,nx))
index_spectra = 0
for idx_var in range(num_var):
    stop = False
    for k in range(np.size(indexes_local)):
        index = indexes_local[k]
        if idx_var == 0:
            s = uc_singlefield[index,:,:]

        elif idx_var == 1:
            s = vc_singlefield[index,:,:]  
        else:
            s = wc_singlefield[index,:,:]
        fftx = numpy.fft.rfft(s,axis=1)*(1/nx)
        ffty = numpy.fft.rfft(s,axis=0)*(1/ny)
        Px = fftx[:,1:] * numpy.conjugate(fftx[:,1:])
        Py = ffty[1:,:] * numpy.conjugate(ffty[1:,:])
        if int(nx % 2) == 0:
            Ex = np.append(2*Px[:,:-1],np.reshape(Px[:,-1],(ny,1)),axis=1)
        else:
            Ex = 2*Px[:,:]
        
        if int(ny % 2) == 0:
            Ey = np.append(2*Py[:-1,:],np.reshape(Py[-1,:],(1,nx)),axis=0)
        else:
            Ey = 2*Py[:,:]
            
        spectra_x[index_spectra,k,:]    = numpy.nanmean(Ex,axis=0) #Average Fourier transform over the direction where it was not calculated
        spectra_y[index_spectra,k,:]    = numpy.nanmean(Ey,axis=1)
        pdf_fields[index_spectra,k,:,:] = s[:,:]

    index_spectra +=1

k_streamwise = np.arange(1,nwave_modes_x+1)
k_spanwise = np.arange(1,nwave_modes_y+1)

#Determine bins for pdfs
num_bins = 100
bin_edges_u = np.linspace(np.nanmin(pdf_fields[0,:,:]),np.nanmax(pdf_fields[0,:,:]), num_bins)
bin_edges_v = np.linspace(np.nanmin(pdf_fields[1,:,:]),np.nanmax(pdf_fields[1,:,:]), num_bins)
bin_edges_w = np.linspace(np.nanmin(pdf_fields[2,:,:]),np.nanmax(pdf_fields[2,:,:]), num_bins)

#Plot balances
for k in range(np.size(indexes_local)):
    figure()
    loglog(k_streamwise[:], (spectra_x[0,k,:] / ustar**2.), 'k-',linewidth=2.0, label='u')
    loglog(k_streamwise[:], (spectra_x[1,k,:] / ustar**2.), 'r-',linewidth=2.0, label='v')
    loglog(k_streamwise[:], (spectra_x[2,k,:] / ustar**2.), 'b-',linewidth=2.0, label='w')
    
    xlabel(r'$\kappa \ [-]$',fontsize = 20)
    ylabel(r'$E \ [-]$',fontsize = 20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([1, 250, 0.000001, 3])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/spectrax_z_" + str(indexes_local[k]) + "_training.png")
    close()
    #
    figure()
    loglog(k_spanwise[:], (spectra_y[0,k,:] / ustar**2.), 'k-',linewidth=2.0, label='u')
    loglog(k_spanwise[:], (spectra_y[1,k,:] / ustar**2.), 'r-',linewidth=2.0, label='v')
    loglog(k_spanwise[:], (spectra_y[2,k,:] / ustar**2.), 'b-',linewidth=2.0, label='w')
    
    xlabel(r'$\kappa \ [-]$',fontsize = 20)
    ylabel(r'$E \ [-]$',fontsize = 20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([1, 250, 0.000001, 3])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/spectray_z_" + str(indexes_local[k]) + "_training.png")
    close()
    #
    figure()
    grid()
    hist(pdf_fields[0,k,:,:].flatten(), bins = bin_edges_u, density = True, histtype = 'step', label = 'u')
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([0, 0.16, 0, 140])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/pdfu_z_" + str(indexes_local[k]) + "_training.png")
    close()
    #
    figure()
    grid()
    hist(pdf_fields[1,k,:,:].flatten(), bins = bin_edges_v, density = True, histtype = 'step', label = 'v')
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    axis([-0.04, 0.04, 0, 140])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/pdfv_z_" + str(indexes_local[k]) + "_training.png")
    close()
    #
    figure()
    grid()
    hist(pdf_fields[2,k,:,:].flatten(), bins = bin_edges_w, density = True, histtype = 'step', label = 'w')
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    axis([-0.03, 0.03, 0, 140])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/pdfw_z_" + str(indexes_local[k]) + "_training.png")
    close()
    #

#Close nc-file
a.close()
