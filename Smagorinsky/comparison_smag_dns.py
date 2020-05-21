import numpy as np
import numpy.ma as ma
import pandas as pd
import netCDF4 as nc
#import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
#from matplotlib import rcParams
mpl.rcParams.update({'figure.autolayout':True})
import matplotlib.pyplot as plt
import argparse
import csv

from matplotlib.ticker import FormatStrFormatter

parser = argparse.ArgumentParser(description='microhh_ML')
parser.add_argument('--dns_file', default=None, \
        help='NetCDF file that contains the residual transports at DNS resolution')
parser.add_argument('--smagorinsky_file', default=None, \
        help='NetCDF file that contains the calculated sub-grid scale transports according to the Smagorinsky-Lilly subgrid model at DNS resolution')
parser.add_argument('--make_table', dest='make_table', default=None, \
        action='store_true', \
        help='Make table with all correlation coefficients between the DNS residual fluxes and Smagorinsky')
parser.add_argument('--make_plots', dest='make_plots', default=None, \
        action='store_true', \
        help='Make plots of DNS residual fluxes and Smagorinsky')
args = parser.parse_args()

###Fetch Smagorinsky fluxes and DNS residual fluxes
dns = nc.Dataset(args.dns_file,'r')
smag = nc.Dataset(args.smagorinsky_file,'r')

#Both files contain only one time step
time_step = 0 #NOTE: in fact corresponds to time_step 28 from validation set

#Extract smagorinsky fluxes and DNS residual fluxes.
dns_tau_xu   = np.array(dns['unres_tau_xu_turb'][time_step,:,:,:])
dns_tau_yu   = np.array(dns['unres_tau_yu_turb'][time_step,:,:,:])
dns_tau_zu   = np.array(dns['unres_tau_zu_turb'][time_step,:,:,:])
dns_tau_xv   = np.array(dns['unres_tau_xv_turb'][time_step,:,:,:])
dns_tau_yv   = np.array(dns['unres_tau_yv_turb'][time_step,:,:,:])
dns_tau_zv   = np.array(dns['unres_tau_zv_turb'][time_step,:,:,:])
dns_tau_xw   = np.array(dns['unres_tau_xw_turb'][time_step,:,:,:])
dns_tau_yw   = np.array(dns['unres_tau_yw_turb'][time_step,:,:,:])
dns_tau_zw   = np.array(dns['unres_tau_zw_turb'][time_step,:,:,:])
#
#NOTE: in contrast to unres transports calculated from DNS, in the calculation of Smag the staggered grid has been taken into account. Given the high resolution used for DNS, this difference will only have a marginal impact on the calculations. For this reason, in this script the Smagorinsky fluxes are assumed to be at the same location as the DNS fluxes (i.e. the grid centers)
dns_tau_xu_smag  = np.array(smag['smag_tau_xu'][time_step,:,:,:])
dns_tau_yu_smag  = np.array(smag['smag_tau_yu'][time_step,:,:-1,:-1])
dns_tau_zu_smag  = np.array(smag['smag_tau_zu'][time_step,:-1,:,:-1])
dns_tau_xv_smag  = np.array(smag['smag_tau_xv'][time_step,:,:-1,:-1])
dns_tau_yv_smag  = np.array(smag['smag_tau_yv'][time_step,:,:,:])
dns_tau_zv_smag  = np.array(smag['smag_tau_zv'][time_step,:-1,:-1,:])
dns_tau_xw_smag  = np.array(smag['smag_tau_xw'][time_step,:-1,:,:-1])
dns_tau_yw_smag  = np.array(smag['smag_tau_yw'][time_step,:-1,:-1,:])
dns_tau_zw_smag  = np.array(smag['smag_tau_zw'][time_step,:,:,:])
#
#Extract coordinates
nt = len(dns.dimensions['time'])
z  = np.array(dns['z'][:])
zextra = np.insert(z,0,-z[0], axis=0)
nz = len(z)
zh = np.array(dns['zh'][:])
y  = np.array(dns['y'][:])
yextra = np.insert(y,0,-y[0], axis=0)
ny = len(y)
yh = np.array(dns['yh'][:])
x  = np.array(dns['x'][:])
xextra = np.insert(x,0,-x[0], axis=0)
nx = len(x)
xh = np.array(dns['xh'][:])

#Extract friction velocity
utau_ref = float(dns['utau_ref'][:])

#NOTE: commented out code takes into account additional ghost cells that have been added to the used transport components
#Calculate trace part of subgrid-stress, and subtract this from labels for fair comparison with Smagorinsky fluxes
trace_train = (dns_tau_xu + dns_tau_yv + dns_tau_zw) * (1./3.)
#print('Trace_train: ' + str(trace_train))
dns_tau_xu_traceless = dns_tau_xu - trace_train
dns_tau_yv_traceless = dns_tau_yv - trace_train
dns_tau_zw_traceless = dns_tau_zw - trace_train

###Loop over heights for all components considering the time steps specified below, and make scatterplots of labels vs fluxes (CNN and Smagorinsky) at each height for all specified time steps combined###
#Write all relevant correlation coefficients to a table
if args.make_table:
    print('start making table')
    heights = np.array(z, dtype=object)
    heights = np.insert(heights,0,'zall')
    heights = np.append(heights,'top_wall')
    components = np.array(
               ['tau_uu_smag','tau_vu_smag','tau_wu_smag',
                'tau_uv_smag','tau_vv_smag','tau_wv_smag',
                'tau_uw_smag','tau_vw_smag','tau_ww_smag'])

    #Define arrays for storage
    corrcoef_xu_smag = np.zeros((nz+2,),dtype=np.float32)
    corrcoef_yu_smag = np.zeros((nz+2,),dtype=np.float32)
    corrcoef_zu_smag = np.zeros((nz+2,),dtype=np.float32)
    corrcoef_xv_smag = np.zeros((nz+2,),dtype=np.float32)
    corrcoef_yv_smag = np.zeros((nz+2,),dtype=np.float32)
    corrcoef_zv_smag = np.zeros((nz+2,),dtype=np.float32)
    corrcoef_xw_smag = np.zeros((nz+2,),dtype=np.float32)
    corrcoef_yw_smag = np.zeros((nz+2,),dtype=np.float32)
    corrcoef_zw_smag = np.zeros((nz+2,),dtype=np.float32)

    #Consider all heights over all time steps
    a = ma.masked_invalid(dns_tau_xu_smag.flatten())
    b = ma.masked_invalid(dns_tau_xu_traceless.flatten())
    msk = (~a.mask & ~b.mask)
    corrcoef_xu_smag[0] = np.round(ma.corrcoef(dns_tau_xu_smag.flatten()[msk], dns_tau_xu_traceless.flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    a = ma.masked_invalid(dns_tau_yu_smag.flatten())
    b = ma.masked_invalid(dns_tau_yu.flatten())
    msk = (~a.mask & ~b.mask)
    corrcoef_yu_smag[0] = np.round(ma.corrcoef(dns_tau_yu_smag.flatten()[msk], dns_tau_yu.flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    a = ma.masked_invalid(dns_tau_zu_smag.flatten())
    b = ma.masked_invalid(dns_tau_zu.flatten())
    msk = (~a.mask & ~b.mask)
    corrcoef_zu_smag[0] = np.round(ma.corrcoef(dns_tau_zu_smag.flatten()[msk], dns_tau_zu.flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    a = ma.masked_invalid(dns_tau_xv_smag.flatten())
    b = ma.masked_invalid(dns_tau_xv.flatten())
    msk = (~a.mask & ~b.mask)
    corrcoef_xv_smag[0] = np.round(ma.corrcoef(dns_tau_xv_smag.flatten()[msk], dns_tau_xv.flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    a = ma.masked_invalid(dns_tau_yv_smag.flatten())
    b = ma.masked_invalid(dns_tau_yv_traceless.flatten())
    msk = (~a.mask & ~b.mask)
    corrcoef_yv_smag[0] = np.round(ma.corrcoef(dns_tau_yv_smag.flatten()[msk], dns_tau_yv_traceless.flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    a = ma.masked_invalid(dns_tau_zv_smag.flatten())
    b = ma.masked_invalid(dns_tau_zv.flatten())
    msk = (~a.mask & ~b.mask)
    corrcoef_zv_smag[0] = np.round(ma.corrcoef(dns_tau_zv_smag.flatten()[msk], dns_tau_zv.flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    a = ma.masked_invalid(dns_tau_xw_smag.flatten())
    b = ma.masked_invalid(dns_tau_xw.flatten())
    msk = (~a.mask & ~b.mask)
    corrcoef_xw_smag[0] = np.round(ma.corrcoef(dns_tau_xw_smag.flatten()[msk], dns_tau_xw.flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    a = ma.masked_invalid(dns_tau_yw_smag.flatten())
    b = ma.masked_invalid(dns_tau_yw.flatten())
    msk = (~a.mask & ~b.mask)
    corrcoef_yw_smag[0] = np.round(ma.corrcoef(dns_tau_yw_smag.flatten()[msk], dns_tau_yw.flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    a = ma.masked_invalid(dns_tau_zw_smag.flatten())
    b = ma.masked_invalid(dns_tau_zw.flatten())
    msk = (~a.mask & ~b.mask)
    corrcoef_zw_smag[0] = np.round(ma.corrcoef(dns_tau_zw_smag.flatten()[msk], dns_tau_zw_traceless.flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
 
    #Consider each individual height
    for k in range(nz+1): #+1 needed to calculate corr_coefs at top wall for appropriate components
        if k == nz: #Ensure only arrays with additional cell for top wall are accessed, put the others to NaN
            corrcoef_xu_smag[k+1] = np.nan
            corrcoef_yu_smag[k+1] = np.nan
            #a = ma.masked_invalid(dns_tau_zu_smag[k,:,:].flatten())
            #b = ma.masked_invalid(dns_tau_zu[k,:,:].flatten())
            #msk = (~a.mask & ~b.mask)
            #corrcoef_zu_smag[k+1] = np.round(ma.corrcoef(dns_tau_zu_smag[k,:,:].flatten()[msk], dns_tau_zu[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_zu_smag[k+1] = np.nan
            corrcoef_xv_smag[k+1] = np.nan
            corrcoef_yv_smag[k+1] = np.nan
            #a = ma.masked_invalid(dns_tau_zv_smag[k,:,:].flatten())
            #b = ma.masked_invalid(dns_tau_zv[k,:,:].flatten())
            #msk = (~a.mask & ~b.mask)
            #corrcoef_zv_smag[k+1] = np.round(ma.corrcoef(dns_tau_zv_smag[k,:,:].flatten()[msk], dns_tau_zv[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_zv_smag[k+1] = np.nan
            corrcoef_xw_smag[k+1] = np.nan
            corrcoef_yw_smag[k+1] = np.nan
            corrcoef_zw_smag[k+1] = np.nan

        else:
            a = ma.masked_invalid(dns_tau_xu_smag[k,:,:].flatten())
            b = ma.masked_invalid(dns_tau_xu_traceless[k,:,:].flatten())
            msk = (~a.mask & ~b.mask)
            corrcoef_xu_smag[k+1] = np.round(ma.corrcoef(dns_tau_xu_smag[k,:,:].flatten()[msk], dns_tau_xu_traceless[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            a = ma.masked_invalid(dns_tau_yu_smag[k,:,:].flatten())
            b = ma.masked_invalid(dns_tau_yu[k,:,:].flatten())
            msk = (~a.mask & ~b.mask)
            corrcoef_yu_smag[k+1] = np.round(ma.corrcoef(dns_tau_yu_smag[k,:,:].flatten()[msk], dns_tau_yu[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            a = ma.masked_invalid(dns_tau_zu_smag[k,:,:].flatten())
            b = ma.masked_invalid(dns_tau_zu[k,:,:].flatten())
            msk = (~a.mask & ~b.mask)
            corrcoef_zu_smag[k+1] = np.round(ma.corrcoef(dns_tau_zu_smag[k,:,:].flatten()[msk], dns_tau_zu[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            a = ma.masked_invalid(dns_tau_xv_smag[k,:,:].flatten())
            b = ma.masked_invalid(dns_tau_xv[k,:,:].flatten())
            msk = (~a.mask & ~b.mask)
            corrcoef_xv_smag[k+1] = np.round(ma.corrcoef(dns_tau_xv_smag[k,:,:].flatten()[msk], dns_tau_xv[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            a = ma.masked_invalid(dns_tau_yv_smag[k,:,:].flatten())
            b = ma.masked_invalid(dns_tau_yv_traceless[k,:,:].flatten())
            msk = (~a.mask & ~b.mask)
            corrcoef_yv_smag[k+1] = np.round(ma.corrcoef(dns_tau_yv_smag[k,:,:].flatten()[msk], dns_tau_yv_traceless[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            a = ma.masked_invalid(dns_tau_zv_smag[k,:,:].flatten())
            b = ma.masked_invalid(dns_tau_zv[k,:,:].flatten())
            msk = (~a.mask & ~b.mask)
            corrcoef_zv_smag[k+1] = np.round(ma.corrcoef(dns_tau_zv_smag[k,:,:].flatten()[msk], dns_tau_zv[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            a = ma.masked_invalid(dns_tau_xw_smag[k,:,:].flatten())
            b = ma.masked_invalid(dns_tau_xw[k,:,:].flatten())
            msk = (~a.mask & ~b.mask)
            corrcoef_xw_smag[k+1] = np.round(ma.corrcoef(dns_tau_xw_smag[k,:,:].flatten()[msk], dns_tau_xw[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            a = ma.masked_invalid(dns_tau_yw_smag[k,:,:].flatten())
            b = ma.masked_invalid(dns_tau_yw[k,:,:].flatten())
            msk = (~a.mask & ~b.mask)
            corrcoef_yw_smag[k+1] = np.round(ma.corrcoef(dns_tau_yw_smag[k,:,:].flatten()[msk], dns_tau_yw[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            a = ma.masked_invalid(dns_tau_zw_smag[k,:,:].flatten())
            b = ma.masked_invalid(dns_tau_zw_traceless[k,:,:].flatten())
            msk = (~a.mask & ~b.mask)
            corrcoef_zw_smag[k+1] = np.round(ma.corrcoef(dns_tau_zw_smag[k,:,:].flatten()[msk], dns_tau_zw_traceless[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix

    #Add correlation coefficients to DataFrame
    corr_coef = np.array(
               [corrcoef_xu_smag,corrcoef_yu_smag,corrcoef_zu_smag,
                corrcoef_xv_smag,corrcoef_yv_smag,corrcoef_zv_smag,
                corrcoef_xw_smag,corrcoef_yw_smag,corrcoef_zw_smag]
               ,dtype=np.float32)
    
    corr_table = pd.DataFrame(np.swapaxes(corr_coef,0,1), index = heights, columns = components)

    #Save table to figure (code taken from StackOverflow)
    def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',bbox=[0, 0, 1, 1], header_columns=0,ax=None, row_color_map=None, corr_table = False, **kwargs):

        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([1, 1])) * np.array([col_width, row_height]) #second numpy array found by trial and error: chosen such that the figure exactly fits the table
            fig = plt.figure(figsize=size)
            ax = plt.gca()
            ax.axis('off')

        #mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels=data.index, **kwargs) #Uncomment when no conditional formatting should be applied
        #Uncomment five lines below when conditional formatting should be applied
        if corr_table:
            normal = np.minimum(np.maximum(data, 0.),1.) #Scale colors in range 0-1
        else:
            normal = np.minimum(np.maximum(data, -1.),1.) #Scale colors in range -1 to 1
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels=data.index, cellColours=plt.cm.jet(normal), **kwargs)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in  mpl_table._cells.items():
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
            #    cell.set_facecolor(row_colors[k[0]%len(row_colors)]) #Uncomment to make polished table for paper
                #Do conditional formatting at line 1248
                pass

        #Save figure
        if corr_table:
            fig.savefig('corr_table_boxfilter.png', bbox_inches='tight')
        else:
            fig.savefig('re_table_boxfilter.png', bbox_inches='tight')

    render_mpl_table(corr_table, header_columns=0, col_width=2.0, bbox=[0.02, 0, 1, 1], corr_table = True)

    print('Finished making tables')

#Define function for making horizontal cross-sections
def make_horcross_heights(values, z, y, x, component, is_lbl, time_step = 0):
    #NOTE1: fourth last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: third last input of this function is a boolean that specifies whether the labels (True) or the NN predictions are being plotted.
    #NOTE3: the second last input of this function is an integer specifying which validation time step stored in the nc-file is plotted (by default the first one, which now corresponds to time step 28 used for validation).
    #NOTE4: the last input of this function is an integer specifying the channel half with [in meter] used to rescale the horizontal dimensions (by default 1m, effectively not rescaling). 

    for k in range(len(z)-1):
        values_height = values[k,:,:] / (utau_ref ** 2.)

        #Make horizontal cross-sections of the values
        plt.figure()
        if not is_lbl:
            plt.pcolormesh(x, y, values_height, vmin=-0.5, vmax=0.5)
        else:
            plt.pcolormesh(x, y, values_height, vmin=-0.5, vmax=0.5)
        #plt.pcolormesh(x * delta, y * delta, values_height, vmin=-0.00015, vmax=0.00015)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(r'$\rm \frac{\tau_{wu}}{u_{\tau}^2} \ [-]$',rotation=270,fontsize=20,labelpad=30)
        plt.xlabel(r'$\rm \frac{x}{\delta} \ [-]$',fontsize=20)
        plt.ylabel(r'$\rm \frac{y}{\delta} \ [-]$',fontsize=20)
        #plt.xticks(fontsize=16, rotation=90)
        plt.xticks(fontsize=16, rotation=0)
        plt.yticks(fontsize=16, rotation=0)
        plt.tight_layout()
        if is_lbl:
            plt.savefig("Horcross_box_dns_tau_" + component + "_" + str((z[k]+z[k+1])/2.) + ".png", dpi = 200)
        else:
            plt.savefig("Horcross_box_smag_tau_" + component + "_" + str((z[k]+z[k+1])/2.) + ".png", dpi = 200)
        plt.close()

#Define function for making spectra
def make_spectra_heights(smag, dns, z, component, time_step = 0):
    #NOTE1: second last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: last input of this function is an integer specifying which validation time step stored in the nc-file is plotted (by default the first one, which now corresponds to time step 28 used for validation).
    for k in range(len(z)):
        

        #ann_height  = ann[k,:,:]  / (utau_ref ** 2.)
        smag_height = smag[k,:,:] / (utau_ref ** 2.)
        dns_height  = dns[k,:,:]  / (utau_ref ** 2.)
        #range_bins = (-2.0,2.0)

        #Hard-code removal of Nan-values for zu component
        if component == 'zu':
            smag_height = smag_height[9:-9,10:-10]
            dns_height = dns_height[9:-9,10:-10]

        #Calculate spectra
        #smag
        nxc = smag_height.shape[1]
        nyc = smag_height.shape[0]
        fftx_smag = np.fft.rfft(smag_height,axis=1)*(1/nxc)
        ffty_smag = np.fft.rfft(smag_height,axis=0)*(1/nyc)
        Px_smag = fftx_smag[:,1:] * np.conjugate(fftx_smag[:,1:])
        Py_smag = ffty_smag[1:,:] * np.conjugate(ffty_smag[1:,:])
        if int(nxc % 2) == 0:
            Ex_smag = np.append(2*Px_smag[:,:-1],np.reshape(Px_smag[:,-1],(nyc,1)),axis=1)
        else:
            Ex_smag = 2*Px_smag[:,:]
        
        if int(nyc % 2) == 0:
            Ey_smag = np.append(2*Py_smag[:-1,:],np.reshape(Py_smag[-1,:],(1,nxc)),axis=0)
        else:
            Ey_smag = 2*Py_smag[:,:]

        smag_spec_x = np.nanmean(Ex_smag,axis=0) #Average FT over direction where it was not calculated
        smag_spec_y = np.nanmean(Ey_smag,axis=1)
        #DNS
        nxc = dns_height.shape[1]
        nyc = dns_height.shape[0]
        fftx_dns = np.fft.rfft(dns_height,axis=1)*(1/nxc)
        ffty_dns = np.fft.rfft(dns_height,axis=0)*(1/nyc)
        Px_dns = fftx_dns[:,1:] * np.conjugate(fftx_dns[:,1:])
        Py_dns = ffty_dns[1:,:] * np.conjugate(ffty_dns[1:,:])
        if int(nxc % 2) == 0:
            Ex_dns = np.append(2*Px_dns[:,:-1],np.reshape(Px_dns[:,-1],(nyc,1)),axis=1)
        else:
            Ex_dns = 2*Px_dns[:,:]
        
        if int(nyc % 2) == 0:
            Ey_dns = np.append(2*Py_dns[:-1,:],np.reshape(Py_dns[-1,:],(1,nxc)),axis=0)
        else:
            Ey_dns = 2*Py_dns[:,:]

        dns_spec_x = np.nanmean(Ex_dns,axis=0) #Average FT over direction where it was not calculated
        dns_spec_y = np.nanmean(Ey_dns,axis=1)

        #Plot spectra
        plt.figure()
        #plt.plot(ann_spec_x,  label = 'ANN')
        plt.plot(smag_spec_x, label = 'Smagorinsky')
        plt.plot(dns_spec_x,  label = 'DNS')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.ylabel(r'$\rm E \ [-]$',fontsize=20)
        plt.xlabel(r'$\rm \kappa \ [-]$',fontsize=20)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16, rotation=0)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig("Spectrax_tau_" + component + "_" + str(z[k]) + ".png", dpi = 200)
        plt.close()
        #
        plt.figure()
        #plt.plot(ann_spec_y,  label = 'ANN')
        plt.plot(smag_spec_y, label = 'Smagorinsky')
        plt.plot(dns_spec_y,  label = 'DNS')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.ylabel(r'$\rm E \ [-]$',fontsize=20)
        plt.xlabel(r'$\rm \kappa \ [-]$',fontsize=20)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16, rotation=0)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig("Spectray_tau_" + component + "_" + str(z[k]) + ".png", dpi = 200)
        plt.close()

#Define function for making pdfs
def make_pdfs_heights(smag, labels, z, component, time_step = 0):
    #NOTE1: second last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: last input of this function is an integer specifying which validation time step stored in the nc-file is plotted (by default the first one, which now corresponds to time step 28 used for validation).
    for k in range(len(z)+1):
        if k == len(z):
            #values_height = values[:,:,:] / (utau_ref ** 2.)
            smag_height   = smag[:,:,:] / (utau_ref ** 2.)
            labels_height = labels[:,:,:] / (utau_ref ** 2.)
            #range_bins = (0.6,0.6)
            
            #Hard-code removal of Nan-values for zu component
            if component == 'zu':
                smag_height = smag_height[23:-23,9:-9,10:-10]
                labels_height = labels_height[23:-23,9:-9,10:-10]

            #Flatten array
            smag_height = smag_height.flatten()
            labels_height = labels_height.flatten()

        else:
            #values_height = values[k,:,:] / (utau_ref ** 2.)
            smag_height   = smag[k,:,:] / (utau_ref ** 2.)
            labels_height = labels[k,:,:] / (utau_ref ** 2.)
            #range_bins = (-2.0,2.0)

            #Hard-code removal of Nan-values for zu component
            if component == 'zu':
                smag_height = smag_height[9:-9,10:-10]
                labels_height = labels_height[9:-9,10:-10]
        
            #Flatten array
            smag_height = smag_height.flatten()
            labels_height = labels_height.flatten()
        
        #Determine bins
        num_bins = 100
        min_val = min(smag_height.min(), labels_height.min())
        max_val = max(smag_height.max(), labels_height.max())
        bin_edges = np.linspace(min_val, max_val, num_bins)

        #Make pdfs of the values and labels
        plt.figure()
        #plt.hist(values_height, bins = bin_edges, density = True, histtype = 'step', label = 'ANN')
        plt.hist(smag_height, bins = bin_edges, density = True, histtype = 'step', label = 'Smagorinsky')
        plt.hist(labels_height, bins = bin_edges, density = True, histtype = 'step', label = 'DNS')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.00008)
        ax.set_xlim(left=-5, right=5)
        plt.ylabel(r'$\rm Probability\ density\ [-]$',fontsize=20)
        plt.xlabel(r'$\rm \frac{\tau_{wu}}{u_{\tau}^2} \ [-]$',fontsize=20)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16, rotation=0)
        plt.legend(loc='upper right')
        plt.tight_layout()
        if k == len(z):
            plt.savefig("PDF_box_tau_" + component + ".png", dpi = 200)
        else:
            plt.savefig("PDF_box_tau_" + component + "_" + str(z[k]) + ".png", dpi = 200)
        plt.close()

#Define function for making pdfs
def make_vertprof(smag, labels, z, component, time_step = 0):
    #NOTE1: second last input of this function is a string indicating the name of the component being plotted.
    #last input of this function is an integer specifying which validation time step stored in the nc-file is plotted (by default the first one, which now corresponds to time step 28 used for validation).

    #Make vertical profile
    plt.figure()
    #plt.plot(z, values[:] / (utau_ref ** 2.), label = 'ANN', marker = 'o', markersize = 2.0)
    plt.plot(z, smag[:] / (utau_ref ** 2.), label = 'Smagorinsky', marker = 'o', markersize = 2.0)
    plt.plot(z, labels[:] / (utau_ref ** 2.), label = 'DNS', marker = 'o', markersize = 2.0)
    #ax = plt.gca()
    #ax.set_yscale('log')
    plt.ylabel(r'$\rm \frac{\tau_{wu}}{u_{\tau}^2} \ [-]$',fontsize=20)
    plt.xlabel(r'$\rm \frac{z}{\delta} \ [-]$',fontsize=20)
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16, rotation=0)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("vertprof_box_tau_" + component + ".png", dpi = 200)
    plt.close()

#Define function for making scatterplots
def make_scatterplot_heights(preds, lbls, preds_horavg, lbls_horavg, heights, component, time_step):
    #NOTE1: third last input of this function is a string indicating the name of the component being plotted.
    for k in range(len(heights)+1):
        if k == len(heights):
            preds_height = preds_horavg[:] / (utau_ref ** 2.)
            lbls_height  = lbls_horavg[:] / (utau_ref ** 2.)
        else:
            preds_height = preds[k,:,:] / (utau_ref ** 2.)
            lbls_height  = lbls[k,:,:] / (utau_ref ** 2.)

        preds_height = preds_height.flatten()
        lbls_height  = lbls_height.flatten()
        
        #Make scatterplots of Smagorinsky/CNN fluxes versus labels
        corrcoef = np.round(ma.corrcoef(preds_height, lbls_height)[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
        plt.figure()
        plt.scatter(lbls_height, preds_height, s=6, marker='o', alpha=0.2)
        if k == len(heights):
            #plt.xlim([-0.004, 0.004])
            #plt.ylim([-0.004, 0.004])
            #plt.xlim([-0.000004, 0.000004])
            #plt.ylim([-0.000004, 0.000004])
            plt.xlim([-2.0, 2.0])
            plt.ylim([-2.0, 2.0])
        else:
            plt.xlim([-2.0, 2.0])
            plt.ylim([-2.0, 2.0])
            #plt.xlim([-15.0, 15.0])
            #plt.ylim([-15.0, 15.0])
            #plt.xlim([-40.0, 40.0])
            #plt.ylim([-40.0, 40.0])
            #plt.xlim([-0.0005, 0.0005])
            #plt.ylim([-0.0005, 0.0005])
        axes = plt.gca()
        plt.plot(axes.get_xlim(),axes.get_ylim(),'b--')
        #plt.gca().set_aspect('equal',adjustable='box')
        plt.xlabel(r'$\rm \frac{\tau_{wu}^{DNS}}{u_{\tau}^2} \,\ {[-]}$',fontsize = 20)
        plt.ylabel(r'$\rm \frac{\tau_{wu}^{smag}}{u_{\tau}^2} \,\ {[-]}$',fontsize = 20)
        #plt.title("ρ = " + str(corrcoef),fontsize = 20)
        plt.axhline(c='black')
        plt.axvline(c='black')
        plt.xticks(fontsize = 16, rotation = 90)
        plt.yticks(fontsize = 16, rotation = 0)
        if k == len(heights):
            plt.savefig("Scatter_Smagorinsky_tau_" + component + "_horavg.png", dpi = 200)
        else:
            plt.savefig("Scatter_Smagorinsky_tau_" + component + "_" + str(heights[k]) + ".png", dpi = 200)
        plt.tight_layout()
        plt.close()


#Call function multiple times to make all plots for smagorinsky and CNN
if args.make_plots:
    print('start making plots')

    #Average fields in horizontal directions for some of the plots
    #dns_tau_xu_horavg = np.nanmean(dns_tau_xu, axis=(1,2), keepdims=False)
    #dns_tau_yu_horavg = np.nanmean(dns_tau_yu, axis=(1,2), keepdims=False)
    dns_tau_zu_horavg = np.nanmean(dns_tau_zu, axis=(1,2), keepdims=False)
    #dns_tau_xv_horavg = np.nanmean(dns_tau_xv, axis=(1,2), keepdims=False)
    #dns_tau_yv_horavg = np.nanmean(dns_tau_yv, axis=(1,2), keepdims=False)
    #dns_tau_zv_horavg = np.nanmean(dns_tau_zv, axis=(1,2), keepdims=False)
    #dns_tau_xw_horavg = np.nanmean(dns_tau_xw, axis=(1,2), keepdims=False)
    #dns_tau_yw_horavg = np.nanmean(dns_tau_yw, axis=(1,2), keepdims=False)
    #dns_tau_zw_horavg = np.nanmean(dns_tau_zw, axis=(1,2), keepdims=False)
    #dns_tau_xu_smag_horavg = np.nanmean(dns_tau_xu_smag, axis=(1,2), keepdims=False)
    #dns_tau_yu_smag_horavg = np.nanmean(dns_tau_yu_smag, axis=(1,2), keepdims=False)
    dns_tau_zu_smag_horavg = np.nanmean(dns_tau_zu_smag, axis=(1,2), keepdims=False)
    #dns_tau_xv_smag_horavg = np.nanmean(dns_tau_xv_smag, axis=(1,2), keepdims=False)
    #dns_tau_yv_smag_horavg = np.nanmean(dns_tau_yv_smag, axis=(1,2), keepdims=False)
    #dns_tau_zv_smag_horavg = np.nanmean(dns_tau_zv_smag, axis=(1,2), keepdims=False)
    #dns_tau_xw_smag_horavg = np.nanmean(dns_tau_xw_smag, axis=(1,2), keepdims=False)
    #dns_tau_yw_smag_horavg = np.nanmean(dns_tau_yw_smag, axis=(1,2), keepdims=False)
    #dns_tau_zw_smag_horavg = np.nanmean(dns_tau_zw_smag, axis=(1,2), keepdims=False)
    
    #Make spectra of labels and MLP predictions
    #make_spectra_heights(dns_tau_xu_smag, dns_tau_xu, z,       'xu', time_step = 0)
    #make_spectra_heights(dns_tau_yu_smag, dns_tau_yu, z,       'yu', time_step = 0)
    #make_spectra_heights(dns_tau_zu_smag, dns_tau_zu, z,      'zu', time_step = 0)
    #make_spectra_heights(dns_tau_xv_smag, dns_tau_xv, z,       'xv', time_step = 0)
    #make_spectra_heights(dns_tau_yv_smag, dns_tau_yv, z,       'yv', time_step = 0)
    #make_spectra_heights(dns_tau_zv_smag, dns_tau_zv, z,      'zv', time_step = 0)
    #make_spectra_heights(dns_tau_xw_smag, dns_tau_xw, z,      'xw', time_step = 0)
    #make_spectra_heights(dns_tau_yw_smag, dns_tau_yw, z,      'yw', time_step = 0)
    #make_spectra_heights(dns_tau_zw_smag, dns_tau_zw, z,       'zw', time_step = 0)
    #
    ##Plot vertical profiles
    #make_vertprof(dns_tau_xu_smag_horavg, dns_tau_xu_horavg, z,      'xu', time_step = 0)
    #make_vertprof(dns_tau_yu_smag_horavg, dns_tau_yu_horavg, z,      'yu', time_step = 0)
    #make_vertprof(dns_tau_zu_smag_horavg, dns_tau_zu_horavg, z,     'zu', time_step = 0)
    #make_vertprof(dns_tau_xv_smag_horavg, dns_tau_xv_horavg, z,      'xv', time_step = 0)
    #make_vertprof(dns_tau_yv_smag_horavg, dns_tau_yv_horavg, z,      'yv', time_step = 0)
    #make_vertprof(dns_tau_zv_smag_horavg, dns_tau_zv_horavg, z,     'zv', time_step = 0)
    #make_vertprof(dns_tau_xw_smag_horavg, dns_tau_xw_horavg, z,     'xw', time_step = 0)
    #make_vertprof(dns_tau_yw_smag_horavg, dns_tau_yw_horavg, z,     'yw', time_step = 0)
    #make_vertprof(dns_tau_zw_smag_horavg, dns_tau_zw_horavg, z,      'zw', time_step = 0)

    ##Make PDFs of labels and MLP predictions
    #make_pdfs_heights(dns_tau_xu_smag, dns_tau_xu, z,       'xu', time_step = 0)
    #make_pdfs_heights(dns_tau_yu_smag, dns_tau_yu, z,       'yu', time_step = 0)
    make_pdfs_heights(dns_tau_zu_smag, dns_tau_zu, z,      'zu', time_step = 0)
    #make_pdfs_heights(dns_tau_xv_smag, dns_tau_xv, z,       'xv', time_step = 0)
    #make_pdfs_heights(dns_tau_yv_smag, dns_tau_yv, z,       'yv', time_step = 0)
    #make_pdfs_heights(dns_tau_zv_smag, dns_tau_zv, z,      'zv', time_step = 0)
    #make_pdfs_heights(dns_tau_xw_smag, dns_tau_xw, z,      'xw', time_step = 0)
    #make_pdfs_heights(dns_tau_yw_smag, dns_tau_yw, z,      'yw', time_step = 0)
    #make_pdfs_heights(dns_tau_zw_smag, dns_tau_zw, z,       'zw', time_step = 0)
    
    ##Make horizontal cross-sections
    ##NOTE1: some transport components are adjusted to convert them in a consistent way to equal shapes.
    #make_horcross_heights(dns_tau_xu, zh, yh, xh,       'xu', True, time_step = 0)
    #make_horcross_heights(dns_tau_yu, zh, yh, xh, 'yu', True, time_step = 0)
    make_horcross_heights(dns_tau_zu, zh, yh, xh, 'zu', True, time_step = 0)
    #make_horcross_heights(dns_tau_xv, zh, yh, xh, 'xv', True, time_step = 0)
    #make_horcross_heights(dns_tau_yv, zh, yh, xh,       'yv', True, time_step = 0)
    #make_horcross_heights(dns_tau_zv, zh, yh, xh, 'zv', True, time_step = 0)
    #make_horcross_heights(dns_tau_xw, zh, yh, xh, 'xw', True, time_step = 0)
    #make_horcross_heights(dns_tau_yw, zh, yh, xh, 'yw', True, time_step = 0)
    #make_horcross_heights(dns_tau_zw, zh, yh, xh,       'zw', True, time_step = 0)
    #make_horcross_heights(dns_tau_xu_smag, zh, yh, xh,       'xu', False, time_step = 0)
    #make_horcross_heights(dns_tau_yu_smag, zh, yh, xh, 'yu', False, time_step = 0)
    make_horcross_heights(dns_tau_zu_smag, zh, yh, xh, 'zu', False, time_step = 0)
    #make_horcross_heights(dns_tau_xv_smag, zh, yh, xh, 'xv', False, time_step = 0)
    #make_horcross_heights(dns_tau_yv_smag, zh, yh, xh,       'yv', False, time_step = 0)
    #make_horcross_heights(dns_tau_zv_smag, zh, yh, xh, 'zv', False, time_step = 0)
    #make_horcross_heights(dns_tau_xw_smag, zh, yh, xh, 'xw', False, time_step = 0)
    #make_horcross_heights(dns_tau_yw_smag, zh, yh, xh, 'yw', False, time_step = 0)
    #make_horcross_heights(dns_tau_zw_smag, zh, yh, xh,       'zw', False, time_step = 0)
    #
    ##Make scatterplots
    ##NOTE: some transport components are adjusted to convert them in a consistent way to equal shapes.
    #
    #make_scatterplot_heights(dns_tau_xu_smag, dns_tau_xu, dns_tau_xu_smag_horavg, dns_tau_xu_horavg, z,  'xu', time_step = 0)
    #make_scatterplot_heights(dns_tau_yu_smag, dns_tau_yu, dns_tau_yu_smag_horavg, dns_tau_yu_horavg, z,  'yu', time_step = 0)
    make_scatterplot_heights(dns_tau_zu_smag, dns_tau_zu, dns_tau_zu_smag_horavg, dns_tau_zu_horavg, z, 'zu', time_step = 0)
    #make_scatterplot_heights(dns_tau_xv_smag, dns_tau_xv, dns_tau_xv_smag_horavg, dns_tau_xv_horavg, z,  'xv', time_step = 0)
    #make_scatterplot_heights(dns_tau_yv_smag, dns_tau_yv, dns_tau_yv_smag_horavg, dns_tau_yv_horavg, z,  'yv', time_step = 0)
    #make_scatterplot_heights(dns_tau_zv_smag, dns_tau_zv, dns_tau_zv_smag_horavg, dns_tau_zv_horavg, z, 'zv', time_step = 0)
    #make_scatterplot_heights(dns_tau_xw_smag, dns_tau_xw, dns_tau_xw_smag_horavg, dns_tau_xw_horavg, z, 'xw', time_step = 0)
    #make_scatterplot_heights(dns_tau_yw_smag, dns_tau_yw, dns_tau_yw_smag_horavg, dns_tau_yw_horavg, z, 'yw', time_step = 0)
    #make_scatterplot_heights(dns_tau_zw_smag, dns_tau_zw, dns_tau_zw_smag_horavg, dns_tau_zw_horavg, z,  'zw', time_step = 0)
    #
    
#Close files
dns.close()
smag.close()
print('Finished')
