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
#NOTE: remove -1's later!!!
dns_tau_xu   = np.array(dns['unres_tau_xu_turb'][time_step,:,:,:-1])
dns_tau_yu   = np.array(dns['unres_tau_yu_turb'][time_step,:,:,:])
dns_tau_zu   = np.array(dns['unres_tau_zu_turb'][time_step,:,:,:])
dns_tau_xv   = np.array(dns['unres_tau_xv_turb'][time_step,:,:,:])
dns_tau_yv   = np.array(dns['unres_tau_yv_turb'][time_step,:,:-1,:])
dns_tau_zv   = np.array(dns['unres_tau_zv_turb'][time_step,:,:,:])
dns_tau_xw   = np.array(dns['unres_tau_xw_turb'][time_step,:,:,:])
dns_tau_yw   = np.array(dns['unres_tau_yw_turb'][time_step,:,:,:])
dns_tau_zw   = np.array(dns['unres_tau_zw_turb'][time_step,:-1,:,:])
#
dns_tau_xu_smag  = np.array(smag['smag_tau_xu'][time_step,:,:,:])
dns_tau_yu_smag  = np.array(smag['smag_tau_yu'][time_step,:,:,:])
dns_tau_zu_smag  = np.array(smag['smag_tau_zu'][time_step,:,:,:])
dns_tau_xv_smag  = np.array(smag['smag_tau_xv'][time_step,:,:,:])
dns_tau_yv_smag  = np.array(smag['smag_tau_yv'][time_step,:,:,:])
dns_tau_zv_smag  = np.array(smag['smag_tau_zv'][time_step,:,:,:])
dns_tau_xw_smag  = np.array(smag['smag_tau_xw'][time_step,:,:,:])
dns_tau_yw_smag  = np.array(smag['smag_tau_yw'][time_step,:,:,:])
dns_tau_zw_smag  = np.array(smag['smag_tau_zw'][time_step,:,:,:])
#
#Extract coordinates
nt = len(dns.dimensions['time'])
z  = np.array(dns['z'][:])
nz = len(z)
zh = np.array(dns['zh'][:])
y  = np.array(dns['y'][:])
ny = len(y)
yh = np.array(dns['yh'][:])
x  = np.array(dns['x'][:])
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
            a = ma.masked_invalid(dns_tau_zu_smag[k,:,:].flatten())
            b = ma.masked_invalid(dns_tau_zu[k,:,:].flatten())
            msk = (~a.mask & ~b.mask)
            corrcoef_zu_smag[k+1] = np.round(ma.corrcoef(dns_tau_zu_smag[k,:,:].flatten()[msk], dns_tau_zu[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_xv_smag[k+1] = np.nan
            corrcoef_yv_smag[k+1] = np.nan
            a = ma.masked_invalid(dns_tau_zv_smag[k,:,:].flatten())
            b = ma.masked_invalid(dns_tau_zv[k,:,:].flatten())
            msk = (~a.mask & ~b.mask)
            corrcoef_zv_smag[k+1] = np.round(ma.corrcoef(dns_tau_zv_smag[k,:,:].flatten()[msk], dns_tau_zv[k,:,:].flatten()[msk])[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
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
def make_horcross_heights(values, z, y, x, component, is_lbl, is_smag = False, time_step = 0, delta = 1):
    #NOTE1: fifth last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: fourth last input of this function is a boolean that specifies whether the labels (True) or the NN predictions are being plotted.
    #NOTE3: thirth last input of this function is a boolean that specifies whether the Smagorinsky fluxes are plotted (True) or not (False)
    #NOTE4: the second last input of this function is an integer specifying which validation time step stored in the nc-file is plotted (by default the first one, which now corresponds to time step 28 used for validation).
    #NOTE5: the last input of this function is an integer specifying the channel half with [in meter] used to rescale the horizontal dimensions (by default 1m, effectively not rescaling). 

    #Check that component is not both specified as label and Smagorinsky value
    if is_lbl and is_smag:
        raise RuntimeError("Value specified as both label and Smagorinsky value, which is not possible.")

    for k in range(len(z)-1):
        values_height = values[time_step,k,:,:] / (utau_ref ** 2.)

        #Make horizontal cross-sections of the values
        plt.figure()
        if is_smag:
            plt.pcolormesh(x, y, values_height, vmin=-0.5, vmax=0.5)
        else:
            plt.pcolormesh(x, y, values_height, vmin=-5.0, vmax=5.0)
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
            plt.savefig("Horcross_label_tau_" + component + "_" + str((z[k]+z[k+1])/2.) + ".png", dpi = 200)
        elif is_smag:
            plt.savefig("Horcross_smag_tau_" + component + "_" + str((z[k]+z[k+1])/2.) + ".png", dpi = 200)
        else:
            plt.savefig("Horcross_tau_" + component + "_" + str((z[k]+z[k+1])/2.) + ".png", dpi = 200)
        plt.close()

#Define function for making spectra
def make_spectra_heights(ann, smag, dns, z, component, time_step = 0):
    #NOTE1: second last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: last input of this function is an integer specifying which validation time step stored in the nc-file is plotted (by default the first one, which now corresponds to time step 28 used for validation).
    for k in range(len(z)):
        
        ann_height  = ann[time_step,k,:,:]  / (utau_ref ** 2.)
        smag_height = smag[time_step,k,:,:] / (utau_ref ** 2.)
        dns_height  = dns[time_step,k,:,:]  / (utau_ref ** 2.)
        #range_bins = (-2.0,2.0)

        #Calculate spectra
        #ANN
        nxc = ann_height.shape[1]
        nyc = ann_height.shape[0]
        fftx_ann = np.fft.rfft(ann_height,axis=1)*(1/nxc)
        ffty_ann = np.fft.rfft(ann_height,axis=0)*(1/nyc)
        Px_ann = fftx_ann[:,1:] * np.conjugate(fftx_ann[:,1:])
        Py_ann = ffty_ann[1:,:] * np.conjugate(ffty_ann[1:,:])
        if int(nxc % 2) == 0:
            Ex_ann = np.append(2*Px_ann[:,:-1],np.reshape(Px_ann[:,-1],(nyc,1)),axis=1)
        else:
            Ex_ann = 2*Px_ann[:,:]
        
        if int(nyc % 2) == 0:
            Ey_ann = np.append(2*Py_ann[:-1,:],np.reshape(Py_ann[-1,:],(1,nxc)),axis=0)
        else:
            Ey_ann = 2*Py_ann[:,:]

        ann_spec_x = np.nanmean(Ex_ann,axis=0) #Average FT over direction where it was not calculated
        ann_spec_y = np.nanmean(Ey_ann,axis=1)
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
        plt.plot(ann_spec_x,  label = 'ANN')
        #plt.plot(smag_spec_x, label = 'Smagorinsky')
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
        plt.plot(ann_spec_y,  label = 'ANN')
        #plt.plot(smag_spec_y, label = 'Smagorinsky')
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
def make_pdfs_heights(values, smag, labels, z, component, time_step = 0):
    #NOTE1: second last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: last input of this function is an integer specifying which validation time step stored in the nc-file is plotted (by default the first one, which now corresponds to time step 28 used for validation).
    for k in range(len(z)+1):
        if k == len(z):
            values_height = values[time_step,:,:,:].flatten() / (utau_ref ** 2.)
            smag_height   = smag[time_step,:,:,:].flatten() / (utau_ref ** 2.)
            labels_height = labels[time_step,:,:,:].flatten() / (utau_ref ** 2.)
            #range_bins = (0.6,0.6)
        else:
            values_height = values[time_step,k,:,:].flatten() / (utau_ref ** 2.)
            smag_height   = smag[time_step,k,:,:].flatten() / (utau_ref ** 2.)
            labels_height = labels[time_step,k,:,:].flatten() / (utau_ref ** 2.)
            #range_bins = (-2.0,2.0)

        #Determine bins
        num_bins = 100
        min_val = min(values_height.min(), labels_height.min())
        max_val = max(values_height.max(), labels_height.max())
        bin_edges = np.linspace(min_val, max_val, num_bins)

        #Make pdfs of the values and labels
        plt.figure()
        plt.hist(values_height, bins = bin_edges, density = True, histtype = 'step', label = 'ANN')
        #plt.hist(smag_height, bins = bin_edges, density = True, histtype = 'step', label = 'Smagorinsky')
        plt.hist(labels_height, bins = bin_edges, density = True, histtype = 'step', label = 'DNS')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.008)
        ax.set_xlim(left=-10, right=10)
        plt.ylabel(r'$\rm Probability\ density\ [-]$',fontsize=20)
        plt.xlabel(r'$\rm \frac{\tau_{wu}}{u_{\tau}^2} \ [-]$',fontsize=20)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16, rotation=0)
        plt.legend(loc='upper right')
        plt.tight_layout()
        if k == len(z):
            plt.savefig("PDF_tau_" + component + ".png", dpi = 200)
        else:
            plt.savefig("PDF_tau_" + component + "_" + str(z[k]) + ".png", dpi = 200)
        plt.close()

#Define function for making pdfs
def make_vertprof(values, smag, labels, z, component, time_step = 0):
    #NOTE1: second last input of this function is a string indicating the name of the component being plotted.
    #last input of this function is an integer specifying which validation time step stored in the nc-file is plotted (by default the first one, which now corresponds to time step 28 used for validation).

    #Make vertical profile
    plt.figure()
    plt.plot(z, values[time_step,:] / (utau_ref ** 2.), label = 'ANN', marker = 'o', markersize = 2.0)
    #plt.plot(z, smag[time_step,:] / (utau_ref ** 2.), label = 'Smagorinsky', marker = 'o', markersize = 2.0)
    plt.plot(z, labels[time_step,:] / (utau_ref ** 2.), label = 'DNS', marker = 'o', markersize = 2.0)
    #ax = plt.gca()
    #ax.set_yscale('log')
    plt.ylabel(r'$\rm \frac{\tau_{wu}}{u_{\tau}^2} \ [-]$',fontsize=20)
    plt.xlabel(r'$\rm \frac{z}{\delta} \ [-]$',fontsize=20)
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16, rotation=0)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("vertprof_tau_" + component + ".png", dpi = 200)
    plt.close()

#Define function for making scatterplots
def make_scatterplot_heights(preds, lbls, preds_horavg, lbls_horavg, heights, component, is_smag, time_step):
    #NOTE1: third last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: second last input of this function is a boolean that specifies whether the Smagorinsky fluxes are being plotted (True) or the CNN fluxes (False).
    for k in range(len(heights)+1):
        if k == len(heights):
            preds_height = preds_horavg[time_step,:] / (utau_ref ** 2.)
            lbls_height  = lbls_horavg[time_step,:] / (utau_ref ** 2.)
        else:
            preds_height = preds[time_step,k,:,:] / (utau_ref ** 2.)
            lbls_height  = lbls[time_step,k,:,:] / (utau_ref ** 2.)
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
            plt.xlim([-15.0, 15.0])
            plt.ylim([-15.0, 15.0])
            #plt.xlim([-40.0, 40.0])
            #plt.ylim([-40.0, 40.0])
            #plt.xlim([-0.0005, 0.0005])
            #plt.ylim([-0.0005, 0.0005])
        axes = plt.gca()
        plt.plot(axes.get_xlim(),axes.get_ylim(),'b--')
        #plt.gca().set_aspect('equal',adjustable='box')
        plt.xlabel(r'$\rm \frac{\tau_{wu}^{DNS}}{u_{\tau}^2} \,\ {[-]}$',fontsize = 20)
        if is_smag:
            plt.ylabel(r'$\rm \frac{\tau_{wu}^{smag}}{u_{\tau}^2} \,\ {[-]}$',fontsize = 20)
        else:
            plt.ylabel(r'$\rm \frac{\tau_{wu}^{ANN}}{u_{\tau}^2} \,\ {[-]}$',fontsize = 20)
        #plt.title("ρ = " + str(corrcoef),fontsize = 20)
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
                plt.savefig("Scatter_tau_" + component + "_horavg.png", dpi = 200)
            else:
                plt.savefig("Scatter_tau_" + component + "_" + str(heights[k]) + ".png", dpi = 200)
        plt.tight_layout()
        plt.close()


#Call function multiple times to make all plots for smagorinsky and CNN
if args.make_plots:
    print('start making plots')
    
    ##Make spectra of labels and MLP predictions
    #make_spectra_heights(unres_tau_xu_CNN, unres_tau_xu_smag, unres_tau_xu, zc,       'xu', time_step = 0)
    #make_spectra_heights(unres_tau_yu_CNN, unres_tau_yu_smag, unres_tau_yu, zc,       'yu', time_step = 0)
    #make_spectra_heights(unres_tau_zu_CNN, unres_tau_zu_smag, unres_tau_zu, zhc,      'zu', time_step = 0)
    #make_spectra_heights(unres_tau_xv_CNN, unres_tau_xv_smag, unres_tau_xv, zc,       'xv', time_step = 0)
    #make_spectra_heights(unres_tau_yv_CNN, unres_tau_yv_smag, unres_tau_yv, zc,       'yv', time_step = 0)
    #make_spectra_heights(unres_tau_zv_CNN, unres_tau_zv_smag, unres_tau_zv, zhc,      'zv', time_step = 0)
    #make_spectra_heights(unres_tau_xw_CNN, unres_tau_xw_smag, unres_tau_xw, zhcless,  'xw', time_step = 0)
    #make_spectra_heights(unres_tau_yw_CNN, unres_tau_yw_smag, unres_tau_yw, zhcless,  'yw', time_step = 0)
    #make_spectra_heights(unres_tau_zw_CNN, unres_tau_zw_smag, unres_tau_zw, zc,       'zw', time_step = 0)
    #
    ##Plot vertical profiles
    #make_vertprof(unres_tau_xu_CNN_horavg, unres_tau_xu_smag_horavg, unres_tau_xu_horavg, zc,      'xu', time_step = 0)
    #make_vertprof(unres_tau_yu_CNN_horavg, unres_tau_yu_smag_horavg, unres_tau_yu_horavg, zc,      'yu', time_step = 0)
    #make_vertprof(unres_tau_zu_CNN_horavg, unres_tau_zu_smag_horavg, unres_tau_zu_horavg, zhc,     'zu', time_step = 0)
    #make_vertprof(unres_tau_xv_CNN_horavg, unres_tau_xv_smag_horavg, unres_tau_xv_horavg, zc,      'xv', time_step = 0)
    #make_vertprof(unres_tau_yv_CNN_horavg, unres_tau_yv_smag_horavg, unres_tau_yv_horavg, zc,      'yv', time_step = 0)
    #make_vertprof(unres_tau_zv_CNN_horavg, unres_tau_zv_smag_horavg, unres_tau_zv_horavg, zhc,     'zv', time_step = 0)
    #make_vertprof(unres_tau_xw_CNN_horavg, unres_tau_xw_smag_horavg, unres_tau_xw_horavg, zhcless, 'xw', time_step = 0)
    #make_vertprof(unres_tau_yw_CNN_horavg, unres_tau_yw_smag_horavg, unres_tau_yw_horavg, zhcless, 'yw', time_step = 0)
    #make_vertprof(unres_tau_zw_CNN_horavg, unres_tau_zw_smag_horavg, unres_tau_zw_horavg, zc,      'zw', time_step = 0)

    ##Make PDFs of labels and MLP predictions
    #make_pdfs_heights(unres_tau_xu_CNN, unres_tau_xu_smag, unres_tau_xu, zc,       'xu', time_step = 0)
    #make_pdfs_heights(unres_tau_yu_CNN, unres_tau_yu_smag, unres_tau_yu, zc,       'yu', time_step = 0)
    #make_pdfs_heights(unres_tau_zu_CNN, unres_tau_zu_smag, unres_tau_zu, zhc,      'zu', time_step = 0)
    #make_pdfs_heights(unres_tau_xv_CNN, unres_tau_xv_smag, unres_tau_xv, zc,       'xv', time_step = 0)
    #make_pdfs_heights(unres_tau_yv_CNN, unres_tau_yv_smag, unres_tau_yv, zc,       'yv', time_step = 0)
    #make_pdfs_heights(unres_tau_zv_CNN, unres_tau_zv_smag, unres_tau_zv, zhc,      'zv', time_step = 0)
    #make_pdfs_heights(unres_tau_xw_CNN, unres_tau_xw_smag, unres_tau_xw, zhcless,  'xw', time_step = 0)
    #make_pdfs_heights(unres_tau_yw_CNN, unres_tau_yw_smag, unres_tau_yw, zhcless,  'yw', time_step = 0)
    #make_pdfs_heights(unres_tau_zw_CNN, unres_tau_zw_smag, unres_tau_zw, zc,       'zw', time_step = 0)
    
    ##Make horizontal cross-sections
    ##NOTE1: some transport components are adjusted to convert them in a consistent way to equal shapes.
    #make_horcross_heights(unres_tau_xu, zhc, yhc, xhc,           'xu', True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_yu, zhc, ygcextra, xgcextra, 'yu', True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_zu, zgcextra, yhc, xgcextra, 'zu', True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_xv, zhc, ygcextra, xgcextra, 'xv', True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_yv, zhc, yhc, xhc,           'yv', True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_zv, zgcextra, ygcextra, xhc, 'zv', True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_xw, zgcextra, yhc, xgcextra, 'xw', True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_yw, zgcextra, ygcextra, xhc, 'yw', True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_zw, zhc, yhc, xhc,           'zw', True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_xu_CNN, zhc, yhc, xhc,           'xu', False, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_yu_CNN, zhc, ygcextra, xgcextra, 'yu', False, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_zu_CNN, zgcextra, yhc, xgcextra, 'zu', False, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_xv_CNN, zhc, ygcextra, xgcextra, 'xv', False, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_yv_CNN, zhc, yhc, xhc,           'yv', False, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_zv_CNN, zgcextra, ygcextra, xhc, 'zv', False, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_xw_CNN, zgcextra, yhc, xgcextra, 'xw', False, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_yw_CNN, zgcextra, ygcextra, xhc, 'yw', False, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_zw_CNN, zhc, yhc, xhc,           'zw', False, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_xu_smag, zhc, yhc, xhc,           'xu', True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_yu_smag, zhc, ygcextra, xgcextra, 'yu', True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_zu_smag, zgcextra, yhc, xgcextra, 'zu', False, True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_xv_smag, zhc, ygcextra, xgcextra, 'xv', False, True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_yv_smag, zhc, yhc, xhc,           'yv', False, True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_zv_smag, zgcextra, ygcextra, xhc, 'zv', False, True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_xw_smag, zgcextra, yhc, xgcextra, 'xw', False, True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_yw_smag, zgcextra, ygcextra, xhc, 'yw', False, True, time_step = 0, delta = delta_height)
    #make_horcross_heights(unres_tau_zw_smag, zhc, yhc, xhc,           'zw', False, True, time_step = 0, delta = delta_height)
    #
    ##Make scatterplots
    ##NOTE: some transport components are adjusted to convert them in a consistent way to equal shapes.
    #make_scatterplot_heights(unres_tau_xu_CNN, unres_tau_xu, unres_tau_xu_CNN_horavg, unres_tau_xu_horavg, zc,  'xu', False, time_step = 0)
    #make_scatterplot_heights(unres_tau_yu_CNN, unres_tau_yu, unres_tau_yu_CNN_horavg, unres_tau_yu_horavg, zc,  'yu', False, time_step = 0)
    #make_scatterplot_heights(unres_tau_zu_CNN, unres_tau_zu, unres_tau_zu_CNN_horavg, unres_tau_zu_horavg, zhc, 'zu', False, time_step = 0)
    #make_scatterplot_heights(unres_tau_xv_CNN, unres_tau_xv, unres_tau_xv_CNN_horavg, unres_tau_xv_horavg, zc,  'xv', False, time_step = 0)
    #make_scatterplot_heights(unres_tau_yv_CNN, unres_tau_yv, unres_tau_yv_CNN_horavg, unres_tau_yv_horavg, zc,  'yv', False, time_step = 0)
    #make_scatterplot_heights(unres_tau_zv_CNN, unres_tau_zv, unres_tau_zv_CNN_horavg, unres_tau_zv_horavg, zhc, 'zv', False, time_step = 0)
    #make_scatterplot_heights(unres_tau_xw_CNN, unres_tau_xw, unres_tau_xw_CNN_horavg, unres_tau_xw_horavg, zhcless, 'xw', False, time_step = 0)
    #make_scatterplot_heights(unres_tau_yw_CNN, unres_tau_yw, unres_tau_yw_CNN_horavg, unres_tau_yw_horavg, zhcless, 'yw', False, time_step = 0)
    #make_scatterplot_heights(unres_tau_zw_CNN, unres_tau_zw, unres_tau_zw_CNN_horavg, unres_tau_zw_horavg, zc,  'zw', False, time_step = 0)
    #
    #make_scatterplot_heights(unres_tau_xu_smag, unres_tau_xu, unres_tau_xu_smag_horavg, unres_tau_xu_horavg, zc,  'xu', True, time_step = 0)
    #make_scatterplot_heights(unres_tau_yu_smag, unres_tau_yu, unres_tau_yu_smag_horavg, unres_tau_yu_horavg, zc,  'yu', True, time_step = 0)
    #make_scatterplot_heights(unres_tau_zu_smag, unres_tau_zu, unres_tau_zu_smag_horavg, unres_tau_zu_horavg, zhc, 'zu', True, time_step = 0)
    #make_scatterplot_heights(unres_tau_xv_smag, unres_tau_xv, unres_tau_xv_smag_horavg, unres_tau_xv_horavg, zc,  'xv', True, time_step = 0)
    #make_scatterplot_heights(unres_tau_yv_smag, unres_tau_yv, unres_tau_yv_smag_horavg, unres_tau_yv_horavg, zc,  'yv', True, time_step = 0)
    #make_scatterplot_heights(unres_tau_zv_smag, unres_tau_zv, unres_tau_zv_smag_horavg, unres_tau_zv_horavg, zhc, 'zv', True, time_step = 0)
    #make_scatterplot_heights(unres_tau_xw_smag, unres_tau_xw, unres_tau_xw_smag_horavg, unres_tau_xw_horavg, zhcless, 'xw', True, time_step = 0)
    #make_scatterplot_heights(unres_tau_yw_smag, unres_tau_yw, unres_tau_yw_smag_horavg, unres_tau_yw_horavg, zhcless, 'yw', True, time_step = 0)
    #make_scatterplot_heights(unres_tau_zw_smag, unres_tau_zw, unres_tau_zw_smag_horavg, unres_tau_zw_horavg, zc,  'zw', True, time_step = 0)
    #
    
#Close files
dns.close()
smag.close()
print('Finished')
