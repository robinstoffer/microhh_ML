#Script that calculates turbulent fluxes according to Smagorinsky sub-grid model.
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import numpy as np
import netCDF4 as nc

def calculate_turbulent_fluxes_dns(flowfields_filepath = 'dns_boxfilter.nc', eddy_diffusivity_filepath = 'eddy_diffusivity_dns.nc', output_filepath = 'smagorinsky_fluxes_dns.nc'):
    '''Calculate all nine turbulent transport components of the Reynolds stress tensor for stored flow fields according to the Smagorinsky sub-grid model, and store them in a netCDF-file. The inputs are as follows: \\
        -flowfields_filepath: string specifying the filepath of the file where the flow fields needed for the flux calculations are stored.\\
        -eddy_diffusivity_filepath: string specifying the filepath of the file in which the eddy diffusivities for the flux calculations are stored. \\
        NOTE: the filepaths have to refer to netCDF-files created with func_generate_training.py and calculate_eddy_diffusivity.py. \\
        -output_filepath: string specifying the filepath, which should include a name for the netCDF-file created in this script and a directory where it needs to be placed. '''

    #Check types input
    if not isinstance(flowfields_filepath,str):
        raise TypeError("Specified flow fields filepath should be a string.")

    if not isinstance(eddy_diffusivity_filepath,str):
        raise TypeError("Specified eddy diffusivity filepath should be a string.")
    
    if not isinstance(output_filepath,str):
        raise TypeError("Specified output filepath should be a string.")


    #Fetch grid information from flowfields_filepath
    a = nc.Dataset(flowfields_filepath, 'r')

    zh = np.array(a['zh'][:])
    z  = np.array(a['z'][:])
    yh = np.array(a['yh'][:])
    y  = np.array(a['y'][:])
    xh = np.array(a['xh'][:])
    x  = np.array(a['x'][:])
    nt = len(a.dimensions['time'])
    nz = len(z)
    ny = len(y)
    nx = len(x)

    #Open file with eddy diffusivity
    b = nc.Dataset(eddy_diffusivity_filepath, 'r')

    #Loop over timesteps
    create_file = True
    for t in range(nt):
    #for t in range(1,2): #NOTE:FOR TESTING PURPOSES ONLY!
        
        #Fetch eddy diffusivity coefficients
        evisc = b["eddy_diffusivity"][t,:,:,:]
        
        #Fetch flow fields
        u = np.array(a['uc'][t,:,:,:])
        v = np.array(a['vc'][t,:,:,:])
        w = np.array(a['wc'][t,:,:,:])

        #Open/create netCDF-file for storage
        if create_file:
            turbulentflux_file = nc.Dataset(output_filepath, 'w')
            create_file      = False
            create_variables = True
        else:
            turbulentflux_file = nc.Dataset(output_filepath, 'r+')
            create_variables = False #Don't define variables when netCDF file already exists, because it should already contain those variables.

        #Define arrays for storage
        smag_tau_xu = np.empty((nz  ,ny  ,nx)  , dtype=float)
        smag_tau_xu.fill(np.nan)
        smag_tau_yu = np.empty((nz  ,ny+1,nx+1), dtype=float)
        smag_tau_yu.fill(np.nan)
        smag_tau_zu = np.empty((nz+1,ny  ,nx+1), dtype=float)
        smag_tau_zu.fill(np.nan)
        smag_tau_xv = np.empty((nz  ,ny+1,nx+1), dtype=float)
        smag_tau_xv.fill(np.nan)
        smag_tau_yv = np.empty((nz  ,ny  ,nx)  , dtype=float)
        smag_tau_yv.fill(np.nan)
        smag_tau_zv = np.empty((nz+1,ny+1,nx)  , dtype=float)
        smag_tau_zv.fill(np.nan)
        smag_tau_xw = np.empty((nz+1,ny  ,nx+1), dtype=float)
        smag_tau_xw.fill(np.nan)
        smag_tau_yw = np.empty((nz+1,ny+1,nx)  , dtype=float)
        smag_tau_yw.fill(np.nan)
        smag_tau_zw = np.empty((nz  ,ny  ,nx)  , dtype=float)
        smag_tau_zw.fill(np.nan)

        #Loop over grid cells to calculate the fluxes
        #NOTE: loop indices chosen such that no ghost cells are needed: bottom and top layer are left empy, because they are not needed for the comparison with DNS
        for k in range(1,len(z)-1):
            #k_stag = k - kgc_center #Take into account that the staggered vertical dimension does not contain one ghost cell
            dz    = zh[k+1]- zh[k]
            dzi   = 1./dz
            dzhib = 1./(z[k] - z[k-1])
            #dzhit = 1./(z[k+1] - z[k])

            for j in range(1,len(y)-1):
                dy    = yh[j+1]- yh[j]
                dyi   = 1./dy
                dyhib = 1./(y[j] - y[j-1])
                #dyhit = 1./(y[j+1] - y[j])

                for i in range(1, len(x)-1):
                    dx    = xh[i+1]- xh[i]
                    dxi   = 1./dx
                    dxhib = 1./(x[i] - x[i-1])
                    #dxhit = 1./(x[i+1] - x[i])

                    #Calculate eddy viscosity coefficients on corners
                    eviscnu = 0.25*(evisc[k,j,i-1]   + evisc[k,j,i]   + evisc[k,j+1,i-1] + evisc[k,j+1,i])
                    eviscsu = 0.25*(evisc[k,j-1,i-1] + evisc[k,j-1,i] + evisc[k,j,i-1]   + evisc[k,j,i])
                    evisctu = 0.25*(evisc[k,j,i-1]   + evisc[k,j,i]   + evisc[k+1,j,i-1] + evisc[k+1,j,i])
                    eviscbu = 0.25*(evisc[k-1,j,i-1] + evisc[k-1,j,i] + evisc[k,j,i-1]   + evisc[k,j,i])
                    eviscev = 0.25*(evisc[k,j-1,i]   + evisc[k,j,i]   + evisc[k,j-1,i+1] + evisc[k,j,i+1])
                    eviscwv = 0.25*(evisc[k,j-1,i-1] + evisc[k,j,i-1] + evisc[k,j-1,i]   + evisc[k,j,i])
                    evisctv = 0.25*(evisc[k,j-1,i]   + evisc[k,j,i]   + evisc[k+1,j-1,i] + evisc[k+1,j,i])
                    eviscbv = 0.25*(evisc[k-1,j-1,i] + evisc[k-1,j,i] + evisc[k,j-1,i]   + evisc[k,j,i]) 
                    eviscew = 0.25*(evisc[k-1,j,i]   + evisc[k,j,i]   + evisc[k-1,j,i+1] + evisc[k,j,i+1])
                    eviscww = 0.25*(evisc[k-1,j,i-1] + evisc[k,j,i-1] + evisc[k-1,j,i]   + evisc[k,j,i])
                    eviscnw = 0.25*(evisc[k-1,j,i]   + evisc[k,j,i]   + evisc[k-1,j+1,i] + evisc[k,j+1,i])
                    eviscsw = 0.25*(evisc[k-1,j-1,i] + evisc[k,j-1,i] + evisc[k-1,j,i]   + evisc[k,j,i])

                    #Calculate turbulent fluxes accoring to Smagorinsky-Lilly model.
                    smag_tau_xu[k,j,i] = -2. * evisc[k,j,i]  * (u[k,j,i+1] - u[k,j,i]) * dxi
                    smag_tau_xv[k,j,i] = -1. * eviscsu * ((u[k,j,i] - u[k,j-1,i]) * dyhib + (v[k,j,i] - v[k,j,i-1]) * dxhib)
                    smag_tau_xw[k,j,i] = -1. * eviscbu * ((u[k,j,i] - u[k-1,j,i]) * dzhib + (w[k,j,i] - w[k,j,i-1]) * dxhib)
                    smag_tau_yu[k,j,i] = -1. * eviscwv * ((v[k,j,i] - v[k,j,i-1]) * dxhib + (u[k,j,i] - u[k,j-1,i]) * dyhib)
                    smag_tau_yv[k,j,i] = -2. * evisc[k,j,i] * (v[k,j+1,i] - v[k,j,i]) * dyi
                    smag_tau_yw[k,j,i] = -1. * eviscbv * ((v[k,j,i] - v[k-1,j,i]) * dzhib + (w[k,j,i] - w[k,j-1,i]) * dyhib)
                    smag_tau_zu[k,j,i] = -1. * eviscww * ((w[k,j,i] - w[k,j,i-1]) * dxhib + (u[k,j,i] - u[k-1,j,i]) * dzhib)
                    smag_tau_zv[k,j,i] = -1. * eviscsw * ((w[k,j,i] - w[k,j-1,i]) * dyhib + (v[k,j,i] - v[k-1,j,i]) * dzhib)
                    smag_tau_zw[k,j,i] = -2. * evisc[k,j,i] * (w[k+1,j,i] - w[k,j,i]) * dzi
                    
#        #If there is no flux at the bottom/top boundaries (i.e. when zero_w_topbottom = True), the fluxes located at the bottom and top are set to 0.
#        if zero_w_topbottom:
#
#            #Set fluxes at bottom boundary to 0 (the calculations below ensure that the fluxes are also set to 0 at the top boundary).
#            smag_tau_zu[0,:,:] = 0.
#            smag_tau_zv[0,:,:] = 0.
#            smag_tau_xw[0,:,:] = 0.
#            smag_tau_yw[0,:,:] = 0.
#        
        ##Add one top/downstream cell to fluxes in the directions where they are located on the grid edges, assuming they are the same as the one at the bottom/upstream edge of the domain.
        ##NOTE: this should work when the horizontal directions have periodic BCs, and the vertical direction has a no-slip BC (ONLY when resolved viscous flux is not added, as is done now).
        ##z-direction
        #smag_tau_zu[-1,:,:] = smag_tau_zu[0,:,:]
        #smag_tau_zv[-1,:,:] = smag_tau_zv[0,:,:]
        #smag_tau_xw[-1,:,:] = smag_tau_xw[0,:,:]
        #smag_tau_yw[-1,:,:] = smag_tau_yw[0,:,:]

        ##y-direction
        #smag_tau_yu[:,-1,:] = smag_tau_yu[:,0,:]
        #smag_tau_xv[:,-1,:] = smag_tau_xv[:,0,:]
        #smag_tau_zv[:,-1,:] = smag_tau_zv[:,0,:]
        #smag_tau_yw[:,-1,:] = smag_tau_yw[:,0,:]

        ##x-direction
        #smag_tau_yu[:,:,-1] = smag_tau_yu[:,:,0]
        #smag_tau_zu[:,:,-1] = smag_tau_zu[:,:,0]
        #smag_tau_xv[:,:,-1] = smag_tau_xv[:,:,0]
        #smag_tau_xw[:,:,-1] = smag_tau_xw[:,:,0]

        #Store calculated values in nc-file
        if create_variables:
            
            #Create new dimensions
            turbulentflux_file.createDimension("nt",nt)
            turbulentflux_file.createDimension("zh",len(zh))
            turbulentflux_file.createDimension("z",len(z))
            turbulentflux_file.createDimension("yh",len(yh))
            turbulentflux_file.createDimension("y",len(y))
            turbulentflux_file.createDimension("xh",len(xh))
            turbulentflux_file.createDimension("x",len(x))

            #Create new variables
            varsmag_tau_xu = turbulentflux_file.createVariable("smag_tau_xu","f8",("nt","z","y","x"))
            varsmag_tau_yu = turbulentflux_file.createVariable("smag_tau_yu","f8",("nt","z","yh","xh"))
            varsmag_tau_zu = turbulentflux_file.createVariable("smag_tau_zu","f8",("nt","zh","y","xh"))
            varsmag_tau_xv = turbulentflux_file.createVariable("smag_tau_xv","f8",("nt","z","yh","xh"))
            varsmag_tau_yv = turbulentflux_file.createVariable("smag_tau_yv","f8",("nt","z","y","x"))
            varsmag_tau_zv = turbulentflux_file.createVariable("smag_tau_zv","f8",("nt","zh","yh","x"))
            varsmag_tau_xw = turbulentflux_file.createVariable("smag_tau_xw","f8",("nt","zh","y","xh"))
            varsmag_tau_yw = turbulentflux_file.createVariable("smag_tau_yw","f8",("nt","zh","yh","x"))
            varsmag_tau_zw = turbulentflux_file.createVariable("smag_tau_zw","f8",("nt","z","y","x"))
            varzh         = turbulentflux_file.createVariable("zh","f8",("zh"))
            varz          = turbulentflux_file.createVariable("z","f8",("z"))
            varyh         = turbulentflux_file.createVariable("yh","f8",("yh"))
            vary          = turbulentflux_file.createVariable("y","f8",("y"))
            varxh         = turbulentflux_file.createVariable("xh","f8",("xh"))
            varx          = turbulentflux_file.createVariable("x","f8",("x"))

            #Store coordinate variables
            varzh[:]      = zh[:]
            varz[:]       = z[:]
            varyh[:]      = yh[:]
            vary[:]       = y[:]
            varxh[:]      = xh[:]
            varx[:]       = x[:]

            create_variables = False #Make sure the variables are only created once

        #Store variables
        varsmag_tau_xu[t,:,:,:] = smag_tau_xu[:,:,:]
        varsmag_tau_yu[t,:,:,:] = smag_tau_yu[:,:,:]
        varsmag_tau_zu[t,:,:,:] = smag_tau_zu[:,:,:]
        varsmag_tau_xv[t,:,:,:] = smag_tau_xv[:,:,:]
        varsmag_tau_yv[t,:,:,:] = smag_tau_yv[:,:,:]
        varsmag_tau_zv[t,:,:,:] = smag_tau_zv[:,:,:]
        varsmag_tau_xw[t,:,:,:] = smag_tau_xw[:,:,:]
        varsmag_tau_yw[t,:,:,:] = smag_tau_yw[:,:,:]
        varsmag_tau_zw[t,:,:,:] = smag_tau_zw[:,:,:]

        #Close file
        turbulentflux_file.close()

    #Close files
    a.close()
    b.close()
