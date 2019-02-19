import numpy as np
import netCDF4 as nc

def calculate_turbulent_fluxes(flowfields_filepath = 'training_data.nc',eddy_diffusivity_filepath = 'eddy_diffusivity.nc', output_filepath = 'smagorinsky_fluxes.nc'):
    '''Calculate all nine turbulent transport components of the Reynolds stress tensor.'''

    #Check types input
    if not isinstance(flowfields_filepath,str):
        raise TypeError("Specified flow fields filepath should be a string.")

    if not isinstance(eddy_diffusivity_filepath,str):
        raise TypeError("Specified eddy diffusivity filepath should be a string.")
    
    if not isinstance(output_filepath,str):
        raise TypeError("Specified output filepath should be a string.")

    #Fetch grid information from flowfields_filepath
    a = nc.Dataset(flowfields_filepath, 'r')
    nt, nz, ny, nx = a['unres_tau_xu'].shape # NOTE1: nt should be the same for all variables. NOTE2: nz,ny,nx are considered from unres_tau_xu because it is located on the grid centers in all three directions and does not contain ghost cells.
    igc            = int(a['igc'][:])
    jgc            = int(a['jgc'][:])
    kgc_center     = int(a['kgc_center'][:])
    iend           = int(a['iend'][:])
    jend           = int(a['jend'][:])
    kend           = int(a['kend'][:])
    cells_around_centercell = int(a['cells_around_centercell'][:])
    if not cells_around_centercell >= 1:
        raise ValueError("The samples should at least consist of 3*3*3 grid cells because no ghost cells in the vertical direction have been implemented in the eddy diffusivity coefficients. This makes it in the current implementation impossible to calculate the Smagorinsky fluxes at the wall, which would be required if the samples are smaller than 3*3*3 grid cells.")

    zhgc = np.array(a['zhgc'][:])
    zgc  = np.array(a['zgc'][:])
    yhgc  = np.array(a['yhgc'][:])
    ygc  = np.array(a['ygc'][:])
    xhgc = np.array(a['xhgc'][:])
    xgc  = np.array(a['xgc'][:])

    #Open file with eddy diffusivity
    b = nc.Dataset(eddy_diffusivity_filepath, 'r')

    #Loop over timesteps
    create_file = True
    for t in range(nt):
        
        #Fetch eddy diffusivity coefficients
        evisc = b["eddy_diffusivity"][t,:,:,:]

        #Fetch flow fields
        u = a['uc'][t,:,:,:]
        v = a['vc'][t,:,:,:]
        w = a['wc'][t,:,:,:]

        #Open/create netCDF-file for storage
        if create_file:
            turbulentflux_file = nc.Dataset(output_filepath, 'w')
            create_file      = False
            create_variables = True
        else:
            turbulentflux_file = nc.Dataset(output_filepath, 'r+')
            create_variables = False #Don't define variables when netCDF file already exists, because it should already contain those variables.

        #Define arrays for storage
        smag_tau_xu = np.zeros((nz,ny,nx), dtype=float)
        smag_tau_yu = np.zeros((nz,ny+1,nx+1), dtype=float)
        smag_tau_zu = np.zeros((nz+1,ny,nx+1), dtype=float)
        smag_tau_xv = np.zeros((nz,ny+1,nx+1), dtype=float)
        smag_tau_yv = np.zeros((nz,ny,nx), dtype=float)
        smag_tau_zv = np.zeros((nz+1,ny+1,nx), dtype=float)
        smag_tau_xw = np.zeros((nz+1,ny,nx+1), dtype=float)
        smag_tau_yw = np.zeros((nz+1,ny+1,nx), dtype=float)
        smag_tau_zw = np.zeros((nz,ny,nx), dtype=float)

        #Loop over grid cells to calculate the fluxes
        for k in range(kgc_center,kend):
            k_stag = k - 1 #Take into account that the staggered vertical dimension does not contain one ghost cell
            dz    = zhgc[k_stag+1]- zhgc[k_stag]
            dzi   = 1./dz
            dzhib = 1./(zgc[k] - zgc[k-1])
            #dzhit = 1./(zgc[k+1] - zgc[k])

            for j in range(jgc,jend):
                dy    = yhgc[j+1]- yhgc[j]
                dyi   = 1./dy
                dyhib = 1./(ygc[j] - ygc[j-1])
                #dyhit = 1./(ygc[j+1] - ygc[j])

                for i in range(igc, iend):
                    dx    = xhgc[i+1]- xhgc[i]
                    dxi   = 1./dx
                    dxhib = 1./(xgc[i] - xgc[i-1])
                    #dxhit = 1./(xgc[i+1] - xgc[i])

                    #Calculate eddy viscosity coefficients on corners
                    eviscn = 0.25*(evisc[k,j,i-1] + evisc[k,j,i] + evisc[k,j+1,i-1] + evisc[k,j+1,i])
                    eviscs = 0.25*(evisc[k,j-1,i-1] + evisc[k,j-1,i] + evisc[k,j,i-1] + evisc[k,j,i])
                    evisce = 0.25*(evisc[k,j-1,i] + evisc[k,j,i] + evisc[k,j-1,i+1] + evisc[k,j,i+1])
                    eviscw = 0.25*(evisc[k,j-1,i-1] + evisc[k,j,i-1] + evisc[k,j-1,i] + evisc[k,j,i])
                    evisct = 0.25*(evisc[k,j,i-1] + evisc[k,j,i] + evisc[k+1,j,i-1] + evisc[k+1,j,i])
                    eviscb = 0.25*(evisc[k-1,j,i-1] + evisc[k-1,j,i] + evisc[k,j,i-1] + evisc[k,j,i])

                    #Calculate turbulent fluxes accoring to Smagorinsky-Lilly model. NOTE: take into account that the Smagorinsky fluxes do not contain ghost cells
                    smag_tau_xu[k-kgc_center,j-jgc,i-igc] = evisc[k,j,i]  * (u[k,j,i+1] - u[k,j,i]) * dxi
                    smag_tau_xv[k-kgc_center,j-jgc,i-igc] = eviscs * ((u[k,j,i] - u[k,j-1,i]) * dyhib + (v[k,j,i] - v[k,j,i-1]) * dxhib)
                    smag_tau_xw[k-kgc_center,j-jgc,i-igc] = eviscb * ((u[k,j,i] - u[k-1,j,i]) * dzhib + (w[k_stag,j,i] - w[k_stag,j,i-1]) * dxhib)
                    smag_tau_yu[k-kgc_center,j-jgc,i-igc] = eviscw * ((v[k,j,i] - v[k,j,i-1]) * dxhib + (u[k,j,i] - u[k,j-1,i]) * dyhib)
                    smag_tau_yv[k-kgc_center,j-jgc,i-igc] = evisc[k,j,i] * (v[k,j+1,i] - v[k,j,i]) * dyi
                    smag_tau_yw[k-kgc_center,j-jgc,i-igc] = eviscb * ((v[k,j,i] - v[k-1,j,i]) * dzhib + (w[k_stag,j,i] - w[k_stag,j-1,i]) * dyhib)
                    smag_tau_zu[k-kgc_center,j-jgc,i-igc] = eviscw * ((w[k_stag,j,i] - w[k_stag,j,i-1]) * dxhib + (u[k,j,i] - u[k-1,j,i]) * dzhib)
                    smag_tau_zv[k-kgc_center,j-jgc,i-igc] = eviscs * ((w[k_stag,j,i] - w[k_stag,j-1,i]) * dyhib + (v[k,j,i] - v[k-1,j,i]) * dzhib)
                    smag_tau_zw[k-kgc_center,j-jgc,i-igc] = evisc[k,j,i] * (w[k_stag+1,j,i] - w[k_stag,j,i]) * dzi
                    
        #Add one top/downstream cell to fluxes in the directions where they are located on the grid edges, assuming they are the same as the one at the bottom/upstream edge of the domain.
        #z-direction
        smag_tau_zu[khend-kgc_center,:,:] = smag_tau_zu[0,:,:]
        smag_tau_zv[khend-kgc_center,:,:] = smag_tau_zv[0,:,:]
        smag_tau_xw[khend-kgc_center,:,:] = smag_tau_xw[0,:,:]
        smag_tau_yw[khend-kgc_center,:,:] = smag_tau_yw[0,:,:]

        #y-direction
        smag_tau_yu[:,jhend-jgc,:] = smag_tau_yu[:,0,:]
        smag_tau_xv[:,jhend-jgc,:] = smag_tau_xv[:,0,:]
        smag_tau_zv[:,jhend-jgc,:] = smag_tau_zv[:,0,:]
        smag_tau_yw[:,jhend-jgc,:] = smag_tau_yw[:,0,:]

        #x-direction
        smag_tau_yu[:,:,ihend-igc] = smag_tau_yu[:,:,0]
        smag_tau_zu[:,:,ihend-igc] = smag_tau_zu[:,:,0]
        smag_tau_xv[:,:,ihend-igc] = smag_tau_xv[:,:,0]
        smag_tau_xw[:,:,ihend-igc] = smag_tau_xw[:,:,0]

        #Store calculated values in nc-file
        if create_variables:
            
            #Create new dimensions
            turbulentflux_file.createDimension("nt",nt)
            turbulentflux_file.createDimension("nzhc",nz+1)
            turbulentflux_file.createDimension("nzc",nz)
            turbulentflux_file.createDimension("nyhc",ny+1)
            turbulentflux_file.createDimension("nyc",ny)
            turbulentflux_file.createDimension("nxhc",nx+1)
            turbulentflux_file.createDimension("nxc",nx)

            #Create new variables
            varsmag_tau_xu = turbulentflux_file.createVariable("smag_tau_xu","f8",("nt","nzc","nyc","nxc"))
            varsmag_tau_yu = turbulentflux_file.createVariable("smag_tau_yu","f8",("nt","nzc","nyhc","nxhc"))
            varsmag_tau_zu = turbulentflux_file.createVariable("smag_tau_zu","f8",("nt","nzhc","nyc","nxhc"))
            varsmag_tau_xv = turbulentflux_file.createVariable("smag_tau_xv","f8",("nt","nzc","nyhc","nxhc"))
            varsmag_tau_yv = turbulentflux_file.createVariable("smag_tau_yv","f8",("nt","nzc","nyc","nxc"))
            varsmag_tau_zv = turbulentflux_file.createVariable("smag_tau_zv","f8",("nt","nzhc","nyhc","nxc"))
            varsmag_tau_xw = turbulentflux_file.createVariable("smag_tau_xw","f8",("nt","nzhc","nyc","nxhc"))
            varsmag_tau_yw = turbulentflux_file.createVariable("smag_tau_yw","f8",("nt","nzhc","nyhc","nxc"))
            varsmag_tau_zw = turbulentflux_file.createVariable("smag_tau_zw","f8",("nt","nzc","nyc","nxc"))

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
