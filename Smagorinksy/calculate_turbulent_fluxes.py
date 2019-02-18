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
    cells_around_centercell = int(a['cells_around_centercell'])
    if not cells around_centercell >= 1:
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
        uc_singlefield = a['uc'][t,:,:,:]
        vc_singlefield = a['vc'][t,:,:,:]
        wc_singlefield = a['wc'][t,:,:,:]

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
            dz    = zhgc[k+1]- zhgc[k]
            dzi   = 1./dz
            dzhib = 1./(zgc[k] - zgc[k-1])
            dzhit = 1./(zgc[k+1] - zgc[k])
            k_stag = k - 1 #Take into account that the staggered vertical dimension does not contain one ghost cell
            dzi_stag = 1./(zhgc[k_stag+1] - zhgc[k_stag])

            for j in range(jgc,jend):
                dy    = yhgc[j+1]- yhgc[j]
                dyi   = 1./dy
                dyhib = 1./(ygc[j] - ygc[j-1])
                dyhit = 1./(ygc[j+1] - ygc[j])

                for i in range(igc, iend):
                    dx    = xhgc[i+1]- xhgc[i]
                    dxi   = 1./dx
                    dxhib = 1./(xgc[i] - xgc[i-1])
                    dxhit = 1./(xgc[i+1] - xgc[i])

                    #Calculate eddy viscosity coefficients on corners
                    eviscn = 0.25*(evisc[k,j,i-1] + evisc[k,j,i] + evisc[k,j+1,i-1] + evisc[k,j+1,i])
                    eviscs = 0.25*(evisc[k,j-1,i-1] + evisc[k,j-1,i] + evisc[k,j,i-1] + evisc[k,j,i])
                    evisce = 0.25*(evisc[k,j-1,i] + evisc[k,j,i] + evisc[k,j-1,i+1] + evisc[k,j,i+1])
                    eviscw = 0.25*(evisc[k,j-1,i-1] + evisc[k,j,i-1] + evisc[k,j-1,i] + evisc[k,j,i])
                    evisct = 0.25*(evisc[k,j,i-1] + evisc[k,j,i] + evisc[k+1,j,i-1] + evisc[k+1,j,i])
                    eviscb = 0.25*(evisc[k-1,j,i-1] + evisc[k-1,j,i] + evisc[k,j,i-1] + evisc[k,j,i])

                    #Calculate turbulent fluxes accoring to Smagorinsky-Lilly model
                    smag_tau_xu[k,j,i] = evisc[k,j,i]  * (u[k,j,i+1] - u[k,j,i]) * dxi
                    smag_tau_xv[k,j,i] = eviscs * ((u[k,j,i] - u[k,j-1,i]) * dyhi + (v[k,j,i] - v[k,j,i-1]) * dxhi)
                    smag_tau_xw[k,j,i] = eviscb * ((u[k,j,i] - u[k-1,j,i]) * dzhi + (w[k,j,i] - w[k,j,i-1]) * dxhi)
                    smag_tau_yu[k,j,i] = eviscw * ((v[k,j,i] - v[k,j,i-1]) * dxhi + (u[k,j,i] - u[k,j-1,i]) * dyhi)
                    smag_tau_yv[k,j,i] = evisc[k,j,i] * (v[k,j+1,i] - v[k,j,i]) * dyi
                    smag_tau_yw[k,j,i] = eviscb * ((v[k,j,i] - v[k-1,j,i]) * dzhi + (w[k,j,i] - w[k,j-1,i]) * dyhi)
                    smag_tau_zu[k,j,i] = eviscw * ((w[k_stag,j,i] - w[k_stag,j,i-1]) * dxhi + (u[k,j,i] - u[k-1,j,i]) * dzhi)
                    smag_tau_zv[k,j,i] = eviscs * ((w[k_stag,j,i] - w[k-stag,j-1,i]) * dyhi + (v[k,j,i] - v[k-1,j,i]) * dzhi)
                    smag_tau_zw[k,j,i] = evisc[k,j,i] * (w[k_stag+1,j,i] - w[k_stag,j,i]) * dzi_stag
                    
        #Add ghostcells to fluxes
