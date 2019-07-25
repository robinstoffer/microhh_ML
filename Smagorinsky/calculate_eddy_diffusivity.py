#Script that calculate the eddy diffusivities for the Smagorinsky sub-grid model
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import numpy as np
import netCDF4 as nc

def _calculate_strain2(strain2,mlen,u,v,w,igc,jgc,kgc_center,iend,jend,kend,xgc,ygc,zgc,xhgc,yhgc,zhgc,utau_ref,mvisc):
    '''Calculates the squared strain rate tensor and mixing lengths required for the Smagorinksy sub-grid model. \\
            NOTE1: strain2 and mlen should be predefined numpy arrays with any default values; it is overwritten with new calculated values based on the fed velocity fields. \\ 
            NOTE2: xgc, ygc, zgc, xhgc, yhgc, zhgc should be numpy arrays with the coordinates in each direction with ghost cells; they are used to calculate dx, dy, and dz. \\
            NOTE3: the squared strain rate is calculated at the grid centers. \\
            NOTE4: Assumes periodic BC in horizontal directions.'''
    
    #Define coefficients
    cs = 0.10 #Default Smagorinsky coefficient for turbulent channel flow
    vandriest_coef = 26 #Default Van Driest damping coeffcient

    #Check whether at least 1 ghost cell is present in each direction
    if not (igc >= 1 and jgc >= 1 and kgc_center >= 1):
        raise ValueError("There should be at least one ghost cell be present in each coordinate direction.")

    #Loop over gridcells to calculate the squared strain rate tensor
    for k in range(kgc_center, kend):
        k_stag = k - 1 #Take into account that the staggered vertical dimension does not contain one ghost cell
        dz    = zhgc[k_stag+1]- zhgc[k_stag]
        dzi   = 1./dz
        dzhib = 1./(zgc[k] - zgc[k-1])
        dzhit = 1./(zgc[k+1] - zgc[k])

        #Incoroporate Van Driest wall damping function
        zplus = (zgc[k] * utau_ref) / mvisc
        damp_coef = 1 - np.exp(-zplus/vandriest_coef) #NOTE: Implement damping function also at top wall?

        for j in range(jgc, jend):
            dy    = yhgc[j+1]- yhgc[j]
            dyi   = 1./dy
            dyhib = 1./(ygc[j] - ygc[j-1])
            dyhit = 1./(ygc[j+1] - ygc[j])

            for i in range(igc, iend):
                dx    = xhgc[i+1]- xhgc[i]
                dxi   = 1./dx
                dxhib = 1./(xgc[i] - xgc[i-1])
                dxhit = 1./(xgc[i+1] - xgc[i])
                
                mlen[k,j,i]    = (damp_coef * cs * ((dx*dy*dz) ** (1/3))) ** 2
                #mlen[k,j,i]    = (damp_coef * cs * ((dx**2 + dz**2 + 4*(dy**2))**0.5)) ** 2
                #mlen[k,j,i]    = (cs * ((dx*dy*dz) ** (1/3))) ** 2 #NOTE: FOR TESTING PURPOSES NO DAMPING FUNCTION INCLUDED!

                strain2[k,j,i] = 2.*(
                        # du/dx + du/dx
                        (((u[k,j,i+1] - u[k,j,i])*dxi) ** 2)
                        
                        # dv/dy + dv/dy
                        + (((v[k,j+1,i] - v[k,j,i])*dyi) ** 2)

                        # dw/dz + dw/dz
                        + (((w[k_stag+1,j,i] - w[k_stag,j,i])*dzi) ** 2)

                        # du/dy + dv/dx
                        + 0.125* (((u[k,j,i] - u[k,j-1,i])*dyhib + (v[k,j,i] - v[k,j,i-1])*dxhib) ** 2)
                        + 0.125* (((u[k,j,i+1] - u[k,j-1,i+1])*dyhib + (v[k,j,i+1] - v[k,j,i])*dxhit) ** 2)
                        + 0.125* (((u[k,j+1,i] - u[k,j,i])*dyhit + (v[k,j+1,i] - v[k,j+1,i-1])*dxhib) ** 2)
                        + 0.125* (((u[k,j+1,i+1] - u[k,j,i+1])*dyhit + (v[k,j+1,i+1] - v[k,j+1,i])*dxhit) ** 2)

                        # du/dz + dw/dx
                        + 0.125* (((u[k,j,i] - u[k-1,j,i])*dzhib + (w[k_stag,j,i] - w[k_stag,j,i-1])*dxhib) ** 2)
                        + 0.125* (((u[k,j,i+1] - u[k-1,j,i+1])*dzhib + (w[k_stag,j,i+1] - w[k_stag,j,i])*dxhit) ** 2)
                        + 0.125* (((u[k+1,j,i] - u[k,j,i])*dzhit + (w[k_stag+1,j,i] - w[k_stag+1,j,i-1])*dxhib) ** 2)
                        + 0.125* (((u[k+1,j,i+1] - u[k,j,i+1])*dzhit + (w[k_stag+1,j,i+1] - w[k_stag+1,j,i])*dxhit) ** 2)
                        
                        # dv/dz + dw/dy
                        + 0.125* (((v[k,j,i] - v[k-1,j,i])*dzhib + (w[k_stag,j,i] - w[k_stag,j-1,i])*dyhib) ** 2)
                        + 0.125* (((v[k,j+1,i] - v[k-1,j+1,i])*dzhib + (w[k_stag,j+1,i] - w[k_stag,j,i])*dyhit) ** 2)
                        + 0.125* (((v[k+1,j,i] - v[k,j,i])*dzhit + (w[k_stag+1,j,i] - w[k_stag+1,j-1,i])*dyhib) ** 2)
                        + 0.125* (((v[k+1,j+1,i] - v[k,j+1,i])*dzhit + (w[k_stag+1,j+1,i] - w[k_stag+1,j,i])*dyhit) ** 2))

                #Add a small number to avoid zero division
                strain2[k,j,i] += float(1e-09)

    #Make use of periodic BC to add ghost cells in horizontal directions
    strain2[:,:,0:igc]         = strain2[:,:,iend-igc:iend]
    strain2[:,:,iend:iend+igc] = strain2[:,:,igc:igc+igc]
    strain2[:,0:jgc,:]         = strain2[:,jend-jgc:jend,:]
    strain2[:,jend:jend+jgc,:] = strain2[:,jgc:jgc+jgc,:]

    mlen[:,:,0:igc]         = mlen[:,:,iend-igc:iend]
    mlen[:,:,iend:iend+igc] = mlen[:,:,igc:igc+igc]
    mlen[:,0:jgc,:]         = mlen[:,jend-jgc:jend,:]
    mlen[:,jend:jend+jgc,:] = mlen[:,jgc:jgc+jgc,:]

    return strain2, mlen


def calculate_eddy_diffusivity(input_filepath = 'training_data.nc', output_filepath = 'eddy_diffusivity.nc'):
    '''Calculates the dimensionless eddy diffusivity [-] required in the Smagorinsky sub-grid model to calculate the sub-grid scale turbulent fluxes. The specified input and output filepaths should be strings that indicate name and location of netCDF files. The input file should be produced by func_generate_training.py. Note that periodic BCs are assumed in the horizontal directions and it is assumed the eddy diffusivity is equal to the molecular kinematic viscosity at the bottom and top walls.'''

    #Check types input 
    if not isinstance(input_filepath,str):
        raise TypeError("Specified input filepath should be a string.")

    if not isinstance(output_filepath,str):
        raise TypeError("Specified output filepath should be a string.")

#    if not isinstance(mvisc,float):
#        raise TypeError("Specified molecular viscosity should be a float with units in m2s-1.")
#
#    if not isinstance(utau_ref,float):
#        raise TypeError("Specified reference friction velocity should be a float.")
#
#    if not (isinstance(half_channel_width,float) or isinstance(half_channel_width,int)):
#        raise TypeError("Specified half channel width should be either a float or integer.")

    #Fetch training data
    a = nc.Dataset(input_filepath, 'r')

    #Get normalized molecular viscosity and friction velocity
    mvisc_ref = float(a['mvisc_ref'][:])
    #mvisc     = float(a['mvisc'][:])
    mvisc = 0. #Don't take molecular contribution into account to be consistent with the training data of the MLP.
    utau_ref  = float(a['utau_ref'][:])

    #Extract information about the grid
    igc            = int(a['igc'][:])
    jgc            = int(a['jgc'][:])
    kgc_center     = int(a['kgc_center'][:])
    iend           = int(a['iend'][:])
    jend           = int(a['jend'][:])
    kend           = int(a['kend'][:])
    nt, nz, ny, nx = a['unres_tau_xu_tot'].shape # NOTE1: nt should be the same for all variables. NOTE2: nz,ny,nx are considered from unres_tau_xu because it is located on the grid centers in all three directions and does not contain ghost cells.
  
    zhgc = np.array(a['zhgc'][:])
    zgc  = np.array(a['zgc'][:])
    nzc  = len(zgc)
    yhgc = np.array(a['yhgc'][:])
    ygc  = np.array(a['ygc'][:])
    nyc  = len(ygc)
    xhgc = np.array(a['xhgc'][:])
    xgc  = np.array(a['xgc'][:])
    nxc  = len(xgc)

    #Loop over timesteps
    create_file = True #Flag to ensure output file is only created once
    for t in range(nt):
    #for t in range(1,2): #NOTE:FOR TESTING PURPOSES ONLY!
        
        #Open/create netCDF-file for storage
        if create_file:
            smagorinsky_file = nc.Dataset(output_filepath, 'w')
            create_file      = False
            create_variables = True
        else:
            smagorinsky_file = nc.Dataset(output_filepath, 'r+')
            create_variables = False #Don't define variables when netCDF file already exists, because it should already contain those variables.
       
        #Define variables for storage
        strain2          = np.zeros((nzc,nyc,nxc))
        mlen             = np.zeros((nzc,nyc,nxc))
        eddy_diffusivity = np.zeros((nzc,nyc,nxc))

        #Extract flowfields for timestep t
        uc_singlefield = np.array(a['uc'][t,:,:,:])
        vc_singlefield = np.array(a['vc'][t,:,:,:])
        wc_singlefield = np.array(a['wc'][t,:,:,:])

        ##QUICK AND DIRTY: RESCALE VELOCITY FIELDS WITH FRICTION VELOCITY
        uc_singlefield = uc_singlefield * utau_ref
        vc_singlefield = vc_singlefield * utau_ref
        wc_singlefield = wc_singlefield * utau_ref

        #Calculate squared strain rate tensor
        strain2, mlen = _calculate_strain2(strain2,mlen,uc_singlefield,vc_singlefield,wc_singlefield,igc,jgc,kgc_center,iend,jend,kend,xgc,ygc,zgc,xhgc,yhgc,zhgc,utau_ref,mvisc)
        #eddy_diffusivity = mlen * np.sqrt(strain2) + mvisc_ref
        eddy_diffusivity = mlen * np.sqrt(strain2) + mvisc

        #For a resolved wall the viscosity at the wall is needed. For now, assume that the eddy viscosity is zero, so set ghost cell such that the viscosity interpolated to the surface equals the molecular viscosity
        #if kgc_center != 1:
        #    raise ValueError("The Smagorinsky filter has been implemented only for 1 ghost cell in the vertical direction. Please change the specified ghost cells in the vertical direction accordingly.")

        eddy_diffusivity[0:kgc_center,:,:]         = 2 * mvisc - np.flip(eddy_diffusivity[kgc_center:kgc_center+kgc_center,:,:], axis = 0)
        eddy_diffusivity[kend:kend+kgc_center,:,:] = 2 * mvisc - np.flip(eddy_diffusivity[kend-kgc_center:kend,:,:], axis = 0)

        #Store calculated values in nc-file
        if create_variables:
            
            #Create new dimensions
            smagorinsky_file.createDimension("nt",nt)
            smagorinsky_file.createDimension("zgc",len(zgc))
            smagorinsky_file.createDimension("ygc",len(ygc))
            smagorinsky_file.createDimension("xgc",len(xgc))

            #Create new variables
            varzgc              = smagorinsky_file.createVariable("zgc","f8",("zgc"))
            varygc              = smagorinsky_file.createVariable("ygc","f8",("ygc"))
            varxgc              = smagorinsky_file.createVariable("xgc","f8",("xgc"))
            varmlen             = smagorinsky_file.createVariable("mlen","f8",("nt","zgc","ygc","xgc"))
            varstrain2          = smagorinsky_file.createVariable("strain2","f8",("nt","zgc","ygc","xgc"))
            vareddy_diffusivity = smagorinsky_file.createVariable("eddy_diffusivity","f8",("nt","zgc","ygc","xgc"))

            #Store coordinate variables
            varzgc[:]           = zgc[:]
            varygc[:]           = ygc[:]
            varxgc[:]           = xgc[:]

            create_variables = False #Make sure the variables are only created once

        #Store variables
        varmlen[t,:,:,:]             = mlen[:,:,:]
        varstrain2[t,:,:,:]          = strain2[:,:,:]
        vareddy_diffusivity[t,:,:,:] = eddy_diffusivity[:,:,:]
        
        #Close file
        smagorinsky_file.close()

    #Close file
    a.close()
