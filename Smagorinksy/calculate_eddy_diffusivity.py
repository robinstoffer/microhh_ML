import numpy as np
import netCDF4 as nc

def _calculate_strain2(strain2,mlen,u,v,w,igc,jgc,kgc_center,iend,jend,kend,xgc,ygc,zgc):
    '''Calculates the squared strain rate tensor and mixing lengths required for the Smagorinksy sub-grid model. NOTE1: strain2 and mlen should be predefined numpy arrays with any default values; it is overwritten with new calculated values based on the fed velocity fields. NOTE2: xgc, ygc, zgc should be numpy arrays with the coordinates in each direction with ghost cells; they are used to calculate dx, dy, and dz. NOTE3: the squared strain rate is calculated at the grid centers.'''
    
    #Define Smagorinksy coefficient
    cs = 0.17

    #Check whether at least 1 ghost cell is present in each direction
    if not (igc >= 1 and jgc >= 1 and kgc_center >= 1):
        raise ValueError("There should be at least one ghost cell be present in each coordinate direction.")

    #Loop over gridcells to calculate the squared strain rate tensor
    for k in range(kgc_center, kend):
        dzib = 1./(zgc[k]   - zgc[k-1])
        dzit = 1./(zgc[k+1] - zgc[k])

        for j in range(jgc, jend):
            dyib = 1./(ygc[j]   - ygc[j-1])
            dyit = 1./(ygc[j+1] - ygc[j])

            for i in range(igc, iend):
                dxib = 1./(xgc[i]   - xgc[i-1])
                dxit = 1./(xgc[i+1] - xgc[i])
                
                mlen[k,j,i]    = (cs * ((dxib*dyib*dzib) ** (1/3))) ** 2

                strain2[k,j,i] = 2.*(
                        # du/dx + du/dx
                        (((u[k,j,i+1] - u[k,j,i])*dxib) ** 2)
                        
                        # dv/dy + dv/dy
                        + (((v[k,j+1,i] - v[k,j,i])*dyib) ** 2)

                        # dw/dz + dw/dz
                        + (((w[k+1,j,i] - w[k,j,i])*dzib) ** 2)

                        # du/dy + dv/dx
                        + 0.125* (((u[k,j,i] - u[k,j-1,i])*dyib + (v[k,j,i] - v[k,j,i-1])*dxib) ** 2)
                        + 0.125* (((u[k,j,i+1] - u[k,j-1,i+1])*dyib + (v[k,j,i+1] - v[k,j,i])*dxit) ** 2)
                        + 0.125* (((u[k,j+1,i] - u[k,j,i])*dyit + (v[k,j+1,i] - v[k,j+1,i-1])*dxib) ** 2)
                        + 0.125* (((u[k,j+1,i+1] - u[k,j,i+1])*dyit + (v[k,j+1,i+1] - v[k,j+1,i])*dxit) ** 2)

                        # du/dz + dw/dx
                        + 0.125* (((u[k,j,i] - u[k-1,j,i])*dzib + (w[k,j,i] - w[k,j,i-1])*dxib) ** 2)
                        + 0.125* (((u[k,j,i+1] - u[k-1,j,i+1])*dzib + (w[k,j,i+1] - w[k,j,i])*dxit) ** 2)
                        + 0.125* (((u[k+1,j,i] - u[k,j,i])*dzit + (w[k+1,j,i] - w[k+1,j,i-1])*dxib) ** 2)
                        + 0.125* (((u[k+1,j,i+1] - u[k,j,i+1])*dzit + (w[k+1,j,i+1] - w[k+1,j,i])*dxit) ** 2)
                        
                        # dv/dz + dw/dy
                        + 0.125* (((v[k,j,i] - v[k-1,j,i])*dzib + (w[k,j,i] - w[k,j-1,i])*dyib) ** 2)
                        + 0.125* (((v[k,j+1,i] - v[k-1,j+1,i])*dzib + (w[k,j+1,i] - w[k,j,i])*dyit) ** 2)
                        + 0.125* (((v[k+1,j,i] - v[k,j,i])*dzit + (w[k+1,j,i] - w[k+1,j-1,i])*dyib) ** 2)
                        + 0.125* (((v[k+1,j+1,i] - v[k,j+1,i])*dzit + (w[k+1,j+1,i] - w[k+1,j,i])*dyit) ** 2))

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
    #NOTE: NO ghost cells yet added in vertical direction!

    return strain2, mlen


def calculate_eddy_diffusivity(input_filepath = 'training_data.nc', output_filepath = 'eddy_diffusivity.nc'):
    '''Calculates the eddy diffusivity required in the Smagorinsky sub-grid model to calculate the sub-grid scale turbulent fluxes.'''

    #Check types input 
    if not isinstance(input_filepath,str):
        raise TypeError("Specified input filepath should be a string.")

    if not isinstance(eddy_diffusivity_filepath,str):
        raise TypeError("Specified output filepath should be a string.")

    #Fetch training data
    a = nc.Dataset(training_filepath, 'r')

    #Extract information about the grid
    igc            = int(a['igc'][:])
    jgc            = int(a['jgc'][:])
    kgc_center     = int(a['kgc_center'][:])
    iend           = int(a['iend'][:])
    jend           = int(a['jend'][:])
    kend           = int(a['kend'][:])
    nt, nz, ny, nx = ['unres_tau_xu'].shape # NOTE1: nt should be the same for all variables. NOTE2: nz,ny,nx are considered from unres_tau_xu because it is located on the grid centers in all three directions and does not contain ghost cells.
  
    #zhc  = np.array(a['zhgc'][:])
    zgc  = np.array(a['zgc'][:])
    nzc  = len(zgc)
    #yhc  = np.array(a['yhgc'][:])
    ygc  = np.array(a['ygc'][:])
    nyc  = len(ygc)
    #xhgc = np.array(a['xhgc'][:])
    xgc  = np.array(a['xgc'][:])
    nxc  = len(xgc) 

    #Loop over timesteps
    create_file = True #Flag to ensure output file is only created once
    for t in range(nt):
        
        #Open/create netCDF-file for storage
        if create_file:
            smagorinsky_file = nc.Dataset(output_filepath, 'w')
            create_file      = False
            create_variables = True
        else:
            smagorinsky_file = nc.Dataset(output_filepath, 'r+')
            create_variables = False #Don't define variables when netCDF file already exists, because it should already contain those variables.
       
        #Define variables for storage
        strain2          = np.zeros((nz,nyc,nxc))
        mlen             = np.zeros((nz,nyc,nxc))
        eddy_diffusivity = np.zeros((nz,nyc,nxc))

        #Extract flowfields for timestep t
        uc_singlefield = np.array(a['uc'][t,:,:,:])
        vc_singlefield = np.array(a['vc'][t,:,:,:])
        wc_singlefield = np.array(a['wc'][t,:,:,:])

        #Calculate squared strain rate tensor
        strain2, mlen = _calculate_strain2(strain2,mlen,u,v,w,igc,jgc,kgc_center,iend,jend,kend,xgc,ygc,zgc)
        eddy_diffusivity = mlen * np.sqrt(strain2) # + mvisc

        #Store calculated values in nc-file
        if create_variables:
            
            #Create new dimensions
            smagorinsky_file.createDimension("nt",nt)
            smagorinsky_file.createDimension("nz",nz) #+2*igc/jgc added because of the implemented ghost cells (see _calculate_strain2 function above).
            smagorinsky_file.createDimension("nyc",ny+2*jgc)
            smagorinsky_file.createDimension("nxc",nx+2*igc)

            #Create new variables
            varmlen             = smagorinsky_file.createVariable("mlen","f8",("nt","nz","nyc","nxc"))
            varstrain2          = smagorinsky_file.createVariable("strain2","f8",("nt","nz","nyc","nxc"))
            vareddy_diffusivity = smagorinsky_file.createVariable("eddy_diffusivity","f8",("nt","nz","nyc","nxc"))

            create_variables = False #Make sure the variables are only created once

        #Store variables
        varmlen[t,:,:,:]             = mlen[:,:,:]
        varstrain2[t,:,:,:]          = strain2[:,:,:]
        vareddy_diffusivity[t,:,:,:] = eddy_diffusivity[:,:,:]
        
        #Close file
        smagorinsky_file.close()
