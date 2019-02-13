import numpy as np
import netCDF4 as nc

def _calculate_strain2(strain2,u,v,w,igc,jgc,kgc_center,iend,jend,kend,xgc,ygc,zgc):
    '''Calculates the squared strain rate tensor required for the Smagorinksy sub-grid model. NOTE1: strain2 should be a predefined numpy array with any default values; it is overwritten with new calculated values based on the fed velocity fields. NOTE2: xgc, ygc, zgc should be numpy arrays with the coordinates in each direction with ghost cells; they are used to calculate dx, dy, and dz. NOTE3: the squared strain rate is calculated at the grid centers.'''
    
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

                strain2[k,j,i] = 2.*( 


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
  
    #zhc = np.array(a['zhgc'][:])
    zgc  = np.array(a['zgc'][:])
    #yhc  = np.array(a['yhgc'][:])
    ygc  = np.array(a['ygc'][:])
    #xhgc = np.array(a['xhgc'][:])
    xgc  = np.array(a['xgc'][:])

    #Loop over timesteps
    create_file = True #Flag to ensure output file is only created once
    for t in range(nt):

        #Define some auxilary variables to keep track of sample numbers
        tot_sample_begin = tot_sample_num #
        sample_num = 0
        
        #Open/create netCDF-file for storage
        if create_file and create_netcdf:
            samples_file = nc.Dataset(samples_filepath, 'w')
            create_file = False
            create_variables = True
        elif create_netcdf:
            samples_file = nc.Dataset(samples_filepath, 'r+')
            create_variables = False #Don't define variables when netCDF file already exists, because it should already contain those variables.
       
       #Define variables for storage
       strain2          = np.zeros((nz,ny,nx))
       eddy_diffusivity = np.zeros((nz,ny,nx))

       #Extract flowfields for timestep t
       uc_singlefield = np.array(a['uc'][t,:,:,:])
       vc_singlefield = np.array(a['vc'][t,:,:,:])
       wc_singlefield = np.array(a['wc'][t,:,:,:])

       #Calculate squared strain rate tensor
       strain2 = _calculate_strain2(strain2,u,v,w,igc,jgc,kgc_center,iend,jend,kend,xgc,ygc,zgc)
