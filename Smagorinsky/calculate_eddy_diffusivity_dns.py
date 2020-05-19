#Script that calculate the eddy diffusivities for the Smagorinsky sub-grid model
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import numpy as np
import netCDF4 as nc

def _calculate_strain2(strain2,mlen,u,v,w,x,y,z,xh,yh,zh,utau_ref,mvisc):
    '''Calculates the squared strain rate tensor and mixing lengths required for the Smagorinksy sub-grid model. \\
            NOTE1: strain2 and mlen should be predefined numpy arrays with any default values; it is overwritten with new calculated values based on the fed velocity fields. \\ 
            NOTE2: x, y, z, xh, yh, zh should be numpy arrays with the coordinates in each direction with ghost cells; they are used to calculate dx, dy, and dz. \\
            NOTE3: the squared strain rate is calculated at the grid centers. \\
            NOTE4: Assumes periodic BC in horizontal directions.'''
    
    #Define coefficients
    cs = 0.10 #Default Smagorinsky coefficient for turbulent channel flow
    vandriest_coef = 26 #Default Van Driest damping coeffcient
    height_channel = 2.0

    #Hard-code filter widths for now
    dxf = (2* np.pi) / 96.0
    dyf = np.pi /48.0
    dzf = 2.0 / 64.0
    
    ##Check whether at least 1 ghost cell is present in each direction
    #if not (igc >= 1 and jgc >= 1 and kgc_center >= 1):
    #    raise ValueError("There should be at least one ghost cell be present in each coordinate direction.")

    #Loop over gridcells to calculate the squared strain rate tensor
    #NOTE: loop indices chosen such that no ghost cells are needed: bottom and top layer are left empy, because they are not needed for the comparison with DNS
    for k in range(1, len(z)-1):
        #k_stag = k - 1 #Take into account that the staggered vertical dimension does not contain one ghost cell
        #dz    = zh[k_stag+1]- zh[k_stag]
        dz    = zh[k+1]- zh[k]
        dzi   = 1./dz
        dzhib = 1./(z[k] - z[k-1])
        dzhit = 1./(z[k+1] - z[k])

        #Incoroporate Van Driest wall damping function
        z_absdist = min(z[k], height_channel - z[k]) #Take shortest absolute distance to wall
        zplus = (z_absdist * utau_ref) / mvisc
        damp_coef = 1 - np.exp(-zplus/vandriest_coef)

        for j in range(1, len(y)-1):
            dy    = yh[j+1]- yh[j]
            dyi   = 1./dy
            dyhib = 1./(y[j] - y[j-1])
            dyhit = 1./(y[j+1] - y[j])

            for i in range(1, len(x)-1):
                dx    = xh[i+1]- xh[i]
                dxi   = 1./dx
                dxhib = 1./(x[i] - x[i-1])
                dxhit = 1./(x[i+1] - x[i])

                mlen[k,j,i]    = (damp_coef * cs * ((dxf*dyf*dzf) ** (1/3))) ** 2
                #mlen[k,j,i]    = (damp_coef * cs * ((dx*dy*dz) ** (1/3))) ** 2
                #mlen[k,j,i]    = (damp_coef * cs * ((dx**2 + dz**2 + 4*(dy**2))**0.5)) ** 2
                #mlen[k,j,i]    = (cs * ((dx*dy*dz) ** (1/3))) ** 2 #NOTE: FOR TESTING PURPOSES NO DAMPING FUNCTION INCLUDED!

                strain2[k,j,i] = 2.*(
                        # du/dx + du/dx
                        (((u[k,j,i+1] - u[k,j,i])*dxi) ** 2)
                        
                        # dv/dy + dv/dy
                        + (((v[k,j+1,i] - v[k,j,i])*dyi) ** 2)

                        # dw/dz + dw/dz
                        + (((w[k+1,j,i] - w[k,j,i])*dzi) ** 2)

                        # du/dy + dv/dx
                        + 0.125* (((u[k,j,i] - u[k,j-1,i])*dyhib + (v[k,j,i] - v[k,j,i-1])*dxhib) ** 2)
                        + 0.125* (((u[k,j,i+1] - u[k,j-1,i+1])*dyhib + (v[k,j,i+1] - v[k,j,i])*dxhit) ** 2)
                        + 0.125* (((u[k,j+1,i] - u[k,j,i])*dyhit + (v[k,j+1,i] - v[k,j+1,i-1])*dxhib) ** 2)
                        + 0.125* (((u[k,j+1,i+1] - u[k,j,i+1])*dyhit + (v[k,j+1,i+1] - v[k,j+1,i])*dxhit) ** 2)

                        # du/dz + dw/dx
                        + 0.125* (((u[k,j,i] - u[k-1,j,i])*dzhib + (w[k,j,i] - w[k,j,i-1])*dxhib) ** 2)
                        + 0.125* (((u[k,j,i+1] - u[k-1,j,i+1])*dzhib + (w[k,j,i+1] - w[k,j,i])*dxhit) ** 2)
                        + 0.125* (((u[k+1,j,i] - u[k,j,i])*dzhit + (w[k+1,j,i] - w[k+1,j,i-1])*dxhib) ** 2)
                        + 0.125* (((u[k+1,j,i+1] - u[k,j,i+1])*dzhit + (w[k+1,j,i+1] - w[k+1,j,i])*dxhit) ** 2)
                        
                        # dv/dz + dw/dy
                        + 0.125* (((v[k,j,i] - v[k-1,j,i])*dzhib + (w[k,j,i] - w[k,j-1,i])*dyhib) ** 2)
                        + 0.125* (((v[k,j+1,i] - v[k-1,j+1,i])*dzhib + (w[k,j+1,i] - w[k,j,i])*dyhit) ** 2)
                        + 0.125* (((v[k+1,j,i] - v[k,j,i])*dzhit + (w[k+1,j,i] - w[k+1,j-1,i])*dyhib) ** 2)
                        + 0.125* (((v[k+1,j+1,i] - v[k,j+1,i])*dzhit + (w[k+1,j+1,i] - w[k+1,j,i])*dyhit) ** 2))

                #Add a small number to avoid zero division
                strain2[k,j,i] += float(1e-09)

    ##Make use of periodic BC to add ghost cells in horizontal directions
    #strain2[:,:,0:igc]         = strain2[:,:,iend-igc:iend]
    #strain2[:,:,iend:iend+igc] = strain2[:,:,igc:igc+igc]
    #strain2[:,0:jgc,:]         = strain2[:,jend-jgc:jend,:]
    #strain2[:,jend:jend+jgc,:] = strain2[:,jgc:jgc+jgc,:]

    #mlen[:,:,0:igc]         = mlen[:,:,iend-igc:iend]
    #mlen[:,:,iend:iend+igc] = mlen[:,:,igc:igc+igc]
    #mlen[:,0:jgc,:]         = mlen[:,jend-jgc:jend,:]
    #mlen[:,jend:jend+jgc,:] = mlen[:,jgc:jgc+jgc,:]

    return strain2, mlen


def calculate_eddy_diffusivity_dns(input_filepath = 'dns_boxfilter.nc', output_filepath = 'eddy_diffusivity.nc'):
    '''Calculates the dimensionless eddy diffusivity [-] required in the Smagorinsky sub-grid model to calculate the sub-grid scale turbulent fluxes. The specified input and output filepaths should be strings that indicate name and location of netCDF files. The input file should be produced by func_boxfilter.py. Note that periodic BCs are assumed in the horizontal directions and it is assumed the eddy diffusivity is equal to the molecular kinematic viscosity at the bottom and top walls. For consistency however with the DNS filtering, the molecular kinematic viscosity is set to 0: the resolved viscous flux is not taken into account.'''

    #Check types input 
    if not isinstance(input_filepath,str):
        raise TypeError("Specified input filepath should be a string.")

    if not isinstance(output_filepath,str):
        raise TypeError("Specified output filepath should be a string.")

    #Fetch input file
    a = nc.Dataset(input_filepath, 'r')

    #Get molecular viscosity and friction velocity
    mvisc = float(a['mvisc'][:])
    utau_ref = float(a['utau_ref'][:])
    mvisc_smag = 0. #Don't take molecular contribution into account to be consistent with the DNS filtering, which does not include the resolved viscous flux.

    #Extract information about the grid
    zh = np.array(a['zh'])
    z  = np.array(a['z'])
    yh = np.array(a['yh'])
    y  = np.array(a['y'])
    xh = np.array(a['xh'])
    x  = np.array(a['x'])
    nt = len(a.dimensions['time'])
    nz = len(z)
    ny = len(y)
    nx = len(x)
  
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
        strain2          = np.empty((nz,ny,nx))
        strain2.fill(np.nan)
        mlen             = np.empty((nz,ny,nx))
        mlen.fill(np.nan)
        eddy_diffusivity = np.empty((nz,ny,nx))
        eddy_diffusivity.fill(np.nan)

        #Extract flowfields for timestep t
        uc_singlefield = np.array(a['uc'][t,:,:,:])
        vc_singlefield = np.array(a['vc'][t,:,:,:])
        wc_singlefield = np.array(a['wc'][t,:,:,:])

        #Calculate squared strain rate tensor
        strain2, mlen = _calculate_strain2(strain2,mlen,uc_singlefield,vc_singlefield,wc_singlefield,x,y,z,xh,yh,zh,utau_ref,mvisc)
        #eddy_diffusivity = mlen * np.sqrt(strain2) + mvisc_ref
        eddy_diffusivity = mlen * np.sqrt(strain2) + mvisc_smag

        ##For a resolved wall the viscosity at the wall is needed. For now, assume that the eddy viscosity is zero, so set ghost cell such that the viscosity interpolated to the surface equals the molecular viscosity
        ##if kgc_center != 1:
        ##    raise ValueError("The Smagorinsky filter has been implemented only for 1 ghost cell in the vertical direction. Please change the specified ghost cells in the vertical direction accordingly.")

        #eddy_diffusivity[0:kgc_center,:,:]         = 2 * mvisc_smag - np.flip(eddy_diffusivity[:kgc_center+kgc_center,:,:], axis = 0)
        #eddy_diffusivity[kend:kend+kgc_center,:,:] = 2 * mvisc_smag - np.flip(eddy_diffusivity[kend-kgc_center:kend,:,:], axis = 0)

        #Store calculated values in nc-file
        if create_variables:
            
            #Create new dimensions
            smagorinsky_file.createDimension("nt",nt)
            smagorinsky_file.createDimension("z",len(z))
            smagorinsky_file.createDimension("y",len(y))
            smagorinsky_file.createDimension("x",len(x))

            #Create new variables
            varz              = smagorinsky_file.createVariable("z","f8",("z"))
            vary              = smagorinsky_file.createVariable("y","f8",("y"))
            varx              = smagorinsky_file.createVariable("x","f8",("x"))
            varmlen             = smagorinsky_file.createVariable("mlen","f8",("nt","z","y","x"))
            varstrain2          = smagorinsky_file.createVariable("strain2","f8",("nt","z","y","x"))
            vareddy_diffusivity = smagorinsky_file.createVariable("eddy_diffusivity","f8",("nt","z","y","x"))

            #Store coordinate variables
            varz[:]           = z[:]
            vary[:]           = y[:]
            varx[:]           = x[:]

            create_variables = False #Make sure the variables are only created once

        #Store variables
        varmlen[t,:,:,:]             = mlen[:,:,:]
        varstrain2[t,:,:,:]          = strain2[:,:,:]
        vareddy_diffusivity[t,:,:,:] = eddy_diffusivity[:,:,:]
        
        #Close file
        smagorinsky_file.close()

    #Close file
    a.close()
