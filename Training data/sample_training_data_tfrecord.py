import numpy as np
import netCDF4 as nc
import os
import tensorflow as tf
from sklearn.utils import shuffle

def generate_samples(output_directory, training_filepath = 'training_data.nc', samples_filepath = 'samples_training.nc', means_stdev_filepath = 'means_stdevs_allfields.nc', create_binary = True, create_netcdf = True, store_means_stdevs = True):
    
    #Check types input
    if not isinstance(output_directory,str):
        raise TypeError("Specified output directory should be a string.")
        
    if not isinstance(training_filepath,str):
        raise TypeError("Specified training filepath should be a string.")
        
    if not isinstance(samples_filepath,str):
        raise TypeError("Specified samples filepath should be a string.")
    
    if not isinstance(means_stdev_filepath,str):
        raise TypeError("Specified means_stdev filepath should be a string.")
    
    if not isinstance(create_binary,bool):
        raise TypeError("Specified create_binary flag should be a boolean.")
    
    if not isinstance(create_netcdf,bool):
        raise TypeError("Specified create_netcdf flag should be a boolean.")

    if not isinstance(store_means_stdevs,bool):
        raise TypeError("Specified store_means_stdevs flag should be a boolean.")

    #Fetch training data
    a = nc.Dataset(training_filepath, 'r')
    
    #Define shapes of output arrays based on stored training data
    nt,nz,ny,nx = a['unres_tau_xu'].shape # NOTE1: nt should be the same for all variables. NOTE2: nz,ny,nx are considered from unres_tau_xu because it is located on the grid centers in all three directions and does not contain ghost cells.
#    nt = 3 ###NOTE:FOR TESTING PURPOSES!!! REMOVE LATER ON!!!!
    size_samples = int(a['size_samples'][:])
    size_samples_gradients = size_samples - 2 #NOTE:To calculate the gradients, 2 additional grid points need to be used in each direction. To ensure that the considered region is the same for the absolute wind velocities and the gradients, the samples of the gradients therefore need to be 2 grid points shorter.
    if size_samples_gradients < 1:
        raise ValueError('The samples of the wind velocity components are too smal to calculate gradients. Please increase the sample size.')
    cells_around_centercell = int(a['cells_around_centercell'][:])
    cells_around_centercell_gradients = int(size_samples_gradients // 2.0)
    if nz < size_samples:
        raise ValueError("The number of vertical layers should at least be equal to the specified sample size.")
    nsamples = (nz - 2*cells_around_centercell) * ny * nx #NOTE: '-2*size_samples' needed to account for the vertical layers that are discarded in the sampling
    
    #Define arrays to store means and stdevs according to store_means_stdevs flag
    if store_means_stdevs:
        mean_uc = np.zeros((nt,1))
        mean_vc = np.zeros((nt,1))
        mean_wc = np.zeros((nt,1))
        mean_pc = np.zeros((nt,1))
        mean_ugradx = np.zeros((nt,1))
        mean_ugrady = np.zeros((nt,1))
        mean_ugradz = np.zeros((nt,1))
        mean_vgradx = np.zeros((nt,1))
        mean_vgrady = np.zeros((nt,1))
        mean_vgradz = np.zeros((nt,1))
        mean_wgradx = np.zeros((nt,1))
        mean_wgrady = np.zeros((nt,1))
        mean_wgradz = np.zeros((nt,1))
        mean_pgradx = np.zeros((nt,1))
        mean_pgrady = np.zeros((nt,1))
        mean_pgradz = np.zeros((nt,1))
        #
        mean_unres_tau_xu = np.zeros((nt,1))
        mean_unres_tau_yu = np.zeros((nt,1))
        mean_unres_tau_zu = np.zeros((nt,1))
        mean_unres_tau_xv = np.zeros((nt,1))
        mean_unres_tau_yv = np.zeros((nt,1))
        mean_unres_tau_zv = np.zeros((nt,1))
        mean_unres_tau_xw = np.zeros((nt,1))
        mean_unres_tau_yw = np.zeros((nt,1))
        mean_unres_tau_zw = np.zeros((nt,1))
        #
        stdev_uc = np.zeros((nt,1))
        stdev_vc = np.zeros((nt,1))
        stdev_wc = np.zeros((nt,1))
        stdev_pc = np.zeros((nt,1))
        stdev_ugradx = np.zeros((nt,1))
        stdev_ugrady = np.zeros((nt,1))
        stdev_ugradz = np.zeros((nt,1))
        stdev_vgradx = np.zeros((nt,1))
        stdev_vgrady = np.zeros((nt,1))
        stdev_vgradz = np.zeros((nt,1))
        stdev_wgradx = np.zeros((nt,1))
        stdev_wgrady = np.zeros((nt,1))
        stdev_wgradz = np.zeros((nt,1))
        stdev_pgradx = np.zeros((nt,1))
        stdev_pgrady = np.zeros((nt,1))
        stdev_pgradz = np.zeros((nt,1))
        #
        stdev_unres_tau_xu = np.zeros((nt,1))
        stdev_unres_tau_yu = np.zeros((nt,1))
        stdev_unres_tau_zu = np.zeros((nt,1))
        stdev_unres_tau_xv = np.zeros((nt,1))
        stdev_unres_tau_yv = np.zeros((nt,1))
        stdev_unres_tau_zv = np.zeros((nt,1))
        stdev_unres_tau_xw = np.zeros((nt,1))
        stdev_unres_tau_yw = np.zeros((nt,1))
        stdev_unres_tau_zw = np.zeros((nt,1))


    #Loop over timesteps, take for each timestep samples of 5x5x5
    tot_sample_num = 0
    create_file = True
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
        uc_samples = np.zeros((nsamples,size_samples,size_samples,size_samples))
        vc_samples = np.zeros((nsamples,size_samples,size_samples,size_samples))
        wc_samples = np.zeros((nsamples,size_samples,size_samples,size_samples))
        pc_samples = np.zeros((nsamples,size_samples,size_samples,size_samples))
        ugradx_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        ugrady_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        ugradz_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        vgradx_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        vgrady_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        vgradz_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        wgradx_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        wgrady_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        wgradz_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        pgradx_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        pgrady_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        pgradz_samples = np.zeros((nsamples,size_samples_gradients,size_samples_gradients,size_samples_gradients))
        unres_tau_xu_samples = np.zeros((nsamples,1))
        unres_tau_xv_samples = np.zeros((nsamples,1))
        unres_tau_xw_samples = np.zeros((nsamples,1))
        unres_tau_yu_samples = np.zeros((nsamples,1))
        unres_tau_yv_samples = np.zeros((nsamples,1))
        unres_tau_yw_samples = np.zeros((nsamples,1))
        unres_tau_zu_samples = np.zeros((nsamples,1))
        unres_tau_zv_samples = np.zeros((nsamples,1))
        unres_tau_zw_samples = np.zeros((nsamples,1))
        
        #Read variables from netCDF-file
        igc                     = int(a['igc'][:])
        jgc                     = int(a['jgc'][:])
        kgc_center              = int(a['kgc_center'][:])
        iend                    = int(a['iend'][:])
        jend                    = int(a['jend'][:])
        kend                    = int(a['kend'][:])
   
        uc_singlefield = np.array(a['uc'][t,:,:,:])
        vc_singlefield = np.array(a['vc'][t,:,:,:])
        wc_singlefield = np.array(a['wc'][t,:,:,:])
        pc_singlefield = np.array(a['pc'][t,:,:,:])

        unres_tau_xu_singlefield = np.array(a["unres_tau_xu"][t,:,:,:])
        unres_tau_xv_singlefield = np.array(a["unres_tau_xv"][t,:,:,:])
        unres_tau_xw_singlefield = np.array(a["unres_tau_xw"][t,:,:,:])
        unres_tau_yu_singlefield = np.array(a["unres_tau_yu"][t,:,:,:])
        unres_tau_yv_singlefield = np.array(a["unres_tau_yv"][t,:,:,:])
        unres_tau_yw_singlefield = np.array(a["unres_tau_yw"][t,:,:,:])
        unres_tau_zu_singlefield = np.array(a["unres_tau_zu"][t,:,:,:])
        unres_tau_zv_singlefield = np.array(a["unres_tau_zv"][t,:,:,:])
        unres_tau_zw_singlefield = np.array(a["unres_tau_zw"][t,:,:,:])

        #Calculate gradients of wind speed and pressure fields#
        #NOTE1: retains dimensions of original flow field, so gradients are still located on the same locations as the corresponding velocities. It uses second-order central differences in the interior, and second-order forward/backward differences at the edges.
        #NOTE2: if the half-channel width delta is not equal to 1, revise the gradients calculation below!
        zhgc = np.array(a['zhgc'][:])
        zgc  = np.array(a['zgc'][:])
        yhgc  = np.array(a['yhgc'][:])
        ygc  = np.array(a['ygc'][:])
        xhgc = np.array(a['xhgc'][:])
        xgc  = np.array(a['xgc'][:])
        pgradz,pgrady,pgradx = np.gradient(pc_singlefield,zgc,ygc,xgc,edge_order=2)
        wgradz,wgrady,wgradx = np.gradient(wc_singlefield,zhgc,ygc,xgc,edge_order=2)
        vgradz,vgrady,vgradx = np.gradient(vc_singlefield,zgc,yhgc,xgc,edge_order=2)
        ugradz,ugrady,ugradx = np.gradient(uc_singlefield,zgc,ygc,xhgc,edge_order=2)

        #Calculate means and stdevs according to store_means_stdevs flag
        if store_means_stdevs:
            mean_uc[t] = np.mean(uc_singlefield)
            mean_vc[t] = np.mean(vc_singlefield)
            mean_wc[t] = np.mean(wc_singlefield)
            mean_pc[t] = np.mean(pc_singlefield)
            mean_ugradx[t] = np.mean(ugradx)
            mean_ugrady[t] = np.mean(ugrady)
            mean_ugradz[t] = np.mean(ugradz)
            mean_vgradx[t] = np.mean(vgradx)
            mean_vgrady[t] = np.mean(vgrady)
            mean_vgradz[t] = np.mean(vgradz)
            mean_wgradx[t] = np.mean(wgradx)
            mean_wgrady[t] = np.mean(wgrady)
            mean_wgradz[t] = np.mean(wgradz)
            mean_pgradx[t] = np.mean(pgradx)
            mean_pgrady[t] = np.mean(pgrady)
            mean_pgradz[t] = np.mean(pgradz)
            #
            mean_unres_tau_xu[t] = np.mean(unres_tau_xu_singlefield)
            mean_unres_tau_yu[t] = np.mean(unres_tau_yu_singlefield)
            mean_unres_tau_zu[t] = np.mean(unres_tau_zu_singlefield)
            mean_unres_tau_xv[t] = np.mean(unres_tau_xv_singlefield)
            mean_unres_tau_yv[t] = np.mean(unres_tau_yv_singlefield)
            mean_unres_tau_zv[t] = np.mean(unres_tau_zv_singlefield)
            mean_unres_tau_xw[t] = np.mean(unres_tau_xw_singlefield)
            mean_unres_tau_yw[t] = np.mean(unres_tau_yw_singlefield)
            mean_unres_tau_zw[t] = np.mean(unres_tau_zw_singlefield)
            #
            stdev_uc[t] = np.std(uc_singlefield)
            stdev_vc[t] = np.std(vc_singlefield)
            stdev_wc[t] = np.std(wc_singlefield)
            stdev_pc[t] = np.std(pc_singlefield)
            stdev_ugradx[t] = np.std(ugradx)
            stdev_ugrady[t] = np.std(ugrady)
            stdev_ugradz[t] = np.std(ugradz)
            stdev_vgradx[t] = np.std(vgradx)
            stdev_vgrady[t] = np.std(vgrady)
            stdev_vgradz[t] = np.std(vgradz)
            stdev_wgradx[t] = np.std(wgradx)
            stdev_wgrady[t] = np.std(wgrady)
            stdev_wgradz[t] = np.std(wgradz)
            stdev_pgradx[t] = np.std(pgradx)
            stdev_pgrady[t] = np.std(pgrady)
            stdev_pgradz[t] = np.std(pgradz)
            #
            stdev_unres_tau_xu[t] = np.std(unres_tau_xu_singlefield)
            stdev_unres_tau_yu[t] = np.std(unres_tau_yu_singlefield)
            stdev_unres_tau_zu[t] = np.std(unres_tau_zu_singlefield)
            stdev_unres_tau_xv[t] = np.std(unres_tau_xv_singlefield)
            stdev_unres_tau_yv[t] = np.std(unres_tau_yv_singlefield)
            stdev_unres_tau_zv[t] = np.std(unres_tau_zv_singlefield)
            stdev_unres_tau_xw[t] = np.std(unres_tau_xw_singlefield)
            stdev_unres_tau_yw[t] = np.std(unres_tau_yw_singlefield)
            stdev_unres_tau_zw[t] = np.std(unres_tau_zw_singlefield)

        ###Do the actual sampling.###
        for index_z in range(kgc_center + cells_around_centercell, kend - cells_around_centercell): #Make sure no vertical levels are sampled where because of the vicintiy to the wall no samples of 5X5X5 grid cells can be taken.
            #NOTE: Since the grid edges in the vertical direction contain no ghost cells, no use can be made of the ghost cell present for the grid centers in the vertical direction.
            index_zlow                    = index_z - cells_around_centercell
            index_zhigh                   = index_z + cells_around_centercell + 1 #NOTE: +1 needed to ensure that in the slicing operation the selected number of grid cells above the center grid cell, is equal to the number of grid cells selected below the center grid cell.
            index_z_noghost               = index_z - kgc_center
            # Idem for gradients
            index_zlow_gradients          = index_z - cells_around_centercell_gradients
            index_zhigh_gradients         = index_z + cells_around_centercell_gradients + 1
            
            for index_y in range(jgc,jend): 
                index_ylow                = index_y - cells_around_centercell
                index_yhigh               = index_y + cells_around_centercell + 1 #NOTE: +1 needed to ensure that in the slicing operation the selected number of grid cells above the center grid cell, is equal to the number of grid cells selected below the center grid cell.
                index_y_noghost           = index_y - jgc
                # Idem for gradients
                index_ylow_gradients      = index_y - cells_around_centercell_gradients
                index_yhigh_gradients     = index_y + cells_around_centercell_gradients + 1
    
                for index_x in range(igc,iend):
                    index_xlow            = index_x - cells_around_centercell
                    index_xhigh           = index_x + cells_around_centercell + 1 #NOTE: +1 needed to ensure that in the slicing operation the selected number of grid cells above the center grid cell, is equal to the number of grid cells selected below the center grid cell.
                    index_x_noghost       = index_x - igc
                    # Idem for gradients
                    index_xlow_gradients  = index_x - cells_around_centercell_gradients
                    index_xhigh_gradients = index_x + cells_around_centercell_gradients + 1
    
                    #Take samples of 5x5x5 and store them
                    uc_samples[sample_num,:,:,:]     = uc_singlefield[index_zlow:index_zhigh,index_ylow:index_yhigh,index_xlow:index_xhigh]
                    vc_samples[sample_num,:,:,:]     = vc_singlefield[index_zlow:index_zhigh,index_ylow:index_yhigh,index_xlow:index_xhigh]
                    wc_samples[sample_num,:,:,:]     = wc_singlefield[index_zlow-1:index_zhigh-1,index_ylow:index_yhigh,index_xlow:index_xhigh] #NOTE: -1 needed to account for missing ghost cells in vertical direction when considering the grid edges.
                    pc_samples[sample_num,:,:,:]     = pc_singlefield[index_zlow:index_zhigh,index_ylow:index_yhigh,index_xlow:index_xhigh]
                    ugradx_samples[sample_num,:,:,:] = ugradx[index_zlow_gradients:index_zhigh_gradients,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
                    ugrady_samples[sample_num,:,:,:] = ugrady[index_zlow_gradients:index_zhigh_gradients,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
                    ugradz_samples[sample_num,:,:,:] = ugradz[index_zlow_gradients:index_zhigh_gradients,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
                    vgradx_samples[sample_num,:,:,:] = vgradx[index_zlow_gradients:index_zhigh_gradients,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
                    vgrady_samples[sample_num,:,:,:] = vgrady[index_zlow_gradients:index_zhigh_gradients,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
                    vgradz_samples[sample_num,:,:,:] = vgradz[index_zlow_gradients:index_zhigh_gradients,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
                    wgradx_samples[sample_num,:,:,:] = wgradx[index_zlow_gradients-1:index_zhigh_gradients-1,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
                    wgrady_samples[sample_num,:,:,:] = wgrady[index_zlow_gradients-1:index_zhigh_gradients-1,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
                    wgradz_samples[sample_num,:,:,:] = wgradz[index_zlow_gradients-1:index_zhigh_gradients-1,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
                    pgradx_samples[sample_num,:,:,:] = pgradx[index_zlow_gradients:index_zhigh_gradients,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
                    pgrady_samples[sample_num,:,:,:] = pgrady[index_zlow_gradients:index_zhigh_gradients,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
                    pgradz_samples[sample_num,:,:,:] = pgradz[index_zlow_gradients:index_zhigh_gradients,index_ylow_gradients:index_yhigh_gradients,index_xlow_gradients:index_xhigh_gradients]
        
                    #Store corresponding unresolved transports
                    unres_tau_xu_samples[sample_num] = unres_tau_xu_singlefield[index_z_noghost,index_y_noghost,index_x_noghost]
                    unres_tau_xv_samples[sample_num] = unres_tau_xv_singlefield[index_z_noghost,index_y_noghost,index_x_noghost]
                    unres_tau_xw_samples[sample_num] = unres_tau_xw_singlefield[index_z_noghost,index_y_noghost,index_x_noghost]
                    unres_tau_yu_samples[sample_num] = unres_tau_yu_singlefield[index_z_noghost,index_y_noghost,index_x_noghost]
                    unres_tau_yv_samples[sample_num] = unres_tau_yv_singlefield[index_z_noghost,index_y_noghost,index_x_noghost]
                    unres_tau_yw_samples[sample_num] = unres_tau_yw_singlefield[index_z_noghost,index_y_noghost,index_x_noghost]                
                    unres_tau_zu_samples[sample_num] = unres_tau_zu_singlefield[index_z_noghost,index_y_noghost,index_x_noghost]
                    unres_tau_zv_samples[sample_num] = unres_tau_zv_singlefield[index_z_noghost,index_y_noghost,index_x_noghost]
                    unres_tau_zw_samples[sample_num] = unres_tau_zw_singlefield[index_z_noghost,index_y_noghost,index_x_noghost]
    
                    sample_num +=1
                    tot_sample_num+=1
        ###
        #Randomly shuffle the input and output data in a consistent manner, such there is no specific order in the samples stored. This will likely help the machine learning algorithm to converge faster towards the global minimum.
        uc_samples, vc_samples, wc_samples, pc_samples, \
        ugradx_samples, ugrady_samples, ugradz_samples, \
        vgradx_samples, vgrady_samples, vgradz_samples, \
        wgradx_samples, wgrady_samples, wgradz_samples, \
        pgradx_samples, pgrady_samples, pgradz_samples, \
        unres_tau_xu_samples, unres_tau_xv_samples, unres_tau_xw_samples, \
        unres_tau_yu_samples, unres_tau_yv_samples, unres_tau_yw_samples, \
        unres_tau_zu_samples, unres_tau_zv_samples, unres_tau_zw_samples \
        = shuffle(
        uc_samples, vc_samples, wc_samples, pc_samples,
        ugradx_samples, ugrady_samples, ugradz_samples,
        vgradx_samples, vgrady_samples, vgradz_samples,
        wgradx_samples, wgrady_samples, wgradz_samples,
        pgradx_samples, pgrady_samples, pgradz_samples,
        unres_tau_xu_samples, unres_tau_xv_samples, unres_tau_xw_samples,
        unres_tau_yu_samples, unres_tau_yv_samples, unres_tau_yw_samples,
        unres_tau_zu_samples, unres_tau_zv_samples, unres_tau_zw_samples)



        #Store samples in nc-file if required by create_netcdf flag
        if create_netcdf:

            ##Store samples for each timestep
            tot_sample_eind = tot_sample_num
            dummy,zsize,ysize,xsize = pc_samples.shape
            
            if create_variables:
                #Create new dimensions
                #dim_xhs = samples_file.createDimension("xhs",xsize-1) 
                #dim_xs = samples_file.createDimension("xs",xsize)
                #dim_yhs = samples_file.createDimension("yhs",ysize-1)
                #dim_ys = samples_file.createDimension("ys",ysize)
                #dim_zhs = samples_file.createDimension("zhs",zsize-1)
                #dim_zs = samples_file.createDimension("zs",zsize)
                samples_file.createDimension("ns"  , None)
                samples_file.createDimension("boxx", size_samples)
                samples_file.createDimension("boxy", size_samples)
                samples_file.createDimension("boxz", size_samples)
                samples_file.createDimension("boxx_gradients", size_samples_gradients)
                samples_file.createDimension("boxy_gradients", size_samples_gradients)
                samples_file.createDimension("boxz_gradients", size_samples_gradients)
    
    
                #Create new variables
                varuc           = samples_file.createVariable("uc_samples","f8",("ns","boxz","boxy","boxx"))
                varvc           = samples_file.createVariable("vc_samples","f8",("ns","boxz","boxy","boxx"))
                varwc           = samples_file.createVariable("wc_samples","f8",("ns","boxz","boxy","boxx"))
                varpc           = samples_file.createVariable("pc_samples","f8",("ns","boxz","boxy","boxx"))
                varugradx       = samples_file.createVariable("ugradx_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varugrady       = samples_file.createVariable("ugrady_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varugradz       = samples_file.createVariable("ugradz_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varvgradx       = samples_file.createVariable("vgradx_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varvgrady       = samples_file.createVariable("vgrady_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varvgradz       = samples_file.createVariable("vgradz_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varwgradx       = samples_file.createVariable("wgradx_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varwgrady       = samples_file.createVariable("wgrady_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varwgradz       = samples_file.createVariable("wgradz_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varpgradx       = samples_file.createVariable("pgradx_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varpgrady       = samples_file.createVariable("pgrady_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varpgradz       = samples_file.createVariable("pgradz_samples","f8",("ns","boxz_gradients","boxy_gradients","boxx_gradients"))
                varunres_tau_xu = samples_file.createVariable("unres_tau_xu_samples","f8",("ns",))
                varunres_tau_xv = samples_file.createVariable("unres_tau_xv_samples","f8",("ns",))
                varunres_tau_xw = samples_file.createVariable("unres_tau_xw_samples","f8",("ns",))
                varunres_tau_yu = samples_file.createVariable("unres_tau_yu_samples","f8",("ns",))
                varunres_tau_yv = samples_file.createVariable("unres_tau_yv_samples","f8",("ns",))
                varunres_tau_yw = samples_file.createVariable("unres_tau_yw_samples","f8",("ns",))
                varunres_tau_zu = samples_file.createVariable("unres_tau_zu_samples","f8",("ns",))
                varunres_tau_zv = samples_file.createVariable("unres_tau_zv_samples","f8",("ns",))
                varunres_tau_zw = samples_file.createVariable("unres_tau_zw_samples","f8",("ns",))
    
    
            create_variables = False #Make sure the variables are only created once    
    
            #Store variables
            varuc[tot_sample_begin:tot_sample_eind,:,:,:]     = uc_samples[:,:,:,:]
            varvc[tot_sample_begin:tot_sample_eind,:,:,:]     = vc_samples[:,:,:,:]
            varwc[tot_sample_begin:tot_sample_eind,:,:,:]     = wc_samples[:,:,:,:]
            varpc[tot_sample_begin:tot_sample_eind,:,:,:]     = pc_samples[:,:,:,:]
            varugradx[tot_sample_begin:tot_sample_eind,:,:,:] = ugradx_samples[:,:,:,:]
            varugrady[tot_sample_begin:tot_sample_eind,:,:,:] = ugrady_samples[:,:,:,:]
            varugradz[tot_sample_begin:tot_sample_eind,:,:,:] = ugradz_samples[:,:,:,:]
            varvgradx[tot_sample_begin:tot_sample_eind,:,:,:] = vgradx_samples[:,:,:,:]
            varvgrady[tot_sample_begin:tot_sample_eind,:,:,:] = vgrady_samples[:,:,:,:]
            varvgradz[tot_sample_begin:tot_sample_eind,:,:,:] = vgradz_samples[:,:,:,:]
            varwgradx[tot_sample_begin:tot_sample_eind,:,:,:] = wgradx_samples[:,:,:,:]
            varwgrady[tot_sample_begin:tot_sample_eind,:,:,:] = wgrady_samples[:,:,:,:]
            varwgradz[tot_sample_begin:tot_sample_eind,:,:,:] = wgradz_samples[:,:,:,:]
            varpgradx[tot_sample_begin:tot_sample_eind,:,:,:] = pgradx_samples[:,:,:,:]
            varpgrady[tot_sample_begin:tot_sample_eind,:,:,:] = pgrady_samples[:,:,:,:]
            varpgradz[tot_sample_begin:tot_sample_eind,:,:,:] = pgradz_samples[:,:,:,:]
            varunres_tau_xu[tot_sample_begin:tot_sample_eind] = unres_tau_xu_samples[:]
            varunres_tau_xv[tot_sample_begin:tot_sample_eind] = unres_tau_xv_samples[:]
            varunres_tau_xw[tot_sample_begin:tot_sample_eind] = unres_tau_xw_samples[:]
            varunres_tau_yu[tot_sample_begin:tot_sample_eind] = unres_tau_yu_samples[:]
            varunres_tau_yv[tot_sample_begin:tot_sample_eind] = unres_tau_yv_samples[:]
            varunres_tau_yw[tot_sample_begin:tot_sample_eind] = unres_tau_yw_samples[:]
            varunres_tau_zu[tot_sample_begin:tot_sample_eind] = unres_tau_zu_samples[:]
            varunres_tau_zv[tot_sample_begin:tot_sample_eind] = unres_tau_zv_samples[:]
            varunres_tau_zw[tot_sample_begin:tot_sample_eind] = unres_tau_zw_samples[:]
    
            #close storage file
            samples_file.close()
    
        #Create binary tfrecord files (partly copied from script imagenet_to_gcs.py provided by the Tensorflow authors, 2017
        if create_binary:
    
            def _int64_feature(value):
                """Wrapper for inserting int64 features into Example proto."""
                if not isinstance(value, list):
                    value = [value]
                return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
            def _float_feature(value):
                """Wrapper for inserting float features into Example proto."""
                if not isinstance(value,list):
                    value = [value]
                return tf.train.Feature(float_list=tf.train.FloatList(value=value))        
            
            def _convert_to_example(uc_sample, vc_sample, wc_sample, pc_sample, unres_tau_xu_sample, unres_tau_xv_sample, unres_tau_xw_sample, unres_tau_yu_sample, unres_tau_yv_sample, unres_tau_yw_sample, unres_tau_zu_sample, unres_tau_zv_sample, unres_tau_zw_sample,x_sample_size,y_sample_size,z_sample_size):
                """Build an Example proto for an example.
                Args:
            
                  uc_sample: float, sample of u-component on coarse grid
    
                  vc_sample: float, sample of v-component on coarse grid
    
                  wc_sample: float, sample of w-component on coarse grid
    
                  pc_sample: float, sample of p on coarse grid
            
                  unres_tau_xu_sample: float, sample of unres_tau_xu on coarse grid
    
                  unres_tau_xv_sample: float, sample of unres_tau_xv on coarse grid
    
                  unres_tau_xw_sample: float, sample of unres_tau_xw on coarse grid
    
                  unres_tau_yu_sample: float, sample of unres_tau_yu on coarse grid
    
                  unres_tau_yv_sample: float, sample of unres_tau_yv on coarse grid
    
                  unres_tau_yw_sample: float, sample of unres_tau_yw on coarse grid
    
                  unres_tau_zu_sample: float, sample of unres_tau_zu on coarse grid
    
                  unres_tau_zv_sample: float, sample of unres_tau_zv on coarse grid
    
                  unres_tau_zw_sample: float, sample of unres_tau_zw on coarse grid
    
                  x_sample_size: integer, number of grid cells in x-direction selected in sample
    
                  y_sample_size: integer, number of grid cells in y-direction selected in sample
    
                  z_sample_size: integer, number of grid cells in z-direction selected in sample
     
                Returns:
                  Example proto
                """
            
                example = tf.train.Example(features=tf.train.Features(feature={
                               'uc_sample': _float_feature(uc_sample.flatten().tolist()),
                               'vc_sample': _float_feature(vc_sample.flatten().tolist()),
                               'wc_sample': _float_feature(wc_sample.flatten().tolist()),
                               'pc_sample': _float_feature(pc_sample.flatten().tolist()),
                               'unres_tau_xu_sample': _float_feature(unres_tau_xu_sample.flatten().tolist()),
                               'unres_tau_xv_sample': _float_feature(unres_tau_xv_sample.flatten().tolist()),
                               'unres_tau_xw_sample': _float_feature(unres_tau_xw_sample.flatten().tolist()),
                               'unres_tau_yu_sample': _float_feature(unres_tau_yu_sample.flatten().tolist()),
                               'unres_tau_yv_sample': _float_feature(unres_tau_yv_sample.flatten().tolist()),
                               'unres_tau_yw_sample': _float_feature(unres_tau_yw_sample.flatten().tolist()),
                               'unres_tau_zu_sample': _float_feature(unres_tau_zu_sample.flatten().tolist()),
                               'unres_tau_zv_sample': _float_feature(unres_tau_zv_sample.flatten().tolist()),
                               'unres_tau_zw_sample': _float_feature(unres_tau_zw_sample.flatten().tolist()),
                               'x_sample_size': _int64_feature(x_sample_size),
                               'y_sample_size': _int64_feature(y_sample_size),
                               'z_sample_size': _int64_feature(z_sample_size)}))        
                return example
            
            def _process_image_files_batch(output_file,uc_samples, vc_samples, wc_samples, pc_samples, unres_tau_xu_samples, unres_tau_xv_samples, unres_tau_xw_samples, unres_tau_yu_samples, unres_tau_yv_samples, unres_tau_yw_samples, unres_tau_zu_samples, unres_tau_zv_samples, unres_tau_zw_samples,x_sample_size,y_sample_size,z_sample_size):
            
                """Processes and saves list of images as TFRecords.
            
                Args:
            
                  output_file: string, name of file in which the samples are stored
            
                  uc_samples: float, samples of u-component on coarse grid
    
                  vc_samples: float, samples of v-component on coarse grid
    
                  wc_samples: float, samples of w-component on coarse grid
    
                  pc_samples: float, samples of p on coarse grid
    
                  unres_tau_xu_samples: float, samples of unres_tau_xu on coarse grid
    
                  unres_tau_xv_samples: float, samples of unres_tau_xv on coarse grid
    
                  unres_tau_xw_samples: float, samples of unres_tau_xw on coarse grid
    
                  unres_tau_yu_samples: float, samples of unres_tau_yu on coarse grid
    
                  unres_tau_yv_samples: float, samples of unres_tau_yv on coarse grid
    
                  unres_tau_yw_samples: float, samples of unres_tau_yw on coarse grid
    
                  unres_tau_zu_samples: float, samples of unres_tau_zu on coarse grid
    
                  unres_tau_zv_samples: float, samples of unres_tau_zv on coarse grid
    
                  unres_tau_zw_samples: float, samples of unres_tau_zw on coarse grid
    
                  x_sample_size: integer, number of grid cells in x-direction selected in samples
    
                  y_sample_size: integer, number of grid cells in y-direction selected in samples
    
                  z_sample_size: integer, number of grid cells in z-direction selected in samples
                """
                writer = tf.python_io.TFRecordWriter(output_file)
    
                for uc_sample, vc_sample, wc_sample, pc_sample, unres_tau_xu_sample, unres_tau_xv_sample, unres_tau_xw_sample, unres_tau_yu_sample, unres_tau_yv_sample, unres_tau_yw_sample, unres_tau_zu_sample, unres_tau_zv_sample, unres_tau_zw_sample in zip(uc_samples, vc_samples, wc_samples, pc_samples, unres_tau_xu_samples, unres_tau_xv_samples, unres_tau_xw_samples, unres_tau_yu_samples, unres_tau_yv_samples, unres_tau_yw_samples, unres_tau_zu_samples, unres_tau_zv_samples, unres_tau_zw_samples):
                    example = _convert_to_example(uc_sample, vc_sample, wc_sample, pc_sample, unres_tau_xu_sample, unres_tau_xv_sample, unres_tau_xw_sample, unres_tau_yu_sample, unres_tau_yv_sample, unres_tau_yw_sample, unres_tau_zu_sample, unres_tau_zv_sample, unres_tau_zw_sample,x_sample_size,y_sample_size,z_sample_size)
                    writer.write(example.SerializeToString())
            
                writer.close()
            

            def _convert_to_example_gradients(ugradx_sample, ugrady_sample, ugradz_sample, vgradx_sample, vgrady_sample, vgradz_sample, wgradx_sample, wgrady_sample, wgradz_sample, pgradx_sample, pgrady_sample, pgradz_sample, unres_tau_xu_sample, unres_tau_xv_sample, unres_tau_xw_sample, unres_tau_yu_sample, unres_tau_yv_sample, unres_tau_yw_sample, unres_tau_zu_sample, unres_tau_zv_sample, unres_tau_zw_sample,x_sample_size,y_sample_size,z_sample_size):
                """Build an Example proto for an example.
                Args:
            
                  ugradx_sample: float, sample of u-gradient w.r.t x on coarse grid

                  ugrady_sample: float, sample of u-gradient w.r.t y on coarse grid

                  ugradz_sample: float, sample of u-gradient w.r.t z on coarse grid

                  vgradx_sample: float, sample of v-gradient w.r.t x on coarse grid

                  vgrady_sample: float, sample of v-gradient w.r.t y on coarse grid

                  vgradz_sample: float, sample of v-gradient w.r.t z on coarse grid

                  wgradx_sample: float, sample of w-gradient w.r.t x on coarse grid

                  wgrady_sample: float, sample of w-gradient w.r.t y on coarse grid

                  wgradz_sample: float, sample of w-gradient w.r.t z on coarse grid

                  pgradx_sample: float, sample of p-gradient w.r.t x on coarse grid

                  pgrady_sample: float, sample of p-gradient w.r.t y on coarse grid
    
                  pgradz_sample: float, sample of p-gradient w.r.t z on coarse grid

                  unres_tau_xu_sample: float, sample of unres_tau_xu on coarse grid
    
                  unres_tau_xv_sample: float, sample of unres_tau_xv on coarse grid
    
                  unres_tau_xw_sample: float, sample of unres_tau_xw on coarse grid
    
                  unres_tau_yu_sample: float, sample of unres_tau_yu on coarse grid
    
                  unres_tau_yv_sample: float, sample of unres_tau_yv on coarse grid
    
                  unres_tau_yw_sample: float, sample of unres_tau_yw on coarse grid
    
                  unres_tau_zu_sample: float, sample of unres_tau_zu on coarse grid
    
                  unres_tau_zv_sample: float, sample of unres_tau_zv on coarse grid
    
                  unres_tau_zw_sample: float, sample of unres_tau_zw on coarse grid
    
                  x_sample_size: integer, number of grid cells in x-direction selected in sample
    
                  y_sample_size: integer, number of grid cells in y-direction selected in sample
    
                  z_sample_size: integer, number of grid cells in z-direction selected in sample
     
                Returns:
                  Example proto
                """
            
                example = tf.train.Example(features=tf.train.Features(feature={
                               'ugradx_sample': _float_feature(ugradx_sample.flatten().tolist()),
                               'ugrady_sample': _float_feature(ugrady_sample.flatten().tolist()),
                               'ugradz_sample': _float_feature(ugradz_sample.flatten().tolist()),
                               'vgradx_sample': _float_feature(vgradx_sample.flatten().tolist()),
                               'vgrady_sample': _float_feature(vgrady_sample.flatten().tolist()),
                               'vgradz_sample': _float_feature(vgradz_sample.flatten().tolist()),
                               'wgradx_sample': _float_feature(wgradx_sample.flatten().tolist()),
                               'wgrady_sample': _float_feature(wgrady_sample.flatten().tolist()),
                               'wgradz_sample': _float_feature(wgradz_sample.flatten().tolist()),
                               'pgradx_sample': _float_feature(pgradx_sample.flatten().tolist()),
                               'pgrady_sample': _float_feature(pgrady_sample.flatten().tolist()),
                               'pgradz_sample': _float_feature(pgradz_sample.flatten().tolist()),
                               'unres_tau_xu_sample': _float_feature(unres_tau_xu_sample.flatten().tolist()),
                               'unres_tau_xv_sample': _float_feature(unres_tau_xv_sample.flatten().tolist()),
                               'unres_tau_xw_sample': _float_feature(unres_tau_xw_sample.flatten().tolist()),
                               'unres_tau_yu_sample': _float_feature(unres_tau_yu_sample.flatten().tolist()),
                               'unres_tau_yv_sample': _float_feature(unres_tau_yv_sample.flatten().tolist()),
                               'unres_tau_yw_sample': _float_feature(unres_tau_yw_sample.flatten().tolist()),
                               'unres_tau_zu_sample': _float_feature(unres_tau_zu_sample.flatten().tolist()),
                               'unres_tau_zv_sample': _float_feature(unres_tau_zv_sample.flatten().tolist()),
                               'unres_tau_zw_sample': _float_feature(unres_tau_zw_sample.flatten().tolist()),
                               'x_sample_size': _int64_feature(x_sample_size),
                               'y_sample_size': _int64_feature(y_sample_size),
                               'z_sample_size': _int64_feature(z_sample_size)}))        
                return example
            
            def _process_image_files_batch_gradient(output_file,ugradx_samples, ugrady_samples, ugradz_samples, vgradx_samples, vgrady_samples, vgradz_samples, wgradx_samples, wgrady_samples, wgradz_samples, pgradx_samples, pgrady_samples, pgradz_samples, unres_tau_xu_samples, unres_tau_xv_samples, unres_tau_xw_samples, unres_tau_yu_samples, unres_tau_yv_samples, unres_tau_yw_samples, unres_tau_zu_samples, unres_tau_zv_samples, unres_tau_zw_samples, x_sample_size, y_sample_size, z_sample_size):
            
                """Processes and saves list of images as TFRecords.
            
                Args:
            
                  output_file: string, name of file in which the samples are stored

                  ugradx_samples: float, samples of u-gradient w.r.t x on coarse grid

                  ugrady_samples: float, samples of u-gradient w.r.t y on coarse grid

                  ugradz_samples: float, samples of u-gradient w.r.t z on coarse grid

                  vgradx_samples: float, samples of v-gradient w.r.t x on coarse grid

                  vgrady_samples: float, samples of v-gradient w.r.t y on coarse grid

                  vgradz_samples: float, samples of v-gradient w.r.t z on coarse grid

                  wgradx_samples: float, samples of w-gradient w.r.t x on coarse grid

                  wgrady_samples: float, samples of w-gradient w.r.t y on coarse grid

                  wgradz_samples: float, samples of w-gradient w.r.t z on coarse grid

                  pgradx_samples: float, samples of p-gradient w.r.t x on coarse grid

                  pgrady_samples: float, samples of p-gradient w.r.t y on coarse grid
    
                  pgradz_samples: float, samples of p-gradient w.r.t z on coarse grid
    
                  unres_tau_xu_samples: float, samples of unres_tau_xu on coarse grid
    
                  unres_tau_xv_samples: float, samples of unres_tau_xv on coarse grid
    
                  unres_tau_xw_samples: float, samples of unres_tau_xw on coarse grid
    
                  unres_tau_yu_samples: float, samples of unres_tau_yu on coarse grid
    
                  unres_tau_yv_samples: float, samples of unres_tau_yv on coarse grid
    
                  unres_tau_yw_samples: float, samples of unres_tau_yw on coarse grid
    
                  unres_tau_zu_samples: float, samples of unres_tau_zu on coarse grid
    
                  unres_tau_zv_samples: float, samples of unres_tau_zv on coarse grid
    
                  unres_tau_zw_samples: float, samples of unres_tau_zw on coarse grid
    
                  x_sample_size: integer, number of grid cells in x-direction selected in samples
    
                  y_sample_size: integer, number of grid cells in y-direction selected in samples
    
                  z_sample_size: integer, number of grid cells in z-direction selected in samples
                """
                writer = tf.python_io.TFRecordWriter(output_file)
    
                for ugradx_sample, ugrady_sample, ugradz_sample, vgradx_sample, vgrady_sample, vgradz_sample, wgradx_sample, wgrady_sample, wgradz_sample,  pgradx_sample, pgrady_sample, pgradz_sample, unres_tau_xu_sample, unres_tau_xv_sample, unres_tau_xw_sample, unres_tau_yu_sample, unres_tau_yv_sample, unres_tau_yw_sample, unres_tau_zu_sample, unres_tau_zv_sample, unres_tau_zw_sample in zip(ugradx_samples, ugrady_samples, ugradz_samples, vgradx_samples, vgrady_samples, vgradz_samples, wgradx_samples, wgrady_samples, wgradz_samples, pgradx_samples, pgrady_samples, pgradz_samples, unres_tau_xu_samples, unres_tau_xv_samples, unres_tau_xw_samples, unres_tau_yu_samples, unres_tau_yv_samples, unres_tau_yw_samples, unres_tau_zu_samples, unres_tau_zv_samples, unres_tau_zw_samples):

                    example = _convert_to_example_gradients(ugradx_sample, ugrady_sample, ugradz_sample, vgradx_sample, vgrady_sample, vgradz_sample, wgradx_sample, wgrady_sample, wgradz_sample,  pgradx_sample, pgrady_sample, pgradz_sample, unres_tau_xu_sample, unres_tau_xv_sample, unres_tau_xw_sample, unres_tau_yu_sample, unres_tau_yv_sample, unres_tau_yw_sample, unres_tau_zu_sample, unres_tau_zv_sample, unres_tau_zw_sample, x_sample_size,y_sample_size,z_sample_size)
                    writer.write(example.SerializeToString())
            
                writer.close()

            #Create training data based on absolute wind velocities
            output_file = os.path.join(output_directory, '{}_time_step_{}_of_{}.tfrecords'.format('training', t+1, nt))
            _process_image_files_batch(output_file,uc_samples, vc_samples, wc_samples, pc_samples, unres_tau_xu_samples, unres_tau_xv_samples, unres_tau_xw_samples, unres_tau_yu_samples, unres_tau_yv_samples, unres_tau_yw_samples, unres_tau_zu_samples, unres_tau_zv_samples, unres_tau_zw_samples, size_samples, size_samples, size_samples)
            print('Finished writing file: %s' % output_file)

            #Create training data based on gradients
            output_file_gradients = os.path.join(output_directory, '{}_time_step_{}_of_{}_gradients.tfrecords'.format('training', t+1,nt)) 
            _process_image_files_batch_gradient(output_file_gradients, ugradx_samples, ugrady_samples,ugradz_samples, vgradx_samples, vgrady_samples, vgradz_samples, wgradx_samples, wgrady_samples, wgradz_samples, pgradx_samples, pgrady_samples, pgradz_samples, unres_tau_xu_samples, unres_tau_xv_samples, unres_tau_xw_samples, unres_tau_yu_samples, unres_tau_yv_samples, unres_tau_yw_samples, unres_tau_zu_samples, unres_tau_zv_samples, unres_tau_zw_samples, size_samples_gradients, size_samples_gradients, size_samples_gradients)

    #Close data file
    a.close()

    #Store means and stdevs of all fields in separate nc-file if required by store_means_stdevs flag.
    #NOTE: arrays exist of single dimension, ranked according to time step.
    if store_means_stdevs:
        means_stdev_file = nc.Dataset(means_stdev_filepath, 'w')
        
        #Create new dimension
        means_stdev_file.createDimension("nt", nt)

        #Create new variables
        varmeanuc            = means_stdev_file.createVariable("mean_uc","f8",("nt",))
        varmeanvc            = means_stdev_file.createVariable("mean_vc","f8",("nt",))
        varmeanwc            = means_stdev_file.createVariable("mean_wc","f8",("nt",))
        varmeanpc            = means_stdev_file.createVariable("mean_pc","f8",("nt",))
        varmeanugradx        = means_stdev_file.createVariable("mean_ugradx","f8",("nt",))
        varmeanugrady        = means_stdev_file.createVariable("mean_ugrady","f8",("nt",))
        varmeanugradz        = means_stdev_file.createVariable("mean_ugradz","f8",("nt",))
        varmeanvgradx        = means_stdev_file.createVariable("mean_vgradx","f8",("nt",))
        varmeanvgrady        = means_stdev_file.createVariable("mean_vgrady","f8",("nt",))
        varmeanvgradz        = means_stdev_file.createVariable("mean_vgradz","f8",("nt",))
        varmeanwgradx        = means_stdev_file.createVariable("mean_wgradx","f8",("nt",))
        varmeanwgrady        = means_stdev_file.createVariable("mean_wgrady","f8",("nt",))
        varmeanwgradz        = means_stdev_file.createVariable("mean_wgradz","f8",("nt",))
        varmeanpgradx        = means_stdev_file.createVariable("mean_pgradx","f8",("nt",))
        varmeanpgrady        = means_stdev_file.createVariable("mean_pgrady","f8",("nt",))
        varmeanpgradz        = means_stdev_file.createVariable("mean_pgradz","f8",("nt",))
        varmeanunrestauxu    = means_stdev_file.createVariable("mean_unres_tau_xu_sample","f8",("nt",))
        varmeanunrestauyu    = means_stdev_file.createVariable("mean_unres_tau_yu_sample","f8",("nt",))
        varmeanunrestauzu    = means_stdev_file.createVariable("mean_unres_tau_zu_sample","f8",("nt",))
        varmeanunrestauxv    = means_stdev_file.createVariable("mean_unres_tau_xv_sample","f8",("nt",))
        varmeanunrestauyv    = means_stdev_file.createVariable("mean_unres_tau_yv_sample","f8",("nt",))
        varmeanunrestauzv    = means_stdev_file.createVariable("mean_unres_tau_zv_sample","f8",("nt",))
        varmeanunrestauxw    = means_stdev_file.createVariable("mean_unres_tau_xw_sample","f8",("nt",))
        varmeanunrestauyw    = means_stdev_file.createVariable("mean_unres_tau_yw_sample","f8",("nt",))
        varmeanunrestauzw    = means_stdev_file.createVariable("mean_unres_tau_zw_sample","f8",("nt",))

        varstdevuc           = means_stdev_file.createVariable("stdev_uc","f8",("nt",))
        varstdevvc           = means_stdev_file.createVariable("stdev_vc","f8",("nt",))
        varstdevwc           = means_stdev_file.createVariable("stdev_wc","f8",("nt",))
        varstdevpc           = means_stdev_file.createVariable("stdev_pc","f8",("nt",))
        varstdevugradx       = means_stdev_file.createVariable("stdev_ugradx","f8",("nt",))
        varstdevugrady       = means_stdev_file.createVariable("stdev_ugrady","f8",("nt",))
        varstdevugradz       = means_stdev_file.createVariable("stdev_ugradz","f8",("nt",))
        varstdevvgradx       = means_stdev_file.createVariable("stdev_vgradx","f8",("nt",))
        varstdevvgrady       = means_stdev_file.createVariable("stdev_vgrady","f8",("nt",))
        varstdevvgradz       = means_stdev_file.createVariable("stdev_vgradz","f8",("nt",))
        varstdevwgradx       = means_stdev_file.createVariable("stdev_wgradx","f8",("nt",))
        varstdevwgrady       = means_stdev_file.createVariable("stdev_wgrady","f8",("nt",))
        varstdevwgradz       = means_stdev_file.createVariable("stdev_wgradz","f8",("nt",))
        varstdevpgradx       = means_stdev_file.createVariable("stdev_pgradx","f8",("nt",))
        varstdevpgrady       = means_stdev_file.createVariable("stdev_pgrady","f8",("nt",))
        varstdevpgradz       = means_stdev_file.createVariable("stdev_pgradz","f8",("nt",))
        varstdevunrestauxu   = means_stdev_file.createVariable("stdev_unres_tau_xu_sample","f8",("nt",))
        varstdevunrestauyu   = means_stdev_file.createVariable("stdev_unres_tau_yu_sample","f8",("nt",))
        varstdevunrestauzu   = means_stdev_file.createVariable("stdev_unres_tau_zu_sample","f8",("nt",))
        varstdevunrestauxv   = means_stdev_file.createVariable("stdev_unres_tau_xv_sample","f8",("nt",))
        varstdevunrestauyv   = means_stdev_file.createVariable("stdev_unres_tau_yv_sample","f8",("nt",))
        varstdevunrestauzv   = means_stdev_file.createVariable("stdev_unres_tau_zv_sample","f8",("nt",))
        varstdevunrestauxw   = means_stdev_file.createVariable("stdev_unres_tau_xw_sample","f8",("nt",))
        varstdevunrestauyw   = means_stdev_file.createVariable("stdev_unres_tau_yw_sample","f8",("nt",))
        varstdevunrestauzw   = means_stdev_file.createVariable("stdev_unres_tau_zw_sample","f8",("nt",))

        #Store variables
        varmeanuc[:]         = mean_uc[:]
        varmeanvc[:]         = mean_vc[:]
        varmeanwc[:]         = mean_wc[:]
        varmeanpc[:]         = mean_pc[:]
        varmeanugradx[:]     = mean_ugradx[:]
        varmeanugrady[:]     = mean_ugrady[:]
        varmeanugradz[:]     = mean_ugradz[:]
        varmeanvgradx[:]     = mean_vgradx[:]
        varmeanvgrady[:]     = mean_vgrady[:]
        varmeanvgradz[:]     = mean_vgradz[:]
        varmeanwgradx[:]     = mean_wgradx[:]
        varmeanwgrady[:]     = mean_wgrady[:]
        varmeanwgradz[:]     = mean_wgradz[:]
        varmeanpgradx[:]     = mean_pgradx[:]
        varmeanpgrady[:]     = mean_pgrady[:]
        varmeanpgradz[:]     = mean_pgradz[:]
        varmeanunrestauxu[:] = mean_unres_tau_xu[:]
        varmeanunrestauyu[:] = mean_unres_tau_yu[:]
        varmeanunrestauzu[:] = mean_unres_tau_zu[:]
        varmeanunrestauxv[:] = mean_unres_tau_xv[:]
        varmeanunrestauyv[:] = mean_unres_tau_yv[:]
        varmeanunrestauzv[:] = mean_unres_tau_zv[:]
        varmeanunrestauxw[:] = mean_unres_tau_xw[:]
        varmeanunrestauyw[:] = mean_unres_tau_yw[:]
        varmeanunrestauzw[:] = mean_unres_tau_zw[:]

        varstdevuc[:]         = stdev_uc[:]
        varstdevvc[:]         = stdev_vc[:]
        varstdevwc[:]         = stdev_wc[:]
        varstdevpc[:]         = stdev_pc[:]
        varstdevugradx[:]     = stdev_ugradx[:]
        varstdevugrady[:]     = stdev_ugrady[:]
        varstdevugradz[:]     = stdev_ugradz[:]
        varstdevvgradx[:]     = stdev_vgradx[:]
        varstdevvgrady[:]     = stdev_vgrady[:]
        varstdevvgradz[:]     = stdev_vgradz[:]
        varstdevwgradx[:]     = stdev_wgradx[:]
        varstdevwgrady[:]     = stdev_wgrady[:]
        varstdevwgradz[:]     = stdev_wgradz[:]
        varstdevpgradx[:]     = stdev_pgradx[:]
        varstdevpgrady[:]     = stdev_pgrady[:]
        varstdevpgradz[:]     = stdev_pgradz[:]
        varstdevunrestauxu[:] = stdev_unres_tau_xu[:]
        varstdevunrestauyu[:] = stdev_unres_tau_yu[:]
        varstdevunrestauzu[:] = stdev_unres_tau_zu[:]
        varstdevunrestauxv[:] = stdev_unres_tau_xv[:]
        varstdevunrestauyv[:] = stdev_unres_tau_yv[:]
        varstdevunrestauzv[:] = stdev_unres_tau_zv[:]
        varstdevunrestauxw[:] = stdev_unres_tau_xw[:]
        varstdevunrestauyw[:] = stdev_unres_tau_yw[:]
        varstdevunrestauzw[:] = stdev_unres_tau_zw[:]

        #Close storage file
        means_stdev_file.close()
