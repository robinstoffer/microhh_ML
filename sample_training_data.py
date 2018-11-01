import numpy as np
import netCDF4 as nc

#Fetch training data
a = nc.Dataset('training_data.nc', 'r')

nt,nz,ny,nx = a['pc'].shape # nt should be the same for all variables, here defined from p. Note that nz,ny,nx can be 1 point smaller for velocity components, taken into account in the for-loops below
nsamples = (nz-4-1)*(ny-4-1)*(nx-4-1) #-4 needed to account for the four vertical layers that are discarded in the sampling, -1 to take into account smaller lengths velocity components

#Loop over timesteps, take for each timestep samples of 5x5x5
tot_sample_num = 0
create_variables = True
create_file = True
for time_step in range(nt):
    #Open/create netCDF-file for storage
    if create_file:
        samples_file = nc.Dataset('samples_training.nc', 'w')
        create_file = False
    else:
        samples_file = nc.Dataset('samples_training.nc', 'r+')

    #Define variables for storage
    uc_samples = np.zeros((nsamples,5,5,5))
    vc_samples = np.zeros((nsamples,5,5,5))
    wc_samples = np.zeros((nsamples,5,5,5))
    pc_samples = np.zeros((nsamples,5,5,5))
    unres_tau_xu_samples = np.zeros((nsamples,1))
    unres_tau_xv_samples = np.zeros((nsamples,1))
    unres_tau_xw_samples = np.zeros((nsamples,1))
    unres_tau_yu_samples = np.zeros((nsamples,1))
    unres_tau_yv_samples = np.zeros((nsamples,1))
    unres_tau_yw_samples = np.zeros((nsamples,1))
    unres_tau_zu_samples = np.zeros((nsamples,1))
    unres_tau_zv_samples = np.zeros((nsamples,1))
    unres_tau_zw_samples = np.zeros((nsamples,1))

    tot_sample_begin = tot_sample_num
    sample_num = 0
    uc_singlefield = a['uc'][time_step,:,:,:]
    vc_singlefield = a['vc'][time_step,:,:,:]
    wc_singlefield = a['wc'][time_step,:,:,:]
    pc_singlefield = a['pc'][time_step,:,:,:]
    #total_tau_xu_singlefield = a["total_tau_xu"][time_step,:,:,:]
    #res_tau_xu_singlefield = a["res_tau_xu"][time_step,:,:,:]    
    unres_tau_xu_singlefield = a["unres_tau_xu"][time_step,:,:,:] 
    #total_tau_xv_singlefield = a["total_tau_xv"][time_step,:,:,:]
    #res_tau_xv_singlefield = a["res_tau_xv"][time_step,:,:,:]
    unres_tau_xv_singlefield = a["unres_tau_xv"][time_step,:,:,:]
    #total_tau_xw_singlefield = a["total_tau_xw"][time_step,:,:,:]
    #res_tau_xw_singlefield = a["res_tau_xw"][time_step,:,:,:]
    unres_tau_xw_singlefield = a["unres_tau_xw"][time_step,:,:,:]
    #total_tau_yu_singlefield = a["total_tau_yu"][time_step,:,:,:]
    #res_tau_yu_singlefield = a["res_tau_yu"][time_step,:,:,:]
    unres_tau_yu_singlefield = a["unres_tau_yu"][time_step,:,:,:]
    #total_tau_yv_singlefield = a["total_tau_yv"][time_step,:,:,:]
    #res_tau_yv_singlefield = a["res_tau_yv"][time_step,:,:,:]
    unres_tau_yv_singlefield = a["unres_tau_yv"][time_step,:,:,:]
    #total_tau_yw_singlefield = a["total_tau_yw"][time_step,:,:,:]
    #res_tau_yw_singlefield = a["res_tau_yw"][time_step,:,:,:]
    unres_tau_yw_singlefield = a["unres_tau_yw"][time_step,:,:,:]
    #total_tau_zu_singlefield = a["total_tau_zu"][time_step,:,:,:]
    #res_tau_zu_singlefield = a["res_tau_zu"][time_step,:,:,:]
    unres_tau_zu_singlefield = a["unres_tau_zu"][time_step,:,:,:]
    #total_tau_zv_singlefield = a["total_tau_zv"][time_step,:,:,:]
    #res_tau_zv_singlefield = a["res_tau_zv"][time_step,:,:,:]
    unres_tau_zv_singlefield = a["unres_tau_zv"][time_step,:,:,:]
    #total_tau_zw_singlefield = a["total_tau_zw"][time_step,:,:,:]
    #res_tau_zw_singlefield = a["res_tau_zw"][time_step,:,:,:]
    unres_tau_zw_singlefield = a["unres_tau_zw"][time_step,:,:,:]


    for index_z in range(3,nz-2): #Make sure no vertical levels are sampled where because of the vicintiy to the wall no samples of 5X5X5 grid cells can be taken.
        index_zlow = index_z-2
        index_zhigh = index_z+3
        for index_y in range(3,ny-2): #Make sure no positions close to the edges are sampled to keep things easy.
            index_ylow = index_y-2
            index_yhigh = index_y+3

            for index_x in range(3,nx-2): #Make sure no positions close to the edges are sampled to keep things easy.
                index_xlow = index_x-2
                index_xhigh = index_x+3

                #Take samples of 5x5x5 (depending on whether either x or y is negative and store them
                uc_samples[sample_num,:,:,:] = uc_singlefield[index_zlow:index_zhigh,index_ylow:index_yhigh,index_xlow-1:index_xhigh-1]
                vc_samples[sample_num,:,:,:] = vc_singlefield[index_zlow:index_zhigh,index_ylow-1:index_yhigh-1,index_xlow:index_xhigh]
                wc_samples[sample_num,:,:,:] = wc_singlefield[index_zlow-1:index_zhigh-1,index_ylow:index_yhigh,index_xlow:index_xhigh]
                pc_samples[sample_num,:,:,:] = pc_singlefield[index_zlow:index_zhigh,index_ylow:index_yhigh,index_xlow:index_xhigh]
                unres_tau_xu_samples[sample_num] = unres_tau_xu_singlefield[index_z,index_y,index_x-1]
                unres_tau_xv_samples[sample_num] = unres_tau_xv_singlefield[index_z,index_y,index_x-1]
                unres_tau_xw_samples[sample_num] = unres_tau_xw_singlefield[index_z,index_y,index_x-1]
                unres_tau_yu_samples[sample_num] = unres_tau_yu_singlefield[index_z,index_y-1,index_x]
                unres_tau_yv_samples[sample_num] = unres_tau_yv_singlefield[index_z,index_y-1,index_x]
                unres_tau_yw_samples[sample_num] = unres_tau_yw_singlefield[index_z,index_y-1,index_x]                
                unres_tau_zu_samples[sample_num] = unres_tau_zu_singlefield[index_z-1,index_y,index_x]
                unres_tau_zv_samples[sample_num] = unres_tau_zv_singlefield[index_z-1,index_y,index_x]
                unres_tau_zw_samples[sample_num] = unres_tau_zw_singlefield[index_z-1,index_y,index_x]

                sample_num +=1
                tot_sample_num+=1

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
        dim_ns = samples_file.createDimension("ns",None)
        dim_boxx = samples_file.createDimension("boxx",5)
        dim_boxy = samples_file.createDimension("boxy",5)
        dim_boxz = samples_file.createDimension("boxz",5)


        #Create new variables
        varuc = samples_file.createVariable("uc_samples","f8",("ns","boxz","boxy","boxx"))
        varvc = samples_file.createVariable("vc_samples","f8",("ns","boxz","boxy","boxx"))
        varwc = samples_file.createVariable("wc_samples","f8",("ns","boxz","boxy","boxx"))
        varpc = samples_file.createVariable("pc_samples","f8",("ns","boxz","boxy","boxx"))
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
    varuc[tot_sample_begin:tot_sample_eind,:,:,:] = uc_samples[:,:,:,:]
    varvc[tot_sample_begin:tot_sample_eind,:,:,:] = vc_samples[:,:,:,:]
    varwc[tot_sample_begin:tot_sample_eind,:,:,:] = wc_samples[:,:,:,:]
    varpc[tot_sample_begin:tot_sample_eind,:,:,:] = pc_samples[:,:,:,:]
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

#Close data file
a.close()

