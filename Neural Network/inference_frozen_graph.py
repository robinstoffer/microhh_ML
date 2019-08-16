#Script to load frozen model and do inference on flow snapshots stored in training file.
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import argparse
#import tensorflow as tf
import numpy as np
import netCDF4 as nc
import time
from diff_manual_python import diff_U

class MLP:
    '''Class to manually build MLP and subsequently do inference. NOTE: should be completely equivalent to MLP defined in MLP2_estimator.py!!!'''

    def __init__(self,ndense): #Specify number of neurons in dense layer when instantiating the MLP
        
        self.ndense             = ndense
        
        #Load all weights and other variables from text files created with extract_variables_graph function located in load_frozen_graph.py
        self.means_inputs       = np.loadtxt('means_inputs.txt')
        self.stdevs_inputs      = np.loadtxt('stdevs_inputs.txt')
        self.means_labels       = np.loadtxt('means_labels.txt')
        self.stdevs_labels      = np.loadtxt('stdevs_labels.txt')
        self.MLPu_hidden_kernel = np.loadtxt('MLPu_hidden_kernel.txt').transpose()
        self.MLPu_hidden_bias   = np.loadtxt('MLPu_hidden_bias.txt')
        self.MLPu_hidden_alpha  = np.loadtxt('MLPu_hidden_alpha.txt')
        self.MLPu_output_kernel = np.loadtxt('MLPu_output_kernel.txt').transpose()
        self.MLPu_output_bias   = np.loadtxt('MLPu_output_bias.txt')
        self.MLPv_hidden_kernel = np.loadtxt('MLPv_hidden_kernel.txt').transpose()
        self.MLPv_hidden_bias   = np.loadtxt('MLPv_hidden_bias.txt')
        self.MLPv_hidden_alpha  = np.loadtxt('MLPv_hidden_alpha.txt')
        self.MLPv_output_kernel = np.loadtxt('MLPv_output_kernel.txt').transpose()
        self.MLPv_output_bias   = np.loadtxt('MLPv_output_bias.txt')
        self.MLPw_hidden_kernel = np.loadtxt('MLPw_hidden_kernel.txt').transpose()
        self.MLPw_hidden_bias   = np.loadtxt('MLPw_hidden_bias.txt')
        self.MLPw_hidden_alpha  = np.loadtxt('MLPw_hidden_alpha.txt')
        self.MLPw_output_kernel = np.loadtxt('MLPw_output_kernel.txt').transpose()
        self.MLPw_output_bias   = np.loadtxt('MLPw_output_bias.txt')
        self.output_denorm_utau2= np.loadtxt('output_denorm_utau2.txt')

#        self.iteration=0

    #Define private function to make input variables non-dimensionless and standardize them
    def _standardization(self, input_variable, mean_variable, stdev_variable, scaling_factor):
        input_variable = np.divide(input_variable, scaling_factor)
        input_variable = np.subtract(input_variable, mean_variable)
        input_variable = np.divide(input_variable, stdev_variable)
        return input_variable

    
     #Define private function that executes a separate MLP.
     def _single_MLP(self, inputs, name_MLP, params):
         '''Function to execute a MLP with specified input. Inputs should be a list of numpy arrays containing the individual variables.'''
     
         #Make input layer
         input_layer = np.concatenate(inputs, axis=1).flatten()
     
         #Execute hidden layer with Leaky Relu activation function
         hidden_neurons = np.dot(self.MLPu_hidden_kernel, input_layer) + self.MLPu_hidden_bias
         y1 = (hidden_neurons > 0) * hidden_neurons
         y2 = (hidden_neurons <= 0) * hidden_neurons * self.MLPu_hidden_alpha
         hidden_activations = y1 + y2
         #CONTINUE HERE!!!!!!!!!!!!!!!!!
         #Execute output layer with no activation function
         output_activations = np.dot(self.MLPu_hidden_kernel, input_layer) + self.MLPu_hidden_bias
         return output_layer

    def predict(self, input_u, input_v, input_w, input_utau_ref):
        
        #Standardize input variables
        input_u_stand  = self._standardization(input_u, self.means_inputs[0], self.stdevs_inputs[0], input_utau_ref)
        input_v_stand  = self._standardization(input_v, self.means_inputs[1], self.stdevs_inputs[1], input_utau_ref)
        input_w_stand  = self._standardization(input_w, self.means_inputs[2], self.stdevs_inputs[2], input_utau_ref)
        
#        self.iteration += 1
#        print(self.iteration)

        #Specify for now some dummy values
        output_denorm = np.ones((1,18))
        return output_denorm

class Grid:
    '''Class to store information about grid.'''

    def __init__(self,coord_center,coord_edge,gc,end_ind_center,end_ind_edge):
        self.zc      = coord_center[0]
        self.yc      = coord_center[1]
        self.xc      = coord_center[2]
        self.ktot    = len(self.zc)
        self.jtot    = len(self.yc)
        self.itot    = len(self.xc)
        self.zhc     = coord_edge[0]
        self.yhc     = coord_edge[1]
        self.xhc     = coord_edge[2]
        self.zsize   = self.zhc[-1]
        self.ysize   = self.yhc[-1]
        self.xsize   = self.xhc[-1]
        self.kstart  = gc[0]
        self.jstart  = gc[1]
        self.istart  = gc[2]
        self.kend    = end_ind_center[0]
        self.jend    = end_ind_center[1]
        self.iend    = end_ind_center[2]
        self.khend   = end_ind_edge[0]
        self.jhend   = end_ind_edge[1]
        self.ihend   = end_ind_edge[2]
        self.kcells  = self.kend   + self.kstart
        self.khcells = self.khend  + self.kstart
        self.jcells  = self.jend   + self.jstart
        self.icells  = self.iend   + self.istart
        self.ijcells = self.icells * self.jcells
        self.dx      = self.xsize  / self.itot
        self.dy      = self.ysize  / self.jtot


if __name__ == '__main__':
    # Pass filenames and batch size as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_graph_filename", default="frozen_inference_graph.pb")
    parser.add_argument("--training_filename", default="training_data.nc")
    parser.add_argument("--inference_filename", default="inference_reconstructed_fields.nc")
    parser.add_argument("--store_variables", default=None, action="store_true", help="Store all tendencies in a netCDF-file when specified.")
    #parser.add_argument("--batch_size", default=1000)
    args = parser.parse_args()
    #batch_size = int(args.batch_size)

    ###Extract flow fields and from netCDF file###
    #Specify time steps NOTE: SHOULD BE 27 TO 30 to access validation fields, CHECK WHETHER THIS IS STILL CONSISTENT!
    #NOTE: flow fields were normalized stored, UNDO first normalisation!
    tstart = 27
    tend   = 30
    tstep_unique = np.linspace(tstart,tend-1, num=3)
    nt = tend - tstart
    #
    flowfields = nc.Dataset(args.training_filename)
    utau_ref_channel = np.array(flowfields['utau_ref'][:],dtype='f4')
    u = np.array(flowfields['uc'][tstart:tend,:,:,:]) * utau_ref_channel
    v = np.array(flowfields['vc'][tstart:tend,:,:,:]) * utau_ref_channel
    w = np.array(flowfields['wc'][tstart:tend,:,:,:]) * utau_ref_channel
    utau_ref = 0.2 #in [m/s]
    #
    #Extract coordinates, shape fields, and ghost cells
    zc       = np.array(flowfields['zc'][:])
    zgc      = np.array(flowfields['zgc'][:])
    nz       = len(zc)
    zhc      = np.array(flowfields['zhc'][:])
    zgcextra = np.array(flowfields['zgcextra'][:])
    yc       = np.array(flowfields['yc'][:])
    ny       = len(yc)
    yhc      = np.array(flowfields['yhc'][:])
    ygcextra = np.array(flowfields['ygcextra'][:])
    xc       = np.array(flowfields['xc'][:])
    nx       = len(xc)
    xhc      = np.array(flowfields['xhc'][:])
    xgcextra = np.array(flowfields['xgcextra'][:])
    zhcless  = zhc[:-1]
    yhcless  = yhc[:-1]
    xhcless  = xhc[:-1]
    igc      = int(flowfields['igc'][:])
    jgc      = int(flowfields['jgc'][:])
    kgc      = int(flowfields['kgc_center'][:])
    iend     = int(flowfields['iend'][:])
    ihend    = int(flowfields['ihend'][:])
    jend     = int(flowfields['jend'][:])
    jhend    = int(flowfields['jhend'][:])
    kend     = int(flowfields['kend'][:])
    khend    = int(flowfields['khend'][:])
    #

    #Store grid information in a class object called grid.
    grid = Grid(coord_center = (zc,yc,xc), coord_edge = (zhc,yhc,xhc), gc = (kgc,jgc,igc), end_ind_center = (kend,jend,iend), end_ind_edge = (khend,jhend,ihend))
    
    #Calculate height differences, ASSUMING a second-order numerical scheme for calculation of the heights
    zgc1     = zgc[grid.kstart-1:grid.kend+1] #Include one ghost cell at each side
    dzh      = np.zeros(grid.ktot + 2) #Include two ghost cells for heights
    dzh[1:]  = zgc1[1:] - zgc1[:-1]
    dzh[0]   = dzh[2]
    dzhi     = 1. / dzh
    #
    dz       = np.zeros(grid.ktot + 2)
    dz[1:-1] = grid.zhc[1:]  - grid.zhc[:-1]
    dz[0]    = dz[1]
    dz[-1]   = dz[-2]
    dzi      = 1. / dz
    
    if args.store_variables:
        ###Create file for inference results###
        inference = nc.Dataset(args.inference_filename, 'w')

        #Create dimensions for storage in nc-file
        inference.createDimension("zc", len(zc))
        inference.createDimension("zgcextra", len(zgcextra))
        inference.createDimension("zhc",len(zhc))
        inference.createDimension("zhcless",len(zhcless))
        inference.createDimension("yc", len(yc))
        inference.createDimension("ygcextra", len(ygcextra))
        inference.createDimension("yhc",len(yhc))
        inference.createDimension("yhcless",len(yhcless))
        inference.createDimension("xc", len(xc))
        inference.createDimension("xgcextra", len(xgcextra))
        inference.createDimension("xhc",len(xhc))
        inference.createDimension("xhcless",len(xhcless))
        inference.createDimension("tstep_unique",len(tstep_unique))

        #Create variables for dimensions and store them
        var_zc           = inference.createVariable("zc",           "f8", ("zc",))
        var_zgcextra     = inference.createVariable("zgcextra",     "f8", ("zgcextra",))
        var_zhc          = inference.createVariable("zhc",          "f8", ("zhc",))
        var_zhcless      = inference.createVariable("zhcless",      "f8", ("zhcless",))
        var_yc           = inference.createVariable("yc",           "f8", ("yc",))
        var_ygcextra     = inference.createVariable("ygcextra",     "f8", ("ygcextra",))
        var_yhc          = inference.createVariable("yhc",          "f8", ("yhc",))
        var_yhcless      = inference.createVariable("yhcless",      "f8", ("yhcless",))
        var_xc           = inference.createVariable("xc",           "f8", ("xc",))
        var_xgcextra     = inference.createVariable("xgcextra",     "f8", ("xgcextra",))
        var_xhc          = inference.createVariable("xhc",          "f8", ("xhc",))
        var_xhcless      = inference.createVariable("xhcless",      "f8", ("xhcless",))
        var_tstep_unique = inference.createVariable("tstep_unique", "f8", ("tstep_unique",))

        var_zc[:]            = zc
        var_zgcextra[:]      = zgcextra
        var_zhc[:]           = zhc
        var_zhcless[:]       = zhcless
        var_yc[:]            = yc
        var_ygcextra[:]      = ygcextra
        var_yhc[:]           = yhc
        var_yhcless[:]       = yhcless
        var_xc[:]            = xc
        var_xgcextra[:]      = xgcextra
        var_xhc[:]           = xhc
        var_xhcless[:]       = xhcless
        var_tstep_unique[:]  = tstep_unique
        
        #Initialize variables for storage inference results
        #var_unres_tau_xu_CNN = inference.createVariable("unres_tau_xu_CNN","f8",("tstep_unique","zc","yc","xgcextra"))      
        #var_unres_tau_xv_CNN = inference.createVariable("unres_tau_xv_CNN","f8",("tstep_unique","zc","yhcless","xhc"))     
        #var_unres_tau_xw_CNN = inference.createVariable("unres_tau_xw_CNN","f8",("tstep_unique","zhcless","yc","xhc"))     
        #var_unres_tau_yu_CNN = inference.createVariable("unres_tau_yu_CNN","f8",("tstep_unique","zc","yhc","xhcless"))
        #var_unres_tau_yv_CNN = inference.createVariable("unres_tau_yv_CNN","f8",("tstep_unique","zc","ygcextra","xc"))
        #var_unres_tau_yw_CNN = inference.createVariable("unres_tau_yw_CNN","f8",("tstep_unique","zhcless","yhc","xc"))
        #var_unres_tau_zu_CNN = inference.createVariable("unres_tau_zu_CNN","f8",("tstep_unique","zhc","yc","xhcless"))
        #var_unres_tau_zv_CNN = inference.createVariable("unres_tau_zv_CNN","f8",("tstep_unique","zhc","yhcless","xc"))
        #var_unres_tau_zw_CNN = inference.createVariable("unres_tau_zw_CNN","f8",("tstep_unique","zc","yc","xc"))
        #
        var_ut = inference.createVariable("u_tendency","f8",("tstep_unique","zc","yc","xhcless"))
        var_vt = inference.createVariable("v_tendency","f8",("tstep_unique","zc","yhcless","xc"))
        var_wt = inference.createVariable("w_tendency","f8",("tstep_unique","zhcless","yc","xc"))
    
    #Instantiate manual MLP class for making predictions
    MLP = MLP(ndense = 107)
    
    #Loop over flow fields, for each time step in tstep_unique (giving 4 loops in total).
    #For each alternating grid cell, store transport components by calling the 'frozen' MLP within a tf.Session().
    #for t in range(nt):
    for t in range(1): #NOTE:FOR TESTING PURPOSES ONLY!
        
        #Select flow fields of time step
        u_singletimestep = u[t,:,:,:-1].flatten()#Flatten and remove ghost cells in horizontal staggered dimensions to make shape consistent to arrays in MicroHH
        v_singletimestep = v[t,:,:-1,:].flatten()
        w_singletimestep = w[t,:,:,:].flatten()
       
        #Define loop indices
        #ii = 1
        #jj = icells
        #kk = ijcells
        blocksize = 5 #size of block used as input for MLP
        b = blocksize // 2 

        ###The code in the block below is the only part that has to be executed when doing inference in MicroHH###
        t1_start = time.perf_counter()

        ##Initialize empty arrays for temporary storage transport components
        #unres_tau_xu_CNN = np.full((len(zc),len(yc),len(xgcextra)), np.nan, dtype=np.float32)
        #unres_tau_xv_CNN = np.full((len(zc),len(yhcless),len(xhc)), np.nan, dtype=np.float32)
        #unres_tau_xw_CNN = np.full((len(zhcless),len(yc),len(xhc)), np.nan, dtype=np.float32)
        #unres_tau_yu_CNN = np.full((len(zc),len(yhc),len(xhcless)), np.nan, dtype=np.float32)
        #unres_tau_yv_CNN = np.full((len(zc),len(ygcextra),len(xc)), np.nan, dtype=np.float32)
        #unres_tau_yw_CNN = np.full((len(zhcless),len(yhc),len(xc)), np.nan, dtype=np.float32)
        #unres_tau_zu_CNN = np.full((len(zhc),len(yc),len(xhcless)), np.nan, dtype=np.float32)
        #unres_tau_zv_CNN = np.full((len(zhc),len(yhcless),len(xc)), np.nan, dtype=np.float32)
        #unres_tau_zw_CNN = np.full((len(zc),len(yc),len(xc)),       np.nan, dtype=np.float32)

        #NOTE: assignment of unresolved transports happens as a side-effect within the function below!
        ut, vt, wt = diff_U(u = u_singletimestep, v = v_singletimestep, w = w_singletimestep, utau_ref = utau_ref_channel, frozen_graph_filename = args.frozen_graph_filename, dzi = dzi, dzhi = dzhi, grid = grid, MLP = MLP, b = b) #Call for scripts based on manual implementation MLP
        #ut, vt, wt = diff_U(u = u_singletimestep, v = v_singletimestep, w = w_singletimestep, utau_ref = utau_ref_channel, frozen_graph_filename = args.frozen_graph_filename, dzi = dzi, dzhi = dzhi, grid = grid, b = b) #Call for script based on frozen graph
        t1_end = time.perf_counter()

        print("Elapsed time during one iteration through a single flow field: ", t1_end - t1_start, " seconds.")

        ###End code block which has to be executed in MicroHH###

        if args.store_variables:
            ##Store temporary arrays containing transport components in nc-file, rescale with realistic friction velocity
            #var_unres_tau_xu_CNN[t,:,:,:] =  unres_tau_xu_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_yu_CNN[t,:,:,:] =  unres_tau_yu_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_zu_CNN[t,:,:,:] =  unres_tau_zu_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_xv_CNN[t,:,:,:] =  unres_tau_xv_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_yv_CNN[t,:,:,:] =  unres_tau_yv_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_zv_CNN[t,:,:,:] =  unres_tau_zv_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_xw_CNN[t,:,:,:] =  unres_tau_xw_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_yw_CNN[t,:,:,:] =  unres_tau_yw_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            #var_unres_tau_zw_CNN[t,:,:,:] =  unres_tau_zw_CNN[:,:,:] * ((utau_ref ** 2)/(utau_ref_channel ** 2))
            ##
            var_ut[t,:,:,:] = ut[:,:,:]
            var_vt[t,:,:,:] = vt[:,:,:]
            var_wt[t,:,:,:] = wt[:,:,:]
            #

    if args.store_variables:
        #Close inference file
        inference.close()

    #Close flow fields file
    flowfields.close()
