#Test script for func_generate_training.py and microhh_tools_robinst.py

#Load modules
import numpy as np
#import netCDF4 as nc
#import struct as st
#import glob
#import re
#import matplotlib as mpl
#mpl.use('Agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
#import matplotlib.pyplot as plt
#import scipy.interpolate

#Load scripts to be tested
from grid_objects_training import Finegrid, Coarsegrid
from func_generate_training import generate_training_data

#Do testing
finegrid = Finegrid(read_grid_flag = False, coordx = np.array([1,2.5,3]), xsize = 4, coordy = np.array([0.5,1.5,3.5,6]), ysize = 7, coordz = np.array([0.1,0.3,1]), zsize = 1.2)
#finegrid = Finegrid()
finegrid.create_variables('u', np.reshape(np.arange(0,36), (3,4,3)), bool_edge_gridcell = (False, False, True))
finegrid.create_variables('v', np.reshape(np.arange(0,36), (3,4,3)), bool_edge_gridcell = (False, True, False))
finegrid.create_variables('w', np.reshape(np.arange(0,36), (3,4,3)), bool_edge_gridcell = (True, False, False))
finegrid.create_variables('p', np.reshape(np.arange(0,36), (3,4,3)), bool_edge_gridcell = (False, False, False))

coarsegrid = Coarsegrid((2,3,2), finegrid)
coarsegrid.downsample('u')
coarsegrid.downsample('v')
coarsegrid.downsample('w')
coarsegrid.downsample('p')
#t = 2
#finegrid.read_binary_variables('u', t, bool_edge_gridcell = (False, False, True))
#finegrid.read_binary_variables('v', t, bool_edge_gridcell = (False, True,  False))
#finegrid.read_binary_variables('w', t, bool_edge_gridcell = (True , False, False))
#finegrid.read_binary_variables('p', t, bool_edge_gridcell = (False, False, False))

#finegrid.add_ghostcells_hor('u', 1, 1, bool_edge_gridcell = (False, False, True))
#finegrid.add_ghostcells_hor('v', 4, 1, bool_edge_gridcell = (False, True, False))
#finegrid.add_ghostcells_hor('w', 0, 2, bool_edge_gridcell = (True, False, False))
#finegrid.add_ghostcells_hor('p', 3, 2, bool_edge_gridcell = (False, False, False))

#coarsegrid = Coarsegrid((32,16,64), finegrid)
#coarsegrid = Coarsegrid((2,3,2), finegrid)
#coarsegrid['output']['u'] = np.reshape(np.arange(0,18), (2,3,3)) #NOTE: For coarsegrid object ghostcell at downstream/ top boundary must be included for variable
#coarsegrid['output']['u'][:,:,-1] = coarsegrid['output']['u'][:,:,0] #NOTE: In the xh-direction the last grid cell should be equal to the first one.
#coarsegrid.add_ghostcells_hor('u', 2, 2, bool_edge_gridcell = (False, False, True))

#generate_training_data((2,3,2))
