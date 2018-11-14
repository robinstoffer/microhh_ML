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
coordx = np.array([0.25,0.75])
xsize = 1.0
coordy = np.array([0.25,0.75])
ysize = 1.0
coordz = np.array([0.01,0.5,0.95])
zsize = 1.0

#coordx = np.array([0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75])
#xsize = 4.0
#coordy = np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5])
#ysize = 10.0
#coordz = np.array([0.01,0.05,0.1,13.7,13.8,13.85,13.9,14.0])
#zsize = 20.0

finegrid = Finegrid(read_grid_flag = False, fourth_order = False, coordx = coordx, xsize = xsize, coordy = coordy, ysize = ysize, coordz = coordz, zsize = zsize, periodic_bc = (False, True, True), no_slip = True)
#finegrid = Finegrid()
#finegrid = Finegrid(read_grid_flag = False, fourth_order = False, coordx = np.array([0.25, 0.75]), xsize = 1.0, coordy = np.array([0.25,0.75]), ysize = 1.0, coordz = np.array([0.5,0.75]), zsize = 1.0, periodic_bc = (False, True, True), no_slip = True)

output_shape = (3,2,2)
#start_value = 0
#end_value = start_value + output_shape[0]*output_shape[1]*output_shape[2]
#output_array = np.reshape(np.arange(start_value,end_value), output_shape)
output_array = np.ones(output_shape)
#output_0level = np.zeros((output_shape[1], output_shape[2]))
output_1level = np.ones((output_shape[1], output_shape[2]))
output_array = np.stack([output_1level, 2*output_1level, 3*output_1level], axis = 0)

finegrid.create_variables('u', output_array, bool_edge_gridcell = (False, False, True))
finegrid.create_variables('v', output_array, bool_edge_gridcell = (False, True, False))
finegrid.create_variables('w', output_array, bool_edge_gridcell = (True, False, False))
finegrid.create_variables('p', output_array, bool_edge_gridcell = (False, False, False))

coarsegrid = Coarsegrid((3,2,2), finegrid)
coarsegrid.downsample('u')
coarsegrid.downsample('v')
coarsegrid.downsample('w')
coarsegrid.downsample('p')

volume_finegrid_u = np.round(finegrid.volume_integral('u'), finegrid.sgn_digits)
volume_coarsegrid_u = np.round(coarsegrid.volume_integral('u'), coarsegrid.sgn_digits)
print('Volume integral u finegrid: ' + str(volume_finegrid_u))
print('Volume integral u coarsegrid: ' + str(volume_coarsegrid_u))

volume_finegrid_v = np.round(finegrid.volume_integral('v'), finegrid.sgn_digits)
volume_coarsegrid_v = np.round(coarsegrid.volume_integral('v'), coarsegrid.sgn_digits)
print('Volume integral v finegrid: ' + str(volume_finegrid_v))
print('Volume integral v coarsegrid: ' + str(volume_coarsegrid_v))

volume_finegrid_w = np.round(finegrid.volume_integral('w'), finegrid.sgn_digits)
volume_coarsegrid_w = np.round(coarsegrid.volume_integral('w'), coarsegrid.sgn_digits)
print('Volume integral w finegrid: ' + str(volume_finegrid_w))
print('Volume integral w coarsegrid: ' + str(volume_coarsegrid_w))

volume_finegrid_p = finegrid.volume_integral('p')
volume_coarsegrid_p = coarsegrid.volume_integral('p')
print('Volume integral p finegrid: ' + str(volume_finegrid_p))
print('Volume integral p coarsegrid: ' + str(volume_coarsegrid_p))

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
