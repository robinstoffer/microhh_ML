# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:54:17 2018

Test script to check convergence for downsampling done in test_training.py, preferably define arrays that are homogenous in the x,y-directions and only have a gradient in the z-direction.
NOTE: Run this script AFTER test_training.py

@author: stoff013
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

test = finegrid['output']['w']['variable'][finegrid.kgc_edge:finegrid.khend, finegrid.jgc:finegrid.jend, finegrid.igc:finegrid.iend]
test2 = np.mean(test,axis = -1)
test3 = np.mean(test2,axis = -1)
test4 = test3.copy()
z = finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend]
z = np.insert(z, 0, 0.0)
z = np.append(z, finegrid['grid']['zsize'])
test3 = np.insert(test3, 0, test3[0])
plt.figure()
plt.step(z, test3, label = 'fine')
z_grid = finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend]
plt.plot(z_grid, test4,'C0o')

test = coarsegrid['output']['w']['variable'][coarsegrid.kgc_edge:coarsegrid.khend, coarsegrid.jgc:coarsegrid.jend, coarsegrid.igc:coarsegrid.iend]
test2 = np.mean(test,axis = -1)
test3 = np.mean(test2,axis = -1)
test4 = test3.copy()
z = coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend]
z = np.insert(z, 0, 0.0)
z = np.append(z, coarsegrid['grid']['zsize'])
test3 = np.insert(test3, 0, test3[0])
plt.step(z, test3, label = 'coarse')
z_grid = coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend]
plt.plot(z_grid, test4,'C1o')

plt.xlabel('z')
plt.ylabel('value')
plt.title('w, coarsegrid shape: ' + str(test.shape))
plt.legend()


test = finegrid['output']['p']['variable'][finegrid.kgc_center:finegrid.kend, finegrid.jgc:finegrid.jend, finegrid.igc:finegrid.iend]
test2 = np.mean(test,axis = -1)
test3 = np.mean(test2,axis = -1)
#test3 = np.append(test3, test3[-1])
test4 = test3.copy()
test3 = np.insert(test3, 0, test3[0])
z = finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend]
plt.figure()
plt.step(z, test3, label = 'fine')
z_grid = finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend]
plt.plot(z_grid, test4,'C0o')

test = coarsegrid['output']['p']['variable'][coarsegrid.kgc_center:coarsegrid.kend, coarsegrid.jgc:coarsegrid.jend, coarsegrid.igc:coarsegrid.iend]
test2 = np.mean(test,axis = -1)
test3 = np.mean(test2,axis = -1)
test4 = test3.copy()
test3 = np.insert(test3, 0, test3[0])
#test3 = np.append(test3, test3[-1])
z = coarsegrid['grid']['zh'][coarsegrid.kgc_edge:coarsegrid.khend]
plt.step(z, test3, label = 'coarse')
z_grid = coarsegrid['grid']['z'][coarsegrid.kgc_center:coarsegrid.kend]
plt.plot(z_grid, test4,'C1o')

plt.xlabel('z')
plt.ylabel('value')
plt.title('p, coarsegrid shape: ' + str(test.shape))
plt.legend()