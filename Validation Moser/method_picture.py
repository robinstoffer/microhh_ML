import netCDF4 as nc
import numpy as np
#from scipy.interpolate import RectBivariateSpline as rbs
import matplotlib as mpl
mpl.use('agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
import matplotlib.pyplot as plt

training_data = nc.Dataset('training_data.nc','r')
highres_data = nc.Dataset('u.nc','r')
highres_coord_data = nc.Dataset('v.nc','r') #Additional coordinates needed for plotting
delta = 500 #Half-channel width in [m]

u = highres_data.variables['u'][2,38,:,:]
xh = highres_data.variables['xh'][:]*delta
y = highres_data.variables['y'][:]*delta
z = highres_data.variables['z'][:]*delta
x = highres_coord_data.variables['x'][:]*delta
yh = highres_coord_data.variables['yh'][:]*delta
#yh = np.insert(yh,0,0)
x = np.insert(x,0,x[0]-xh[1])
yh = np.append(yh,np.pi*delta)
print(z[38])

uc = training_data.variables['uc'][2,2,:,:]
xhc =  training_data.variables['xhc'][:]*delta
xc = training_data.variables['xc'][:]*delta
yhc = training_data.variables['yhc'][:]*delta
yc = training_data.variables['yc'][:]*delta
zc = training_data.variables['zc'][:]*delta
yhc = np.insert(yhc,0,0)
yhc = np.append(yhc,np.pi*delta)
print(zc[2])

total_tau_xu = training_data.variables['total_tau_xu'][2,2,:,:]
res_tau_xu = training_data.variables['res_tau_xu'][2,2,:,:]
unres_tau_xu = training_data.variables['unres_tau_xu'][2,2,:,:]
#print(total_tau_xu.shape)
#print(xhc.shape)
#print(yc.shape)

lines = []
#itot = int(x[-1] / 10)
#jtot = int(y[-1] / 10)
for i in xhc:
    lx = np.array([i, i])
    ly = np.array([0, yhc[-1]])
    lines.append((lx, ly))
for j in yhc:
    lx = np.array([0, xc[-1]])
    ly = np.array([j, j])
    lines.append((lx, ly))

plt.figure()
#plt.subplot(121)
plt.pcolormesh(x, yh, u)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\rm {[m\ s^{-1}}]$',rotation=270,fontsize=20,labelpad=30)
for l in lines:
    plt.plot(*l, '#dddddd', linewidth=1.0)
plt.xlabel('x [m]',fontsize=20)
plt.ylabel('y [m]',fontsize=20)
#plt.xlim(0, 104)
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'a) $u$', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_high_res_u.png')


#plt.subplot(122)
plt.figure()
plt.pcolormesh(xc, yhc, uc)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[\rm {m\ s^{-1}}]$',rotation=270,fontsize=20,labelpad=30)
#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=2.0)
plt.xlabel('x [m]',fontsize=20)
plt.ylabel('y [m]',fontsize=20)
plt.xlim(0, xc[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'b) $u_c$', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_low_res_u.png')

plt.figure()
plt.pcolormesh(xc, yhc, total_tau_xu)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[\rm {m^{2}\ s^{-2}}]$',rotation=270,fontsize=20,labelpad=30)
#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=2.0)
plt.xlabel('x [m]',fontsize=20)
plt.ylabel('y [m]',fontsize=20)
plt.xlim(0, xc[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'c)$\tau_{uu,tot}$', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_total_transport.png')

plt.figure()
plt.pcolormesh(xc, yhc, res_tau_xu)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[\rm {m^{2}\ s^{-2}}]$',rotation=270,fontsize=20,labelpad=30)
#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=2.0)
plt.xlabel('x [m]',fontsize=20)
plt.ylabel('y [m]',fontsize=20)
plt.xlim(0, xc[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'd)$\tau_{uu,res}$', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_res_transport.png')

plt.figure()
plt.pcolormesh(xc, yhc, unres_tau_xu)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[\rm {m^{2}\ s^{-2}}]$',rotation=270,fontsize=20,labelpad=30)

#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=2.0)
plt.xlabel('x [m]',fontsize=20)
plt.ylabel('y [m]',fontsize=20)
plt.xlim(0, xc[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'e)$\tau_{uu,unres}$', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_unres_transport.png')

training_data.close()
