import netCDF4 as nc
import numpy as np
#from scipy.interpolate import RectBivariateSpline as rbs
import matplotlib as mpl
mpl.use('agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
import matplotlib.pyplot as plt

training_data = nc.Dataset('/projects/1/flowsim/simulation1/lesscoarse/training_data.nc','r')
highres_data = nc.Dataset('/projects/1/flowsim/simulation1/u.nc','r')
highres_coord_data = nc.Dataset('/projects/1/flowsim/simulation1/v.nc','r')
delta = 500 #Half-channel width in [m], 500 m is representative for a realistic ABL
#delta = 1 NOTE: half-channel width for Moser case
utau_ref = 0.2 #Friction velocity in [m/s], 0.2 m/s is representative for a realistic ABL. This is applied to undo the normalisation and rescale the values to realistic numbers.
utau_ref_moser = training_data['utau_ref'][:] #NOTE: friction velocity for Moser case, needed to normalize high-res velocity fields. The low-resolution fields were already normalized.
#utau_ref_moser = 0.0059

u = (highres_data.variables['u'][0,38,:,:] / utau_ref_moser) * utau_ref
xh = highres_data.variables['xh'][:]*delta
y = highres_data.variables['y'][:]*delta
z = highres_data.variables['z'][:]*delta
x = highres_coord_data.variables['x'][:]*delta
yh = highres_coord_data.variables['yh'][:]*delta
#yh = np.insert(yh,0,0)
x = np.insert(x,0,x[0]-xh[1])
yh = np.append(yh,np.pi*delta)
print(z[38])

uc = training_data.variables['uc'][0,training_data.variables['kgc_center'][:]:training_data.variables['kend'][:],training_data.variables['jgc'][:]:training_data.variables['jend'][:],training_data.variables['igc'][:]:training_data.variables['ihend'][:]]
uc = uc[2,:,:] * utau_ref
#xhc =  training_data.variables['xhc'][training_data.variables['igc'][:]:training_data.variables['ihend'][:]]*delta
xc = training_data.variables['xgc'][training_data.variables['igc'][:]-1:training_data.variables['iend'][:]+1]*delta
#yhc = training_data.variables['yghc'][training_data.variables['jgc'][:]:training_data.variables['jhend'][:]]*delta
#yc = training_data.variables['ygc'][training_data.variables['jgc'][:]:training_data.variables['jend'][:]+1]*delta
zgc = training_data.variables['zgc'][training_data.variables['kgc_center'][:]:training_data.variables['kend'][:]]*delta
xhc =  training_data.variables['xhc'][:]*delta
#xc  = training_data.variables['xc'][:]*delta
yhc = training_data.variables['yhc'][:]*delta
yc  = training_data.variables['yc'][:]*delta
zc  = training_data.variables['zc'][:]*delta
#yhc = np.append(yhc,np.pi*delta)
print(zgc[2])

#NOTE: only turbulence contribution considered, not the total transport as the viscous contribution should not be added.
total_tau_xu = training_data.variables['total_tau_xu_turb'][0,2,:,:] * (utau_ref ** 2)
res_tau_xu = training_data.variables['res_tau_xu_turb'][0,2,:,:] * (utau_ref ** 2) 
unres_tau_xu = training_data.variables['unres_tau_xu_turb'][0,2,:,:] * (utau_ref ** 2)
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
plt.pcolormesh(x, yh, u, vmin=1.0, vmax=4.0)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\rm {[m\ s^{-1}}]$',rotation=270,fontsize=20,labelpad=30)
#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=1.0)
plt.xlabel('x [m]',fontsize=20)
plt.ylabel('y [m]',fontsize=20)
plt.xlim(0, x[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
#plt.title(r'a) $u$', loc='center', fontsize=20)
plt.title(r'a) high-resolution flow field', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_high_res_u.png', dpi = 400)


#plt.subplot(122)
plt.figure()
plt.pcolormesh(xc, yhc, uc, vmin=1.0, vmax=4.0)
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
#plt.title(r'b) $u_c$', loc='center', fontsize=20)
plt.title(r'b) coarse-grained flow field', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_low_res_u.png', dpi = 400)

plt.figure()
plt.pcolormesh(xhc, yhc, total_tau_xu, vmin=3.0, vmax=16.0)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[\rm {m^{2}\ s^{-2}}]$',rotation=270,fontsize=20,labelpad=30)
#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=2.0)
plt.xlabel('x [m]',fontsize=20)
plt.ylabel('y [m]',fontsize=20)
plt.xlim(0, xhc[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
#plt.title(r'c)$\tau_{uu,tot}$', loc='center', fontsize=20)
plt.title(r'c) Total turbulent transport', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_total_transport.png', dpi = 400)

plt.figure()
plt.pcolormesh(xhc, yhc, res_tau_xu, vmin=3.0, vmax=16.0)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[\rm {m^{2}\ s^{-2}}]$',rotation=270,fontsize=20,labelpad=30)
#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=2.0)
plt.xlabel('x [m]',fontsize=20)
plt.ylabel('y [m]',fontsize=20)
plt.xlim(0, xhc[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
#plt.title(r'd)$\tau_{uu,res}$', loc='center', fontsize=20)
plt.title(r'd) Resolved turbulent transport', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_res_transport.png', dpi = 400)

plt.figure()
plt.pcolormesh(xhc, yhc, unres_tau_xu, vmin=-0.5, vmax=0.5)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[\rm {m^{2}\ s^{-2}}]$',rotation=270,fontsize=20,labelpad=30)

#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=2.0)
plt.xlabel('x [m]',fontsize=20)
plt.ylabel('y [m]',fontsize=20)
plt.xlim(0, xhc[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
#plt.title(r'e)$\tau_{uu,unres}$', loc='center', fontsize=20)
plt.title(r'e) Unresolved/subgrid turbulent transport', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_unres_transport.png', dpi = 400)

training_data.close()
