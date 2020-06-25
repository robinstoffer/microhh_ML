import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

data = nc.Dataset('qtpath.xy.nc', 'r')
L = 480.e3

time = data.variables['time'][:]
x = data.variables['x'][:]
y = data.variables['y'][:]

k = (2.*np.pi/L)*np.arange(x.size//2+1)
l = (2.*np.pi/L)*np.arange(y.size)
K = np.empty((l.size, k.size))

l_calc = (np.where(l<y.size/2, l, l-y.size))
K[:,:] = np.hypot(k[np.newaxis,:], l_calc[:,np.newaxis])

qtpath_spectra = np.zeros((time.size, l.size, k.size))

for n in range(time.size):
    qtpath = data.variables['qtpath'][n,:,:]
    qtpath_fft = np.fft.rfft2(qtpath) / (x.size*y.size)
    qtpath_fft[0,0] = 0.
    qtpath_fft_power = abs(qtpath_fft)**2
    qtpath_fft_power[:,1:-1] *= 2
    qtpath_spectra[n,:,:] = qtpath_fft_power

bin_seq = (2.*np.pi/L)*np.arange(1., k.size+1, 1.)
n, bin_edges = np.histogram(K.flatten(), bin_seq, weights=(qtpath_spectra[0,:,:]).flatten())
pdfy = np.zeros((time.size, n.size))
pdfx = 0.5*(bin_edges[0:-1] + bin_edges[:1])

print(2.*np.pi/pdfx[0])
print(2.*np.pi/bin_edges[0])

for i in range(time.size):
    n, bins = np.histogram(K.flatten(), bin_seq, weights=(qtpath_spectra[i,:,:]).flatten())
    pdfy[i,:] = n[:]

plt.figure()
for i in range(10, 41, 5):
    plt.semilogx(pdfx, pdfx*pdfy[i,:], label="{0}".format(time[i]/3600.))
plt.legend(frameon=False, loc=0)

plt.figure()
for i in range(0,10,1):
    plt.semilogy(time/3600., pdfy[:,i], label='{}'.format(2.*np.pi/pdfx[i]))
plt.legend(loc=0, frameon=False)
plt.show()
