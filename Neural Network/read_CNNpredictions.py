import numpy as np
import netCDF4 as nc
#import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
#from matplotlib import rcParams
mpl.rcParams.update({'figure.autolayout':True})
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='microhh_ML')
parser.add_argument('--prediction_file', default=None, \
        help='NetCDF file that contains the predictions')
args = parser.parse_args()

#Fetch predictions made by CNN
a=nc.Dataset(args.prediction_file,'r')

#Read variables
preds_values = a['preds_values'][:]
preds_values_random = a['preds_values_random'][:]
lbls_values = a['lbls_values'][:]
residuals = a['residuals'][:]
residuals_random = a['residuals_random'][:]

#Make scatterplots of predictions versus labels
plt.figure()
plt.scatter(lbls_values,preds_values,s=6,marker='o')
plt.gca().axis('Equal')
#plt.xlim(min(lbls_values)*0.8,max(lbls_values)*1.2)
#plt.ylim(min(lbls_values)*0.8,max(lbls_values)*1.2)
#plt.xlim(-0.0010,0.0010)
#plt.ylim(-0.0010,0.0010)
plt.xlim(-25.0,25.0)
plt.ylim(-25.0,25.0)
plt.plot(plt.gca().get_xlim(),plt.gca().get_ylim(),'b--')
#plt.gca().set_aspect('equal',adjustable='box')
plt.xlabel("Labels",fontsize = 20)
plt.ylabel("Predictions CNN",fontsize = 20) 
plt.axhline(c='black')
plt.axvline(c='black')
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Scatter_CNNpredictions_vs_labels.png")
plt.close()

plt.figure()
plt.scatter(lbls_values,preds_values_random,s=6,marker='o')
plt.gca().axis('Equal')
#plt.xlim(min(lbls_values)*0.8,max(lbls_values)*1.2)
#plt.ylim(min(lbls_values)*0.8,max(lbls_values)*1.2)
#plt.xlim(-0.0010,0.0010)
#plt.ylim(-0.0010,0.0010)
plt.xlim(-25.0,25.0)
plt.ylim(-25.0,25.0)
plt.plot(plt.gca().get_xlim(),plt.gca().get_ylim(),'b--')
#plt.gca().set_aspect('equal',adjustable='box')
plt.xlabel("Labels",fontsize = 20)
plt.ylabel("Predictions random",fontsize = 20)
plt.axhline(c='black')
plt.axvline(c='black')
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Scatter_randompredictions_vs_labels.png")
plt.close()

plt.figure()
plt.hist(residuals,bins=20)
#plt.xlim(0,0.001)
plt.xlim(0,25.0)
#plt.ylim(-0.01,0.01)
plt.xlabel("Magnitude",fontsize = 20)
plt.ylabel("Count",fontsize = 20)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Hist_CNNpredictions_vs_labels.png")
plt.close()

plt.figure()
plt.hist(residuals_random,bins=20)
#plt.xlim(0,0.001)
plt.xlim(0,25.0)
#plt.ylim(-0.01,0.01)
plt.xlabel("Magnitude",fontsize = 20)
plt.ylabel("Count",fontsize = 20)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Hist_randompredictions_vs_labels.png")
plt.close()

plt.figure()
plt.hist(preds_values,bins=20)
#plt.xlim(-0.001,0.001)
plt.xlim(-25.0,25.0)
#plt.ylim(-0.01,0.01)
plt.xlabel("Magnitude",fontsize = 20)
plt.ylabel("Count",fontsize = 20)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Hist_CNNpredictions.png")
plt.close()

plt.figure()
plt.hist(lbls_values,bins=20)
#plt.xlim(-0.001,0.001)
plt.xlim(-25.0,25.0)
#plt.ylim(-0.01,0.01)
plt.xlabel("Magnitude",fontsize = 20)
plt.ylabel("Count",fontsize = 20)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Hist_labels.png")
plt.close()

plt.figure()
plt.hist(preds_values_random,bins=20)
#plt.xlim(-0.001,0.001)
plt.xlim(-25.0,25.0)
#plt.ylim(-0.01,0.01)
plt.xlabel("Magnitude",fontsize = 20)
plt.ylabel("Count",fontsize = 20)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Hist_randompredictions.png")
plt.close()
