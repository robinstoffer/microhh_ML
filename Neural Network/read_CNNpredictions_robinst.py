import numpy as np
import netCDF4 as nc
#import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
#from matplotlib import rcParams
mpl.rcParams.update({'figure.autolayout':True})
import matplotlib.pyplot as plt

#Fetch predictions made by CNN
a=nc.Dataset('/home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_CNN1/CNN_predictions.nc','r')

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
plt.xlim(-0.01,0.01)
plt.ylim(-0.01,0.01)
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
plt.xlim(-0.01,0.01)
plt.ylim(-0.01,0.01)
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
plt.xlim(-0.01,0.01)
#plt.ylim(-0.01,0.01)
plt.xlabel("Magnitude",fontsize = 20)
plt.ylabel("Count",fontsize = 20)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Hist_CNNpredictions_vs_labels.png")
plt.close()

plt.figure()
plt.hist(residuals_random,bins=20)
plt.xlim(-0.01,0.01)
#plt.ylim(-0.01,0.01)
plt.xlabel("Magnitude",fontsize = 20)
plt.ylabel("Count",fontsize = 20)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Hist_randompredictions_vs_labels.png")
plt.close()

plt.figure()
plt.hist(preds_values,bins=20)
plt.xlim(-0.01,0.01)
#plt.ylim(-0.01,0.01)
plt.xlabel("Magnitude",fontsize = 20)
plt.ylabel("Count",fontsize = 20)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Hist_CNNpredictions.png")
plt.close()

plt.figure()
plt.hist(lbls_values,bins=20)
plt.xlim(-0.01,0.01)
#plt.ylim(-0.01,0.01)
plt.xlabel("Magnitude",fontsize = 20)
plt.ylabel("Count",fontsize = 20)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Hist_labels.png")
plt.close()

plt.figure()
plt.hist(preds_values_random,bins=20)
plt.xlim(-0.01,0.01)
#plt.ylim(-0.01,0.01)
plt.xlabel("Magnitude",fontsize = 20)
plt.ylabel("Count",fontsize = 20)
plt.xticks(fontsize = 16, rotation = 90)
plt.yticks(fontsize = 16, rotation = 0)
plt.savefig("Hist_randompredictions.png")
plt.close()
