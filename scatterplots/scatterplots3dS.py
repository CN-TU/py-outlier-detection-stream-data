#!/usr/bin/env

import numpy as np
import pandas as pd

import sys
from glob import glob
import os
import re
import ntpath
import time
from scipy.io.arff import loadarff 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

inpath  = sys.argv[1]
jitter  = sys.argv[2]

def rand_jitter(arr):
    stdev = .005*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

# Create directory for plots
if os.path.exists('plots') == False:
    os.mkdir('plots') 


#for filename in glob(os.path.join(inpath, '*.arff')):

for dirpath, dirs, files in os.walk(inpath):  
    for filename in files: 
        fname = os.path.join(dirpath,filename)
        if fname.endswith('.arff'): 
            print("Data file (.arff)", fname,"\n")
            arffdata = loadarff(fname)
            df_data = pd.DataFrame(arffdata[0])

            df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))
            labels = df_data['class'].values
            labels = np.array(labels, dtype=int)
            data = df_data.values
            labels[labels>0] = -1
            labels = labels + 1
            data = np.array(data, dtype=np.float64)

            fig = plt.figure(1, figsize=(20, 4), dpi=100,)
            ax = fig.add_subplot(111, projection='3d')

            x_max, x_min = np.max(data[:,0]), np.min(data[:,0])
            y_max, y_min = np.max(data[:,1]), np.min(data[:,1])
            xr = np.arange(x_min, x_max, .1)
            yr = np.arange(y_min, y_max, .1)

            #Plane
            point1  = np.array([0.5, 0.5, 6000])
            point2  = np.array([0.5, 0.5, 7001])
            normal = np.array([0, 0, 1])
            d1 = -point1.dot(normal)
            d2 = -point2.dot(normal)
            xx, yy = np.meshgrid(xr, yr)
            z1 = (-normal[0] * xx - normal[1] * yy - d1) * 1. /normal[2]
            z2 = (-normal[0] * xx - normal[1] * yy - d2) * 1. /normal[2]
            ax.plot_surface(z1, xx, yy, alpha=0.1)
            ax.plot_surface(z2, xx, yy, alpha=0.1)


            alpha=np.arange(0,data.shape[0])
            if jitter=='1':
                ax.scatter(alpha, rand_jitter(data[:,0]), rand_jitter(data[:,1]), s=1, cmap='Blues', c=alpha)
            else:
                ax.scatter(alpha[labels==0], data[(labels==0),0], data[(labels==0),1], s=1, cmap='Blues', c=alpha[labels==0])
                ax.scatter(alpha[labels==1], data[(labels==1),0], data[(labels==1),1], s=1, cmap='Reds', c=alpha[labels==1])
            ax.view_init(azim=-75, elev=15)

            plt.tight_layout()
            plt.savefig('plots/'+filename+'_long'+'.png')

            plt.close()
            fig = plt.figure(1, figsize=(4, 4), dpi=100,)

            if jitter=='1':
                plt.scatter(rand_jitter(data[6000:7001,0]), rand_jitter(data[6000:7001,1]), s=1, cmap='Blues', c=alpha[6000:7001])
            else:
                datas=data[6000:7001,:]
                labelss=labels[6000:7001]
                alphas=alpha[6000:7001]
                plt.scatter(datas[(labelss==0),0], datas[(labelss==0),1], s=1, cmap='Blues', c=alphas[labelss==0])
                plt.colorbar(ticks=[np.min(alphas), np.max(alphas)])
                plt.scatter(datas[(labelss==1),0], datas[(labelss==1),1], s=1, cmap='Reds', c=alphas[labelss==1])
            plt.savefig('plots/'+filename+'_short'+'.png')

            plt.close()
