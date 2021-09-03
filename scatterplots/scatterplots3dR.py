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

from sklearn.preprocessing import StandardScaler,RobustScaler

inpath  = sys.argv[1]
jitter  = sys.argv[2]

def rand_jitter(arr):
    stdev = .005*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev
  
# Create directory for plots
if os.path.exists('plots') == False:
    os.mkdir('plots') 

# Selected space features
feat_pairs = [(0,1),(1,2),(2,3),(0,2)]

for dirpath, dirs, files in os.walk(inpath):  
    for filename in files: 
        fname = os.path.join(dirpath,filename)
        if fname.endswith('.arff'): 

            # RETRIEVE DATASET            
            print("Data file (.arff)", fname,"\n")
            arffdata = loadarff(fname)
            df_data = pd.DataFrame(arffdata[0])

            #df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))
            labels = df_data['class'].values
            labels = np.array(labels, dtype=int)
            data = df_data.values
            #labels[labels>0] = -1
            #labels = labels + 1
            data = np.array(data, dtype=np.float64)

            scaler = StandardScaler()
            #scaler = RobustScaler()
            data = scaler.fit_transform(data)

            for f0,f1 in feat_pairs:
                # DRAW 3D SCATTER PLOT
                fig = plt.figure(1, figsize=(20, 4), dpi=100,)
                ax = fig.add_subplot(111, projection='3d')

                x_max, x_min = np.max(data[:,f0]), np.min(data[:,f0])
                y_max, y_min = np.max(data[:,f1]), np.min(data[:,f1])
                
                xr = np.arange(x_min, x_max, (x_max-x_min)/100) 
                yr = np.arange(y_min, y_max, (y_max-y_min)/100) 

                #Plane
                ini = int(0.4*data.shape[0]/1000)*1000
                fin = int(0.5*data.shape[0]/1000)*1000
                point1  = np.array([0.5, 0.5,  ini]) 
                point2  = np.array([0.5, 0.5, fin]) 
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
                    ax.scatter(alpha, rand_jitter(data[:,f0]), rand_jitter(data[:,f1]), s=1, cmap='Blues', c=alpha)
                else:
                    ax.scatter(alpha[labels==0], data[(labels==0),f0], data[(labels==0),f1], s=1, cmap='Blues', c=alpha[labels==0])
                    ax.scatter(alpha[labels==1], data[(labels==1),f0], data[(labels==1),f1], s=1, cmap='Reds', c=alpha[labels==1])
                ax.view_init(azim=-75, elev=15)

                plt.tight_layout()
                plt.savefig('plots/'+filename+str(f0)+str(f1)+'_long'+'.png')

                plt.close()

                fig = plt.figure(1, figsize=(4, 4), dpi=100,)

                if jitter=='1':
                    plt.scatter(rand_jitter(data[ini:fin+1,f0]), rand_jitter(data[ini:fin+1,f1]), s=1, cmap='Blues', c=alpha[ini:fin+1])
                else:
                    datas=data[ini:fin+1,:]
                    labelss=labels[ini:fin+1]
                    alphas=alpha[ini:fin+1]
                    plt.scatter(datas[(labelss==0),f0], datas[(labelss==0),f1], s=1, cmap='Blues', c=alphas[labelss==0])
                    plt.colorbar(ticks=[ini, fin])
                    plt.scatter(datas[(labelss==1),f0], datas[(labelss==1),f1], s=1, cmap='Reds', c=alphas[labelss==1])
                plt.savefig('plots/'+filename+str(f0)+str(f1)+'_short'+'.png')

                plt.close()
