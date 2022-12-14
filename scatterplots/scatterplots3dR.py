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
from matplotlib import gridspec

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

            dsname = filename.split('.')[0]
            if dsname == 'optout':
                xtlab = ['0','0','0.5M','1M','1.5M','2M']
            elif dsname == 'shuttle':
                xtlab = ['0','0','10k','20k','30k','40k','50k']
            elif dsname == 'yahoo':
                xtlab = ['0','0','20k','40k','60k','80k']
            elif dsname == 'swan':
                xtlab = ['0','0','50k','100k','150k','200k','250k','300k']



            for f0,f1 in feat_pairs:
                # DRAW 3D SCATTER PLOT

                x_max, x_min = np.max(data[:,f0]), np.min(data[:,f0])
                y_max, y_min = np.max(data[:,f1]), np.min(data[:,f1])
                
                xr = np.arange(x_min, x_max, (x_max-x_min)/100) 
                yr = np.arange(y_min, y_max, (y_max-y_min)/100) 

                fig = plt.figure(figsize=(15,4))
                gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.09, hspace=0.0, top=1.3, bottom=-0.2, left=-0.05, right=0.98) 

                alpha=np.arange(0,data.shape[0])

                for i in range(3):
                    ax = plt.subplot(gs[i],projection='3d')

                    ax.scatter3D(alpha[labels==0], data[(labels==0),f0], data[(labels==0),f1], s=1, c='blue', rasterized=True, alpha=0.5)
                    ax.scatter3D(alpha[labels==1], data[(labels==1),f0], data[(labels==1),f1], s=1, c='red', rasterized=True, alpha=0.5)

                    ax.set_xlabel("time ", fontsize=14, linespacing=5)
                    ax.set_ylabel("f"+str(f0), fontsize=14)
                    ax.set_zlabel("f"+str(f1), fontsize=14)

                    #ax.set_ylim(0,1)
                    #ax.set_zlim(0,1)
                    ax.view_init(azim=-90+40*i, elev=10)
                    ax.set_xticklabels(xtlab)

                plt.savefig('plots/'+filename+str(f0)+str(f1)+'_long'+'.pdf')

                plt.close()


