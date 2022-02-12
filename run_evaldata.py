#!/usr/bin/env python3

# install dependencies to a local subdirectory
import dependencies
dependencies.assert_pkgs({
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
    'xStream': 'contrib/xStream.tar.bz2',
    'pysdo': 'https://github.com/CN-TU/pysdo/archive/master.zip',
    'dSalmon': 'git+https://github.com/CN-TU/dSalmon'
})

import numpy as np
import pandas as pd

import sys
import glob
import os
import re
import ntpath
import time
import subprocess
import pickle
import gzip
from scipy.io.arff import loadarff 

import pysdo
import indices
from dSalmon.trees import MTree

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from KDEpy import FFTKDE
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# if no parameters are provided, the script will be run with these configurations
DEFAULT_RUNS = '''
./datasets/synthetic/base/arff 1
./datasets/synthetic/nonstat/arff 1
./datasets/synthetic/overlap/arff 1
./datasets/synthetic/close/arff 1
./datasets/synthetic/dens-diff/arff 1
./datasets/synthetic/h-out/arff 1
./datasets/synthetic/m-out/arff 1
./datasets/synthetic/moving/arff 1
./datasets/synthetic/sequential/arff 1
./datasets/real/shuttle 0 
./datasets/real/swan 0
./datasets/real/yahoo 0
./datasets/real/optout 0 
'''


if len(sys.argv) < 2:
    print ('No arguments given. Running default configurations.')
    runs = [ [sys.executable, __file__] + params.split(' ') for params in DEFAULT_RUNS.split('\n') if params ]
    for run in runs:
        print (' '.join(run))
        subprocess.call(run)
    quit()

inpath  = sys.argv[1]
reverse_class = (sys.argv[2] == '1')

np.random.seed(0)

#print("\nData folder:",inpath)

res = []

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*.arff'))):
    #print("\nData file (.arff)", filename)
    #print("Data file index: ", idf)
    m = re.match('.*_([0-9]+).arff', filename)
    dataset_idx = m.group(1) if m else 0
    filename_short = ntpath.basename(filename)
    arffdata = loadarff(filename)
    df_data = pd.DataFrame(arffdata[0])

    if(df_data['class'].dtypes == 'object'):
        df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))
    labels = df_data['class'].values
    labels = np.array(labels, dtype=int)
    data = df_data.values
    data = MinMaxScaler().fit_transform(data)
    timestamps = np.arange(0,data.shape[0])
    timestamps = timestamps[:,None]

    if reverse_class:
        labels[labels>0] = -1
        labels = labels + 1
    data = np.array(data[:,0:-1], dtype=np.float64)

    #print("Dataset size:", data.shape)

    lda = LDA(n_components=1)
    lda_data = lda.fit_transform(data, labels)

    #x0, y0 = FFTKDE(bw='silverman').fit(lda_data[labels==0]).evaluate(200)
    #x1, y1 = FFTKDE(bw='silverman').fit(lda_data[labels==1]).evaluate(200)

    #plt.fill_between(x0, y0, color="lightpink", alpha=0.5, label='inliers')
    #plt.fill_between(x1, y1, color="skyblue", alpha=0.5, label='outliers')
    #plt.legend()

    bins = np.linspace(min(lda_data), max(lda_data), 50).flatten().tolist()
    n0, b0, p0 = plt.hist(lda_data[labels==0], bins, density=True, alpha=0.75)
    n1, b1, p1 = plt.hist(lda_data[labels==1], bins, density=True, alpha=0.75)

    tree = MTree()
    inserted_keys = tree.insert(data)
    din,dout = [],[]

    for idx, x in enumerate(data):
        neighbor_keys, neighbor_distances, _ = tree.knn(tree.ix[idx], k=3)
        if labels[idx]==0:
            din.append(neighbor_distances[-1])
        else:
            dout.append(neighbor_distances[-1])

    aux = (np.quantile(dout, 0.75)-np.quantile(dout, 0.25))/(np.quantile(din, 0.75)-np.quantile(din, 0.25))
    rho = 1/np.log2(1+aux)

    phi = sum(np.minimum(n0,n1))/sum(np.maximum(n0,n1))
    print(filename_short, ", idx:",idf, ", shape:",data.shape, ", #outliers:",sum(labels), ", phi:", phi, ", rho:", rho)
    #plt.show()

