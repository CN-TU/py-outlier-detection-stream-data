#!/usr/bin/env python3

# checking package dependencies
from dependencies import check_pkgs
check_pkgs()

import numpy as np
import pandas as pd

import sys
import glob
import subprocess
import os
import re
import ntpath
from scipy.io.arff import loadarff 

import src.pamsearch as pams
import src.data as dt

from dSalmon.trees import MTree

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

res = []

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*.arff'))):
    data, labels, timestamps ,_ ,_ = dt.retrieve_data(filename=filename, idf=idf, collection=False, non_predictors=3, reverse_class=reverse_class)
    filename_short = ntpath.basename(filename)

    lda = LDA(n_components=1)
    lda_data = lda.fit_transform(data, labels)

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

