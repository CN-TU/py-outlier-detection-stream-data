#!/usr/bin/env python3

# checking package dependencies
from dependencies import check_pkgs
check_pkgs()

import numpy as np
import pandas as pd

import sys
import glob
import os
import subprocess
import pickle
import gzip

import src.pamsearch as pams
import src.data as dt

import pysdo
from src.indices import get_indices
from xStream import xStream
from dSalmon import outlier

import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

DEFAULT_RUNS = '''
datasets/example/ tests/example/ xstream 2000 0 0 1
datasets/example/ tests/example/ sdo 2000 0 0 1
datasets/example/ tests/example/ sdostream 2000 0 0 1
datasets/example/ tests/example/ swknn 2000 0 0 1
datasets/example/ tests/example/ swlof 2000 0 0 1
datasets/example/ tests/example/ rrct 2000 0 0 1
datasets/example/ tests/example/ rshash 2000 0 0 1
datasets/example/ tests/example/ loda 2000 0 0 1
'''


if len(sys.argv) < 2:
    print ('No arguments given. Running default configurations.')
    runs = [ [sys.executable, __file__] + params.split(' ') for params in DEFAULT_RUNS.split('\n') if params ]
    for run in runs:
        print (' '.join(run))
        os.makedirs(run[3], exist_ok=True)
        subprocess.call(run)
    quit()

assert len(sys.argv) >= 8, 'Wrong number of arguments'

inpath  = sys.argv[1]                   # folder with arff datasets to process
outpath = sys.argv[2]                   # folder to save .csv files with results
alg = sys.argv[3]                       # selected algorithm id: secod, sdo, sdostream, loda, rshash, swlof, xstream, rrct
timepam = int(sys.argv[4])              # time/memory parameter, observation/sliding window
jumble_data = (sys.argv[5] == '1')      # if '1' data is jumbled before processing (currently unused)
reverse_class = (sys.argv[6] == '1')    # enter '1' if outliers are labeled as '0' in the dataset
nrep = int(sys.argv[7])                 # number of repetition to average results for non-deterministic algorithms


np.random.seed(0)

alg_names =  {"xstream": "xStream", "secod": "SECOD", "sdo": "SDO", "sdostream": "SDOstream", "swknn": "SWKNN", "swlof": "SWLOF", "rrct": "RRCT", "rshash": "RSHash", "loda": "LODA"}
algn = alg_names[alg]

print("\nData folder:",inpath)
print("Result folder:",outpath)
print("Jumbling:",jumble_data)

res = []

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*.arff'))):
    data, labels, timestamps ,_ ,_ = dt.retrieve_data(filename=filename, idf=0, collection=False, non_predictors=3, reverse_class=reverse_class)

    print("Dataset size:", data.shape)
    training_len = 2000
    blocksize = 2000
    print("Training size:", training_len)
    print("Blocksize:", blocksize)

    best_param = pams.parameter_selection_2D(data, labels, timestamps, training_len, blocksize, alg, timepam)

    for idx in range(nrep):
        detector = pams.adjust_algorithm(alg, best_param, idx, timepam, blocksize)
        print("Run ",idx , " of ", nrep)

        scores = np.zeros(data.shape[0])

        ### TRAINING
        scores[0:training_len] = detector.fit_predict(data[0:training_len,:])
  
        ### TEST
        for i in range(training_len,data.shape[0],blocksize):
            if alg=='sdo':
                scores[i:(i+blocksize)] = detector.predict(data[i:(i+blocksize),:])
            else: 
                scores[i:(i+blocksize)] = detector.fit_predict(data[i:(i+blocksize),:])
            print(".", end='', flush=True)
            if ((i / blocksize) % 30 == 0):
                print("Datapoints: ", i)

        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0

        if np.any(np.isinf(scores)):
            print ('Warning: marking inf as non-outlier')
            scores[np.isinf(scores)] = 0

        top_outliers = 100
        scores_test = scores[training_len:]
        scores_test = -1/(1+scores_test)
        scores_test = (scores_test-np.min(scores_test))/(np.max(scores_test)-np.min(scores_test))
        ind = np.argsort(scores_test)[::-1]
        labels_test = np.zeros(data.shape[0]-training_len)
        labels_test[ind[:top_outliers]] = 1

        fig = plt.figure(1, figsize=(4, 4), dpi=100,)
        plt.scatter(data[training_len:,0],data[training_len:,1], s=1, cmap='coolwarm', c=scores_test)
        plt.colorbar(ticks=[np.min(scores_test), np.max(scores_test)])
        plt.title(algn+' scores', fontsize=14)
        plt.savefig(outpath+'ex'+alg+'.pdf')

        fig = plt.figure(2, figsize=(4, 4), dpi=100,)
        plt.scatter(data[training_len:,0],data[training_len:,1], s=1, cmap='Reds', c=labels_test)
        plt.colorbar(ticks=[np.min(labels_test), np.max(labels_test)])
        plt.title(algn + ' top ' + str(top_outliers) + ' outliers', fontsize=14)
        plt.savefig(outpath+'ex_bin_'+alg+'.pdf')

fig = plt.figure(0, figsize=(3.13, 4), dpi=100,)
ax = plt.gca()
plt.scatter(data[0:training_len,0],data[0:training_len,1], s=1, cmap='coolwarm', c=labels[0:training_len])
#plt.colorbar(ticks=[np.min(labels[0:training_len]), np.max(labels[0:training_len])])
ax.set_xlim((-0.05, 1.01))
ax.set_ylim((-0.05, 1.06))
plt.title('training dataset', fontsize=14)
plt.savefig(outpath+'ex_base.pdf')

fig = plt.figure(3, figsize=(3.13, 4), dpi=100,)
ax = plt.gca()
plt.scatter(data[training_len:,0],data[training_len:,1], s=1, cmap='Greys', c=0.5*np.ones([training_len]))
#plt.colorbar(ticks=[0,1])
plt.scatter(data[training_len:,0],data[training_len:,1], s=1, cmap='Greys', c='black')
ax.set_xlim((-0.05, 1.01))
ax.set_ylim((-0.05, 1.06))
plt.title('evaluation dataset', fontsize=14)
plt.savefig(outpath+'ex_eval.pdf')

