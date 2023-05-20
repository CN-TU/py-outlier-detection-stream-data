#!/usr/bin/env python3

# checking package dependencies
from dependencies import check_pkgs
check_pkgs()

import numpy as np
import pandas as pd

import sys
import glob
import os
import time
import subprocess
import pickle
import gzip

import src.pamsearch as pams
import src.data as dt

import pysdo
from src.indices import get_indices
from xStream import xStream
from dSalmon import outlier
from src.streamECOD import sECOD

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# if no parameters are provided, the script will be run with these configurations
DEFAULT_RUNS = '''
./datasets/synthetic/base/arff tests/base/ sdo 500 0 1 10
./datasets/synthetic/nonstat/arff tests/nonstat/ sdo 500 0 1 10
./datasets/synthetic/overlap/arff tests/overlap/ sdo 500 0 1 10
./datasets/synthetic/close/arff tests/close/ sdo 500 0 1 10
./datasets/synthetic/dens-diff/arff tests/dens-diff/ sdo 500 0 1 10
./datasets/synthetic/h-out/arff tests/h-out/ sdo 500 0 1 10
./datasets/synthetic/m-out/arff tests/m-out/ sdo 500 0 1 10
./datasets/synthetic/moving/arff tests/moving/ sdo 500 0 1 10
./datasets/synthetic/sequential/arff tests/sequential/ sdo 500 0 1 10
./datasets/synthetic/base/arff tests/base/ sdostream 500 0 1 10
./datasets/synthetic/nonstat/arff tests/nonstat/ sdostream 500 0 1 10
./datasets/synthetic/overlap/arff tests/overlap/ sdostream 500 0 1 10
./datasets/synthetic/close/arff tests/close/ sdostream 500 0 1 10
./datasets/synthetic/dens-diff/arff tests/dens-diff/ sdostream 500 0 1 10
./datasets/synthetic/h-out/arff tests/h-out/ sdostream 500 0 1 10
./datasets/synthetic/m-out/arff tests/m-out/ sdostream 500 0 1 10
./datasets/synthetic/moving/arff tests/moving/ sdostream 500 0 1 10
./datasets/synthetic/sequential/arff tests/sequential/ sdostream 500 0 1 10
./datasets/synthetic/base/arff tests/base/ swknn 500 0 1 1
./datasets/synthetic/nonstat/arff tests/nonstat/ swknn 500 0 1 1
./datasets/synthetic/overlap/arff tests/overlap/ swknn 500 0 1 1
./datasets/synthetic/close/arff tests/close/ swknn 500 0 1 1
./datasets/synthetic/dens-diff/arff tests/dens-diff/ swknn 500 0 1 1
./datasets/synthetic/h-out/arff tests/h-out/ swknn 500 0 1 1
./datasets/synthetic/m-out/arff tests/m-out/ swknn 500 0 1 1
./datasets/synthetic/moving/arff tests/moving/ swknn 500 0 1 1
./datasets/synthetic/sequential/arff tests/sequential/ swknn 500 0 1 1
./datasets/synthetic/base/arff tests/base/ rrct 500 0 1 10
./datasets/synthetic/nonstat/arff tests/nonstat/ rrct 500 0 1 10
./datasets/synthetic/overlap/arff tests/overlap/ rrct 500 0 1 10
./datasets/synthetic/close/arff tests/close/ rrct 500 0 1 10
./datasets/synthetic/dens-diff/arff tests/dens-diff/ rrct 500 0 1 10
./datasets/synthetic/h-out/arff tests/h-out/ rrct 500 0 1 10
./datasets/synthetic/m-out/arff tests/m-out/ rrct 500 0 1 10
./datasets/synthetic/moving/arff tests/moving/ rrct 500 0 1 10
./datasets/synthetic/sequential/arff tests/sequential/ rrct 500 0 1 10
./datasets/synthetic/base/arff tests/base/ xstream 500 0 1 10
./datasets/synthetic/nonstat/arff tests/nonstat/ xstream 500 0 1 10
./datasets/synthetic/overlap/arff tests/overlap/ xstream 500 0 1 10
./datasets/synthetic/close/arff tests/close/ xstream 500 0 1 10
./datasets/synthetic/dens-diff/arff tests/dens-diff/ xstream 500 0 1 10
./datasets/synthetic/h-out/arff tests/h-out/ xstream 500 0 1 10
./datasets/synthetic/m-out/arff tests/m-out/ xstream 500 0 1 10
./datasets/synthetic/moving/arff tests/moving/ xstream 500 0 1 10
./datasets/synthetic/sequential/arff tests/sequential/ xstream 500 0 1 10
./datasets/synthetic/base/arff tests/base/ swlof 500 0 1 1
./datasets/synthetic/nonstat/arff tests/nonstat/ swlof 500 0 1 1
./datasets/synthetic/overlap/arff tests/overlap/ swlof 500 0 1 1
./datasets/synthetic/close/arff tests/close/ swlof 500 0 1 1
./datasets/synthetic/dens-diff/arff tests/dens-diff/ swlof 500 0 1 1
./datasets/synthetic/h-out/arff tests/h-out/ swlof 500 0 1 1
./datasets/synthetic/m-out/arff tests/m-out/ swlof 500 0 1 1
./datasets/synthetic/moving/arff tests/moving/ swlof 500 0 1 1
./datasets/synthetic/sequential/arff tests/sequential/ swlof 500 0 1 1
./datasets/synthetic/base/arff tests/base/ rshash 500 0 1 10
./datasets/synthetic/nonstat/arff tests/nonstat/ rshash 500 0 1 10
./datasets/synthetic/overlap/arff tests/overlap/ rshash 500 0 1 10
./datasets/synthetic/close/arff tests/close/ rshash 500 0 1 10
./datasets/synthetic/dens-diff/arff tests/dens-diff/ rshash 500 0 1 10
./datasets/synthetic/h-out/arff tests/h-out/ rshash 500 0 1 10
./datasets/synthetic/m-out/arff tests/m-out/ rshash 500 0 1 10
./datasets/synthetic/moving/arff tests/moving/ rshash 500 0 1 10
./datasets/synthetic/sequential/arff tests/sequential/ rshash 500 0 1 10
./datasets/synthetic/base/arff tests/base/ loda 500 0 1 10
./datasets/synthetic/nonstat/arff tests/nonstat/ loda 500 0 1 10
./datasets/synthetic/overlap/arff tests/overlap/ loda 500 0 1 10
./datasets/synthetic/close/arff tests/close/ loda 500 0 1 10
./datasets/synthetic/dens-diff/arff tests/dens-diff/ loda 500 0 1 10
./datasets/synthetic/h-out/arff tests/h-out/ loda 500 0 1 10
./datasets/synthetic/m-out/arff tests/m-out/ loda 500 0 1 10
./datasets/synthetic/moving/arff tests/moving/ loda 500 0 1 10
./datasets/synthetic/sequential/arff tests/sequential/ loda 500 0 1 10
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

print("\nData folder:",inpath)
print("Result folder:",outpath)
print("Jumbling:",jumble_data)

os.makedirs(outpath, exist_ok=True)

res = []
training_size_factor = 0.2
block_size_factor = 0.01
save_scores = False

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*.arff'))):

    data, labels, timestamps, m, dataset_idx = dt.retrieve_data(filename=filename, idf=0, collection=True, non_predictors=3, reverse_class=reverse_class)

    print("Dataset size:", data.shape)
    training_len=int(training_size_factor*data.shape[0])
    blocksize = int(block_size_factor*data.shape[0])
    print("Training size:", training_len)
    print("Blocksize:", blocksize)

    # defining ranges for the parameter search and finding best parameters
    best_param = pams.parameter_selection(data, labels, timestamps, training_len, blocksize, alg, timepam)

    # initializing selected algorithm with best parameters
    for idx in range(nrep):
        detector = pams.adjust_algorithm(alg, best_param, idx, timepam, blocksize)
        print("Run ",idx , " of ", nrep)

        scores = np.zeros(data.shape[0])

        ### TRAINING
        start_tr = time.time()
        scores[:training_len] = detector.fit_predict(data[:training_len,:])
        end_tr = time.time()
        elapsed_tr = end_tr - start_tr

        ### TEST
        start_ts = time.time()
        for i in range(training_len,data.shape[0],blocksize):
            if alg=='sdo':
                scores[i:(i+blocksize)] = detector.predict(data[i:(i+blocksize),:])
            else: 
                scores[i:(i+blocksize)] = detector.fit_predict(data[i:(i+blocksize),:])
            print(".", end='', flush=True)
            if ((i / blocksize) % 30 == 0):
                print("Datapoints: ", i)
        end_ts = time.time()
        elapsed_ts = end_ts - start_ts

        # saving scores
        if save_scores:
            scores_file = '%sSCORES_%s_%d_T%d_%d.pickle.gz' % (outpath, alg, int(dataset_idx), int(timepam), int(idx))
            with gzip.open(scores_file, 'wb') as f:
                pickle.dump(scores, f)

        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0

        perf = get_indices(labels[training_len:], pams.transform_scores(scores[training_len:]))

        new_row = "dataset: %d, idx: %d, P@n: %.2f, aP@n: %.2f, AP: %.2f, aAP: %.2f, MF1: %.2f, aMF1: %.2f, ROC: %.2f, Ttr: %.3f, Tts: %.3f" % (idf,idx,perf['Patn'],perf['adj_Patn'],perf['ap'],perf['adj_ap'],perf['maxf1'],perf['adj_maxf1'],perf['auc'],elapsed_tr,elapsed_ts)
        res.append(new_row)
        print("\n",new_row)

# saving performances
df = pd.DataFrame(data=res)
outfile = outpath + "SUMMARY_" + alg + "_T" + str(timepam) + ".txt"
print("Summary file (txt):",outfile,"\n")
df.to_csv(outfile, sep=',', header=None)

