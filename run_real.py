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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# if no parameters are provided, the script will be run with these configurations
DEFAULT_RUNS = '''
./datasets/real/shuttle tests/real/shuttle/ sdo 100 0 0 10
./datasets/real/shuttle tests/real/shuttle/ sdo 500 0 0 10
./datasets/real/shuttle tests/real/shuttle/ sdo 1000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ sdo 5000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ sdo 10000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ sdostream 100 0 0 10
./datasets/real/shuttle tests/real/shuttle/ sdostream 500 0 0 10
./datasets/real/shuttle tests/real/shuttle/ sdostream 1000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ sdostream 5000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ sdostream 10000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ swknn 100 0 0 1
./datasets/real/shuttle tests/real/shuttle/ swknn 500 0 0 1
./datasets/real/shuttle tests/real/shuttle/ swknn 1000 0 0 1
./datasets/real/shuttle tests/real/shuttle/ swknn 5000 0 0 1
./datasets/real/shuttle tests/real/shuttle/ swknn 10000 0 0 1
./datasets/real/shuttle tests/real/shuttle/ rrct 100 0 0 10
./datasets/real/shuttle tests/real/shuttle/ rrct 500 0 0 10
./datasets/real/shuttle tests/real/shuttle/ rrct 1000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ rrct 5000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ rrct 10000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ xstream 100 0 0 10
./datasets/real/shuttle tests/real/shuttle/ xstream 500 0 0 10
./datasets/real/shuttle tests/real/shuttle/ xstream 1000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ xstream 5000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ xstream 10000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ swlof 100 0 0 1
./datasets/real/shuttle tests/real/shuttle/ swlof 500 0 0 1
./datasets/real/shuttle tests/real/shuttle/ swlof 1000 0 0 1
./datasets/real/shuttle tests/real/shuttle/ swlof 5000 0 0 1
./datasets/real/shuttle tests/real/shuttle/ swlof 10000 0 0 1
./datasets/real/shuttle tests/real/shuttle/ rshash 100 0 0 10
./datasets/real/shuttle tests/real/shuttle/ rshash 500 0 0 10
./datasets/real/shuttle tests/real/shuttle/ rshash 1000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ rshash 5000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ rshash 10000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ loda 100 0 0 10
./datasets/real/shuttle tests/real/shuttle/ loda 500 0 0 10
./datasets/real/shuttle tests/real/shuttle/ loda 1000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ loda 5000 0 0 10
./datasets/real/shuttle tests/real/shuttle/ loda 10000 0 0 10
./datasets/real/swan tests/real/swan/ sdo 100 0 0 10
./datasets/real/swan tests/real/swan/ sdo 500 0 0 10
./datasets/real/swan tests/real/swan/ sdo 1000 0 0 10
./datasets/real/swan tests/real/swan/ sdo 5000 0 0 10
./datasets/real/swan tests/real/swan/ sdo 10000 0 0 10
./datasets/real/swan tests/real/swan/ sdo 50000 0 0 10
./datasets/real/swan tests/real/swan/ sdo 100000 0 0 10
./datasets/real/swan tests/real/swan/ sdostream 100 0 0 10
./datasets/real/swan tests/real/swan/ sdostream 500 0 0 10
./datasets/real/swan tests/real/swan/ sdostream 1000 0 0 10
./datasets/real/swan tests/real/swan/ sdostream 5000 0 0 10
./datasets/real/swan tests/real/swan/ sdostream 10000 0 0 10
./datasets/real/swan tests/real/swan/ sdostream 50000 0 0 10
./datasets/real/swan tests/real/swan/ sdostream 100000 0 0 10
./datasets/real/swan tests/real/swan/ swknn 100 0 0 1
./datasets/real/swan tests/real/swan/ swknn 500 0 0 1
./datasets/real/swan tests/real/swan/ swknn 1000 0 0 1
./datasets/real/swan tests/real/swan/ swknn 5000 0 0 1
./datasets/real/swan tests/real/swan/ swknn 10000 0 0 1
./datasets/real/swan tests/real/swan/ swknn 50000 0 0 1
./datasets/real/swan tests/real/swan/ swknn 100000 0 0 1
./datasets/real/swan tests/real/swan/ rrct 100 0 0 10
./datasets/real/swan tests/real/swan/ rrct 500 0 0 10
./datasets/real/swan tests/real/swan/ rrct 1000 0 0 10
./datasets/real/swan tests/real/swan/ rrct 5000 0 0 10
./datasets/real/swan tests/real/swan/ rrct 10000 0 0 10
./datasets/real/swan tests/real/swan/ rrct 50000 0 0 10
./datasets/real/swan tests/real/swan/ rrct 100000 0 0 10
./datasets/real/swan tests/real/swan/ xstream 100 0 0 10
./datasets/real/swan tests/real/swan/ xstream 500 0 0 10
./datasets/real/swan tests/real/swan/ xstream 1000 0 0 10
./datasets/real/swan tests/real/swan/ xstream 5000 0 0 10
./datasets/real/swan tests/real/swan/ xstream 10000 0 0 10
./datasets/real/swan tests/real/swan/ xstream 50000 0 0 10
./datasets/real/swan tests/real/swan/ xstream 100000 0 0 10
./datasets/real/swan tests/real/swan/ swlof 100 0 0 1
./datasets/real/swan tests/real/swan/ swlof 500 0 0 1
./datasets/real/swan tests/real/swan/ swlof 1000 0 0 1
./datasets/real/swan tests/real/swan/ swlof 5000 0 0 1
./datasets/real/swan tests/real/swan/ swlof 10000 0 0 1
./datasets/real/swan tests/real/swan/ swlof 50000 0 0 1
./datasets/real/swan tests/real/swan/ swlof 100000 0 0 1
./datasets/real/swan tests/real/swan/ rshash 100 0 0 10
./datasets/real/swan tests/real/swan/ rshash 500 0 0 10
./datasets/real/swan tests/real/swan/ rshash 1000 0 0 10
./datasets/real/swan tests/real/swan/ rshash 5000 0 0 10
./datasets/real/swan tests/real/swan/ rshash 10000 0 0 10
./datasets/real/swan tests/real/swan/ rshash 50000 0 0 10
./datasets/real/swan tests/real/swan/ rshash 100000 0 0 10
./datasets/real/swan tests/real/swan/ loda 100 0 0 10
./datasets/real/swan tests/real/swan/ loda 500 0 0 10
./datasets/real/swan tests/real/swan/ loda 1000 0 0 10
./datasets/real/swan tests/real/swan/ loda 5000 0 0 10
./datasets/real/swan tests/real/swan/ loda 10000 0 0 10
./datasets/real/swan tests/real/swan/ loda 50000 0 0 10
./datasets/real/swan tests/real/swan/ loda 100000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdo 100 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdo 500 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdo 1000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdo 5000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdo 10000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdo 50000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdostream 100 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdostream 500 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdostream 1000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdostream 5000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdostream 10000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ sdostream 50000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ swknn 100 0 0 1
./datasets/real/yahoo tests/real/yahoo/ swknn 500 0 0 1
./datasets/real/yahoo tests/real/yahoo/ swknn 1000 0 0 1
./datasets/real/yahoo tests/real/yahoo/ swknn 5000 0 0 1
./datasets/real/yahoo tests/real/yahoo/ swknn 10000 0 0 1
./datasets/real/yahoo tests/real/yahoo/ swknn 50000 0 0 1
./datasets/real/yahoo tests/real/yahoo/ rrct 100 0 0 10
./datasets/real/yahoo tests/real/yahoo/ rrct 500 0 0 10
./datasets/real/yahoo tests/real/yahoo/ rrct 1000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ rrct 5000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ rrct 10000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ rrct 50000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ xstream 100 0 0 10
./datasets/real/yahoo tests/real/yahoo/ xstream 500 0 0 10
./datasets/real/yahoo tests/real/yahoo/ xstream 1000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ xstream 5000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ xstream 10000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ xstream 50000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ swlof 100 0 0 1
./datasets/real/yahoo tests/real/yahoo/ swlof 500 0 0 1
./datasets/real/yahoo tests/real/yahoo/ swlof 1000 0 0 1
./datasets/real/yahoo tests/real/yahoo/ swlof 5000 0 0 1
./datasets/real/yahoo tests/real/yahoo/ swlof 10000 0 0 1
./datasets/real/yahoo tests/real/yahoo/ swlof 50000 0 0 1
./datasets/real/yahoo tests/real/yahoo/ rshash 100 0 0 10
./datasets/real/yahoo tests/real/yahoo/ rshash 500 0 0 10
./datasets/real/yahoo tests/real/yahoo/ rshash 1000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ rshash 5000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ rshash 10000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ rshash 50000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ loda 100 0 0 10
./datasets/real/yahoo tests/real/yahoo/ loda 500 0 0 10
./datasets/real/yahoo tests/real/yahoo/ loda 1000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ loda 5000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ loda 10000 0 0 10
./datasets/real/yahoo tests/real/yahoo/ loda 50000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdo 500 0 0 10
./datasets/real/optout/ tests/real/optout/ sdo 1000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdo 5000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdo 10000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdo 50000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdo 100000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdo 500000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdostream 500 0 0 10
./datasets/real/optout/ tests/real/optout/ sdostream 1000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdostream 5000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdostream 10000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdostream 50000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdostream 100000 0 0 10
./datasets/real/optout/ tests/real/optout/ sdostream 500000 0 0 10
./datasets/real/optout/ tests/real/optout/ swknn 500 0 0 1
./datasets/real/optout/ tests/real/optout/ swknn 1000 0 0 1
./datasets/real/optout/ tests/real/optout/ swknn 5000 0 0 1
./datasets/real/optout/ tests/real/optout/ swknn 10000 0 0 1
./datasets/real/optout/ tests/real/optout/ swknn 50000 0 0 1
./datasets/real/optout/ tests/real/optout/ swknn 100000 0 0 1
./datasets/real/optout/ tests/real/optout/ swknn 500000 0 0 1
./datasets/real/optout/ tests/real/optout/ rrct 500 0 0 10
./datasets/real/optout/ tests/real/optout/ rrct 1000 0 0 10
./datasets/real/optout/ tests/real/optout/ rrct 5000 0 0 10
./datasets/real/optout/ tests/real/optout/ rrct 10000 0 0 10
./datasets/real/optout/ tests/real/optout/ rrct 50000 0 0 10
./datasets/real/optout/ tests/real/optout/ rrct 100000 0 0 10
./datasets/real/optout/ tests/real/optout/ rrct 500000 0 0 10
./datasets/real/optout/ tests/real/optout/ xstream 500 0 0 10
./datasets/real/optout/ tests/real/optout/ xstream 1000 0 0 10
./datasets/real/optout/ tests/real/optout/ xstream 5000 0 0 10
./datasets/real/optout/ tests/real/optout/ xstream 10000 0 0 10
./datasets/real/optout/ tests/real/optout/ xstream 50000 0 0 10
./datasets/real/optout/ tests/real/optout/ xstream 100000 0 0 10
./datasets/real/optout/ tests/real/optout/ xstream 500000 0 0 10
./datasets/real/optout/ tests/real/optout/ swlof 500 0 0 1
./datasets/real/optout/ tests/real/optout/ swlof 1000 0 0 1
./datasets/real/optout/ tests/real/optout/ swlof 5000 0 0 1
./datasets/real/optout/ tests/real/optout/ swlof 10000 0 0 1
./datasets/real/optout/ tests/real/optout/ swlof 50000 0 0 1
./datasets/real/optout/ tests/real/optout/ swlof 100000 0 0 1
./datasets/real/optout/ tests/real/optout/ swlof 500000 0 0 1
./datasets/real/optout/ tests/real/optout/ rshash 500 0 0 10
./datasets/real/optout/ tests/real/optout/ rshash 1000 0 0 10
./datasets/real/optout/ tests/real/optout/ rshash 5000 0 0 10
./datasets/real/optout/ tests/real/optout/ rshash 10000 0 0 10
./datasets/real/optout/ tests/real/optout/ rshash 50000 0 0 10
./datasets/real/optout/ tests/real/optout/ rshash 100000 0 0 10
./datasets/real/optout/ tests/real/optout/ rshash 500000 0 0 10
./datasets/real/optout/ tests/real/optout/ loda 500 0 0 10
./datasets/real/optout/ tests/real/optout/ loda 1000 0 0 10
./datasets/real/optout/ tests/real/optout/ loda 5000 0 0 10
./datasets/real/optout/ tests/real/optout/ loda 10000 0 0 10
./datasets/real/optout/ tests/real/optout/ loda 50000 0 0 10
./datasets/real/optout/ tests/real/optout/ loda 100000 0 0 10
./datasets/real/optout/ tests/real/optout/ loda 500000 0 0 10
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

training_size_factor = 0.2
block_size_factor = 0.01
save_scores = False
res = []

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*.arff'))):
    data, labels, timestamps, m, dataset_idx = dt.retrieve_data(filename=filename, idf=0, collection=False, non_predictors=3, reverse_class=reverse_class)

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

df = pd.DataFrame(data=res)
outfile = outpath + "SUMMARY_" + alg + "_T" + str(timepam) + ".txt"
print("Summary file (txt):",outfile,"\n")
df.to_csv(outfile, sep=',', header=None)


