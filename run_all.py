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
from xStream import xStream
from dSalmon import outlier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

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


inpath  = sys.argv[1]
outpath = sys.argv[2]
alg = sys.argv[3]
timepam = int(sys.argv[4])
jumble_data = (sys.argv[5] == '1') # currently unused
reverse_class = (sys.argv[6] == '1')
nrep = int(sys.argv[7])

np.random.seed(0)

def compute_score(y_true, scores):
    # compute adjusted average precision
    m = y_true.size
    num_outliers = np.sum(y_true)
    assert num_outliers > 0
    perm = np.argsort(scores)[::-1]
    scores_s = scores[perm]
    y_true_s = y_true[perm]
    y_true_cs = np.cumsum(y_true_s[:])
    ap = np.sum( y_true_cs[:num_outliers] / np.arange(1, num_outliers + 1) ) / num_outliers
    return (ap - num_outliers/m) / (1 - num_outliers/m)

def compute_training_score(y, scores):
    # during training don't score the first samples, since the
    # models are not yet trained for those samples, and, hence,
    # outlier scores are random
    if np.sum(y[y.size//2:]) == 0:
        # Return 1 if there are no outliers. This distorts the score
        # computed by parameter search. However, since all parameter
        # combinations use the same cv splits, we still find the parameters
        # based on the remaining 2 splits
        print('Warning: returning AAP=1 for dataset with no outliers')
        return 1
    return compute_score(y[y.size//2:], scores[y.size//2:])

def transform_scores(scores):
    # roc_auc_score() cannot handle inf values. This transforms the scores,
    # so that order is retained, equal scores remain equal, but all scores
    # are finite.
    scores_sorted = np.sort(scores)
    return np.searchsorted(scores_sorted, scores)
    

#################### Algorithm wrapper classes for parameter tuning #################
#####################################################################################

class wSDO(BaseEstimator, ClassifierMixin):
    def __init__(self,**kwargs):
        self.set_params(**kwargs)
    def get_params(self, deep=True):
        return self.params
    def set_params(self, **kwargs):
        self.params = kwargs
        return self
    def fit(self, X, y=None):
        pass
    def score(self, X, y):
        scores = pysdo.SDO(**self.params).fit_predict(X[:,:-1])
        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        return compute_training_score(y, transform_scores(scores))

class wSDOstream(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.set_params(**kwargs)
    def get_params(self, deep=True):
        return self.params
    def set_params(self, **kwargs):
        self.params = kwargs
        return self
    def fit(self, X, y):
        pass
    def score(self, X, y):
        scores = outlier.SDOstream(**self.params).fit_predict(X[:,:-1])
        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        return compute_training_score(y, transform_scores(scores))

class wSWDBOR(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.set_params(**kwargs)
    def get_params(self, deep=True):
        return self.params
    def set_params(self, **kwargs):
        self.params = kwargs
        return self
    def fit(self, X, y):
        pass
    def score(self, X, y):
        scores = outlier.SWDBOR(**self.params).fit_predict(X[:,:-1])
        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        return compute_training_score(y, transform_scores(scores))

class wSWKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.set_params(**kwargs)
    def get_params(self, deep=True):
        return self.params
    def set_params(self, **kwargs):
        self.params = kwargs
        return self
    def fit(self, X, y):
        pass
    def score(self, X, y):
        scores = outlier.SWKNN(**self.params).fit_predict(X[:,:-1])
        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        return compute_training_score(y, transform_scores(scores))

class wSWLOF(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.set_params(**kwargs)
    def get_params(self, deep=True):
        return self.params
    def set_params(self, **kwargs):
        self.params = kwargs
        return self
    def fit(self, X, y):
        pass
    def score(self, X, y):
        scores = outlier.SWLOF(**self.params).fit_predict(X[:,:-1])
        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        return compute_training_score(y, transform_scores(scores))

class wxStream(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.set_params(**kwargs)
    def get_params(self, deep=True):
        return self.params
    def set_params(self, **kwargs):
        self.params = kwargs
        return self
    def fit(self, X, y):
        pass
    def score(self, X, y):
        scores = xStream(**self.params).fit_predict(X[:,:-1])
        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        return compute_training_score(y, transform_scores(scores))

class wRRCT(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.set_params(**kwargs)
    def get_params(self, deep=True):
        return self.params
    def set_params(self, **kwargs):
        self.params = kwargs
        return self
    def fit(self, X, y):
        pass
    def score(self, X, y):
        scores = outlier.SWRRCT(**self.params, n_jobs=1).fit_predict(X[:,:-1])
        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        return compute_training_score(y, transform_scores(scores))

class wRSHash(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.set_params(**kwargs)
    def get_params(self, deep=True):
        return self.params
    def set_params(self, **kwargs):
        self.params = kwargs
        return self
    def fit(self, X, y):
        pass
    def score(self, X, y):
        scores = outlier.RSHash(**self.params).fit_predict(X[:,:-1])
        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        return compute_training_score(y, transform_scores(scores))

class wLODA(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.set_params(**kwargs)
    def get_params(self, deep=True):
        return self.params
    def set_params(self, **kwargs):
        self.params = kwargs
        return self
    def fit(self, X, y):
        pass
    def score(self, X, y):
        scores = outlier.LODA(**self.params).fit_predict(X[:,:-1])
        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        return compute_training_score(y, transform_scores(scores))

#####################################################################################
#####################################################################################

def parameter_tuning(param, X, label, detector):
    print('\nTuning parameters...')
    cv = StratifiedKFold(n_splits=3)
    print(param)
    gs = RandomizedSearchCV(detector, param, cv=cv, verbose=3, n_iter=30)
    print(X.shape,label.shape)
    gs.fit(X, label)
    print('Best parameters...')
    print(gs.best_params_) 
    return gs.best_params_

def sw_gridsearch(detector, window, k_max, X, y):
    # The SWLOF and SWKNN algorithm allow returning outlier scores
    # for a range of possible k values. We use this to perform
    # grid search more efficiently.
    results = np.zeros(k_max)
    cv = StratifiedKFold(n_splits=3)
    for _, test_indices in cv.split(X,y):
        det = detector(window, k_max, k_is_max=True)
        scores = det.fit_predict(X[test_indices,:-1])
        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        results += [ compute_training_score(y[test_indices], transform_scores(scores[:,i])) for i in range(k_max) ]
    best_k = int(np.argmax(results) + 1)
    print ('Best k: %d' % best_k)
    return {'k': best_k}

def sample_size (N, s, e):
    z=1.96
    num = N * pow(z,2) * pow(s,2)
    den = (N-1) * pow(e,2) + pow(z,2) * pow(s,2)
    n = int(np.floor(num/den))
    return n

def SDO_default_k(X):
    m = X.shape[0]
    Xt = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xt)
    sigma = np.std(Xp)
    if sigma<1:
        sigma=1
    error = 0.1*np.std(Xp)
    k = sample_size( m, sigma, error )
    return k

print("\nData folder:",inpath)
print("Result folder:",outpath)
print("Jumbling:",jumble_data)

os.makedirs(outpath, exist_ok=True)

res = []

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*.arff'))):
    print("Data file (.arff)", filename)
    print("Data file index: ", idf,"\n")
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

    print("Dataset size:", data.shape)
    training_len=int(0.2*data.shape[0])
    blocksize = int(0.01*data.shape[0])
    print("Training size:", training_len)
    print("Blocksize:", blocksize)

    if alg=='sdo':
        print("Algorithm: SDO")
        px = [int(x) for x in np.linspace(3, 8, num = 5)]
        kdef=500
        pk = [int(x) for x in np.linspace(kdef/10, 5*kdef, num = 10)]
        pqv = [int(x)/10 for x in np.linspace(1, 5, num = 5)]
        param = dict(k=pk, x=px, qv=pqv, chunksize=[blocksize+1], return_scores = [True], random_state=[0])
        wrapper = wSDO()
        best_param = parameter_tuning(param, np.hstack((data[0:training_len,:],timestamps[0:training_len])), labels[0:training_len], wrapper)
    elif alg=='sdostream':
        print("Algorithm: SDOstream")
        px = [int(x) for x in np.linspace(3, 8, num = 5)]
        kdef=500
        pk = [int(x) for x in np.linspace(kdef/10, 5*kdef, num = 10)]
        pqv = [int(x)/10 for x in np.linspace(1, 5, num = 5)]
        param = dict(k=pk, x=px, qv=pqv, T=[timepam])
        wrapper = wSDOstream()
        best_param = parameter_tuning(param, np.hstack((data[:training_len,:],timestamps[:training_len])), labels[:training_len], wrapper)
    elif alg=='swknn':
        print("Algorithm: SWKNN")
        best_param = sw_gridsearch(outlier.SWKNN, timepam, 50, np.hstack((data[:training_len,:],timestamps[:training_len])), labels[:training_len])
    elif alg=='swlof':
        print("Algorithm: SWLOF")
        best_param = sw_gridsearch(outlier.SWLOF, timepam, 35, np.hstack((data[:training_len,:],timestamps[:training_len])), labels[:training_len])
    elif alg=='rrct':
        print("Algorithm: Robust Random Cut Trees")
        pne = [int(x) for x in np.linspace(10, 300, num = 20)]
        param = dict(n_estimators=pne, window=[timepam])
        wrapper = wRRCT()
        best_param = parameter_tuning(param, np.hstack((data[:training_len,:],timestamps[:training_len,:])), labels[:training_len], wrapper)
    elif alg=='xstream':
        print("Algorithm: xStream")
        pk = [int(x) for x in np.linspace(3, 150, num = 20)]
        pc = [int(x) for x in np.linspace(5, 150, num = 20)]
        pdd = [int(x) for x in np.linspace(5, 25, num = 10)] 
        param = dict(k=pk, c=pc, d=pdd, init_sample=[timepam])
        wrapper = wxStream()
        best_param = parameter_tuning(param, np.hstack((data[:training_len,:],timestamps[:training_len,:])), labels[:training_len], wrapper)
    elif alg=='rshash':
        print("Algorithm: RSHash")
        pne = [int(x) for x in np.linspace(50, 300, num = 30)]
        pnh = [int(x) for x in np.linspace(2, 40, num = 30)]
        pnb = [int(x) for x in np.linspace(500, 50000, num = 30)]
        param = dict(n_estimators=pne, cms_w=pnh, cms_d=pnb, window=[timepam])
        wrapper = wRSHash()
        best_param = parameter_tuning(param, np.hstack((data[0:training_len,:],timestamps[0:training_len,:])), labels[0:training_len], wrapper)
    elif alg=='loda':
        print("Algorithm: LODA")
        pnp = [int(x) for x in np.linspace(10, 500, num = 30)]
        pnb = [int(x) for x in np.linspace(10, 100, num = 30)]
        param = dict(n_projections=pnp, n_bins=pnb, window=[timepam])
        wrapper = wLODA()
        best_param = parameter_tuning(param, np.hstack((data[0:training_len,:],timestamps[0:training_len,:])), labels[0:training_len], wrapper)
    else:
        print("ERROR: Invalid algorithm! \n")
        quit()
    print("--------------")

    for idx in range(nrep):
        if alg=='sdo':
            detector = pysdo.SDO(k=best_param['k'], x=best_param['x'], qv=best_param['qv'], random_state=idx, chunksize=blocksize+1, return_scores = True)
        elif alg=='sdostream':
            detector = outlier.SDOstream(k=best_param['k'], T=timepam, x=best_param['x'], qv=best_param['qv'], seed=idx)
        elif alg=='swknn':
            detector = outlier.SWKNN(k=best_param['k'], window=timepam)
        elif alg=='swlof':
            detector = outlier.SWLOF(k=best_param['k'], window=timepam)
        elif alg=='rrct':
            detector = outlier.SWRRCT(n_estimators=best_param['n_estimators'], window=timepam, seed=idx, n_jobs=1)
        elif alg=='xstream':
            detector = xStream(k=best_param['k'], c=best_param['c'], d=best_param['d'], init_sample=timepam, seed=idx)
        elif alg=='rshash':
            detector = outlier.RSHash(n_estimators=best_param['n_estimators'], cms_w=best_param['cms_w'], cms_d=best_param['cms_d'], window=timepam, seed=idx, n_jobs=1)
        elif alg=='loda':
            detector = outlier.LODA(n_projections=best_param['n_projections'], n_bins=best_param['n_bins'], window=timepam)
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
        scores_file = '%sSCORES_%s_%d_T%d_%d.pickle.gz' % (outpath, alg, int(dataset_idx), int(timepam), int(idx))
        with gzip.open(scores_file, 'wb') as f:
            pickle.dump(scores, f)

        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        perf = indices.get_indices(labels[training_len:], transform_scores(scores[training_len:]))
        new_row = "dataset: %d, idx: %d, P@n: %.2f, aP@n: %.2f, AP: %.2f, aAP: %.2f, MF1: %.2f, aMF1: %.2f, ROC: %.2f, Ttr: %.3f, Tts: %.3f" % (idf,idx,perf['Patn'],perf['adj_Patn'],perf['ap'],perf['adj_ap'],perf['maxf1'],perf['adj_maxf1'],perf['auc'],elapsed_tr,elapsed_ts)
        res.append(new_row)
        print("\n",new_row)

df = pd.DataFrame(data=res)
outfile = outpath + "SUMMARY_" + alg + "_T" + str(timepam) + ".txt"
print("Summary file (txt):",outfile,"\n")
df.to_csv(outfile, sep=',', header=None)


