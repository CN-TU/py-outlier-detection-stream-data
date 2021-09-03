#!/usr/bin/env python3

import dependencies
dependencies.assert_pkgs({
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
    'xStream': 'contrib/xStream.tar.bz2',
    'pysdo': 'https://github.com/CN-TU/pysdo/archive/master.zip',
    'dSalmon': 'https://github.com/CN-TU/dSalmon'
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

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

DEFAULT_RUNS = '''
datasets/example/ tests/example/ sdo 2000 0 0 1
datasets/example/ tests/example/ sdostream 2000 0 0 1
datasets/example/ tests/example/ swknn 2000 0 0 1
datasets/example/ tests/example/ swlof 2000 0 0 1
datasets/example/ tests/example/ rrct 2000 0 0 1
datasets/example/ tests/example/ rshash 2000 0 0 1
datasets/example/ tests/example/ loda 2000 0 0 1
datasets/example/ tests/example/ xstream 2000 0 0 1
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
alg = sys.argv[3] #algorithm
timepam = int(sys.argv[4])
jumble_data = (sys.argv[5] == '1')
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
    # during training don't score the first samples, as the
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
    
def stable_sigmoid(x):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig

def performance(s,m):
    idx = s.argsort()[::-1]
    #l = np.arange(len(s))[::-1]/len(s)
    l = stable_sigmoid(np.linspace(4,-4,len(s)))
    return sum(m[idx]*l)/len(s)


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
        return compute_training_score(y, -1/(1+scores))

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
        return compute_training_score(y, -1/(1+scores))

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
        return compute_training_score(y, -1/(1+scores))

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
        return compute_training_score(y, -1/(1+scores))

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
        return compute_training_score(y, -1/(1+scores))

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
        return compute_training_score(y, -1/(1+scores))

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
        return compute_training_score(y, -1/(1+scores))

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
        return compute_training_score(y, -1/(1+scores))

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
        return compute_training_score(y, -1/(1+scores))


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
        results += [ compute_training_score(y[test_indices], -1/(1+scores[:,i])) for i in range(k_max) ]
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

res = []

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*.arff'))):
    print("Data file (.arff)", filename)
    print("Data file index: ", idf,"\n")
    filename_short = ntpath.basename(filename)
    arffdata = loadarff(filename)
    df_data = pd.DataFrame(arffdata[0])

    if(df_data['class'].dtypes == 'object'):
        df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))
    print(df_data['class'].value_counts())
    labels = df_data['class'].to_numpy().astype(int)
    dens = df_data['den'].to_numpy().astype(int)
    clus = df_data['clus'].to_numpy().astype(int)
    data = df_data.to_numpy()
    data = MinMaxScaler().fit_transform(data)
    timestamps = np.arange(0,data.shape[0])
    timestamps = timestamps[:,None]

    if reverse_class:
        labels[labels>0] = -1
        labels = labels + 1
    data = np.array(data[:,0:-2], dtype=np.float64)

    print("Dataset size:", data.shape)
    training_len=2000
    blocksize = 2000
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
        px = [int(x) for x in np.linspace(3, 15, num = 5)]
        kdef=500
        pk = [int(x) for x in np.linspace(kdef/10, 5*kdef, num = 10)]
        pqv = [int(x)/10 for x in np.linspace(1, 5, num = 5)]
        param = dict(k=pk, x=px, qv=pqv, T=[timepam])
        wrapper = wSDOstream()
        best_param = parameter_tuning(param, np.hstack((data[0:training_len,:],timestamps[0:training_len])), labels[0:training_len], wrapper)
    elif alg=='swknn':
        print("Algorithm: SWKNN")
        best_param = sw_gridsearch(outlier.SWKNN, timepam, 50, np.hstack((data[0:training_len,:],timestamps[0:training_len])), labels[0:training_len])
    elif alg=='swlof':
        print("Algorithm: SWLOF")
        best_param = sw_gridsearch(outlier.SWLOF, timepam, 35, np.hstack((data[0:training_len,:],timestamps[0:training_len])), labels[0:training_len])
    elif alg=='rrct':
        print("Algorithm: Robust Random Cut Trees")
        pne = [int(x) for x in np.linspace(10, 300, num = 20)]
        param = dict(n_estimators=pne, window=[timepam])
        wrapper = wRRCT()
        best_param = parameter_tuning(param, np.hstack((data[0:training_len,:],timestamps[0:training_len,:])), labels[0:training_len], wrapper)
    elif alg=='xstream':
        print("Algorithm: xStream")
        pk = [int(x) for x in np.linspace(3, 500, num = 20)]
        pc = [int(x) for x in np.linspace(5, 500, num = 20)]
        pdd = [int(x) for x in np.linspace(5, 100, num = 10)]
        param = dict(k=pk, c=pc, d=pdd, init_sample=[timepam])
        wrapper = wxStream()
        best_param = parameter_tuning(param, np.hstack((data[0:training_len,:],timestamps[0:training_len,:])), labels[0:training_len], wrapper)
    elif alg=='rshash':
        print("Algorithm: RSHash")
        pne = [int(x) for x in np.linspace(50, 300, num = 30)]
        pnh = [int(x) for x in np.linspace(2, 40, num = 30)]
        pnb = [int(x) for x in np.linspace(10, 100, num = 30)]
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
        scores[0:training_len] = detector.fit_predict(data[0:training_len,:])
        end_tr = time.time()
        elapsed_tr = end_tr - start_tr
        sct = scores[0:training_len]
        lbl = labels[0:training_len]     
        sct[np.isnan(sct)] = 0      
        from numpy import inf
        sct[sct==inf] = 0     
        if alg=='swknn':
            Rth = 1.1*max(sct[lbl==0])

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

        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0

        sce = scores[training_len:]
        sce[np.isnan(sce)] = 0      
        sce[sce==inf] = 0   
        flat = sce.flatten()
        flat.sort()
        th = flat[-100] 
        print("-------")
        print(alg)
        print(np.nanmax(sct[lbl==0]),np.nanmin(sct[lbl==1]),th)    
        print(np.nanmax(sce),np.nanmin(sce),th)              
        print("-------")
        scbin = sce > th

        cle = clus[training_len:]
        dee = dens[training_len:]

        if alg=='swknn':
            idx = sce.argsort()[::-1]
            sce, cle, dee = sce[idx], cle[idx], dee[idx]
            sce = sce > Rth
            sce = sce.astype(np.float)

        s_loc, d_loc = sce, dee
        loc = performance(s_loc, d_loc)
        
        #states = [2, 4]
        #mask = np.in1d(dee, states)
        #s_rel, m_rel = sce[mask], cle[mask]
        #mask = np.in1d(m_rel, states)
        #s_rel, m_rel = s_rel[mask], m_rel[mask]
        s_rel, m_rel = sce, np.copy(cle)
        m_rel[m_rel==2]=-1
        m_rel[m_rel==1]=-1
        m_rel[m_rel==3]=1
        m_rel[m_rel==4]=1
        m_rel[m_rel==0]=1

        rel = performance(s_rel,m_rel)
        print(alg, loc, rel, file=open(outpath+'ex_Loc_Rel.txt', "a"))
        print("Locality: ", loc, ", Relativity: ", rel)

        fig = plt.figure(1, figsize=(4, 4), dpi=100,)
        sc = -1/(1+scores[training_len:])
        sc = sc.reshape(-1, 1) 
        sc2 = MinMaxScaler().fit_transform(sc)
        plt.scatter(data[training_len:,0],data[training_len:,1], s=1, cmap='coolwarm', c=sc2.flatten())
        plt.colorbar(ticks=[np.min(sc2), np.max(sc2)])
        plt.title(alg+' scores', fontsize=14)
        plt.savefig(outpath+'ex'+alg+'.png')
        fig = plt.figure(2, figsize=(4, 4), dpi=100,)
        plt.scatter(data[training_len:,0],data[training_len:,1], s=1, cmap='Reds', c=scbin.flatten())
        plt.colorbar(ticks=[np.min(scbin), np.max(scbin)])
        plt.title(alg+' top 100 outliers', fontsize=14)
        plt.savefig(outpath+'ex_bin_'+alg+'.png')
        img = Image.open(outpath+'ex_bin_'+alg+'.png') 
        left, top, right, bottom = 0, 0, 313, 400
        img_res = img.crop((left, top, right, bottom)) 
        img_fin = Image.new("RGB", (400,400), (255, 255, 255))
        img_fin.paste(img_res, (left, top, right, bottom))
        img_fin.save(outpath+'ex_bin_'+alg+'.png') 

fig = plt.figure(0, figsize=(4, 4), dpi=100,)
ax = plt.gca()
plt.scatter(data[0:training_len,0],data[0:training_len,1], s=1, cmap='coolwarm', c=labels[0:training_len])
plt.colorbar(ticks=[np.min(labels[0:training_len]), np.max(labels[0:training_len])])
ax.set_xlim((-0.05, 1.01))
ax.set_ylim((-0.05, 1.06))
plt.title('training dataset', fontsize=14)
plt.savefig(outpath+'ex_base.png')
img = Image.open(outpath+'ex_base.png') 
left, top, right, bottom = 0, 0, 313, 400
img_res = img.crop((left, top, right, bottom)) 
img_res.save(outpath+'ex_eval.png') 
img_fin = Image.new("RGB", (400,400), (255, 255, 255))
img_fin.paste(img_res, (left, top, right, bottom))
img_fin.save(outpath+'ex_base.png') 

fig = plt.figure(3, figsize=(4, 4), dpi=100,)
ax = plt.gca()
plt.scatter(data[training_len:,0],data[training_len:,1], s=1, cmap='Greys', c=0.5*np.ones([training_len]))
plt.colorbar(ticks=[0,1])
plt.scatter(data[training_len:,0],data[training_len:,1], s=1, cmap='Greys', c='black')
ax.set_xlim((-0.05, 1.01))
ax.set_ylim((-0.05, 1.06))
plt.title('evaluation dataset', fontsize=14)
plt.savefig(outpath+'ex_eval.png')
img = Image.open(outpath+'ex_eval.png') 
left, top, right, bottom = 0, 0, 313, 400
img_res = img.crop((left, top, right, bottom)) 
img_res.save(outpath+'ex_eval.png') 
img_fin = Image.new("RGB", (400,400), (255, 255, 255))
img_fin.paste(img_res, (left, top, right, bottom))
img_fin.save(outpath+'ex_eval.png') 

fig = plt.figure(5, figsize=(4, 4), dpi=100,)
ax = plt.gca()
aux=data[training_len:,:]
caux=clus[training_len:]
plt.scatter(aux[caux==1,0],aux[caux==1,1], s=1, color='red')
plt.scatter(aux[caux==2,0],aux[caux==2,1], s=1, color='blue')
plt.scatter(aux[caux==3,0],aux[caux==3,1], s=1, color='green')
plt.scatter(aux[caux==4,0],aux[caux==4,1], s=1, color='grey')
plt.scatter(aux[caux==0,0],aux[caux==0,1], s=1, color='black')
#plt.scatter(data[training_len:,0],data[training_len:,1], s=1, cmap='Dark2', c=clus[training_len:])
plt.colorbar(ticks=[np.min(clus[training_len:]), np.max(clus[training_len:])])
ax.set_xlim((-0.05, 1.01))
ax.set_ylim((-0.05, 1.06))
plt.title('test dataset_clusters', fontsize=14)
plt.savefig(outpath+'ex_test_clusters.png')
img = Image.open(outpath+'ex_test_clusters.png') 
left, top, right, bottom = 0, 0, 313, 400
img_res = img.crop((left, top, right, bottom)) 
img_res.save(outpath+'ex_test_clusters.png') 
img_fin = Image.new("RGB", (400,400), (255, 255, 255))
img_fin.paste(img_res, (left, top, right, bottom))
img_fin.save(outpath+'ex_test_clusters.png') 

fig = plt.figure(6, figsize=(4, 4), dpi=100,)
ax = plt.gca()
aux=data[training_len:,:]
cdens=dens[training_len:]
plt.scatter(aux[cdens==1,0],aux[cdens==1,1], s=1, color='red')
plt.scatter(aux[cdens==2,0],aux[cdens==2,1], s=1, color='green')
plt.scatter(aux[cdens==3,0],aux[cdens==3,1], s=1, color='blue')
plt.scatter(aux[cdens==4,0],aux[cdens==4,1], s=1, color='black')
#plt.scatter(data[training_len:,0],data[training_len:,1], s=1, cmap='Dark2', c=dens[training_len:])
plt.colorbar(ticks=[np.min(dens[training_len:]), np.max(dens[training_len:])])
ax.set_xlim((-0.05, 1.01))
ax.set_ylim((-0.05, 1.06))
plt.title('test dataset_dens_levs', fontsize=14)
plt.savefig(outpath+'ex_test_denslevs.png')
img = Image.open(outpath+'ex_test_denslevs.png') 
left, top, right, bottom = 0, 0, 313, 400
img_res = img.crop((left, top, right, bottom)) 
img_res.save(outpath+'ex_test_denslevs.png') 
img_fin = Image.new("RGB", (400,400), (255, 255, 255))
img_fin.paste(img_res, (left, top, right, bottom))
img_fin.save(outpath+'ex_test_denslevs.png') 
