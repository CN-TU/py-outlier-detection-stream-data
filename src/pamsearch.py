from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

import numpy as np

import pysdo
from xStream import xStream
from dSalmon import outlier

def adjust_algorithm(alg, best_param, idx, timepam, blocksize):
    detector = None
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
    return detector

def parameter_selection(data, labels, timestamps, training_len, blocksize, alg, timepam):
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
    return best_param

def parameter_selection_2D(data, labels, timestamps, training_len, blocksize, alg, timepam):
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
        pk = [int(x) for x in np.linspace(3, 50, num = 20)]
        pc = [int(x) for x in np.linspace(5, 50, num = 20)]
        pdd = [int(x) for x in np.linspace(5, 50, num = 10)]
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
    return best_param


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
