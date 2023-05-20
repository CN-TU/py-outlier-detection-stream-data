
from sklearn.metrics import roc_auc_score
import numpy as np

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

def get_indices(y_true, scores):
	m = y_true.size
	num_outliers = np.sum(y_true)
	res = {}
	
	perm = np.argsort(scores)[::-1]
	scores_s = scores[perm]
	y_true_s = y_true[perm]

	# P@n
	res['Patn'] = np.sum(y_true_s[:num_outliers]) / num_outliers
	res['adj_Patn'] = (res['Patn'] - num_outliers/m) / (1 - num_outliers/m)
	
	y_true_cs = np.cumsum(y_true_s[:])

	# average precision
	res['ap'] = np.sum( y_true_cs[:num_outliers] / np.arange(1, num_outliers + 1) ) / num_outliers
	res['adj_ap'] = (res['ap'] - num_outliers/m) / (1 - num_outliers/m)
	
	# Max. F1 score
	res['maxf1'] = 2 * np.max(y_true_cs[:m] / np.arange(1 + num_outliers, m + 1 + num_outliers))
	res['adj_maxf1'] = (res['maxf1'] - num_outliers/m) / (1 - num_outliers/m)
	
	 # ROC-AUC
	res['auc'] = roc_auc_score(y_true, scores)
	
	return res
