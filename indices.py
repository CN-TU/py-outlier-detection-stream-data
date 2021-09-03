
from sklearn.metrics import roc_auc_score
import numpy as np

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
