from scipy.io.arff import loadarff 
import ntpath
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

def retrieve_data(filename, idf, collection, non_predictors, reverse_class):

    print("Data file (.arff)", filename)
    print("Data file index: ", idf,"\n")
    filename_short = ntpath.basename(filename)
    arffdata = loadarff(filename)
    df_data = pd.DataFrame(arffdata[0])

    m,dataset_idx = 0,0
    if collection:
        m = re.match('.*_([0-9]+).arff', filename)
        dataset_idx = m.group(1) if m else 0

    if(df_data['class'].dtypes == 'object'):
        df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))

    labels = df_data['class'].to_numpy().astype(int)
    data = df_data.to_numpy()

    data = MinMaxScaler().fit_transform(data)
    timestamps = np.arange(0,data.shape[0])
    timestamps = timestamps[:,None]

    if reverse_class:
        labels[labels>0] = -1
        labels = labels + 1
    data = np.array(data[:,0:-non_predictors], dtype=np.float64)

    return data, labels, timestamps, m, dataset_idx


