#!/usr/bin/env python3

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
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

infile  = sys.argv[1]
outfile  = sys.argv[2]

np.random.seed(0)

df = pd.read_csv(infile, sep=',')

del df['timestamp']

newtsix = df.index[df['value'] == 'value'].tolist()
for ix in newtsix:
    df.iloc[ix+1]['is_anomaly'] = 1
    df.iloc[ix+2]['is_anomaly'] = 1
    df.iloc[ix+3]['is_anomaly'] = 1
    #df.iloc[ix+1]['changepoint'] = '1'

df = df.drop(df[df['value'] == 'value'].index)
df = df.astype(float) 

df = df.rename(columns={"is_anomaly": "class"})
cl = df['class'].to_numpy()
cl[0:3]=1

#df.to_csv('ts.csv', sep=',', header=None, index=False)

del df['class']

#print('Relocating labels in last column...')
#cols = df.columns.tolist()
#cols = cols[::-1]
#cols = cols[-1:] + cols[:-1]  
#df = df[cols] 

df['EMA'] = df.iloc[:,0].ewm(span=2,adjust=False).mean()
df['SMA-20'] = df.iloc[:,0].rolling(20,min_periods=1).mean()
df['val-1'] = df['value'].shift(periods=1, fill_value=0.1)

df['SMA-20'] = np.where((df['SMA-20'] == 0),1,df['SMA-20'])

df['class'] = cl

df['val-1'] = (df['value'] - df['val-1']) / df['SMA-20']
df['EMA'] = (df['value'] - df['EMA']) / df['SMA-20']
df['SMA-20'] = (df['value'] - df['SMA-20']) / df['SMA-20']

del df['value']
#del df['avg']
#del df['dfmean']
#df = df.fillna(0)

#print(df.head(5))
print(df['class'].value_counts())

#print('Min-Max normalizing...')
#scaler = MinMaxScaler()
#df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df['class'] = df['class'].values.astype(int)
print(df.head(5))

df.to_csv(outfile, sep=',', header=None, index=False)

