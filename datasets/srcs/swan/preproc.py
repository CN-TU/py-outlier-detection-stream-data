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


if len(sys.argv) < 2:
    print ('No arguments given')
    quit()

infile  = sys.argv[1]
outfile = sys.argv[2]

np.random.seed(0)


print("\nData file:",infile)
print("Result file:",outfile)

#arffdata = loadarff(infile)
#df = pd.DataFrame(arffdata[0])
df = pd.read_csv(infile, sep='\t')

print('\nSorting based on END_TIME...')
#df['END_TIME'].to_datetime(stamps, format="%Y%m%d:%H:%M:%S.%f").sort_values()
df = df.sort_values(by = 'END_TIME')

print('Removing extra-headers...')
df = df[df['START_TIME'] != 'START_TIME']

print('Removing meta-data...')
del df['NOAA_AR_NO']
del df['START_TIME']
del df['END_TIME']
df = df.rename(columns={"LABEL": "class"})

print('Relocating labels in last column...')
cols = df.columns.tolist()
cols = cols[::-1]
df = df[cols] 
print(df.head(5))

print('Converting labels to binary...')
print(df['class'].value_counts())
df['class'] = df['class'].replace('F','0')
df['class'] = df['class'].replace('C','1')
df['class'] = df['class'].replace('B','1')
df['class'] = df['class'].replace('X','1')
df['class'] = df['class'].replace('M','1')
print(df['class'].value_counts())

#if(df['class'].dtypes == 'object'):
#    df['class'] = df['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))

print('Min-Max normalizing...')
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df['class'] = df['class'].values.astype(int)
print(df.head(5))

df.to_csv(outfile, sep=',', header=None, index=False)

