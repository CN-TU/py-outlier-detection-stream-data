import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import glob
import os
import re

import collections
d = collections.defaultdict(dict)
dci = collections.defaultdict(dict)

inpath  = sys.argv[1]
metric  = sys.argv[2]
show = (sys.argv[3] == '1')

dataset = inpath.split("/")[-2]
if dataset == 'optout':
    dataset = 'cicids2017'

metrics = { "pan": "var5", "apan": "var7", "ap": "var9", "aap": "var11", "mf1": "var13", "amf1": "var15", "roc": "var17", "ttr": "var19", "tts": "var21" }
algs = { "swlof": "LOF", "swknn": "kNN", "xstream": "xS", "sdo": "SDO", "sdostream": "SDOs", "rrct": "RRCT", "rshash": "RSH", "loda": "LODA"}

#dfval = pd.DataFrame()
for filename in glob.glob(os.path.join(inpath, '*.txt')):
    raw_data = pd.read_csv(filename, header = None, prefix="var", sep = ' ') 
    raw_data = raw_data.replace(',', '',regex=True) 
    raw_data = raw_data.replace('\"', '',regex=True) 
    clmn = list(raw_data)   
    #for i in clmn:
    #    if type(raw_data[i]) is not (int or float): 
    #        raw_data[i] = raw_data[i].map(lambda x: x.rstrip(',\"'))
    aux = raw_data[metrics[metric]]
    alg_T = re.sub(inpath, '', filename).lstrip('/SUMMARY_')
    alg_name = alg_T.split('_')[0]
    T = int(alg_T.split('_')[1].rstrip('.txt').lstrip('T'))   
    aux = aux.values
    aux = [float(i) for i in aux]
    d[alg_name][T] = np.nanmean(aux)
    dci[alg_name][T] = np.nanstd(aux)

d = dict(sorted(d.items(), key=lambda item: item[0]))
plt.figure(1, figsize=(7, 4), dpi=100)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

for key in d:
    if (key=='ensemble0' and (metric=='tts' or metric=='ttr')):
        for t, val in d[key].items():
            d[key][t] = d['swknn'][t]+d['sdostream'][t]#+d['xstream'][t]+d['rrct'][t]

import itertools
#marker = itertools.cycle(('P',' ', 'o', '*', 'v', 'x', 'X', 'd', 'p')) 
marker = itertools.cycle(('P',',', '.', '*', 'v', 'x')) 
nmax = 1

for key in d:
    pinds = np.zeros(len(d[key]))
    pvals = np.zeros(len(d[key]))
    perrors = np.zeros(len(d[key]))
    i = 0
    for t, val in d[key].items():
        pinds[i] = int(t)
        pvals[i] = val
        perrors[i] = dci[key][int(t)]
        i+=1
    vals = pvals[np.argsort(pinds)]
    inds = np.sort(pinds) 
    errors = perrors[np.argsort(pinds)]
    #plt.semilogx(inds,vals,label=key, marker = next(marker))
    if key=='ensemble0':
        plt.errorbar(inds,vals, errors,label=algs[key], marker = next(marker), linestyle='dotted')
    else:
        plt.errorbar(inds,vals, errors,label=algs[key], marker = next(marker), linestyle='solid')
    plt.xscale('log')
    #plt.errorbar(inds,vals, errors, linestyle='None', marker = next(marker))
    nmax = np.max([np.max(vals),nmax])

axes = plt.gca()
#axes.set_ylim([0,1])
plt.legend()
if metric=='tts':
    #mmax=10000
    mmax=1.1*nmax
    axes.set_ylim([-0.1,mmax])
plt.title(dataset)
plt.grid(which='both')
plt.xlabel('T (memory span, sliding-window)', fontsize=14)
if metric=='tts':
    plt.ylabel("evaluation time (s)", fontsize=14)
else:
    plt.ylabel(metric, fontsize=14)
plt.tight_layout()
#metric = metric + '_' + inpath.replace('/', '_')
#metric = metric.replace('.', '')
print("Printing..."+'T_'+metric+'_'+dataset+'.png') 
plt.savefig('T_'+metric+'_'+dataset+'.png')
if show:
    plt.show()
