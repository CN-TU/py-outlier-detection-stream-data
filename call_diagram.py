import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import glob
import os
import re

lenind = 200
inpath  = sys.argv[1]
metric  = sys.argv[2]
show = (sys.argv[3] == '1')
if len(sys.argv) > 4: 
    alls = (sys.argv[4] == '1')
    if alls: 
        lenind = 1800

metrics = { "pan": "var5", "apan": "var7", "ap": "var9", "aap": "var11", "mf1": "var13", "amf1": "var15", "roc": "var17", "ttr": "var19", "tts": "var21" }
algs = { "swlof": "LOF", "swknn": "kNN", "xstream": "xS", "sdo": "SDO", "sdostream": "SDOs", "rrc": "RRCT", "rshash": "RSH", "loda": "LODA"}

dfval = pd.DataFrame()
for filename in glob.glob(os.path.join(inpath, '*.txt')):
    raw_data = pd.read_csv(filename, header = None, prefix="var", sep = ' ') 
    alg_name = algs[re.sub(inpath, '', filename).lstrip('/SUMMARY_').rstrip('_T500.txt')]
    clmn = list(raw_data)   
    for i in clmn: 
        raw_data[i] = raw_data[i].map(lambda x: x.rstrip(',\"'))
    aux = raw_data[metrics[metric]]
    aux = [float(x) for x in aux if str(x) != 'nan']
    aux = [float(x) for x in aux if str(x) != 'inf']
    if (alg_name == "LOF" or alg_name == "kNN"):
        aux = np.repeat(aux, repeats=10)

    dfval[alg_name] = pd.Series(aux, index=np.arange(0,lenind)).astype(float)

data = list()
for i, row in dfval.iterrows():
    #print(','.join(map(str, ['kNN',i,row['kNN']])) ) 
    #print(','.join(map(str, ['LOF',i,row['LOF']])) ) 
    #print(','.join(map(str, ['xS',i,row['xS']])) ) 
    #print(','.join(map(str, ['SDO',i,row['SDO']])) ) 
    #print(','.join(map(str, ['SDOs',i,row['SDOs']])) ) 
    #print(','.join(map(str, ['RRCT',i,row['RRCT']])) ) 
    data.append(['kNN',i,row['kNN']]) 
    data.append(['LOF',i,row['LOF']]) 
    data.append(['xS',i,row['xS']]) 
    data.append(['SDO',i,row['SDO']]) 
    data.append(['SDOs',i,row['SDOs']]) 
    data.append(['RRCT',i,row['RRCT']]) 
    data.append(['RSH',i,row['RSH']]) 
    data.append(['LODA',i,row['LODA']]) 

dfdiag = pd.DataFrame(data=data, index=None, columns=['classifier_name','dataset_name','accuracy'], dtype=None, copy=False)

chall = inpath.split('/')
if chall == 'm-out':
    chall = 'med-out'
elif chall == 'h-out':
    chall = 'many-out'
elif chall == 'synall':
    chall = 'All synthetic'

outputF = chall[-1]+'_'+metric+'_CDD'
title=chall[-1]+' (aap)'

import cddiagram
cddiagram.draw_cd_diagram(df_perf=dfdiag, title=title, labels=True, filename=outputF)
