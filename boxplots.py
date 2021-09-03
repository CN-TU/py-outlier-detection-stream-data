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
    print(filename, alg_name)
    dfval[alg_name] = pd.Series(aux, index=np.arange(0,lenind)).astype(float)

plt.figure(1, figsize=(6, 3), dpi=100)
plt.rc('xtick', labelsize=14)
#plt.xticks(rotation=-45)
box_plot_data=dfval.to_numpy() 
plt.boxplot(box_plot_data,patch_artist=True,labels=list(dfval.columns.values), showfliers=False)
axes = plt.gca()
mmax=1.1*np.max(box_plot_data)
if metric=='tts':
    mmax=80
axes.set_ylim([-0.1,mmax])

chall = inpath.split('/')
if chall == 'm-out':
    chall = 'med-out'
elif chall == 'h-out':
    chall = 'many-out'
elif chall == 'synall':
    chall = 'All synthetic'

mtit = metric
if mtit == 'tts':
    mtit = 'eval. time (seconds)'
plt.title(chall[-1])
plt.ylabel(metric, fontsize=14)
plt.tight_layout()
plt.savefig(chall[-1]+'_'+metric+'_BP.png')
if show:
    plt.show()
