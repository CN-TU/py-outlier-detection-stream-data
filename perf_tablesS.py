import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import glob
import os
import re

inpath  = sys.argv[1]
prefix  = sys.argv[2]

metrics = { "pan": "var5", "apan": "var7", "ap": "var9", "aap": "var11", "mf1": "var13", "amf1": "var15", "roc": "var17", "ttr": "var19", "tts": "var21" }
algs = { "swlof": "LOF", "swknn": "kNN", "xstream": "xS", "sdo": "SDO", "sdostream": "SDOs", "rrct": "RRCT", "rshash": "RSHash", "loda": "LODA"}

dcolumns = [ "pan", "apan", "ap", "aap", "mf1", "amf1", "roc"]
dindex = [ "LOF", "kNN", "xS", "SDO", "SDOs", "RRCT"]

dfval = pd.DataFrame(columns=dcolumns,index=dindex)
for filename in glob.glob(os.path.join(inpath, '*.txt')):
    raw_data = pd.read_csv(filename, header = None, prefix="var", sep = ' ') 
    #alg_name = algs[re.sub(inpath, '', filename).lstrip('/SUMMARY_').rstrip('_T500.txt')]
    alg_name = algs[re.sub(inpath, '', filename).split("_")[1]]
    clmn = list(raw_data)   
    for i in clmn: 
        raw_data[i] = raw_data[i].map(lambda x: x.rstrip(',\"'))

    for metric in dcolumns:
        aux = raw_data[metrics[metric]]
        aux = [float(x) for x in aux if str(x) != 'nan']
        aux = [float(x) for x in aux if str(x) != 'inf']
        if (alg_name == "LOF" or alg_name == "kNN"):
            aux = np.repeat(aux, repeats=10)

#        dfval[alg_name] = pd.Series(np.mean(aux), index=np.arange(0,lenind)).astype(float)
        dfval.at[alg_name,metric] = str(int(100*np.mean(aux))/100) +'+/-' + str(int(100*np.std(aux))/100)

chall = inpath.split('/')
if chall == 'm-out':
    chall = 'med-out'
elif chall == 'h-out':
    chall = 'many-out'
elif chall == 'synall':
    chall = 'All synthetic'

out_file = prefix + chall[-1]+'.tex'
out_table = dfval.to_latex(caption=chall[-1])

text_file = open(out_file, "w")
text_file.write(out_table)
text_file.close()
