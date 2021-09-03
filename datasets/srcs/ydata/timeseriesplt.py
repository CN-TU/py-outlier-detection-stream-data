import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

filename  = sys.argv[1]
a  = int(sys.argv[2])
b  = int(sys.argv[3])
show = 1

df = pd.read_csv(filename, header = None, sep = ',') 
val = df[0].to_numpy()
lab = df[1].to_numpy()
x = np.arange(0,len(val))


plt.figure(1, figsize=(16, 6), dpi=300)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

plt.subplot(211)
plt.plot(x[a:b],val[a:b])
ax = plt.subplot(212)
ax.stem(x[a:b],lab[a:b]==2,  linefmt='black', label='change points', markerfmt="", use_line_collection=True)
ax.stem(x[a:b],lab[a:b]==1,  linefmt='lightgray', label='outliers',  markerfmt=" ", use_line_collection=True)
ax.legend()

plt.xlabel('time', fontsize=14)
#plt.ylabel("evaluation time (s)", fontsize=14)
plt.tight_layout()
plt.savefig('ts.png')
if show:
    plt.show()

quit()
