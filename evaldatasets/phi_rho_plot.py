import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from brokenaxes import brokenaxes

csvfile  = sys.argv[1]	
df = pd.read_csv(csvfile, delimiter=',', names=['algorithm', 'overlap', 'odens' ])

#scaler = MinMaxScaler()
#df[['odens', 'overlap']] = scaler.fit_transform(df[['odens', 'overlap']])
#df['overlap'] = 1-df['overlap']

#fig, ax1 = plt.subplots(1, 1, figsize=(3,3), dpi=100,)
fig = plt.figure(figsize=(6,5))
baxes = brokenaxes(xlims=((0,1.2),(3880.9,3881.1)))
X,Y = df['odens'].to_numpy(),df['overlap'].to_numpy()
baxes.scatter(X, Y, c='black')

#df.plot.scatter(x='odens', y='overlap', c='black')
baxes.set_xlabel('$\\rho$', fontsize = 14)
baxes.set_ylabel('$\phi$', fontsize = 14)
#ax1.set_xlim(-0.1, 1.1)
#ax1.set_ylim(-0.1, 1.1)
df['incx'] = [ 0.03, -0.22, 0.01, 0.01,  0.02, -0.05, -0.110, 0.01,  0.02, 0.01,  0.02, 0.01, -0.09]
df['incy'] = [-0.01, -0.02, 0.01, 0.01, -0.02,  0.01, -0.030, 0.01,  0.00, 0.01, -0.01, 0.01, 0.02]
for i, row in df.iterrows():
    x = row['odens']
    y = row['overlap']
    name = row['algorithm']
    incx = row['incx']
    incy = row['incy']
    #plt.text(x + incx, y + incy, name, fontsize=12)
    baxes.text(x+incx, y+incy, name, fontsize=12)

#print(df)
plt.savefig('phi_rho.pdf')
#plt.show()

