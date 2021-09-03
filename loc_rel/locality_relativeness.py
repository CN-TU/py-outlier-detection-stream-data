import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

csvfile  = sys.argv[1]	
df = pd.read_csv(csvfile, delimiter=' ', names=['algorithm', 'locality', 'relativeness'])

scaler = MinMaxScaler()
df[['locality', 'relativeness']] = scaler.fit_transform(df[['locality', 'relativeness']])

fig, ax1 = plt.subplots(1, 1, figsize=(3,3), dpi=100,)
ax1 = df.plot.scatter(x='locality', y='relativeness', c='black')
ax1.set_xlabel('locality', fontsize = 14)
ax1.set_ylabel('relativeness', fontsize = 14)
ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(-0.1, 1.1)
df['incx'] = [-0.06, 0, 0.02, -0.05, 0.01, -0.10, 0.01, 0.05]
df['incy'] = [0.025, 0.02, -0.03, 0.02, 0.01, -0.07, -0.07, -0.03] 
for i, row in df.iterrows():
    x = row['locality']
    y = row['relativeness']
    name = row['algorithm']
    incx = row['incx']
    incy = row['incy']
    plt.text(x + incx, y + incy, name, fontsize=12)

#print(df)
#plt.show()
plt.savefig('locality_relativeness.png', dpi = 100)
