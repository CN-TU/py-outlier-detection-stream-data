import pandas as pd
import sys

csvfile  = sys.argv[1]

df = pd.read_csv(csvfile)
df.loc[(df.Label > 0), 'Label'] = 1

#optout vector
dfout = df[['apply(packetTotalCount,forward)','apply(octetTotalCount,forward)', 'apply(stdev(ipTotalLength),backward)', 'modeCount(destinationIPAddress)', 'modeCount(ipTTL)', 'Label']] 

dfout.to_csv('optout.csv', index=False, header=False)
