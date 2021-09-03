
# Yahoo! Synthetic and real time-series with labeled anomalies, version 1.0 dataset

Downloaded on Sep. 2020 from:

See "README.txt" and "WebScopeReadMe.txt" files.


## Preprocesing

We use only time series from the [A1Benchmark] folder. Since we need longer time data and we want to check longer periodicities, we concatenate time-series and form a long A1-time-series. For consistency, the first 3 points after each concatenation are considered anomalies. 

1. Extract the downloaded package und merge timeseries:

    $ cat A1Benchmark/real_* > a1ts.c

During preprocessing we extract 3 features: a) d1n, b) dEn, c)dMn. 
"d1n" is the normalized increment: d1n_t = (Val_t - Val_t-1) / SMA_t
"dEn" is the normalized difference respect to the Exponential Moving Average (spam=2): EMAn_t = (Val_t - EMA_t) / SMA_t   
"dSn" is the normalized difference respect to Simple Moving Average (calculated over the last 20 instances): SMAn_t = (Val_t - SMA_t) / SMA_t

Since some parts of the timeseries are integer, if SMA_t=0, then SMA_t=1

2. Preprocess data using python:

    $ python3 preproc.py a1ts.csv a1tspr.csv

3. Copy "a1tspr.csv" in the [datasets/real/yahoo] folder. 

3. From the linux comman line, merge it with a proper ARFF header: 

    $ cat head.txt a1tspr.csv > yahoo.arff 

4. Delete "a1tspr.csv"

## Details

- Binary labels: 0-normal, 1-anomaly
- 6 features + 1 label (columns)
- 168000 instances (rows)
