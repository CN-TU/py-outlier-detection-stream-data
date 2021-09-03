
# Shuttle dataset

Downloaded from:

    http://odds.cs.stonybrook.edu/shuttle-dataset/

Original paper:

    https://www.sciencedirect.com/science/article/pii/B9781558602007501215?via%3Dihub

On Sep. 2020

## Preprocesing

1. Load file in MATLAB.

2. From MATLAB, save as "shuttle.csv" and copy it in the [datasets/real/shuttle] folder. 

3. From the linux comman line, merge it with a proper ARFF header: 

    $ cat header.txt shuttle.csv > shuttle.arff 

4. Delete "shuttle.csv"

## Details

- Binary labels: 0-normal, 1-anomaly
- 9 features + 1 label (columns)
- 48997 instances (rows)
