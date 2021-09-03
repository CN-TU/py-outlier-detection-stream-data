
# CICIDS2017-Optout dataset

Original raw data source downloaded from:

    https://www.unb.ca/cic/datasets/ids-2017.html

On 2018.

Features extracted and preprocessed according to:

    https://link.springer.com/chapter/10.1007%2F978-3-030-43887-6_13

*Felix Iglesias, Alexander Hartl, Tanja Zseby, Arthur Zimek, Are Network Attacks Outliers? A Study of Space Representations and Unsupervised Algorithms, Machine Learning and Knowledge Discovery in Databases, 10.1007/978-3-030-43887-6_13, (159-175), (2020).*

## Preprocesing

1. Download the data from the original source (UNB.CA).

2. Download preprocessing tools from:

    https://github.com/CN-TU/Datasets-preprocessing/tree/master/CIC-IDS-2017

3. Follow instructions in the repository (2) for extracting features from the original pcaps (1) into a "CAIAConAGM.csv" file, which is consistent with the *CAIAConAGM feature vector format*. In order to chek that the preprocessing was correct, we provide here a "CAIAConAGM_10rows.csv" with the first 10 extraced flows. 

4. Extract the *optout feature vector*:

    $ python3 CCA2optout.py 

It will automatically generate an output file named "optout.csv".

5. Copy "optout.csv" in the [datasets/real/optout] folder. 

6. From the linux comman line, merge it with a proper ARFF header: 

    $ cat header.txt optout.csv > optout.arff 

7. Delete "optout.csv"


## Details

- Binary labels: 0-normal, 1-anomaly
- 5 features + 1 label (columns)
- 2317922 instances (rows)

Use "sample.py" to extract a stratified 1% subset.
