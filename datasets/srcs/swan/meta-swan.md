
# Space weather dataset

Downloaded on Aug. 2020, from:

    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM

Downloaded files:
- partition1_instances.tar.gz
- partition2_instances.tar.gz
- partition3_instances.tar.gz
- partition4_instances.tar.gz
- partition5_instances.tar.gz

Data-asset published in:

    https://www.nature.com/articles/s41597-020-0548-x?sf235774487=1#Sec20

## Preprocesing


1. Extract the downloaded zips in this folder [srcs/swan].

We use the tools proposed by authors for extracting features from raw data. We extract the same default set of features used in the provided examples.

2. Download the following repo and extract it in the [srcs/swan] folder:

    https://bitbucket.org/gsudmlab/swan_features/src/master/

3. Copy "CONSTANTS.py" and "example.py" in [srcs/swan] within the [srcs/swan/gsudmlab-swan_features] folder (overwrite if necessary).
 
4. From [srcs/swan/gsudmlab-swan_features], open a terminal and run :

    $ python3 example.py

5. Merge outfiles from [swan/gsudmlab-swan_features/data/output] and save it in [srcs/swan]:

    $ cat gsudmlab-swan_features/data/output/raw_features_p* > swan.csv

6. We rearrange data, remove meta-data features, sort it based on END_TIME, and transform classes in a binary label. Copy the resulting file in the [datasets/real/swan] folder.

    $ python3 preproc.py swan.csv swan_bin.csv   

7. From the linux comman line, merge it with a proper ARFF header: 

    $ cat head.txt swan_bin.csv > swan.arff 

8. Delete "swan_bin.csv"

## Details

- Binary labels: 0-normal, 1-anomaly
- 12 features + 1 label (columns)
- 331185 instances (rows)
