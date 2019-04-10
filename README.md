# SDO+RF-for-SENCA
Streaming Emerging New Classes and Anomalies with SDO and RF algorithms.

Experiments conducted for the paper:
"Streaming Emerging Novel Attack Classes in Network Traffic"

April, 2019
Félix Iglesias, TU Wien


## Summary 
The following experiments use the CICIDS2017 dataset for the evaluation of the SDO+RF methodolgy for
the detection of streaming emerging new classes and anomalies. The evaluation consists of two different
scenarios: "lpk" (limited pre-knowledge) and "cpk" (comprehensive pre-knowledge). In every scenario the 
dataset is split into training and test subsets. The "training" subset contains normal traffic and 11 attack 
classes, the test subset contains normal traffic and 16 attack classes. The SENCForest algorithm is
used as benchmark for comparison.

## Requirements 
1. The CICIDS2017 dataset belongs to the Canadian Institute for Cybersecurity. It can be obtained on demand
in the following link:
https://www.unb.ca/cic/datasets/ids-2017.html

2. CICIDS2017 is provided as pcaps and a ground truth file. We extracted AGM features with a feature extractor 
based on Golang. Our feature extractor is pending publication (shortly), but it can be requested on demand by
email here: "felix.iglesias@tuwien.ac.at". In any case, features can be extracted with any feature extractor, 
e.g., tshark. 

We provide a "sample.csv" with the names of the used features and 5 samples. 
The configuraton of the AGM vector extraction is: [flow-key: sourceIPaddress, observation_time: 10 seconds]

Features are:

```
<flowStartSeconds, sourceIPAddress, distinct(destinationIPAddress), mode(destinationIPAddress), modeCount(destinationIPAddress), distinct(sourceTransportPort),
mode(sourceTransportPort), modeCount(sourceTransportPort), distinct(destinationTransportPort), mode(destinationTransportPort), modeCount(destinationTransportPort),
distinct(protocolIdentifier), mode(protocolIdentifier), modeCount(protocolIdentifier) ,distinct(ipTTL),mode(ipTTL), modeCount(ipTTL), distinct(_tcpFlags),
mode(_tcpFlags), modeCount(_tcpFlags), distinct(octetTotalCount), mode(octetTotalCount), modeCount(octetTotalCount), packetTotalCount, Attack, Label>
```

where,

"Attack": text label for attack families or normal traffic, e.g., "Normal", "Botnet:ARES", "Brute Force:FTP-Patator"

"Label": binary label, 1:attack, 0:normal

We call the preprocessed dataset in CSV format as follows: "Full_AGM.csv"

3. SDO can be downloaded from here: https://github.com/CN-TU/sdo-matlab

4. SENCForest can be downloaded from here:
http://lamda.nju.edu.cn/code_sencForest.ashx?AspxAutoDetectCookieSupport=1


## Instructions for replication 

0. Open a terminal

1. Extract the files in the working folder

2. Copy the CICIDS2017 preprocessed dataset according to the AGM format ("Full_AGM.csv") in the working folder. 

3. Generate limited pre-knowledge splits: 
> python split_data.py Full_AGM.csv lpk 0.90

It will create the following files within [lpk]:
"lab_test.csv","test.csv","train.csv","lab_train.csv"

4. Generate limited pre-knowledge splits: 
> python split_data.py Full_AGM.csv cpk 0.30

It will create the following files within [cpk]:
"lab_test.csv","test.csv","train.csv","lab_train.csv"

5. Open MATLAB

6. Add the [SDO folder] to the Path (MATLAB) 

7. Go to the [lpk/sdo-rf] or [cpk/sdo-rf] folder, run "init_simulation" 
(for a faster simulation, you can use "init_simulation_fast" instead, it does the same but samples are given to models in chunks
instead of one-by-one. Models deal with one-by-one or chunk data likewise).

At the end of the process, performance indices are printed out in the matlab command window

8. Add the [SENCForest] to the Path (MATLAB) 

7. Go to the [lpk/iforest] or [cpk/ifores] folder, run "init_simulation_SF" 

At the end of the process, performance indices are printed out in the matlab command window


## Included files 

File | Description
-----|------------- 
README.txt | this file
sample.csv | example of the CICIDS2017 formatted according to the AGM vector [flow-key: sourceIPaddress, observation_time: 10 seconds] header and 5 flows.
split_data.py | python script to prepare training and test splits with normal traffic and 11 attack families in the training subset and normal traffic and 16 attack families in the test split.
[lpk] | folder for the "limited pre-knowledge" experiments
[lpk/sdo-rf] | folder with scripts for using the SDO+RF method in SENCA experiments
lpk/sdo-rf/init_simulation.m  |  script for runing experiments with SDO+RF
lpk/sdo-rf/init_simulation_fast.m  | script for runing experiments with SDO+RF (fast mode)
lpk/sdo-rf/SENCA_performance.m  |  function for evaluation the SDO+RF performance
lpk/sdo-rf/new_class_detector.m  | function implementing the new class detectino for the the Final Class Evaluator
lpk/sdo-rf/optim_param.m  |  script for the parameter optimization of the TreeBagger
[lpk/sencforest]   | folder with scripts for using the SENCForest method in SENCA experiments
lpk/sencforest/init_simulation_SF.m   | script for runing experiments with SENCForest
lpk/sencforest/SENCA_performance_SF.m  | function for evaluation the SENCForest performance
[cpk]  | folder for the "comprehensive pre-knowledge" experiments. Contain the same files as [lpk]

--------

More information about our research at the CN-Group TU Wien:

https://www.cn.tuwien.ac.at/
