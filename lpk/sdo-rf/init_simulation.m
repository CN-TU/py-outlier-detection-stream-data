% ********************* SDO + RF for SENCA in NT *********************
% FIV, Mar 2019

tic;
% ************************* Retrieving  data *************************
fprintf("Loading datasets... ")
y_train=csvread('../lab_train.csv');
y_test=csvread('test_cat.csv');
X_train=csvread('../train.csv',1,1);
X_test=csvread('../test.csv',1,1);

fprintf("Elapsed time: %.2f seconds \n", toc); fprintf("Preprocessing... \n");
% Z-score transformation
[X_train,mu,sigma] = zscore(X_train);
aux = bsxfun(@rdivide, bsxfun(@minus, X_test, mu), sigma);
X_test = aux;

% Build binary arrays  
keys =   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]; %labels
values = [1,1,1,1,1,1,1,1,1,1, 1, 0, 1, 1, 1, 1, 1];
attacks = containers.Map(keys,values);
yb_train = arrayfun( @(x)(attacks(x)),y_train); % yb_train = 1: new attack, 0:known attack or non-attack
yb_test = arrayfun( @(x)(attacks(x)),y_test); % yb_test = 1: new attack, 0:known attack or non-attack

keys =   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]; %labels
values = [1,1,0,1,0,0,0,0,1,0, 0, 0, 0, 0, 0, 1, 0];
new_attacks = containers.Map(keys,values);
new_test = arrayfun( @(x)(new_attacks(x)),y_test); % new_test = 1: new attack, 0: known attack or non-attack
clear aux keys values new_attacks mu sigma

% *************************** Parall. ******************************
mypool = parpool(2);
paroptions = statset('UseParallel',true);

% *************************** TRAINING ******************************
% training SDO
fprintf("Elapsed time: %.2f seconds \n", toc); fprintf("Training SDO... "); 
param.k=1000; param.qv=0.5; param.x=5; param.sd=10; param.hbs=1;
[ rank_train, observers, param ] = sdof( X_train, param );
% observers: SDO model, r_train: outlierness ranks of training samples

% training TreeBaggers
fprintf("Elapsed time: %.2f seconds \n", toc); fprintf("Training TreeBaggers... "); 
%tree_model_Mc = TreeBagger(50,X_train,y_train,'OOBPrediction','On','Method','classification','Options',paroptions);
pTB.minLS = 2; pTB.numPTS = 15;
tree_model_Mc = TreeBagger(50,X_train,y_train,'Method','classification','OOBPrediction','on','MinLeafSize',pTB.minLS,'NumPredictorstoSample',pTB.numPTS,'Options',paroptions);
%tree_model_Mc = classRF_train(X_train,y_train,50);
%tree_model_2c = TreeBagger(50,X_train,yb_train,'OOBPrediction','On','Method','classification');

% Distance thresholds to establish anomalies based on training data
tr.mean_legitimate = mean(rank_train(yb_train==0));
tr.std_legitimate = std(rank_train(yb_train==0));
tr.mean_attack = mean(rank_train(yb_train==0));
tr.std_attack = std(rank_train(yb_train==1));
conf.max_normal = tr.mean_legitimate + 2*tr.mean_legitimate;
conf.max_attack = tr.mean_attack + 2*tr.mean_attack;


% *************************** TEST ******************************
fprintf("Elapsed time: %.2f seconds \n", toc); fprintf("Analyzing test data in streaming (sample by sample). c=100 samples... \n");

la = length(y_test);
% dummy descriptive variables for the test dataset
SDOrank = zeros(la,1); % sample outlierness
TBclass = zeros(la,1); % predicted multiclass (attack type)
TBattack = zeros(la,1); % predicted binary class (non-attack/attack)
legit = zeros(la,1); % if a sample is an legitimate (non-attack)
knownA = zeros(la,1); % if a sample is a known attack (present in training) 
anomaly = zeros(la,1); % if a sample is an anomaly
new_class = zeros(la,1); % if a sample is an anomaly and new class

outlier_rec.anom_id = zeros(la,1); % the ID of the anomaly. Similar anomalies have the same anom_id
outlier_rec.anom_pos = []; % the position of the i-anomaly in the test dataset

% Subset of anomalous instances
outlier_rec.anomalies_set=[];
outlier_rec.an_count=0;

% Distance threshold to group anomalies in a new class
conf.anom_tol = sqrt(size(X_train,2))*0.2; % tolerance, max. distance allowed between two anomalies of the same class
conf.min_group = 10; % minimum number of anomalies to establish a new class

c=0;
for i=1:la
    sample.x = X_test(i,:);
    sample.SDOrank  = sdof_apply_model( sample.x, observers, param.x );
    sample.TBclass = round(str2double(predict(tree_model_Mc,sample.x)));
    %sample.TBclass = classRF_predict(sample.x,tree_model_Mc);
    %sample.TBattack = round(str2double(predict(tree_model_2c,sample.x)));
    sample.TBattack = arrayfun( @(x)(attacks(x)),sample.TBclass);    
    [p, outlier_rec] = new_class_detector(sample, i, outlier_rec, conf);
    SDOrank(i) = sample.SDOrank;
    TBclass(i) = sample.TBclass;
    TBattack(i) = sample.TBattack;
    legit(i) = p.legit;
    knownA(i) = p.known_attack; 
    anomaly(i) = p.anomaly;
    new_class(i) = p.new_class; 
    if (p.new_class_found)
        fprintf("\n Found possible new class! id:%d, sample:%d - pred:%d,  real:%d ",outlier_rec.anom_id(i),i,TBclass(i),y_test(i));
    end
    if (mod(i,100)==0)
        fprintf("c");
    end
    if (mod(i,5000)==0)
        c=c+1; fprintf(" - %d * 5000 samples, Elapsed time: %.2f seconds) \n",c,toc);
    end

end

[results] = SENCA_performance(SDOrank, TBclass, TBattack, legit, knownA, anomaly, new_class, y_test, yb_test, new_test);
delete(gcp('nocreate'))

clear attacks i la sample
clear mypool paroptions pTB 