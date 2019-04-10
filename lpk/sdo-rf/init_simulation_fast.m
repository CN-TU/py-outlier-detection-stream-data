% ********************* SDO + RF for SENCA in NT *********************
% Fast version, stream data is processed in chunks to reduce simulation
% times
% FIV, Mar 2019

tic;
chunk = 10000; % chunk of samples
% ************************* Retrieving  data *************************
fprintf("Loading datasets... ")
y_train=csvread('../lab_train.csv');
%y_test=csvread('test_cat.csv');
y_test=csvread('../lab_test.csv');
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
pTB.minLS = 1; pTB.numPTS = 15;
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
la = length(y_test);
% dummy descriptive variables for the test dataset
SDOrank = []; % zeros(la,1); % sample outlierness
TBclass = []; %zeros(la,1); % predicted multiclass (attack type)
TBattack = []; %zeros(la,1); % predicted binary class (non-attack/attack)
legit = []; %zeros(la,1); % if a sample is an legitimate (non-attack)
knownA = []; %zeros(la,1); % if a sample is a known attack (present in training) 
anomaly = []; %zeros(la,1); % if a sample is an anomaly
new_class = []; %zeros(la,1); % if a sample is an anomaly and new class

outlier_rec.anom_id = zeros(la,1); % the ID of the anomaly. Similar anomalies have the same anom_id
outlier_rec.anom_pos = []; % the position of the i-anomaly in the test dataset

% Subset of anomalous instances
outlier_rec.anomalies_set=[];
outlier_rec.an_count=0;

% Distance threshold to group anomalies in a new class
conf.anom_tol = sqrt(size(X_train,2))*0.2; % tolerance, max. distance allowed between two anomalies of the same class
conf.min_group = 10; % minimum number of anomalies to establish a new class

c = 0;
lb = floor(la/chunk);
fprintf("Elapsed time: %.2f seconds \n", toc); fprintf("Analyzing test data in streaming (sample by sample). '.'=%d samples... \n", chunk);

for i=1:lb+1
    ini = (i-1)*chunk+1;
    if (i<lb+1), fin = chunk*i; else, fin = la; 
    end
    xx=[]; clegit=[]; cknownA=[]; canomaly=[]; cnew_class=[];
    xx = X_test(ini:fin,:);
    cSDOrank  = sdof_apply_model( xx, observers, param.x )';
    cTBclass = round(str2double(predict(tree_model_Mc,xx)));
    %TBclass = classRF_predict(sample.x,tree_model_Mc);
    %TBattack = round(str2double(predict(tree_model_2c,sample.x)));
    cTBattack = arrayfun( @(x)(attacks(x)),cTBclass);  
    for j = 0:length(xx)-1
        samp.x = xx(j+1,:);
        samp.SDOrank = cSDOrank(j+1);
        samp.TBclass = cTBclass(j+1);
        samp.TBattack = cTBattack(j+1);
        [p, outlier_rec] = new_class_detector(samp, ini+j, outlier_rec, conf);
        if (p.new_class_found)
            fprintf("\n Found possible new class! id:%d, sample:%d - pred:%d,  real:%d ",outlier_rec.anom_id(ini+j),ini+j,cTBclass(j+1),y_test(ini+j));
        end
        clegit(j+1,1)=p.legit;
        cknownA(j+1,1)=p.known_attack;
        canomaly(j+1,1)=p.anomaly;
        cnew_class(j+1,1)=p.new_class;
    end
    SDOrank = [SDOrank; cSDOrank];
    TBclass = [TBclass; cTBclass];
    TBattack = [TBattack; cTBattack];
    legit = [legit; clegit];
    knownA = [knownA; cknownA]; 
    anomaly = [anomaly; canomaly];
    new_class = [new_class; cnew_class];
    fprintf(".");
    if (mod(i,10)==0)
        c=c+1; fprintf(" - %d * %d samples (%.2f), Elapsed time: %.2f seconds) \n",c,chunk,100*c*chunk/la,toc);
    end
end

clear cSDOrank cTBclass cTBattack clegit cknownA canomaly cnew_class c new_class_found p samp
clear attacks fin ini i j la

[results] = SENCA_performance(SDOrank, TBclass, TBattack, legit, knownA, anomaly, new_class, y_test, yb_test, new_test);
delete(gcp('nocreate'))

clear mypool paroptions pTB xx