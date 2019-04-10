% SENCForest.

fprintf("Loading datasets... ")
y_train=csvread('../lab_train.csv');
y_test=csvread('../lab_test.csv');
X_train=csvread('../train.csv',1,1);
X_test=csvread('../test.csv',1,1);

newevaluation=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Parametres%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_num=2;  %Known Classes
newclass_num=2;
alltraindata=X_train;
alltraindatalabel=y_train;
streamdata=X_test;
streamdatalabel=y_test;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%trainning process%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NumTree = 100; % number of Tree
NumSub = 100; % subsample size for each class
CurtNumDim=size(alltraindata, 2);
rseed = sum(100 * clock);
set(0,'RecursionLimit',5000)
tic
Model = SENCForest(alltraindata, NumTree, NumSub, CurtNumDim, rseed,alltraindatalabel);
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%testing process%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Para.beta=1;%%pathline
Para.alpha=1;%%%distance
Para.buffersize=10;
tic
[Result]=Testingpro(streamdata,streamdatalabel,Model,Para);% eachclassnum,window);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Evaluation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toc

pred_class=Result(:,1);

% Build binary array  
keys =   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]; %labels
values = [1,1,1,1,1,1,1,1,1,1, 1, 0, 1, 1, 1, 1, 1];
attacks = containers.Map(keys,values);
yb_test = arrayfun( @(x)(attacks(x)),y_test); % yb_test = 1: new attack, 0:known attack or non-attack

pred_class=Result(:,1);
[r] = SENCA_performance_SF(pred_class, y_test, yb_test)
