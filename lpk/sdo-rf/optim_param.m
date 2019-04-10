% ********************* RF and SENCForest param opt *********************
% FIV, Mar 2019

tic;
% ************************* Retrieving  data *************************
fprintf("Loading datasets... ")
y_train=csvread('../lab_train.csv');
X_train=csvread('../train.csv',1,1);

fprintf("Elapsed time: %f seconds \n", toc); fprintf("Preprocessing... "); tic;
% Z-score transformation
[X_train,mu,sigma] = zscore(X_train);

% Build binary arrays  
keys =   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]; %labels
values = [1,1,1,1,1,1,1,1,1,1, 1, 0, 1, 1, 1, 1, 1];
attacks = containers.Map(keys,values);
yb_train = arrayfun( @(x)(attacks(x)),y_train); % yb_train = 1: new attack, 0:known attack or non-attack

clear keys values mu sigma

% *************************** TRAINING ******************************

% training TreeBaggers
fprintf("Elapsed time: %f seconds \n", toc); fprintf("Searching for optimal param... \n "); tic;

maxMinLS = 5;
minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
numPTS = optimizableVariable('numPTS',[1,size(X_train,2)-1],'Type','integer');
hyperparametersRF = [minLS; numPTS];
results = bayesopt(@(params)oobErrRF(params,X_train,y_train),hyperparametersRF,'AcquisitionFunctionName','expected-improvement-plus','Verbose',1);
bestOOBErr = results.MinObjective

fprintf("Elapsed time: %f seconds \n", toc); fprintf("Using best parameters... \n "); tic;
bestHyperparameters = results.XAtMinObjective
Mdl = TreeBagger(50,X_train, y_train,'Method','classification','MinLeafSize',bestHyperparameters.minLS,'NumPredictorstoSample',bestHyperparameters.numPTS);

fprintf("Elapsed time: %f seconds \n", toc); tic;

pred_mc = round(str2double(predict(Mdl,X_train)));
pred_b = arrayfun( @(x)(attacks(x)),pred_mc);
LM = loss(Mdl,X_train,y_train);
accb = 1-xor(pred_b,yb_train)/length(yb_train);

function oobErr = oobErrRF(params,X_train, y_train)
%oobErrRF Trains random forest and estimates out-of-bag quantile error
%   oobErr trains a random forest of 300 regression trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns the out-of-bag quantile error based on the median. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.
randomForest = TreeBagger(50,X_train,y_train,'Method','classification','OOBPrediction','on','MinLeafSize',params.minLS,'NumPredictorstoSample',params.numPTS);
oobErr = oobError(randomForest, 'Mode','ensemble');
end
