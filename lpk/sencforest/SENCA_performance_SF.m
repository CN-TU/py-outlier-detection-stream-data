function [r] = SENCA_performance_IF(pred_class, y_test, yb_test)

%PERFORMANCE Summary of this function goes here
%   Detailed explanation goes here

keys =   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]; %labels
values = [1,1,0,1,0,0,0,0,1,0, 0, 0, 0, 0, 0, 1, 0];
new_attacks = containers.Map(keys,values);
new_test = arrayfun( @(x)(new_attacks(x)),y_test); % new_test = 1: new attack, 0: known attack or non-attack

keys =   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,999]; %labels
values = [0,0,1,0,1,1,1,1,0,1, 1, 0, 1, 1, 1, 0, 1,0];
known_attacks = containers.Map(keys,values);
knownA = arrayfun( @(x)(known_attacks(x)),pred_class); % yb_test = 1: new attack, 0:known attack or non-attack

new_class = pred_class==999;
anomaly = new_class;
legit = pred_class==11;

fprintf("\n--------PERFORMANCE----------\n");
fprintf("Real legit. traffic: %d \n", sum(yb_test==0));
fprintf("Real novel attacks: %d \n", sum(new_test));
fprintf("Real known attacks: %d \n", sum(yb_test)-sum(new_test));
fprintf("-----------------------------\n");
fprintf("Detected known attacks: %d \n", sum(knownA));
fprintf("Detected anomalies: %d \n", sum(anomaly));
fprintf("Detected novel attacks as in novel classes (anomalies): %d \n", sum(new_test & new_class));
fprintf("Detected novel attacks as anomalies: %d \n", sum(new_test & anomaly));
fprintf("Detected novel attacks as anomalies or known attacks: %d \n", sum(new_test & (anomaly | knownA)));
fprintf("Undetected novel attacks: %d \n", sum(new_test)-sum(new_test & (anomaly | knownA)));
fprintf("Undetected novel and old attacks: %d \n", sum(yb_test) - sum(yb_test & (anomaly | knownA)));

fprintf("TNR: %.3f \n", sum(yb_test==0 & legit)/sum(yb_test==0));
y_knownA = yb_test & new_test==0;
fprintf("TPR_ka: %.3f \n", sum(y_knownA & knownA)/sum(knownA==1));
fprintf("STPR_na: %.3f \n", sum(new_test & new_class)/sum(new_test));
fprintf("ETPR_na: %.3f \n", sum(new_test & (new_class | anomaly))/sum(new_test));
fprintf("OTPR_na: %.3f \n", sum(new_test & (new_class | anomaly | knownA))/sum(new_test));
fprintf("OTPR_a: %.3f \n", sum(yb_test & (new_class | anomaly | knownA))/sum(yb_test));
fprintf("OER: %.3f \n", sum((yb_test & legit) + (yb_test==0 & knownA)) / length(yb_test));
fprintf("Rn-in-nc: %.3f \n", sum(yb_test==0 & new_class)/sum(new_class));
fprintf("Rn-in-a: %.3f \n", sum(yb_test==0 & anomaly)/sum(anomaly));
fprintf("Rna-in-nc: %.3f \n", sum(new_test & new_class)/sum(new_class));
fprintf("Rna-in-a: %.3f \n", sum(new_test & anomaly)/sum(anomaly));


r=1;
%novel_attacks=[0,1,3,8,15];
%r.class_sum=[];
%for i=1:max(y_test)+1
%    j = i-1;
%    unk = find(novel_attacks==j);
%    pnorm = sum(y_test==j & legit==1);
%    pkA = sum(y_test==j & knownA==1);
%    po = sum(y_test==j & anomaly==1 & new_class==0);
%    pnc = sum(y_test==j & new_class==1);
%    fprintf("Class: %d, Attack? %d, Known? %d, total: %d, pred_norm: %d, pred_knownA: %d, pred_anom: %d, pred_new-class: %d \n", j, j~=11, isempty(unk), sum(y_test==j), pnorm, pkA, po, pnc);
%    cs = [j, j~=11, isempty(unk), sum(y_test==j), pnorm, pkA, po, pnc];
%    r.class_sum = [r.class_sum; cs];
%end

