function [r] = SENCA_performance(SDOrank, TBclass, TBattack, legit, knownA, anomaly, new_class, y_test, yb_test, new_test)

%PERFORMANCE Summary of this function goes here
%   Detailed explanation goes here

r.r_legit = sum(yb_test==0);
r.r_new_attacks = sum(new_test);
r.r_known_attacks = sum(yb_test)-sum(new_test);
r.p_known_attacks = sum(knownA);
r.p_anomalies = sum(anomaly);
r.p_new_as_newc = sum(new_test & new_class);
r.p_new_as_anom = sum(new_test & anomaly);
r.p_new_as_nora = (new_test & (anomaly | knownA));
r.und_new_attacks = sum(new_test)-sum(new_test & (anomaly | knownA));
r.und_attack = sum(yb_test) - sum(yb_test & (anomaly | knownA));

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

novel_attacks=[0,1,3,8,15];
r.class_sum=[];
for i=1:max(y_test)+1
    j = i-1;
    unk = find(novel_attacks==j);
    pnorm = sum(y_test==j & legit==1);
    pkA = sum(y_test==j & knownA==1);
    po = sum(y_test==j & anomaly==1 & new_class==0);
    pnc = sum(y_test==j & new_class==1);
    fprintf("Class: %d, Attack? %d, Known? %d, total: %d, pred_norm: %d, pred_knownA: %d, pred_anom: %d, pred_new-class: %d \n", j, j~=11, isempty(unk), sum(y_test==j), pnorm, pkA, po, pnc);
    cs = [j, j~=11, isempty(unk), sum(y_test==j), pnorm, pkA, po, pnc];
    r.class_sum = [r.class_sum; cs];
end

%[ indexo ] = performance (ranko', out_test);

%num_outs = sum(out_test);
%a= y_pred~=y_test;
%num_missclas = sum(a);
%B(:,1)=y_pred(a);
%B(:,2)=y_test(a);
%B(:,3)=ranko(a);
%D(:,1)=y_pred(out_test==1);
%D(:,2)=y_test(out_test==1);
%D(:,3)=ranko(out_test==1);

%r.test_mean_TN = mean(ranko(out_test==0));
%r.test_std_TN = std(ranko(out_test==0));
%r.test_mean_MissKnown = mean(ranko(out_test==0 & a==1));
%r.test_std_MissKnown = std(ranko(out_test==0 & a==1));
%r.test_mean_newA = mean(ranko(out_test==1));
%r.test_std_newA = std(ranko(out_test==1));

%r.train_mean_A = mean(y(y_train~=11));
%r.train_std_A = std(y(y_train~=11));
%r.train_mean_N = mean(y(y_train==11));
%r.train_std_N = std(y(y_train==11));

end

