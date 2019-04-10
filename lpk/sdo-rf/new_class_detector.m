
function [p, outlier_rec] = new_class_detector(sample, i, outlier_rec, conf)
% NEW_CLASS_DETECTOR generates final, binry labels,
% based on the inputs: TBattack, TBclass and SDOrank
% it generates the outputs: anomaly, new_class, legitimate, known-attack
%
% FIV, Mar 2019

x = sample.x;
TBattack = sample.TBattack;
TBclass = sample.TBclass;
SDOrank = sample.SDOrank;

anom_id = outlier_rec.anom_id; 
anom_pos = outlier_rec.anom_pos; 
anomalies_set = outlier_rec.anomalies_set;
an_count = outlier_rec.an_count;

anom_tol = conf.anom_tol;
min_group = conf.min_group;
max_normal = conf.max_normal;
max_attack = conf.max_attack;

anomaly=0;
legit=0;
new_class=0;
known_attack=0;

new_class_found=0;

                        
if (TBattack==0) % if predicted as non-attack
        if (SDOrank<max_normal)
            legit=1;
        else % if predicted as outlier
            if (an_count>=1)
                a = pdist2(x,anomalies_set);
                b = find(a<anom_tol);
                if length(b)>1
                    anom_id(i) = anom_id(anom_pos(b(1)));
                    if (sum(anom_id==anom_id(i))==min_group)
                        new_class=1;
                        new_class_found=1;
                    elseif (sum(anom_id==anom_id(i))>min_group)
                        new_class=1;
                    end
                else
                    anom_id(i) = max(anom_id)+1;
                end
            else
                anom_id(i)=1;                
            end
            anomaly=1;
            anomalies_set=[anomalies_set;x];
            an_count=an_count+1;
            anom_pos(an_count)=i;
        end
    else
        if (SDOrank<max_attack)
            known_attack=1;
        else
            if (an_count>=1)
                a = pdist2(x,anomalies_set);
                b = find(a<anom_tol);
                if length(b)>1
                    anom_id(i) = anom_id(anom_pos(b(1)));
                    if (sum(anom_id==anom_id(i))==min_group)
                        new_class=1;
                        new_class_found=1;
                    elseif (sum(anom_id==anom_id(i))>min_group)
                        new_class=1;
                    end
                else
                    anom_id(i) = max(anom_id)+1;
                end
            else
                anom_id(i)=1;                
            end
            anomaly=1;
            anomalies_set=[anomalies_set;x];
            an_count=an_count+1;
            anom_pos(an_count)=i;
        end
end

p.anomaly = anomaly;
p.new_class = new_class;
p.known_attack = known_attack;
p.legit = legit;
p.new_class_found = new_class_found;

outlier_rec.anom_id = anom_id; 
outlier_rec.anom_pos = anom_pos; 
outlier_rec.anomalies_set = anomalies_set;
outlier_rec.an_count = an_count;

end

