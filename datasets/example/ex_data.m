
rng(15)
x1 = normrnd(5,0.8,[995,2]);
x2 = normrnd(2,3,[1000,2]);
x2(:,1)=x2(:,1)+20;
x2(:,2)=x2(:,2)+20;
out = [1,20; 7,14; 25,2; 34,8; 5,25];
x=[x1;x2];
training = [x;out];
yk=[zeros(1,1995),ones(1,5)];
dt=[3*ones(1,995),5*ones(1,1000),ones(1,5)];
ct=[ones(1,995),2*ones(1,1000),zeros(1,5)];
a = randperm(2000);
tr=training(a,:);
ykf=yk(a);
dtf=dt(a);
ctf=ct(a);

x=[];
x1 = normrnd(5,0.8,[900,2]);
x2 = normrnd(2,3,[900,2]);
x2(:,1)=x2(:,1)+20;
x2(:,2)=x2(:,2)+20;
x3 = normrnd(3,2,[95,2]);
x4 = normrnd(2,0.3,[95,2]);
x3(:,1)=x3(:,1)+10;
x4(:,2)=x4(:,2)+15;
x=[x1;x2;x3;x4];
out = [19,38; 30,4; -2,-6; 2,33; 10,30; 33,37; 28,-7; 0,-1; -5,10; 13,35];
eval = [x;out];
ye=[zeros(1,1800),ones(1,200)];
de=[3*ones(1,900),5*ones(1,900),2*ones(1,95),4*ones(1,95),ones(1,10)];
ce=[ones(1,900),2*ones(1,900),3*ones(1,95),4*ones(1,95),zeros(1,10)];
b = randperm(2000);
ev=eval(b,:);
yef=ye(b);
def=de(b);
cef=ce(b);
dens = [dtf,def]';
clus = [ctf,cef]';
fdens = dens;


figure(1)
scatter(tr(:,1),tr(:,2),10, ykf, 'filled');
figure(2)
scatter(ev(:,1),ev(:,2),10, yef, 'filled');

data=[[tr;ev],[ykf';yef']];

figure(3)
scatter(data(:,1),data(:,2),10, dens, 'filled');

a = data(dens==3,1:2);
ca = median(a);
D = pdist2(a,ca,'euclidean');
coef = mean(D)+2*std(D);
la = 13*ones(1,size(a,1));
la(D<coef) = 23;
fdens(dens==3)=la;

a = data(dens==5,1:2);
ca = median(a);
D = pdist2(a,ca,'euclidean');
coef = mean(D)+1.5*std(D);
la = 15*ones(1,size(a,1));
la(D<coef) = 25;
fdens(dens==5)=la;

dens(:)=1;
dens(fdens==23)=4;
dens(fdens==25)=3;
dens(fdens==13)=3;
dens(fdens==4)=4;
dens(fdens==15)=2;
dens(fdens==2)=2;

figure(4);
scatter(data(dens==4,1),data(dens==4,2),10, 'k', 'filled'); hold on;
scatter(data(dens==3,1),data(dens==3,2),10, 'b', 'filled');
scatter(data(dens==2,1),data(dens==2,2),10, 'g', 'filled');
scatter(data(dens==1,1),data(dens==1,2),10, 'r', 'filled');

figure(5);
scatter(data(clus==4,1),data(clus==4,2),10, 'm', 'filled'); hold on;
scatter(data(clus==3,1),data(clus==3,2),10, 'b', 'filled');
scatter(data(clus==2,1),data(clus==2,2),10, 'g', 'filled');
scatter(data(clus==1,1),data(clus==1,2),10, 'r', 'filled');
scatter(data(clus==0,1),data(clus==0,2),10, 'k', 'filled');


data=[data,dens,clus];
csvwrite('ex_data.csv',data);

clear a b eval x1 x2 x3 x4 out x training eval ye yk dt ca coef D fdens la de def dtf