clear ; close all; clc
    counter=1;
    tic;
fprintf('Loading and Visualizing Data ...\n')
load('iris.mat'); 
m=min(data);
mx=max(data);
Temp1=(data-repmat(m,size(data,1),1));
Temp2=repmat((mx-m),size(data,1),1);
D=Temp1./Temp2;

hist(data(:,1))
xlabel('Feature values')
ylabel('Fequency')
legend('First Feature','Second Feature','Third Feature','Fourth Feature')