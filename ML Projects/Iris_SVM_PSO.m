%This example of the big code of SVM
%Nonlinear data
clear ; close all; clc
fprintf('Loading and Visualizing Data ...\n')
load('iris.mat'); 
X=data;
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
Nclass=3;

runs=1;
for R=1:runs
clear Tr Temp1 Temp2 xsup ypred Test TestL TrL

y=randperm(size(X,1));
Temp=ceil(size(X,1)*1/2);
Tr=X(y(1,1:Temp),:);
Test=X(y(1,Temp+1:size(X,1)),:);
TrL=Y(y(1,1:Temp),1);
TestL=Y(y(1,Temp+1:size(X,1)),1);
% Validation=X(y(1,size(X,1)*2/3+1:size(X,1)),:);
% ValidationL=Y(y(1,size(X,1)*2/3+1:size(X,1)),1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PSn=10;
swarm_size = PSn;                       % number of the swarm particles
maxIter = 10;                          % maximum number of iterations
inertia = 0.5;
correction_factor1 = 2;
correction_factor2 = 2;
kernel='gaussian';
kerneloption=1; % I think it is Sigma
verbose=0;
lambda=1e-2;

a = 1:PSn/2;
[X1,Y1] = meshgrid(a,a);
C = cat(2,X1',Y1');
D = reshape(C,[],2);
TempV=[];
for i=1:PSn
   Temp1(i)=rand (1)*rand (1)*rand (1)*rand (1)*rand (1)*1000;
   Temp2(i)=rand (1)*rand (1)*rand (1)*rand (1)*rand (1)*1000;
 end
swarm(1:swarm_size,1,1) = Temp1;  %  Penalty parameter (C)
swarm(1:swarm_size,1,2) = Temp2;  %  RBF Kernel (Sigma)
swarm(:,2,:) = 0;                       % set initial velocity for particles
swarm(:,4,1) = 100;                    % set the best value so far

plotObjFcn = 1;   
% Initialize the population/solutions
for iter = 1:maxIter
    Temp1 = (swarm(:, 1, 1) + swarm(:, 2, 1));       %update x position with the velocity
    Temp2 = (swarm(:, 1, 2) + swarm(:, 2, 2));       %update x position with the velocity
   
     for i=1:size(Temp1,1)
       while(Temp1(i,1)<0)
           Temp1(i)=rand (1)*rand (1)*rand (1)*rand (1)*rand (1)*100000;
       end
    end
    
    for i=1:size(Temp2,1)
       while(Temp2(i,1)<0)
           Temp2(i)=rand (1)*rand (1)*rand (1)*rand (1)*rand (1)*100000;
       end
    end
    
    swarm(:, 1, 1) = Temp1;
    swarm(:, 1, 2) = Temp2;
    C = swarm(:, 1, 1);                                         % get the updated position
    lambda = swarm(:, 1, 2); 
    
    for i=1:size(C,1)
        [xsup,w,b,nbsv]=svmmulticlassoneagainstall(Tr,TrL,Nclass,C(i,1),lambda(i,1),kernel,kerneloption,verbose);
        [ypred] = svmmultival(Test,xsup,w,b,nbsv,kernel,kerneloption); 
        Trainingerror=sum(ypred==TestL)*100/size(TestL,1);
        Fitness(i)=100-Trainingerror;
    end
     for ii = 1:swarm_size
        if Fitness(1,ii) < swarm(ii,4,1)
            swarm(ii, 3, 1) = swarm(ii, 1, 1);                  % update best x position,
            swarm(ii, 3, 2) = swarm(ii, 1, 2);                  % update best x position,
            swarm(ii, 4, 1) = Fitness(1,ii);                       % update the best value so far
        end
     end
    [~, gbest] = min(swarm(:, 4, 1));                           % find the best function value in total
    Gbest(iter,1)=min(Fitness);
    Gworst(iter,1)=max(Fitness);
    
     for i = 1 : swarm_size
        swarm(i, 2, 1) = rand*inertia*swarm(i, 2, 1) + correction_factor1*rand*(swarm(i, 3, 1) - swarm(i, 1, 1)) + correction_factor2*rand*(swarm(gbest, 3, 1) - swarm(i, 1, 1));   %x velocity component
        swarm(i, 2, 2) = rand*inertia*swarm(i, 2, 2) + correction_factor1*rand*(swarm(i, 3, 2) - swarm(i, 1, 2)) + correction_factor2*rand*(swarm(gbest, 3, 2) - swarm(i, 1, 2));   %y velocity component
     end
    
     clf;plot(swarm(:, 3, 1),swarm(:, 3, 2), 'bx');             % drawing swarm movements
%     axis([0 size(Tr,1)+5 0 size(Tr,1)+5]);
    pause(.1);                                                 % un-comment this line to decrease the animation speed
    disp(['iteration: ' num2str(iter)]);    
end

% Output/display
disp(['Number of evaluations: ',num2str(iter)]);
disp(['Best value of Acc=',num2str(min(100-Gbest))]);
clear xsup;
[~,ind]=min(Gbest);
Best_C=swarm(ind,3,1);
Best_Lamda=swarm(ind,3,2);
[xsup,w,b,nbsv]=svmmulticlassoneagainstall(Tr,TrL,Nclass,Best_C,Best_Lamda,kernel,kerneloption,verbose);
[ypred] = svmmultival(Tr,xsup,w,b,nbsv,kernel,kerneloption); 
Trainingerror=sum(ypred==TrL)*100/size(TrL,1);
disp(['Training Accuracy is ' num2str(Trainingerror)])
Temp1=ypred==TrL;
disp(['Number of misclassified Samples in Training ' int2str(sum(~Temp1))])
TrMis(R)=sum(~Temp1);

[ypred] = svmmultival(Test,xsup,w,b,nbsv,kernel,kerneloption); 
Testingerror=sum(ypred==TestL)*100/size(TestL,1);
disp(['Testing Accuracy is ' num2str(Testingerror)])
Temp1=ypred==TestL;
disp(['Number of misclassified Samples in Testing' int2str(sum(~Temp1))])
TestMis(R)=sum(~Temp1);

% To plot misclassified Samples
% Temp1=ypred==Y;
% disp(['Number of misclassified Samples ' int2str(sum(~Temp1))])
% % TTT=unique(xsup);
disp(['Number of support vectors is ' int2str(size(xsup,1))])

TrAcc(R)=Trainingerror;
TestAcc(R)=Testingerror;
SV(R)=size(xsup,1);
end 

[X,Y] = meshgrid(0.01:0.1:10,0.01:0.05:10);
for i=1:size(X,1)
    for j=1:size(X,2)
        [xsup,w,b,nbsv]=svmmulticlassoneagainstall(Tr,TrL,Nclass,X(i,j),Y(i,j),kernel,kerneloption,verbose);
        [ypred] = svmmultival(Tr,xsup,w,b,nbsv,kernel,kerneloption); 
        Trainingerror(i,j)=sum(ypred==TrL)*100/size(TrL,1);        
        [ypred2] = svmmultival(Test,xsup,w,b,nbsv,kernel,kerneloption); 
        Testingerror(i,j)=sum(ypred2==TestL)*100/size(TestL,1);        
        SV(i,j)=size(xsup,1);
        disp([int2str(i) ' and ' int2str(j)])
    end
end


NN=20
[c,h]=contour(X(1:NN,1:NN),Y(1:NN,1:NN),100-Trainingerror(1:NN,1:NN)); hold on
clabel(c,h);
grid;
xlabel('x1');
ylabel('x2');
NN=50
surf(X(1:NN,1:NN), Y(1:NN,1:NN),100-Trainingerror(1:NN,1:NN))

[c,h]=contour(X,Y,100-Testingerror); hold on
clabel(c,h);
grid;
xlabel('x1');
ylabel('x2');
NN=50
surf(X(1:NN,1:NN), Y(1:NN,1:NN),100-Testingerror(1:NN,1:NN))

[c,h]=contour(X,Y,SV); hold on
clabel(c,h);
grid;
xlabel('x1');
ylabel('x2');


%Plot Heat Map of Support vectors
clf
heatmap(SV(1:50,:), [(0.01:0.05:10)], [(0.01:0.05:10)]);
xlabel('C')
ylabel('Sigma');

%Plot Heat Map of Misclassified Samples (Training)
clf
heatmap(100-Trainingerror(1:50,1:50), [(0.01:0.05:10)], [(0.01:0.05:10)]);
xlabel('C')
ylabel('Sigma');
    
%Plot Heat Map of Misclassified Samples (Testing)
clf
heatmap(Testingerror(1:20,1:20), [(0.01:0.05:10)], [(0.01:0.05:10)]);
xlabel('C')
ylabel('Sigma');
    
