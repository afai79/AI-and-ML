%This example of the big code of SVM
%Nonlinear data
clear ; close all; clc
fprintf('Loading and Visualizing Data ...\n')

% Data sets 1)iris   2)iono 3) Liver 4) ORL   5)Yale  6)Sonar 7)Ovarian 8)Wine
% 9)Diabetes 10)Breast 11)TicTacToe 12)Glass

[X,Y,Nclass]=SelectDataSet('Sonar');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y=randperm(size(X,1));
Temp=ceil(size(X,1)*2/3);
Tr=X(y(1,1:Temp),:);
Test=X(y(1,Temp+1:size(X,1)),:);
TrL=Y(y(1,1:Temp),1);
TestL=Y(y(1,Temp+1:size(X,1)),1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PSn=10;
swarm_size = PSn;                       % number of the swarm particles
maxIter = 20;                          % maximum number of iterations
inertia = 1.0;
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
   Temp1(i)=rand (1)*rand (1)*rand (1)*1000;
   Temp2(i)=rand (1)*rand (1)*rand (1)*1000;
 end
swarm(1:swarm_size,1,1) = Temp1;  %  Penalty parameter (C)
swarm(1:swarm_size,1,2) = Temp2;  %  RBF Kernel (Sigma)
swarm(:,2,:) = 0;                       % set initial velocity for particles
swarm(:,4,1) = 100;                    % set the best value so far

plotObjFcn = 1;   
% Initialize the population/solutions
for iter = 1:maxIter
    Temp1 = ((swarm(:, 1, 1) + swarm(:, 2, 1)/1.3));       %update x position with the velocity
    Temp2 = ((swarm(:, 1, 2) + swarm(:, 2, 1)/1.3));       %update x position with the velocity
   
     for i=1:size(Temp1,1)
       while(Temp1(i,1)<0)
           Temp1(i)=rand (1)*rand (1)*rand (1)*1000;
       end
    end
    
    for i=1:size(Temp2,1)
       while(Temp2(i,1)<0)
           Temp2(i)=rand (1)*rand (1)*rand (1)*1000;
       end
    end
    
     swarm(:, 1, 1) = Temp1;
    swarm(:, 1, 2) = Temp2;
    C = swarm(:, 1, 1);                                         % get the updated position
    lambda = swarm(:, 1, 2); 
    
    for i=1:size(C,1)
        [xsup,w,b,nbsv]=svmmulticlassoneagainstall(Tr,TrL,Nclass,C(i,1),lambda(i,1),kernel,kerneloption,verbose);
        [ypred] = svmmultival(Tr,xsup,w,b,nbsv,kernel,kerneloption); 
        Trainingerror=sum(ypred==TrL)*100/size(TrL,1);
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
disp(['Best value of Acc=',num2str(min(Gbest))]);
clear xsup;
[~,ind]=min(Gbest)
Best_C=swarm(ind,3,1)
Best_Lamda=swarm(ind,3,2)
[xsup,w,b,nbsv]=svmmulticlassoneagainstall(Tr,TrL,Nclass,Best_C,Best_Lamda,kernel,kerneloption,verbose);
[ypred] = svmmultival(Tr,xsup,w,b,nbsv,kernel,kerneloption); 
Trainingerror=sum(ypred==TrL)*100/size(TrL,1);
disp(['Training Accuracy is ' num2str(Trainingerror)])
Temp1=ypred==TrL;
disp(['Number of misclassified Samples in Training' int2str(sum(~Temp1))])

[ypred] = svmmultival(Test,xsup,w,b,nbsv,kernel,kerneloption); 
Testingerror=sum(ypred==TestL)*100/size(TestL,1);
disp(['Testing Accuracy is ' num2str(Testingerror)])
Temp1=ypred==TestL;
disp(['Number of misclassified Samples in Testing' int2str(sum(~Temp1))])

% To plot misclassified Samples
% Temp1=ypred==Y;
% disp(['Number of misclassified Samples ' int2str(sum(~Temp1))])
% % TTT=unique(xsup);
disp(['Number of support vectors is ' int2str(size(xsup,1))])
