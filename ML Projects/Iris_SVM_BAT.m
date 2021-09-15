%This example to search for the SVM parameters using BAT algorithm
%iris dataset
clear ; close all; clc
fprintf('Loading and Visualizing Data ...\n')
% load('iris.mat'); 
% X=data;
% Nclass=3;
% Data sets 1)iris   2)iono
[X,Y,Nclass]=SelectDataSet('iris');
runs=2;
for R=1:runs
% clear Tr Temp1 Temp2 xsup ypred Test TestL TrL
clear Sol S best fmin Fitness Fnew t
y=randperm(size(X,1));
Temp=ceil(size(X,1)*1/2);
Tr=X(y(1,1:Temp),:);
Test=X(y(1,Temp+1:size(X,1)),:);
TrL=Y(y(1,1:Temp),1);
TestL=Y(y(1,Temp+1:size(X,1)),1);

n=20;      % Population size, typically 10 to 40
N_gen= 20;  % Number of generations
A=0.5;      % Loudness  (constant or decreasing)
r=0.5;      % Pulse rate (constant or decreasing)
Qmin=0;         % Frequency minimum
Qmax=2; 
N_iter=0;
d=2;
Lb=0.01*ones(1,d);
Ub=1000*ones(1,d);
% Initializing arrays
Q=zeros(n,1);   % Frequency
v=zeros(n,d);   % Velocities
kernel='gaussian';
% kerneloption=1; % I think it is Sigma
verbose=0;
lambda=1e-2;

%To prepare for drawing bats
a = 1:n/2;
[X1,Y1] = meshgrid(a,a);
C = cat(2,X1',Y1');
D = reshape(C,[],2);
TempV=[];

% Initialize the population/solutions
for i=1:n,
    Sol(i,:)=Lb+(Ub-Lb).*rand (1,d).*rand (1,d).*rand (1,d).*rand (1,d).*rand (1,d).*rand(1,d).*rand (1,d);
    [xsup,w,b,nbsv]=svmmulticlassoneagainstall(Tr,TrL,Nclass,Sol(i,1),lambda,kernel,Sol(i,2),verbose);
    [ypred] = svmmultival(Test,xsup,w,b,nbsv,kernel,Sol(i,2)); 
    Trainingerror=sum(ypred==TestL)*100/size(TestL,1);
    Fitness(i)=100-Trainingerror;
end
clf;plot(Sol(:,1),Sol(:,2), 'bx'); 

[fmin,I]=min(Fitness);
best=Sol(I,:);

for t=1:N_gen, 
% Loop over all bats/solutions
        for i=1:n,
          Q(i)=Qmin+(Qmin-Qmax)*rand;
          v(i,:)=v(i,:)+(Sol(i,:)-best)*Q(i);
          S(i,:)=Sol(i,:)+v(i,:); 
          % Check for limits
              for jj=1:d
                  while  (S(i,jj)<Lb(jj) || S(i,jj)>Ub(jj))
                      S(i,jj)=Lb(jj)+(Ub(jj)-Lb(jj)).*rand (1).*rand (1).*rand (1).*rand (1).*rand (1).*rand(1).*rand (1);
                  end
              end
                   
          % Apply simple bounds/limits
%           Sol(i,:)=simplebounds(Sol(i,:),Lb,Ub);
          % Pulse rate
          if rand>r
          % The factor 0.001 limits the step sizes of random walks 
              S(i,1)=best(1,1)+0.01*rand(1);
              S(i,2)=best(1,2)+0.0010*rand(1);              
          end
          % Evaluate new solutions
          [xsup,w,b,nbsv]=svmmulticlassoneagainstall(Tr,TrL,Nclass,S(i,1),lambda,kernel,S(i,2),verbose);
          [ypred] = svmmultival(Tr,xsup,w,b,nbsv,kernel,S(i,2)); 
          Trainingerror=sum(ypred==TrL)*100/size(TrL,1);
          Fnew=100-Trainingerror;
          % Update if the solution improves, or not too loud
          if (Fnew<Fitness(i)) ,
              Sol(i,:)=S(i,:);
              Fitness(i)=Fnew;
          end
          % Update the current best solution
          if (Fnew<fmin && rand<A)
              best=S(i,:);
              fmin=Fnew;
              A=0.9*A;
              r=0.7*(1-exp(-0.8*t)); % here gama=0.8 and r starts with 0.5
              disp(['A= ' num2str(A)])
              disp(['r= ' num2str(r)])
              disp(['t= ' num2str(t)])
          end
        end
        
clf;plot(Sol(:,1),Sol(:,2), 'bx'); hold on
plot(best(:,1),best(:,2), 'rx'); 
pause(0.1);                              
N_iter=N_iter+n;
disp(['Iteration number ' int2str(t)])
end
% Output/display
% disp(['Number of evaluations: ',num2str(iter)]);
disp(['Best value of C=',num2str(best),' fmin=',num2str(100-fmin)]);
clear xsup;
Best_C=best(1,1);
Best_Kernel=best(1,2);
[xsup,w,b,nbsv]=svmmulticlassoneagainstall(Tr,TrL,Nclass,Best_C,lambda,kernel,Best_Kernel,verbose);
[ypred] = svmmultival(Tr,xsup,w,b,nbsv,kernel,Best_Kernel); 
Trainingerror=sum(ypred==TrL)*100/size(TrL,1);
disp(['Training Accuracy is ' num2str(Trainingerror)])
Temp1=ypred==TrL;
disp(['Number of misclassified Samples in Training ' int2str(sum(~Temp1))])
TrMis(R,1)=sum(~Temp1);
TrAcc(R,1)=Trainingerror;

[ypred] = svmmultival(Test,xsup,w,b,nbsv,kernel,Best_Kernel); 
Testingerror=sum(ypred==TestL)*100/size(TestL,1);
disp(['Testing Accuracy is ' num2str(Testingerror)])
Temp1=ypred==TestL;
disp(['Number of misclassified Samples in Testing' int2str(sum(~Temp1))])
TestMis(R,1)=sum(~Temp1);
TestAcc(R,1)=Testingerror;
Best(R,:)=best;
% To plot misclassified Samples
% Temp1=ypred==Y;
% disp(['Number of misclassified Samples ' int2str(sum(~Temp1))])
% % TTT=unique(xsup);
disp(['Number of support vectors is ' int2str(size(xsup,1))])

SV(R,1)=size(xsup,1);
end
mean(TrAcc)
std(TrAcc)
mean(TestAcc)
std(TestAcc)

mean(SV)
std(SV)
%         BEST(counter,:)=100-fmin;

% [X,Y] = meshgrid(0.01:0.1:10,0.01:0.05:10);
% for i=1:size(X,1)
%     for j=1:size(X,2)
%         [xsup,w,b,nbsv]=svmmulticlassoneagainstall(Tr,TrL,Nclass,X(i,j),Y(i,j),kernel,kerneloption,verbose);
%         [ypred] = svmmultival(Tr,xsup,w,b,nbsv,kernel,kerneloption); 
%         Trainingerror(i,j)=sum(ypred==TrL)*100/size(TrL,1);        
%         [ypred2] = svmmultival(Test,xsup,w,b,nbsv,kernel,kerneloption); 
%         Testingerror(i,j)=sum(ypred2==TestL)*100/size(TestL,1);        
%         SV(i,j)=size(xsup,1);
%         disp([int2str(i) ' and ' int2str(j)])
%     end
% end
% 
% 
% NN=20
% [c,h]=contour(X(1:NN,1:NN),Y(1:NN,1:NN),100-Trainingerror(1:NN,1:NN)); hold on
% clabel(c,h);
% grid;
% xlabel('x1');
% ylabel('x2');
% NN=50
% surf(X(1:NN,1:NN), Y(1:NN,1:NN),100-Trainingerror(1:NN,1:NN))
% 
% [c,h]=contour(X,Y,100-Testingerror); hold on
% clabel(c,h);
% grid;
% xlabel('x1');
% ylabel('x2');
% NN=50
% surf(X(1:NN,1:NN), Y(1:NN,1:NN),100-Testingerror(1:NN,1:NN))
% 
% [c,h]=contour(X,Y,SV); hold on
% clabel(c,h);
% grid;
% xlabel('x1');
% ylabel('x2');
% 
% 
% %Plot Heat Map of Support vectors
% clf
% heatmap(SV(1:50,:), [(0.01:0.05:10)], [(0.01:0.05:10)]);
% xlabel('C')
% ylabel('Sigma');
% 
% %Plot Heat Map of Misclassified Samples (Training)
% clf
% heatmap(100-Trainingerror(1:50,1:50), [(0.01:0.05:10)], [(0.01:0.05:10)]);
% xlabel('C')
% ylabel('Sigma');
%     
% %Plot Heat Map of Misclassified Samples (Testing)
% clf
% heatmap(Testingerror(1:20,1:20), [(0.01:0.05:10)], [(0.01:0.05:10)]);
% xlabel('C')
% ylabel('Sigma');
%     
