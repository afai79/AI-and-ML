%This example of the big code of SVM
%Nonlinear data
clear ; close all; clc
fprintf('Loading and Visualizing Data ...\n')
load('iris.mat'); 
X=data;
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
Nclass=3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=4;      % Population size, typically 10 to 40
N_gen= 5;  % Number of generations
A=0.5;      % Loudness  (constant or decreasing)
r=0.5;      % Pulse rate (constant or decreasing)
% set the position of the initial swarm
Qmin=0;         % Frequency minimum
Qmax=2; 
N_iter=0;
d=1;
Lb=0.01*ones(1,d);
Ub=1000*ones(1,d);
% Initializing arrays
Q=zeros(n,1);   % Frequency
v=zeros(n,d);   % Velocities
sigma = 0.1;
kernel='gaussian';
kerneloption=10; % I think it is Sigma
verbose=0;
lambda=1e-2;
% Initialize the population/solutions
for i=1:n,
    Sol(i,:)=Lb+(Ub-Lb).*rand (1,d).*rand (1,d).*rand (1,d).*rand (1,d).*rand (1,d).*rand(1,d).*rand (1,d);
    [xsup,w,b,nbsv]=svmmulticlassoneagainstall(X,Y,Nclass,Sol(i,1),lambda,kernel,kerneloption,verbose);
    [ypred] = svmmultival(X,xsup,w,b,nbsv,kernel,kerneloption); 
    Trainingerror=sum(ypred==Y)*100/size(Y,1);
    Fitness(i)=100-Trainingerror;
end
% Find the initial best solution
[fmin,I]=min(Fitness);
best=Sol(I,:);
for t=1:N_gen, 
% Loop over all bats/solutions
        for i=1:n,
          Q(i)=Qmin+(Qmin-Qmax)*rand;
          v(i,:)=v(i,:)+(Sol(i,:)-best)*Q(i);
          S(i,:)=Sol(i,:)+v(i,:);
          % Apply simple bounds/limits
          Sol(i,:)=simplebounds(Sol(i,:),Lb,Ub);
          % Pulse rate
          if rand>r
          % The factor 0.001 limits the step sizes of random walks 
              S(i,:)=best+0.001*randn(1,d);
          end
          % Evaluate new solutions
          [xsup,w,b,nbsv]=svmmulticlassoneagainstall(X,Y,Nclass,S(i,:),lambda,kernel,kerneloption,verbose);
          [ypred] = svmmultival(X,xsup,w,b,nbsv,kernel,kerneloption); 
          Trainingerror=sum(ypred==Y)*100/size(Y,1);
          Fnew=100-Trainingerror;
          % Update if the solution improves, or not too loud
          if (Fnew<=Fitness(i)) & (rand<A) ,
              Sol(i,:)=S(i,:);
              Fitness(i)=Fnew;
          end
          % Update the current best solution
          if Fnew<=fmin,
              best=S(i,:);
              fmin=Fnew;
          end
        end
        N_iter=N_iter+n;
        disp(['Iteration number ' int2str(t)])
end
% Output/display
disp(['Number of evaluations: ',num2str(N_iter)]);
disp(['Best value of C=',num2str(best),' fmin=',num2str(100-fmin)]);
clear xsup;
[xsup,w,b,nbsv]=svmmulticlassoneagainstall(X,Y,Nclass,best,lambda,kernel,kerneloption,verbose);
[ypred] = svmmultival(X,xsup,w,b,nbsv,kernel,kerneloption); 
Trainingerror=sum(ypred==Y)*100/size(Y,1);

%      Trainingerror=sum(ypred==YY)*100/size(Y,1);
disp(['Training Accuracy is ' num2str(Trainingerror)])
% To plot misclassified Samples
Temp1=ypred==Y;
% plot(X(~Temp1,1),X(~Temp1,2),'rs', 'MarkerSize',10), hold on
disp(['Number of misclassified Samples ' int2str(sum(~Temp1))])
% TTT=unique(xsup);
disp(['Number of support vectors is ' int2str(size(xsup,1))])
