clc;
clear all;
close all
C1=[1 2 3 2 4 3 5 6;
    1 1 4 5 2 6 3 1]
Y1=[1 1 1 1 1 1 1 1]

C2=[6 6 7 5 4 6 3 1;
    3 6 2 4 5 2 5 5]
Y2=[0 0 0 0 0 0 0 0]
C=[C1 C2]
Y=[Y1 Y2]
plotData(C', Y')
Y2=[2 2 2 2 2 2 2 2]
Y=[Y1 Y2]




n=4      % Population size, typically 10 to 40
N_gen=1;  % Number of generations
A=0.5;      % Loudness  (constant or decreasing)
r=0.5;

Qmin=0;         % Frequency minimum
Qmax=2;         % Frequency maximum
% Iteration parameters
N_iter=0;       % Total number of function evaluations
% Dimension of the search variables
d=2;           % Number of dimensions 
% Lower limit/bounds/ a vector
Nclass=2;

Lb=0.01*ones(1,d);
Ub=1000*ones(1,d);
% Initializing arrays
Q=zeros(n,1);   % Frequency
v=zeros(n,d);   % Velocities

sigma = 0.1;
kernel='gaussian';
kerneloption=1; % I think it is Sigma
verbose=0;
lambda=1e-2;

for i=1:n,
    Sol(i,:)=Lb+(Ub-Lb).*rand (1,d).*rand (1,d).*rand (1,d).*rand (1,d).*rand (1,d).*rand(1,d).*rand (1,d);
    [xsup,w,b,nbsv]=svmmulticlassoneagainstall(C',Y',Nclass,Sol(i,1),Sol(i,2),kernel,kerneloption,verbose);
    [ypred] = svmmultival(C',xsup,w,b,nbsv,kernel,kerneloption); 
    Trainingerror=sum(ypred==Y')*100/size(Y',1);
    Fitness(i)=100-Trainingerror;
end


[fmin,I]=min(Fitness);
best=Sol(I,:);

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
              S(i,:)=best+0.001*randn(1,d);
          end
          % Evaluate new solutions
          [xsup,w,b,nbsv]=svmmulticlassoneagainstall(C',Y',Nclass,S(i,1),S(i,2),kernel,kerneloption,verbose);
          [ypred] = svmmultival(C',xsup,w,b,nbsv,kernel,kerneloption); 
          Trainingerror=sum(ypred==Y')*100/size(Y',1);
          Fnew=100-Trainingerror;
          % Update if the solution improves, or not too loud
          if (Fnew<Fitness(i)) & (rand<A) ,
              Sol(i,:)=S(i,:);
              Fitness(i)=Fnew;
          end
          % Update the current best solution
          if Fnew<=fmin,
              best=S(i,:);
              fmin=Fnew;
          end
          XX=Y;    
  XX(XX==2)=0;
%   plotData(C', XX')
  figure,
visualizeBoundary(C', XX',xsup,w,b,nbsv,kernel,kerneloption);
pause();
plot(xsup(:,1),xsup(:,2),'rs');

 end
%     XX=Y;    
%   XX(XX==2)=0;
% %   plotData(C', XX')
%   figure,
% visualizeBoundary(C', XX',xsup,w,b,nbsv,kernel,kerneloption);

