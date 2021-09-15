%This example to search for the SVM parameters using BAT algorithm
clear ; close all; clc
fprintf('Loading and Visualizing Data ...\n')
[X,Y,Nclass]=SelectDataSet('iris');
N_folds=5;
Accuracy_fold=zeros(N_folds,1);
for fold_N = 1:N_folds 
    clear Sol S best fmin Fitness Fnew t
    y=randperm(size(X,1));    X=X(y,:);
    Y=Y(y);
    N=size(X,1); %N=Numeber of Samples;
    n1=ceil(N/N_folds); % find the size of the testing data sets
    last=N-(N_folds-1)*n1;% find the size of the last set (if any)
    if last==0,
        last=n1; % N_folds divides N, all pieces are the same
    end
    if last<n1/2, % if the last piece is smaller than
        % half of the size of the others, % then issue a warning
        fprintf('%s\n','Warning: imbalanced testing sets')
    end
    v=[]; % construct indicator-labels for the N_folds subsets
    for i=1:N_folds-1;
      v=[v;ones(n1,1)*i];
    end
    v=[v;ones(last,1)*N_folds];

    L=v==fold_N;

    TrainPatterns=X(~L,:); % training data
    TrainTargets=Y(~L,:); % training labels

    TestPatterns=X(L,:); % test data
    TestTargets=Y(L,:);
    
    n=10;      % Population size, typically 10 to 40
    N_gen= 50;  % Number of generations
    A=0.5;      % Loudness  (constant or decreasing)
    r=0.5;      % Pulse rate (constant or decreasing)
    Qmin=0;         % Frequency minimum
    Qmax=2; 
    N_iter=0; 
    d=2;
    %     Lb=0.01*ones(1,d);
    Lb=[0.01,0.0001];
    %     Ub=1000*ones(1,d);
    Ub=[1000,32];
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
        [xsup,w,b,nbsv]=svmmulticlassoneagainstall(TrainPatterns,TrainTargets,Nclass,Sol(i,1),lambda,kernel,Sol(i,2),verbose);
        [ypred] = svmmultival(TrainPatterns,xsup,w,b,nbsv,kernel,Sol(i,2)); 
        Trainingerror=sum(ypred==TrainTargets)*100/size(TrainTargets,1);
        Fitness(i)=100-Trainingerror;
    end
    
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
                    S(i,jj)=Lb(jj)+(Ub(jj)-Lb(jj)).*rand (1);
                end
            end
            
            % Pulse rate
            if rand>r
                % The factor 0.001 limits the step sizes of random walks 
                S(i,:)=best+0.01*rand(1);            S(i,2)=best(1,2)+0.0010*rand(1);              
            end
            
            % Evaluate new solutions
            [xsup,w,b,nbsv]=svmmulticlassoneagainstall(TrainPatterns,TrainTargets,Nclass,S(i,1),lambda,kernel,S(i,2),verbose);
            [ypred] = svmmultival(TrainPatterns,xsup,w,b,nbsv,kernel,S(i,2)); 
            Trainingerror=sum(ypred==TrainTargets)*100/size(TrainTargets,1);
            Fnew=100-Trainingerror;
            % Update if the solution improves, or not too loud
            if (Fnew<Fitness(i)) ,
                Sol(i,:)=S(i,:);
                Fitness(i)=Fnew;
            end
            
            % Update the current best solution
            if (Fnew<fmin)
                best=S(i,:);
                fmin=Fnew;
%                 A=0.9*A;
%                 r=r*(1-exp(-0.9*t)); % here gama=0.8 and r starts with 0.5
            end
        end
        N_iter=N_iter+n;
        disp(['Iteration number ' int2str(t)])
    end
    disp(['Best value of C=',num2str(best),' fmin=',num2str(100-fmin)]);
    % clear xsup;
    Best_C=best(1,1);
    Best_Kernel=best(1,2);
    [xsup,w,b,nbsv]=svmmulticlassoneagainstall(TrainPatterns,TrainTargets,Nclass,Best_C,lambda,kernel,Best_Kernel,verbose);
    % [ypred] = svmmultival(Tr,xsup,w,b,nbsv,kernel,Best_Kernel); 
    % Trainingerror=sum(ypred==TrL)*100/size(TrL,1);
    % disp(['Training Accuracy is ' num2str(Trainingerror)])
    % Temp1=ypred==TrL;
    % disp(['Number of misclassified Samples in Training ' int2str(sum(~Temp1))])
    % TrMis(R,1)=sum(~Temp1);
    % TrAcc(R,1)=Trainingerror;
    
    [ypred] = svmmultival(TestPatterns,xsup,w,b,nbsv,kernel,Best_Kernel); 
    Testingerror=sum(ypred==TestTargets)*100/size(TestTargets,1);
    disp(['Testing Accuracy is ' num2str(Testingerror)])
    TestAcc(fold_N,1)=Testingerror;
    Best(fold_N,:)=best;
end
mean(TestAcc)
std(TestAcc)

