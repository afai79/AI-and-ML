clc
clear all
close all;
%% Loading dataset
load('Tumorigenic.mat')
% X=[2,4,5,10,11,12,13,18,19,24,27]; % Quick Tumorigenic
% X=[4,5,10,11,12,13,18,19,20,22,24]; % Matrix Tumorigenic
X=[4,5,6,8,9,10, 12, 13,18 19,24,30]; % Entropy Tumorigenic
% X=1:31

%%%%Mutagenic
% load('Mutagenic.mat')
% X=[1,4,7,8,10,11,12,18,19,20,24,25,30]; % Quick Mutagenic
% X=[4,5,7,10,11,12,13,19,22]; % Matrix Mutagenic
% X=[5,6,7,10,11,12,14,19,22,24,30]; % Entropy Mutagenic

%%%%%Irritant SMOTE=5
% load('Irritant.mat')
% X=[4,5, 6,7, 8,10,11,12,18,19,20, 24,25, 29]; % Quick Irritant
% X=[5,7,8,11,12,13,18,19,20,22, 24]; % Matrix Irritant
% X=[4,5,7,8,10,11,12,13,18,19,20,24,30]; % Entropy Irritant
%%

%%%%ReproductiveEffective SMOTE==1
% load('ReproductiveEffective.mat')
% X=[1,2,4,8,10,11,12,13,19,22,24,25,26,29]; % Quick Irritant
% X=[1,4,5,7,10,11,12,13,18,19,20,22,23,24]; % Matrix Irritant
% X=1:31;%All features Entropy Irritant
%%
%% Feature Selection
Xnew=data(:,[X]);
%%

%% Preprocesssing step
% [Xnew Labels] = SMOTE(Xnew, Labels,0.3);
% Make the data balanced
% Under Sampling
% YY=Labels==1;
% if(sum(YY)~=sum(~YY))
%     if(sum(YY)<sum(~YY))
%         NewLabels1=Labels(YY,1);
%         NewLabels2=Labels(~YY,1);
%         NewLabels2(size(NewLabels1,1)+1:end,:)=[];
%         Labels=[NewLabels1;NewLabels2];
%         
%         Newdata1=Xnew(YY,:);
%         Newdata2=Xnew(~YY,:);
%         Newdata2(size(Newdata1,1)+1:end,:)=[];
%         Xnew=[Newdata1;Newdata2];        
%     end
% end

% OverSampling
%% Make the data balanced (OverSampling)
YY=Labels==1;
data1=Xnew(YY,:);
data2=Xnew(~YY,:);
Lables1=Labels(YY,:);
Lables2=Labels(~YY,:);
Rem=rem(size(data2,1),size(data1,1));
Temp=size(data2,1)-Rem;
Newdata1=[];NewLables1=[];
% for i=1:Temp/size(data1,1)  %% this for can be activated to make the data
% balanced
    Newdata1=[Newdata1;data1];
    NewLables1=[NewLables1;Lables1];
 Newdata1=[Newdata1;data1];
    NewLables1=[NewLables1;Lables1];

% end
Newdata1=[Newdata1;data1(1:Rem,:)];
NewLables1=[NewLables1;Lables1(1:Rem,:)];
Labels=[NewLables1;Lables2];
Xnew=[Newdata1;data2];   
%%
YY=Labels==0;
Labels(YY,1)=2;
Labels(~YY,1)=1;

y=randperm(size(Xnew,1));
Labels=Labels(y);
Xnew=Xnew(y,:);

N_folds=3;
Accuracy_fold_Ensemble=zeros(N_folds,1);

for fold_N = 1:N_folds
    %%% Croos Validation
    %N=size(Patterns,2); %N=Numeber of Samples; %N_folds=Number of partitions
    N=size(Xnew,1); %N=Numeber of Samples;
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
    TrainPatterns=Xnew(~L,:); % training data
    TrainTargets=Labels(~L,:); % training labels
    TestPatterns=Xnew(L,:); % test data
    TestTargets=Labels(L,:);
    
    %% KNN
    mdl = ClassificationKNN.fit(TrainPatterns,TrainTargets,'NumNeighbors',5);
    ypred = predict(mdl,TestPatterns);
    YY=ypred==TestTargets;
    YY1=TestTargets==1;
    YY2=TestTargets==2;
    TP=sum(YY1.*YY);
    TN=sum(YY2.*YY);
    FP=sum(YY1.*~YY);
    FN=sum(YY2.*~YY);
    Sen(fold_N,1)=TP/(TP+FN);
%     FPr=FP/(FP+TN);
%     AUC(fold_N,1)=(1+Sen(fold_N,1)-FPr)/2;
    Spe(fold_N,1)=TN/(TN+FP);
    BAc(fold_N,1)=(TP/(TP+FN)+TN/(TN+FP))/2;
    GM(fold_N,1)=sqrt((TP/(TP+FN))*(TN/(TN+FP)));
    Accuracy(fold_N,1)=(TP+TN)/(TP+TN+FN+FP);    

    %% Naive Bayesian
    O1 = NaiveBayes.fit(TrainPatterns,TrainTargets);
    ypred = O1.predict(TestPatterns);
    YY=ypred==TestTargets;
    YY1=TestTargets==1;
    YY2=TestTargets==2;
    TP=sum(YY1.*YY);
    TN=sum(YY2.*YY);
    FP=sum(YY1.*~YY);
    FN=sum(YY2.*~YY);
    Sen(fold_N,2)=TP/(TP+FN);
%     FPr=FP/(FP+TN);
%     AUC(fold_N,1)=(1+Sen(fold_N,1)-FPr)/2;
    Spe(fold_N,2)=TN/(TN+FP);
    BAc(fold_N,2)=(TP/(TP+FN)+TN/(TN+FP))/2;
    GM(fold_N,2)=sqrt((TP/(TP+FN))*(TN/(TN+FP)));
    Accuracy(fold_N,2)=(TP+TN)/(TP+TN+FN+FP);    
    
      %% LDA
    O1 = ClassificationDiscriminant.fit(TrainPatterns,TrainTargets);
    ypred = predict(O1,TestPatterns);
    YY=ypred==TestTargets;
    YY1=TestTargets==1;
    YY2=TestTargets==2;
    TP=sum(YY1.*YY);
    TN=sum(YY2.*YY);
    FP=sum(YY1.*~YY);
    FN=sum(YY2.*~YY);
    Sen(fold_N,3)=TP/(TP+FN);
%     FPr=FP/(FP+TN);
%     AUC(fold_N,1)=(1+Sen(fold_N,1)-FPr)/2;
    Spe(fold_N,3)=TN/(TN+FP);
    BAc(fold_N,3)=(TP/(TP+FN)+TN/(TN+FP))/2;
    GM(fold_N,3)=sqrt((TP/(TP+FN))*(TN/(TN+FP)));
    Accuracy(fold_N,3)=(TP+TN)/(TP+TN+FN+FP);    

    %%
    % Start WOA
    SearchAgents_no=10; % Number of search agents 
    Max_iteration=50; % Maximum numbef of iterations
    lb=-1000;
    ub=1000;
    dim=2;
    
    Leader_pos(fold_N,1:dim)=zeros(1,dim);
    Leader_score(fold_N)=inf; %change this to -inf for maximization problems
    
    %Initialize the positions of search agents
    Boundary_no= size(ub,2); % numnber of boundaries
    
    % If the boundaries of all variables are equal and user enter a signle
    % number for both ub and lb
    if Boundary_no==1
        Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
    end
    
%     if Boundary_no>1
%         for i=1:dim
%              Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
%         end
%     end
    
    t=1;% Loop counter
    while t<Max_iteration
        for i=1:size(Positions,1)
            % Return back the search agents that go beyond the boundaries of the search space
            Flag4ub=Positions(i,:)>ub;
            Flag4lb=Positions(i,:)<lb;
            Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
            
            % Calculate objective function for each search agent
            kernel='gaussian';  
%             kerneloption=10; % I think it is Sigma
            verbose=0;
            lambda=1e-2;
%             C=0.1;
            [xsup,w,b,nbsv]=svmmulticlassoneagainstall(TrainPatterns,TrainTargets,2,Positions(i,1),lambda,kernel,Positions(i,2),verbose);
            [ypred] = svmmultival(TrainPatterns,xsup,w,b,nbsv,kernel,Positions(i,2)); 
            TrainingAcc=sum(ypred==TrainTargets)*100/size(TrainTargets,1);
            fitness=100-TrainingAcc 
            
            % Update the leader
            if fitness<Leader_score(fold_N) % Change this to > for maximization problem
                Leader_score(fold_N)=fitness; % Update alpha
                Leader_pos(fold_N,:)=Positions(i,:);
            end
        end
%          figure(t)
%          plot(Positions(:, 1), Positions(:,2), 'bx');hold on   % drawing swarm movements
%          plot(Leader_pos(1, 1), Leader_pos(1,2), 'rx');hold on   % drawing swarm movements
%          
%          pause(.2)
    
        a=2-t*((2)/Max_iteration); % a decreases linearly fron 2 to 0 in Eq. (2.3)
        
        % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        a2=-1+t*((-1)/Max_iteration);
    
        % Update the Position of search agents 
        for i=1:size(Positions,1)
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A=2*a*r1-a;  % Eq. (2.3) in the paper
            C=2*r2;      % Eq. (2.4) in the paper       
            
            b=1;               %  parameters in Eq. (2.5)
            l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
            
            p = rand();        % p in Eq. (2.6)
            
            for j=1:size(Positions,2)
                            
                if p<0.5 
                    %% Shrinking encircling mechanism
                if abs(A)>=1
                    %% A random search agent is chosen
                    rand_leader_index = floor(SearchAgents_no*rand()+1);
                    X_rand = Positions(rand_leader_index, :);
                    D_X_rand=abs(C*X_rand(j)-Positions(i,j)); % Eq. (2.7)
                    Positions(i,j)=X_rand(j)-A*D_X_rand;      % Eq. (2.8)
                    
                elseif abs(A)<1
                    %% The best solution is selected
                    D_Leader=abs(C*Leader_pos(fold_N,j)-Positions(i,j)); % Eq. (2.1)
                    Positions(i,j)=Leader_pos(fold_N,j)-A*D_Leader;      % Eq. (2.2)
                end
                elseif p>=0.5
                    %% Spiral updating position
                    
                    distance2Leader=abs(Leader_pos(fold_N,j)-Positions(i,j));
                    % Eq. (2.5)
                    Positions(i,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos(fold_N,j);
                    
                end
            end
        end
        t=t+1;
        Convergence_curve(t)=Leader_score(fold_N);
        [t Leader_score(fold_N)];
    end

%     figure('Position',[269   240   660   290])
%     %Draw search space
%     subplot(1,2,1);
%     x=-100:2:100; y=x
%     L=length(x);
%     f=[];
%     
%     for i=1:L
%         for j=1:L
%             [xsup,w,b,nbsv]=svmmulticlassoneagainstall(TrainPatterns,TrainTargets,2,x(i),lambda,kernel,y(j),verbose);
%             [ypred] = svmmultival(TrainPatterns,xsup,w,b,nbsv,kernel,y(j)); 
%             TrainingAcc=sum(ypred==TrainTargets)*100/size(TrainTargets,1);
%             fitness=100-TrainingAcc; 
%             f(i,j)=fitness;
%             disp(['i= ' int2str(i) ' j = ' int2str(j)])
%         end
%     end
%     title('Parameter space')
%     xlabel('x_1');
%     ylabel('x_2');
%     zlabel([Function_name,'( x_1 , x_2 )'])
%     
%     %Draw objective space
%     subplot(1,2,1);
%     semilogy(Convergence_curve,'Color','r')
%     title('Objective space')
%     xlabel('Iteration');
%     ylabel('Best score obtained so far');
%     
%     axis tight
%     grid on
%     box on
%     
%     legend('WOA')
    
%     display(['The best solution obtained by WOA is : ', num2str(Leader_pos(fold_N,:))]);
%     display(['The best optimal value of the objective funciton found by WOA is : ', num2str(Leader_score(fold_N))]);

    % Train SVM with the best parameters from WOA 

    [xsup,w,b,nbsv]=svmmulticlassoneagainstall(TrainPatterns,TrainTargets,2,Leader_pos(1,1),lambda,kernel,Leader_pos(1,2),verbose);
    [ypred] = svmmultival(TrainPatterns,xsup,w,b,nbsv,kernel,Leader_pos(1,2)); 
    TrainingAcc=sum(ypred==TrainTargets)*100/size(TrainTargets,1);
    [ypred] = svmmultival(TestPatterns,xsup,w,b,nbsv,kernel,Leader_pos(1,2)); 
    TestingAcc=sum(ypred==TestTargets)*100/size(TestTargets,1);

%     Temp=unique(xsup);
%     disp(['Number of support vectors is ' int2str(size(Temp,1))])
%     visualizeBoundary_Alaa(X, y,  xsup,w,b,nbsv)
%     [E] = bagging_train(TrainPatterns,TrainTargets,N_classifiers,2);
%     [OutLables,ind,e] = bagging_classify(E,TestPatterns,TestTargets);
    YY=ypred==TestTargets;
    YY1=TestTargets==1;
    YY2=TestTargets==2;
    TP=sum(YY1.*YY);
    TN=sum(YY2.*YY);
    FP=sum(YY1.*~YY);
    FN=sum(YY2.*~YY);
    Sen(fold_N,4)=TP/(TP+FN);
%     FPr=FP/(FP+TN);
%     AUC(fold_N,1)=(1+Sen(fold_N,1)-FPr)/2;
    Spe(fold_N,4)=TN/(TN+FP);
    BAc(fold_N,4)=(TP/(TP+FN)+TN/(TN+FP))/2;
    GM(fold_N,4)=sqrt((TP/(TP+FN))*(TN/(TN+FP)));
    Accuracy(fold_N,4)=(TP+TN)/(TP+TN+FN+FP);    
    disp(['The fold number ' int2str(fold_N)]);
end

disp(['mean Accuracy = ' num2str(mean(Accuracy))])
disp(['mean Sensitivity = ' num2str(mean(Sen))])
disp(['mean Specificity = ' num2str(mean(Spe))])
disp(['mean GM = ' num2str(mean(GM))])
disp(['mean Balanced Accuracy = ' num2str(mean(BAc))])

%% Heatmap
% clf
% heatmap(f, [(-100:2:100)], [(-100:2:100)]);
% xlabel('C')
% ylabel('Sigma');

%% Contour Plot
% [XX,YY] = meshgrid(-100:2:100,-100:2:100);
% [c,h]=contour(XX,YY,f); hold on
% clabel(c,h);
% grid;
% xlabel('x1');
% ylabel('x2');

%% Surface
% surfc(x,y,f,'LineStyle','none');
% title('Parameter space')
% xlabel('x_1');
% ylabel('x_2');
% zlabel([Function_name,'( x_1 , x_2 )'])
