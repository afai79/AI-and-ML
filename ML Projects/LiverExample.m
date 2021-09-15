%This example of the big code of SVM
%Nonlinear data
clear ; close all; clc
 
%% %%%%%%%%%% Example 1: Demonstration of Linear SVM %%%%%%
% Loading and Visualizing training dataset.
fprintf('Loading and Visualizing Data ...\n')
% Loading dataset from example1.mat: 
% You will have X, y in your environment
%load('Liver.mat'); 
% Loaded dataset from example2 : 
% You will have X, y in your environment
% Plot training data
%X=data;

[X,Y,Nclass]=SelectDataSet('Liver');
y=size(X,1);

% Training SVM with RBF Kernel (Dataset example2) ==========
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
% [aa,ind]=sort(Y);
% XX=X(ind,:);
% YY=Y(ind,1);

    kernel='gaussian';
    C=10;
    kerneloption=1; % I think it is Sigma
    verbose=0;
    lambda=0.01;
    [xsup,w,b,nbsv]=svmmulticlassoneagainstall(X,Y,2,C,lambda,kernel,kerneloption,verbose);
    [ypred] = svmmultival(X,xsup,w,b,nbsv,kernel,kerneloption); 
    Trainingerror=sum(ypred==Y)*100/size(Y,1);
    disp(['Training Accuracy is ' int2str(Trainingerror)])

    disp(['Number of Correct Classified Samples is ' int2str(sum(ypred==Y))])
% disp(['Number of Support vectors is ' int2str(sum(model.alphas>0.005))])
% To plot misclassified Samples
Temp1=ypred==Y;
% plot(X(~Temp1,1),X(~Temp1,2),'rs', 'MarkerSize',10), hold on
disp(['Number of misclassified Samples ' int2str(sum(~Temp1))])
TTT=unique(xsup);
disp(['Number of support vectors is ' int2str(size(TTT,1))])
