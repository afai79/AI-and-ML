%This example of the big code of SVM
%Nonlinear data
clear ; close all; clc
 
%% %%%%%%%%%% Example 1: Demonstration of Linear SVM %%%%%%
% Loading and Visualizing training dataset.
fprintf('Loading and Visualizing Data ...\n')
% Loading dataset from example1.mat: 
% You will have X, y in your environment
load('example2.mat'); 
% Loaded dataset from example2 : 
% You will have X, y in your environment
% Plot training data
figure,
plotData(X, y);
title('Traning Dataset');
% Training SVM with RBF Kernel (Dataset example2) ==========
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
[aa,ind]=sort(y);
XX=X(ind,:);
YY=y(ind,1);
Temp=YY==0;
YY(Temp,1)=2;
YY(~Temp,1)=1;

counter=1;
% for i=1:863
%     if(y(i,1)==1)
%         XX(counter,1)=X(i,1);
%         YY(counter,1)=1

    kernel='gaussian'; 
    kerneloption=0.1; % I think it is Sigma
    verbose=0;
    lambda=1e-2;
    [xsup,w,b,nbsv]=svmmulticlassoneagainstall(XX,YY,2,100000,lambda,kernel,kerneloption,verbose);
    [ypred] = svmmultival(XX,xsup,w,b,nbsv,kernel,kerneloption); 
    Trainingerror=sum(ypred==YY)*100/size(y,1)
    disp(['Number of training Samples is ' int2str(sum(ypred==YY))])

   







C =110.8923; sigma = 0.1;
% I set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
figure,
visualizeBoundary(X, y, model);
title('Decision boundary with RBF kernel');

Temp1=svmPredict(model, X);
Trainingerror=sum(Temp1==y)*100/size(y,1);
disp(['Training Error is ' int2str(Trainingerror)])
disp(['Number of Support vectors is ' int2str(sum(model.alphas>0.005))])
% To plot misclassified Samples
Temp1=Temp1==y;
% plot(X(~Temp1,1),X(~Temp1,2),'rs', 'MarkerSize',10), hold on
disp(['Number of misclassified Samples ' int2str(sum(~Temp1))])

%To draw support vectors
for i=1:size(model.X,1)
if (model.alphas(i)>0.005)
plot(model.X(i,1),model.X(i,2),'gs', 'MarkerSize',10)
end
end

Temp1=svmPredict(model, X);
Temp1=Temp1==y;
plot(X(~Temp1,1),X(~Temp1,2),'rs', 'MarkerSize',12), hold on
