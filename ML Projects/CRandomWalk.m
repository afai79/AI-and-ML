
function [RWs]=CRandomWalk(Dim,max_iter,lb, ub,antlion,current_iter,I)
if size(lb,1) ==1 && size(lb,2)==1 %Check if the bounds are scalar
    lb=ones(1,Dim)*lb;
    ub=ones(1,Dim)*ub;
end

if size(lb,1) > size(lb,2) %Check if boundary vectors are horizontal or vertical
    lb=lb';
    ub=ub';
end


% Dicrease boundaries to converge towards antlion
if current_iter<(max_iter*0.1)
    I=1;

end

lb=I*lb; % Equation (2.10) in the paper
ub=I*ub; % Equation (2.11) in the paper

% Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]
if rand<0.5
    lb=lb+antlion; % Equation (2.8) in the paper
else
    lb=-lb+antlion;
end

if rand>=0.5
    ub=ub+antlion; % Equation (2.9) in the paper
else
    ub=-ub+antlion;
end

% This function creates n random walks and normalize accroding to lb and ub
% vectors 
for i=1:Dim
    X = [0 cumsum(2*(rand(max_iter,1)>0.5)-1)']; % Equation (2.1) in the paper
    %[a b]--->[c d]
    a=min(X);
    b=max(X);
    c=lb(i);
    d=ub(i);      
    X_norm=((X-a).*(d-c))./(b-a)+c; % Equation (2.7) in the paper
    RWs(:,i)=X_norm;
end