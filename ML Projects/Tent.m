function x=Tent(n,level,a,x0)
%Tent map
% x(1)=0.6;

if x0==0
    x0=0.4;
end

if x0<0.7
    x(1)=x0/0.7;
else 
    x(1)=(10/3)*(1-x0);
end

for i=2:n
    if x(i-1)<0.7
        x(i)=x(i-1)/0.7;
    else
        x(i)=(10/3)*(1-x(i-1));
    end
end
% Add normal white noise
x=x+randn(1,n)*level*std(x);