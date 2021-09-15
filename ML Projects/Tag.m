clc
clear all
theta=10:10:720;
Sp=5*((1- cos(theta*3.14/180))+(5/72)*(1-cos(2*theta*3.14/180)));
plot(theta,Sp,'b-');