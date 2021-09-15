plot(BEST1,'b-'), hold on
plot(BEST4,'r-'), hold on   
plot(BEST3,'g-')
xlabel('Number of Iterations')
ylabel('Test Error Rate (%)')
legend('First run','Second run','Third run')

CPUTIME=[6.74 8.37 12.05 21.77 25.02 30.78;
    6.095 9.59 14.88 18.99 20.93 34.54;
    5.295 11.6 16.88 15.38 21.63 25.73]
MCPU=mean(CPUTIME)
SCPU=std(CPUTIME)
X=5:5:30
errorbar(MCPU,SCPU/2)
xlabel('Number of Iterations')
ylabel('CPU Time (secs)')
% legend('First run','Second run','Third run')


X=4:4:40
CPU=[6.98 9.84 12.8 20.43 25.73 26.01 26.3 36.35 35.48 45.33]
Acc=[5.43 5.43 4 4 2.67 2.67 2.67 2.67 2.67 2.67 ]
plot(X,Acc,'b-'), hold on   
xlabel('Number of bats')
ylabel('Training Error rate (%)')

plot(Acc,X,'b-'), hold on   
xlabel('Number of bats')
ylabel('Test Error Rate (%)')
