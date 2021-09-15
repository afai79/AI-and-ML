
 [xsup,w,b,nbsv]=svmmulticlassoneagainstall(Tr,TrL,2,5,lambda,kernel,0.5,verbose);
 [ypred] = svmmultival(Tr,xsup,w,b,nbsv,kernel,0.5); 
 Trainingerror(i,j)=sum(ypred==TrL)*100/size(TrL,1);        
  

[X,Y] = meshgrid(0.01:0.1:10,0.01:0.05:10);  % X=C and Y=gamma
for i = 1:size(X, 2)
   this_X = [X(:, i), X(:, i)];
   [ypred2] = svmmultival(this_X,xsup,w,b,nbsv,kernel,0.5); 
   Testingerror(i,j)=sum(ypred2==TestL)*100/size(TestL,1);        
   SV(i,j)=size(xsup,1);
   disp([int2str(i) ])
end