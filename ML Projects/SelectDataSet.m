function [X,Y,Nclass]=SelectDataSet(dataset)

switch (dataset)
    case 'iris'
        load('iris.mat'); 
        X=data;
        Nclass=3;

    case 'iono'
        load ionosphere
        Temp=cell2mat(Y);
        for i=1:size(Temp,1)
            if(Temp(i,1)=='g')        
                Labels(i,1)=1;
            else
                Labels(i,1)=2;
            end
        end
        Y=Labels;
        clear Temp;
        Nclass=2;
    case 'Liver'
        load Liver
        X=X;Nclass=2;
    case 'ORL'
        load ORL_32;
        X=data';
        c=1;
        for i=1:40
            for j=1:10
                Y(c,1)=i;
                c=c+1;
            end
        end
        Nclass=40;
    case 'Yale'
        load Yale_32
        X=data';
        c=1;
        for i=1:15
            for j=1:11
                Y(c,1)=i;
                c=c+1;
            end
        end
        Nclass=15;
    case 'Sonar'
        load Sonar
        X=data;
        Y=Labels;        
        Nclass=2;
    case 'Ovarian'
        load ovariancancer
        X=obs;
        Temp=cell2mat(grp);
        for i=1:size(Temp,1)
            if(Temp(i,1)=='C')
                Y(i,1)=1;
            else
                Y(i,1)=2;
            end
        end
        Nclass=2;
    case 'Wine'
        load wine_dataset
        wineInputs=wineInputs';
        for i=1:size(wineTargets,2)
            if(wineTargets(1,i)==1)
                Y(i,1)=1;
            elseif (wineTargets(2,i)==1)
                Y(i,1)=2;
            else
                Y(i,1)=3;
            end
        end
        X=wineInputs;        
        Nclass=3;
    case 'Diabetes'
        load Diabetes;
        X=data;
        Y(Y==0)=2;
        Nclass=2;
    case 'Breast'
        load BreastCancer
        X=data;
        Y(Y==4)=1;
        Nclass=2;
    case 'TicTacToe'
        load('TicTacToe.mat')
        X=data;
        Y=Labels;
        Y(Y==0)=2;
        Nclass=2;
         case 'Glass'
             load('Glass.mat')
             X=data;
             Y=Labels;
             Nclass=7;
        

        

        
        
        

        
      end