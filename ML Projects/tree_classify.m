function labels = tree_classify(T, test_data)
% classify test_data using the tree classifier T
 for i = 1 : size(test_data,1)
 index = 1; leaf = 0;
 while leaf == 0,
 if T(index,3) == 0, % leaf is found
 labels(i) = T(index,1); leaf = 1;
 else
 if test_data(i,T(index,1)) <= T(index,2)
 index = T(index,3); %left
 else
 index = T(index,4); %right
 end
 end
 end
 end
%-------------------