function T = tree_build(data, labels, classN, chi2_threshold)
 % --- train tree classifier
 if numel(unique(labels)) == 1 % all data are of the same class
 T = [labels(1),0,0,0]; % make a leaf
 else
 [chosen_feature,threshold] = tree_select_feature(data,labels);
 leftIndex = data(:,chosen_feature) <= threshold;
 chi2 = tree_chi2(leftIndex,labels,classN);
 if chi2 > chi2_threshold % accept the split
 leftIndex = data(:,chosen_feature) <= threshold;
 Tl = tree_build(data(leftIndex,:),labels(leftIndex),...
 classN,chi2_threshold); % left subtree
 Tr = tree_build(data(~ leftIndex,:),labels(~ leftIndex),...
 classN,chi2_threshold); % right subtree
 % merge the two trees
 Tl(:,[3 4]) = Tl(:,[3 4]) + (Tl(:,[3 4]) > 0) * 1;
 Tr(:,[3 4]) = Tr(:,[3 4]) + (Tr(:,[3 4]) > 0) * (size(Tl,1)+1);
 T = [chosen_feature, threshold, 2, size(Tl,1)+2; Tl; Tr];
 else % make a leaf
 T = [mode(labels), 0, 0, 0];
 end
end