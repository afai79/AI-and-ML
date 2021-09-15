function [top_feature, top_thre] = tree_select_feature(data,labels)
 % --- select the best feature
 [n, m] = size(data);
 i_G = Gini(labels); % Gini index of impurity at the parent node
 [D, s] = deal(zeros(1, m)); % preallocate for speed
 for j = 1 : m % check each feature
 if numel(unique(data(:,j))) == 1 % the feature has only 1 value
 D(j) = 0; s(j) = -999; % missing
 else
 Dsrt = sort(data(:,j)); % sort j-th feature
 dde_i = zeros(1, n); % preallocate for speed
 for i = 1 : n-1 % check the n-1 split points
sp = (Dsrt(i) + Dsrt(i+1)) / 2;
 left = data(:,j) <= sp;
 % Make sure that there are points in both children nodes
 if sum(left) > 0 && sum(left) < n
 i_GL = Gini(labels(left));i_GR = Gini(labels(~ left));
dde_i(i) = i_G - mean(left)*i_GL - mean(~ left)*i_GR;
else % one child node is empty
 dde_i(i)=0;
 end
 end
[D(j), index_s] = max(dde_i); % best impurity reduction
 s(j) = (Dsrt(index_s) + Dsrt(index_s+1)) / 2; % threshold
 end
 end
 [~, top_feature] = max(D); top_thre = s(top_feature);
 %......................................................................
 