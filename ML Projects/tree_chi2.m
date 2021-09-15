function chi2 = tree_chi2(left, labels, classN)
 % --- calculate chiˆ2 statistic for the split of labels on "left"
 n = numel(labels); chi2 = 0; n_L = sum(left);
 for i = 1 : classN
 n_i = sum(labels == i); n_iL = sum(labels(left) == i);
 if n_i > 0 && n_L > 0 % add only for non-empty children nodes
 chi2 = chi2 + (n * n_iL - n_i * n_L)^2 /...
 (2 * n_i * (n_L) * (n - n_L));
 end
 end