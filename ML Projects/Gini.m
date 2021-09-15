function i_G = Gini(labels)
 % --- calculate Gini index
 for i = 1 : max(labels)
    P(i) = mean(labels == i);
 end
 i_G = 1 - P * P';