function ind = p_sample(w)
 % sample from distribution w
 N = length(w);
 cdf = cumsum(w);
 r = rand(N,1); % uniform random numbers
 d1 = repmat(r(:),1,N);
 d2 = repmat(cdf(:)',N,1);
 [~, ind] = max((d1 < d2)');
 end
 %---------------------------------------------------------%