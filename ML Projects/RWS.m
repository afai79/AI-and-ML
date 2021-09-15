
function choice = RWS(weights)
%roulette wheel selection

  accumulation = cumsum(weights);
  p = rand()*accumulation(end);
  chosen_index = -1;
  for index = 1 : length(accumulation)
    if (accumulation(index) > p)
      chosen_index = index;
      break;
    end
  end
  choice = chosen_index;