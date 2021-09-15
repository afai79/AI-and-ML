function visualizeBoundary_Alaa(X, y,  xsup,w,b,nbsv)
%VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
%   boundary learned by the SVM and overlays the data on it

% Plot the training data on top of the boundary
YYY=y;
for i=1:size(X,1)
   if(YYY(i,1)==2)
       YYY(i,1)=0;
   end
end
plotData(X, YYY)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmmultival(this_X,xsup,w,b,nbsv,'gaussian',0.1); 
end

% Plot the SVM boundary
for i=1:size(vals,1)
   for j=1:size(vals,1)
    if(vals(i,j)==2)
       vals(i,j)=0;
    end
   end
end
hold on
contour(X1, X2, vals, [0 0], 'Color', 'b', 'LineWidth', 2);


end
