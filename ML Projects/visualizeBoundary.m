function visualizeBoundary(X, y,xsup,w,b,nbsv,kernel,kerneloption)
%VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
%   boundary learned by the SVM and overlays the data on it

% Plot the training data on top of the boundary
plotData(X, y)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmmultival(this_X,xsup,w,b,nbsv,kernel,kerneloption); 
end
vals(vals==2)=0;

% Plot the SVM boundary
hold on
contour(X1, X2, vals, [0 0], 'Color', 'b');


end
