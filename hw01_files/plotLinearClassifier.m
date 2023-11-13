function plotLinearClassifier(X,Y,w,b,fighandle,subplotno)
%plotLinearClassifier
%   Function that takes as input 
%       X: a 2 x n matrix where each column corresponds to a feature vector
%          in R^2
%       Y: a 1 x n vector of binary labels (0,1) for each training vector
%       w: a 2 x 1 normal vector
%       b: an offset for the separating hyperplane
%   and creates a scatter plot of the data along with a plot of the
%   decision boundary determined by (w,b)

figure(fighandle);
idx0 = find(Y==0);
idx1 = find(Y==1);
subplot(subplotno)
scatter(X(1,idx0),X(2,idx0))
hold on
scatter(X(1,idx1),X(2,idx1),'r+')

xlim = get(subplot(subplotno),'xlim');
ylim = get(subplot(subplotno),'ylim');
x = [xlim(1):(xlim(2)-xlim(1))/1000:xlim(2)];
y = -(w(1)*x +b)/w(2);
plot(x,y,'g')
set(subplot(subplotno),'xlim',xlim);
set(subplot(subplotno),'ylim',ylim);