function [w,b] = LRFit(X,Y)
%LRFit 
%   Function that takes as input 
%       X: a d x n matrix where each column corresponds to a feature vector
%          in R^d
%       Y: a 1 x n vector of binary labels (0,1) for each training vector
%   and generates an output of
%       w: a d x 1 normal vector
%       b: an offset for the separating hyperplane
%   using logistic regression

[d,n] = size(X);

Xtilde = [ones(1,n); X];

theta = zeros(d+1,1); % Initialize starting point to zero
t = 0; % Iteration counter
theta_old = ones(d+1,1);
while (norm(theta-theta_old)/norm(theta_old)>0.01)
    
    theta_old = theta;
    
    g = 1./(1+exp(-Xtilde'*theta_old)); 
        
    G = Xtilde*(Y'-g);

    %how to do this per thing
%     H = zeros(size(Xtilde(:,1)*(Xtilde*(g.*(1-g)))'));
%     for xi = Xtilde
%         i = i+1;
%         H = H+xi*(Xtilde*(g.*(1-g)))';
%     end
%     H = -H;

    H = -Xtilde*(Xtilde*diag(g.*(1-g)))';

    theta = theta_old - H\G;
    
    t = t+1;
    % Update other stopping criteria?
    
end

b = theta(1);
w = theta(2:end);