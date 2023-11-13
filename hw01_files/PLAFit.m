function [w,b] = PLAFit(X,Y)
%PLAFit 
%   Function that takes as input 
%       X: a d x n matrix where each column corresponds to a feature vector
%          in R^d
%       Y: a 1 x n vector of binary labels (0,1) for each training vector
%   and generates an output of
%       w: a d x 1 normal vector
%       b: an offset for the separating hyperplane
%   using the perceptron learning algorithm

[d,n] = size(X);

Xtilde = [ones(1,n); X]; 

Y = 2*Y-1; % Convert labels from 0,1 to -1,+1

theta = zeros(d+1,1); % Initialize starting point to zero
t = 0; % Iteration counter
max_t = length(Y)*10;
theta_old = ones(d+1,1);
while (norm(theta-theta_old)/norm(theta_old)>eps && t<max_t)
    idx = randi([1,length(Y)]);
    yi = Y(idx);
    Xtildei = Xtilde(idx);
    
    theta_old = theta;
    
    if not(yi == sign(theta_old.'*Xtildei))
        theta = theta_old+yi*Xtildei;
    end
    
    t = t+1;
    % Update other stopping criteria?
    
end

b = theta(1);
w = theta(2:end);