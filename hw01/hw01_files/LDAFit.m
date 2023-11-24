function [w,b] = LDAFit(X,Y)
%LDAFit 
%   Function that takes as input 
%       X: a d x n matrix where each column corresponds to a feature vector
%          in R^d
%       Y: a 1 x n vector of binary labels (0,1) for each training vector
%   and generates an output of
%       w: a d x 1 normal vector
%       b: an offset for the separating hyperplane
%   using LDA

[d,n] = size(X);

idx0 = find(Y==0); % X(:,idx0) corresponds to all training vectors labeled 0
idx1 = find(Y==1); % X(:,idx1) corresponds to all training vectors labeled 1

n0 = length(idx0); % number of training vectors labeled 0
n1 = length(idx1); % number of training vectors labeled 1

pi_hat_0 = n0/n;
pi_hat_1 = n1/n;

mu_hat_0 = (1/n0)*sum(X(:,idx0),2);
mu_hat_1 = (1/n1)*sum(X(:,idx1),2);

Sigma_hat = (1/n)*(( (X(:,idx0)-mu_hat_0)*(X(:,idx0)-mu_hat_0).' )+( (X(:,idx1)-mu_hat_1)*(X(:,idx1)-mu_hat_1).' ));

w = pinv(Sigma_hat)*(mu_hat_1-mu_hat_0);
b = 0.5*(mu_hat_0.'*pinv(Sigma_hat)*mu_hat_0)-0.5*(mu_hat_1.'*pinv(Sigma_hat)*mu_hat_1)+log(pi_hat_1/pi_hat_0);

