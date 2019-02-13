function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

[m,n] = size(X);
theta_transposed = theta.';
sum = 0;

disp(theta_transposed)
disp(X(1,:).')

for num_rows = 1:m
    h = theta_transposed*(X(num_rows,:).');
    sum = sum + (h-y(num_rows,:))^2
end;

J = (1/(2*m))*sum;


% =========================================================================
end