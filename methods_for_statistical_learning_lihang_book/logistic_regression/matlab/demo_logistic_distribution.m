% demo_logistic_distribution.m
% ------------------------------
% Author: Jesse Chen
% Reference: charpter 6.1, page 77, Lihang's book.

% define logistic distribution and it's PDF.
% x can be vector or scalar, but mu and gamma should be scalars.
logistic_distribution = @(x, mu, gamma)(1.0./(1 + exp(-(x - mu)/gamma)));

logistic_pdf = @(x, mu, gamma)(exp(-(x - mu)/gamma)./(gamma*(1 + exp(-(x - mu)/gamma)).^2));

figure; 
x = -5:0.01:5;
plot(x, logistic_pdf(x, 0, 0.2));