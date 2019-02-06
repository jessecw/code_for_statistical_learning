function [w, b] = perceptron_learning_in_dual_format(T, learning_rate)
% [w, b] = perceptron_learning_in_dual_format(T, learning_rate)
% --------------------------------------------------------------------
% Author: Jesse Chen
% Reference: Algorithm 2.2, Page 33, Lihang's Book
% Notes:
% T: the training dataset, T = \{\left(x_1, y_1\right), \cdots, \left(x_N,
% y_N\right)\}. x_i\in\mathbb{R}^{n}, y_i \in\left\{-1, +1\right\}
% learnng_rate: the rate for learning, it should in the range \left(0, 1\right]
% Please be noted:
% Each element of L = (alpha'*(y.*G) + b)'.*y corresponds to 
% y_i\left(\sum_{j=1}^N \alpha_j y_j x_j\cdot x_i + b\right)


%% determine the model parameter
[training_set_size, model_size] = size(T);
N = model_size - 1;

assert(N > 0);

%% 
x = T(:,1:N);
y = T(:,model_size);

%% calculate Gram Matrix
G = x*x';

%% set up the initial parameters
alpha = zeros(training_set_size, 1);
b = 0;

%% Learning
L = (alpha'*(y.*G) + b)'.*y;

has_fault_cases = sum(double(L <= 0));
while has_fault_cases > 0
  fault_case_index = find(L' <= 0);
  
  alpha(fault_case_index(1)) = alpha(fault_case_index(1)) + learning_rate;
  b = b + learning_rate * y(fault_case_index(1));
  
  L = (alpha'*(y.*G) + b)'.*y;
  has_fault_cases = sum(double(L <= 0));  
end

w = sum(alpha.*(y.*x), 1);
b = alpha'*y;

