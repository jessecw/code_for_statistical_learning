function [w,b] = perceptron_learning_in_naive_format(T, learning_rate)
% [w,b] = perceptron_learning_in_naive_format(T, learning_rate)
% --------------------------------------------------------------
% Author: Jesse Chen
% Reference: Algorithm 2.1, Page 29, Lihang's Book
% Notes:
% T: the training dataset, T = \{\left(x_1, y_1\right), \cdots, \left(x_N,
% y_N\right)\}. x_i\in\mathbb{R}^{n}, y_i \in\left\{-1, +1\right\}
% learnng_rate: the rate for learning, it should in the range \left(0, 1\right]

%% determine the model parameter
[training_set_size, model_size] = size(T);
N = model_size - 1;

assert(N > 0);

%% select the initial parameter set
w = zeros(N, 1);
b = 0;

%% 
x = T(:,1:N);
y = T(:,model_size);

%%
z = x*w + repmat(b, [training_set_size, 1]);
L = z.*y;
has_fault_cases = sum(double(L <= 0));

while has_fault_cases > 0
  fault_case_index = find(L <= 0);
  w = w + learning_rate * y(fault_case_index(1)) * x(fault_case_index(1), :)';
  b = b + learning_rate * y(fault_case_index(1));

  z = x*w + repmat(b, [training_set_size, 1]);
  L = z.*y;
  has_fault_cases = sum(double(L <= 0));
end



