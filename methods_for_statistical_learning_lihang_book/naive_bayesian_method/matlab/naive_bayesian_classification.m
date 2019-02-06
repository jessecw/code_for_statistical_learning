function y = naive_bayesian_classification(T, x)
% y = naive_bayesian_classification(T, x)
% ----------------------------------------
% Author: Jesse Chen
% Reference: chapter 4.2.2, page 50, Lihang's book.
% Notes:
% T: (\mathbf{x}, y). The last column are the class labels and the other
% columns are features.
% x: the input feature to be classified.
% y: the class label with max posterior probability 

%% get the model size and the training set size
[training_set_size, model_size_plus_one] = size(T);

model_size = model_size_plus_one - 1;

assert(model_size > 0);

%% get unique classes
c = unique(T(:,model_size_plus_one));
c_size = numel(c);

%% get class's prior probability
c_counts = hist(T(:,model_size_plus_one), unique(T(:,model_size_plus_one)));
P_c_prior = c_counts / training_set_size;

%% naive beyesian classification
P = zeros(c_size, 1);
for i = 1:c_size
  P(i) = P_c_prior(i);
  for j = 1:model_size
    P(i) = P(i) * (sum(double( (T(:,j) == x(j)) & (T(:,end) == c(i)) )))/c_counts(i);
  end
end

%% find class with max posterior probability
[maxP, y] = max(P); 
y = c(y);
