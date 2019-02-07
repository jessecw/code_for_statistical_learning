function y = bayesian_classification(T, x, lambda)
% y = bayesian_classification(T, x, lambda)
% --------------------------------------
% Author: Jesse Chen
% Reference: chapter 4.2.3, page 51, Lihang's book.
% Notes:
% T: (\mathbf{x}, y). The last column are the class labels and the other
% columns are features.
% x: the input feature to be classified.
% lambda: Laplace smoothing coefficient.
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
P_c_prior = (c_counts + lambda) / (training_set_size + c_size * lambda);

%% get features' all possible value counts
feature_set_count = zeros(model_size, 1);
for i = 1:model_size
 feature_set_count(i) = numel(unique(T(:,i)));
end

%% beyesian classification
P = zeros(c_size, 1);
for i = 1:c_size
  P(i) = P_c_prior(i);
  for j = 1:model_size
    P(i) = P(i) * (sum(double( (T(:,j) == x(j)) & (T(:,end) == c(i)) )) + lambda)/...
        (c_counts(i) + feature_set_count(j)*lambda);
  end
end

%% find class with max posterior probability
[maxP, y] = max(P); 
y = c(y);


