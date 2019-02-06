%% demo_perceptron_learning.m

%% case 1
T = [
  3, 3, 1
  4, 3, 1
  1, 1, -1
];

[w, b] = perceptron_learning_in_naive_format(T, 1);
[w2, b2] = perceptron_learning_in_dual_format(T, 1);