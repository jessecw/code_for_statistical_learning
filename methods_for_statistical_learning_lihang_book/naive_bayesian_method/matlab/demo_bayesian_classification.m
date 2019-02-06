%% demo_bayesian_classification.m

T = [
  1 1 -1  % 1
  1 2 -1  % 2
  1 2  1  % 3
  1 1  1  % 4
  1 1 -1  % 5
  2 1 -1  % 6
  2 2 -1  % 7
  2 2  1  % 8
  2 3  1  % 9
  2 3  1  % 10
  3 3  1  % 11
  3 2  1  % 12
  3 2  1  % 13
  3 3  1  % 14
  3 3 -1  % 15
];

x = [2 1];

c_x = naive_bayesian_classification(T, x);