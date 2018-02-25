function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly
all_theta = zeros(num_labels, n + 1);     # Matriz 10 x n + 1 de Zeros

% Add ones to the X data matrix
X = [ones(m, 1) X];    # Adiciona uma linha de 1s à esquerda, agora X tem uma coluna a mais à esquerda (o theta0)

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda.
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
%
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

# OBS: y é um vetor [10 10 10 10 ... 1 1 1 1 ... 2 2 2 2 ... ]
# Imprimir y para entender melhor
# (y == c) faz um filtro.
# Se c for 2 o filtro retorna [0 0 0 0 ... 0 0 0 0 ... 1 1 1 1 ... ]
# É assim que se faz o treino de um contra todos, o trecho do vetor correxpondente a label 2 (neste caso)
# será treinado contra todos os outros, o treino que der os maiores valores, são os que tem mais chamce de
# ser o número que procuramos (já que aqui estamos treinando para predizer números de mão de 0 à 9)

% Set Initial theta
initial_theta = zeros(n + 1, 1);

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);
for c = 1:num_labels
   # Essa função vai fazer o gradiente dentro dela, é uma função de otimização.
  [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
  all_theta(c, :) = theta';    # Prenche cada linha de all_theta com o vetor transposto resultante de cada treino
end


% =========================================================================


end
