function [Input, True, Pred] = data_preprocess(X, Z)
[M, N, ~] = size(X);
Input = zeros(M, N, 2);
True = zeros(M, N, 2);
for t = 2:N
    z_delta = Z(:, t, :) - Z(:, t-1, :);
    Input(:, t, :)= z_delta;
    x_delta = X(:, t, 1:2) - Z(:, t-1, :);
    True(:, t, :)= x_delta;
    x_pre_delta = X(:, t, 1:2) - X(:, t-1, 1:2);
    Pred(:, t, :) = x_pre_delta;
end

end

