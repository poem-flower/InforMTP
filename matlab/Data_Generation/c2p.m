function X_polar = c2p(X)
[M, N, m] = size(X);
X_polar = zeros(M, N, 2);

for i=1:M
    [X_polar(i, :, 2), X_polar(i, :, 1)] = cart2pol(X(i, :, 1), X(i, :, 2));
end

if m>2
    X_doppler = - X(:, :, 3).*cos(X_polar(:, :, 2)) - X(:, :, 4).*sin(X_polar(:, :, 2));
    X_polar = cat(3, X_polar, X_doppler);
end

end


