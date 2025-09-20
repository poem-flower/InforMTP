function x = p2c(x_polar)
[x(1, :), x(2, :)] = pol2cart(x_polar(2, :), x_polar(1, :));
end