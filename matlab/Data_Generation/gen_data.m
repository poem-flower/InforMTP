function [X, Z_cart] = gen_data(M, N)
config = readyaml('../../config.yaml');
T=config.T;
R_max = config.R_max;
V_min =config.V_min;
V_max = config.V_max;
R_min = 0;
a_max = config.a_max;
alpha_max = config.alpha_max;
r = config.R;
q1 = config.Q1;
q2 = config.Q2;
R = diag([r r]).^2;
Q = diag([q1 q1 q2 q2 0 0]).^2;
H=[1 0 0 0 0 0;
   0 1 0 0 0 0];
lambda = config.Lambda;
%% 生成数据

R0_max = R_max - V_max*N*T;
X = zeros(M, N, 6);
Z_cart = zeros(M, N, 2);

tic
parfor k = 1:M
    x_temp = zeros(6, N);
    z_temp = zeros(2, N);
    x_temp(:, 1) = gen_initial(R_min, R0_max, V_min, V_max, a_max);
    change_id = change(N, lambda);
    F = gen_F(T, alpha_max, change_id);
    num = randi(3) - 1;
    change_v = sort(randsample(N-1, num));
    for t=2:N
        if ismember(t,change_id)
            x_temp(5:6, t-1) = 2*a_max*rand(2, 1) - a_max; 
        end

        if ismember(t, change_v)
            temp = gen_initial(R_min, R0_max, V_min, V_max, a_max);
            x_temp(3:4, t-1) = temp(3:4, :);
        end

        x_temp(:, t) = F{1, t}*x_temp(:, t-1)+sqrt(Q)*randn(6, 1);
    end
    X(k, :, :) = x_temp';
    z_temp = x_temp(1:2, :) + sqrt(R)*randn(2, N);
    Z_cart(k, :, :) = z_temp';
    disp(['Track generated: ' num2str(k)])
end
toc

end