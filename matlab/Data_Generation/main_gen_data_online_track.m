clc
config = readyaml('../../config.yaml');
seq_len = config.seq_len;
pred_len = config.pred_len;
rng(0)
M = config.M_track;
N = config.N_track;
[X, Z_Cart] = gen_data(M, N);
[Input, True, Pred] = data_preprocess(X, Z_Cart);
True_polar = c2p(X);
Input = cat(3, Input, True_polar(:, :, 3)+0.5*randn(M, N, 1));
Input = Input(:, 2:end, :);
True = True(:, 2:end, :);
Pred = Pred(:, 2:end, :);
[Input]=data_scale(Input, 1);
[True]=data_scale(True, 1);
[Pred]=data_scale(Pred, 1);
filepath = "../../data/InforMTP/";
Input_data = Input;
True_data = True;
Pred_data = Pred;
save(strcat(filepath, 'my_data_track'), "Input_data","True_data", "Pred_data")
save ('../online_track_data.mat', 'X', 'Z_Cart');

