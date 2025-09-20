clc
config = readyaml('../../config.yaml');
seq_len = config.seq_len;
pred_len = config.pred_len;
rng(0)
M = config.M;
N = seq_len+pred_len+1;
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
M_train = M*(80/100);
M_test = M*(5/100);
M_val = M*(15/100);
%训练集
Input_data = Input(1:M_train, :, :);
True_data = True(1:M_train, :, :);
Pred_data = Pred(1:M_train, :, :);
save(strcat(filepath, 'my_data_train'), "Input_data","True_data", "Pred_data")
%测试集
Input_data = Input(M_train+1:M_train+M_test, :, :);
True_data = True(M_train+1:M_train+M_test, :, :);
Pred_data = Pred(M_train+1:M_train+M_test, :, :);
save(strcat(filepath, 'my_data_test'), "Input_data","True_data", "Pred_data")
%验证集
Input_data = Input(end-M_val+1:end, :, :);
True_data = True(end-M_val+1:end, :, :);
Pred_data = Pred(end-M_val+1:end, :, :);
save(strcat(filepath, 'my_data_val'), "Input_data","True_data", "Pred_data")
