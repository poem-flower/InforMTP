function [x_revised, If_pre] = data_inv(out_net,z, seq_len, pred_len)
x_revised = zeros(seq_len+1, 2);
If_pre = zeros(pred_len, 2);
x_revised(1, :) = z(1, 1:2);
for t = 2:seq_len+1 
    x_del = out_net(t-1, :);
    x_revised(t, :) = z(t-1, :) + x_del;
end

x_del = out_net(seq_len+1, :);
If_pre(1, :) = x_revised(t, :)+ x_del;
for t = seq_len+3:seq_len+pred_len+1  
    x_del = out_net(t-1, :);
    If_pre(t-seq_len-1, :) = If_pre(t-seq_len-2, :) + x_del;
end

end