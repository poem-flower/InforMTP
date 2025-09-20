function [output]=data_scale(input, flag)
config = readyaml('../../config.yaml');
V_max = config.V_max;
T = config.T;
D_scale = V_max;
if flag==1
    R_scale = V_max*T;
    output(:, :, 1:2) = input(:, :, 1:2)/R_scale;
    try
        output(:, :, 3) = input(:, :, 3)/D_scale;
    end
    
elseif flag==0
    R_scale = V_max*T;
    output = input*R_scale;
end

end