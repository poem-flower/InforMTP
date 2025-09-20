function F =gen_F(T, alpha_max, change_id)
stage_num = size(change_id, 2) - 1;
label = round(rand(1,stage_num)*3)+1;
F = repmat({F_(label(1), T, alpha_max)}, 1, change_id(2)-change_id(1)+1);
for i = 2:stage_num
     F = [F repmat({F_(label(i), T, alpha_max)}, 1, change_id(i+1)-change_id(i))];
end

end

