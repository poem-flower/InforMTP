function X0=gen_initial(R_min, R_max, V_min, V_max, a_max)
R=(R_max-R_min)*rand+R_min;
V=(V_max-V_min)*rand+V_min;
R_phi = 2*pi*rand-pi;
V_phi = 2*pi*rand-pi;
X0(1:2, :) = R.*[cos(R_phi); sin(R_phi)];
X0(3:4, :) = V.*[cos(V_phi); sin(V_phi)];
X0(5:6, :) = 2*a_max*rand(2, 1) - a_max;

end