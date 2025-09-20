function F = F_(label, T, alpha_max)
if (label==1)
    F = [1 0 T 0 0 0;
        0 1 0 T 0 0;
        0 0 1 0 0 0;
        0 0 0 1 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0];
elseif(label==2)
    F = [1 0 T 0 T^2/2 0;
        0 1 0 T 0 T^2/2;
        0 0 1 0 T 0;
        0 0 0 1 0 T;
        0 0 0 0 1 0;
        0 0 0 0 0 1];
elseif(label==3)
    a = 0.5;
    a1 = (a*T-1+exp(-a*T))/(a^2);
    a2 = (1- exp(-a*T))/a;
    a3 = exp(-a*T);
    F = [1 0 T 0 a1 0;
        0 1 0 T 0 a1;
        0 0 1 0 a2 0;
        0 0 0 1 0 a2;
        0 0 0 0 a3 0;
        0 0 0 0 0 a3];

elseif(label==4)
    alpha = (2*alpha_max*rand-alpha_max)*pi/180;
    F = [1 0 sin(alpha*T)/alpha (cos(alpha*T)-1)/alpha 0 0;
               0 1 (1-cos(alpha*T))/alpha sin(alpha*T)/alpha 0 0;
               0 0 cos(alpha*T) -sin(alpha*T) 0 0;
               0 0 sin(alpha*T) cos(alpha*T) 0 0;
               0 0 0 0 0 0;
               0 0 0 0 0 0] ;
end

end