function change_id = change(N, lambda)

id = 1;
change_id = [];

while(id<N)
    change_id = [change_id id];
    id = id + random('Poisson',lambda);
end
change_id = [change_id N];

end