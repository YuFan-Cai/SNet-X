function T = tracematrices(C)

[M, N, D, D] = size(C);
T = zeros(M, N);
for k = 1:D
    T = T + C(:, :, k, k);
end

