function F = expmatrices_getF(G, J, EA, EB)

[M, N, D, D] = size(G);
F = zeros(M, N, D, D);
for i = 1:D
    k = [1:(i-1) (i+1):D];
    phi = 2 * (G(:, :, i, k) - G(:, :, i, i)) .* J(:, :, i, k);
    F(:, :, i, i) = ...
        G(:, :, i, i) .* EB(:, :, i, i) .* EA(:, :, i, i) + ...
        sum(phi .* real(EA(:, :, i, k) .* conj(EB(:, :, i, k))), 4);
    for j = (i+1):D
        k = 1:D;
        phi = (G(:, :, j, k) - G(:, :, i, k)) .* J(:, :, i, j);
        F(:, :, i, j) = ...
            sum(phi .* ...
                (EB(:, :, i, k) .* conj(EA(:, :, j, k)) + ...
                 EA(:, :, i, k) .* conj(EB(:, :, j, k))), 4);
        F(:, :, j, i) = conj(F(:, :, i, j));
    end
end
