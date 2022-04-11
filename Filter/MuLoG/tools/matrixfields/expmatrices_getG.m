function [G, J] = expmatrices_getG(La, eLa)

if ~exist('eLa', 'var')
    eLa = dfunmatrices(La, @(x) exp(x));
end

[M, N, D, D] = size(La);
G = eLa;
J = zeros(M, N, D, D);
mLa = eLa;
MLa = eLa;
for i = 1:D
    for j = (i+1):D
        J(:, :, i, j)   = 1 ./ (La(:, :, j, j) - La(:, :, i, i));
        J(:, :, j, i)   = - J(:, :, i, j);

        mLa(:, :, i, j) = min(eLa(:, :, i, i), eLa(:, :, j, j));
        MLa(:, :, i, j) = max(eLa(:, :, i, i), eLa(:, :, j, j));
        G(:, :, i, j)   = -(eLa(:, :, i, i) - eLa(:, :, j, j)) .* J(:, :, i, j);

        G(:, :, j, i)   = G(:, :, i, j);
        mLa(:, :, j, i) = mLa(:, :, i, j);
        MLa(:, :, j, i) = MLa(:, :, i, j);
    end
end
G(isnan(G) | isinf(G)) = 0; % In case of equals eigenvalues
G(G < mLa) = mLa(G < mLa);
G(G > MLa) = MLa(G > MLa);
