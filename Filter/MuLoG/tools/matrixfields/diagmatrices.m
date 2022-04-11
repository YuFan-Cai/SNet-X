function L = diagmatrices(C)

switch ndims(C)
    case 5
        [M, N, D, D, T] = size(C);
        L = zeros(M, N, D, T);
        for t = 1:T
            L(:, :, :, t) = diagmatrices(C(:, :, :, :, t));
        end
    case 4
        [M, N, D, D] = size(C);
        L = zeros(M, N, D);
        for k = 1:D
            L(:, :, k) = C(:, :, k, k);
        end
    case 3
        [M, N, D] = size(C);
        L = zeros(M, N, D, D);
        for k = 1:D
            L(:, :, k, k) = C(:, :, k);
        end
    case 2
        L  = C;
    otherwise
        error('Unexpected use of diagmatrices');
end
