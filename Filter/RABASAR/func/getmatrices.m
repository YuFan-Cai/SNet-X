function y = getmatrices(x, D)

ratio = 1;
[M, N, ~] = size(x);
y = zeros(D, D, M, N);
d = 1;
for k = 1:D
    y(k, k, :, :) = reshape(x(:, :, d), [1 1 M N]);
    d = d + 1;
end
for k = 1:D
    for l = (k+1):D
        y(k, l, :, :) = y(k, l, :, :) + ...
            reshape(x(:, :, d), [1 1 M N]) / sqrt(2) * ratio;
        y(l, k, :, :) = y(l, k, :, :) + ...
            reshape(x(:, :, d), [1 1 M N]) / sqrt(2) * ratio;
        d = d + 1;
        y(k, l, :, :) = y(k, l, :, :) + ...
            sqrt(-1) * reshape(x(:, :, d), [1 1 M N]) / sqrt(2) * ratio;
        y(l, k, :, :) = y(l, k, :, :) + ...
            -sqrt(-1) * reshape(x(:, :, d), [1 1 M N]) / sqrt(2) * ratio;
        d = d + 1;
    end
end

end
