function y = getmatrices_adj(x, ~)

ratio = 1;
[~, D, M, N] = size(x);
y = zeros(M, N, D^2);
d = 1;
for k = 1:D
    y(:, :, d) = real(squeeze(x(k, k, :, :)));
    d = d + 1;
end
for k = 1:D
    for l = k+1:D
        y(:, :, d) = real(squeeze(x(k, l, :, :))) * sqrt(2) * ratio;
        d = d + 1;
        y(:, :, d) = imag(squeeze(x(k, l, :, :))) * sqrt(2) * ratio;
        d = d + 1;
    end
end

end
