function C = dfunmatrices(C, func)

[~, ~, D, ~] = size(C);

for k = 1:D
    C(:, :, k, k) = func(C(:, :, k, k));
end
