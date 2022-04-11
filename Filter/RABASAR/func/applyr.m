function x = applyr(x, f, M, N, D)

x = reshape(x, [M * N, D^2])';
x = f(x);
x = reshape(x', [M, N, D^2]);

end
