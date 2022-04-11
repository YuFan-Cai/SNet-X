function T = detmatrices(C)

T = spfunscalars(C, @(x) prod(x, 3));
