function C = adjmatrices(C)

C = conj(permute(C, [1, 2, 4, 3]));
