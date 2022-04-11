function C = condloading(C, cnb)
%% Perform conditionment loading as described in
%
%    MuLoG v2: An improved Multi-channel Logarithm with
%    Gaussian denoising algorithm for SAR despeckeling
%    Charles-Alban Deledalle, Loic Denis, Florence Tupin
%
% Input/Output
%
%    C        a M x N field of D x D covariance matrices
%               size D x D x M x N
%
%    cnb        the inverse condion number
%
%
%
% License
%
% This software is governed by the CeCILL license under French law and
% abiding by the rules of distribution of free software. You can use,
% modify and/ or redistribute the software under the terms of the CeCILL
% license as circulated by CEA, CNRS and INRIA at the following URL
% "http://www.cecill.info".
%
% As a counterpart to the access to the source code and rights to copy,
% modify and redistribute granted by the license, users are provided only
% with a limited warranty and the software's author, the holder of the
% economic rights, and the successive licensors have only limited
% liability.
%
% In this respect, the user's attention is drawn to the risks associated
% with loading, using, modifying and/or developing or reproducing the
% software by the user in light of its specific status of free software,
% that may mean that it is complicated to manipulate, and that also
% therefore means that it is reserved for developers and experienced
% professionals having in-depth computer knowledge. Users are therefore
% encouraged to load and test the software's suitability as regards their
% requirements in conditions enabling the security of their systems and/or
% data to be ensured and, more generally, to use and operate it in the
% same conditions as regards security.
%
% The fact that you are presently reading this means that you have had
% knowledge of the CeCILL license and that you accept its terms.
%
% Copyright 2020 Charles Deledalle
% Email charles-alban.deledalle@math.u-bordeaux.fr


[M, N, ~, D] = size(C);
if D == 1
    return
end

Cr = C;
Ma = - inf * ones(M, N, 1, 1);
mi = + inf * ones(M, N, 1, 1);
[U, La] = eigmatrices(Cr);
for k = 1:D
    Ma = max(Ma, abs(La(:, :, k, k)));
    mi = min(mi, abs(La(:, :, k, k)));
end
cond = max(mi ./ Ma, cnb);
for k = 1:D
    La(:, :, k, k) = ...
        + (La(:, :, k, k) - mi) ./ (Ma - mi) .* Ma .* (1 - cond) ...
        + Ma .* cond;
end
UT = adjmatrices(U);
C = mulmatrices(U, La, UT, 'HNndn');
