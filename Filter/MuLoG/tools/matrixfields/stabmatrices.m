function C = stabmatrices(C)
%% Stabilize the input matrix in a way that
%
%  - The modification is small: Output ≈ Input
%  - The result does not have exact zero entries,
%  - The reuslt does not have exact equal eigenvalues,
%  - Functions implementing the procedure described here
%
%    Deledalle, C.A., Denis, L., Tabti, S. and Tupin, F., 2017.
%    Closed-form expression of the eigen decomposition
%    of 2 x 2 and 3 x 3 Hermitian matrices
%
%    do not suffer much from numerical instabilities.
%
%  Warning: using this function repeatidly will accumulate imprecisions
%  that will quickly accumulate and leads to signficant errors.
%  For instance:
%
%      invmatrices(stabmatrices(invmatrices(stabmatrices(C)))) ≉ C
%
%  while
%
%      invmatrices(invmatrices(stabmatrices(C))) ≈ C
%
%  Try not to use that function too often then!
%
%
% Input/Output
%
%    C        a M x N field of D x D covariance matrices
%             size D x D x M x N
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


% Diagonal elements are randomly modified after 6 significant digits.
% Off diagonal elemements are randomly modified in a way that the
% correlations are modified after 6 significant digits.

eps = 1e-6; %1e-6
[M, N, ~, D] = size(C);
state = deterministic('on', 4242);
for i = 1:D
    C(:,:,i,i) = (1 + eps * (2 * rand(M, N) - 1)) .* C(:,:,i,i);
end
for i = 1:D
    for j = (i+1):D
        zeta = sqrt(C(:,:,i,i) .* C(:,:,j,j));
        C(:,:,i,j) = ...
            abs(abs(C(:,:,i,j)) + eps * zeta .* (2 * rand(M, N) - 1)) .* ...
            exp(sqrt(-1) .* angle(C(:,:,i,j)));
        C(:,:,j,i) = conj(C(:,:,i,j));
    end
end
deterministic('off', state);
