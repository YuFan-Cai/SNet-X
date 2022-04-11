function [img, gamma] = diagloading(img, L)
%% Perform diagonal loading as described in
%
%    Deledalle, C.A., Denis, L., Tabti, S. and Tupin, F., 2017.
%    MuLoG, or How to apply Gaussian denoisers to multi-channel SAR speckle reduction?
%
% Input/Output
%
%    img        a M x N field of D x D covariance matrices
%               size D x D x M x N
%
%    L          the number of looks
%               parameter of the Wishart distribution
%               linking C and Sigma
%               For SLC images: L = 1
%               For MLC images: L = ENL
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
% Copyright 2017 Charles Deledalle
% Email charles-alban.deledalle@math.u-bordeaux.fr

[M, N, ~, D] = size(img);

R = max(min(L, D), 1);

[x, y, k, l] = fftgrid(M, N, D, D);
tau       = sqrt(2 * D / R / (4 * pi));
kernel    = (k == 0) .* (l == 0) .* exp(-(x.^2 + y.^2) / (2 * tau^2));
kernel    = kernel ./ sum(kernel(:));
img_conv  = ifftn(fftn(img) .* fftn(kernel));
gamma     = ones(M, N, D, D);
gamma0    = ones(M, N, D, D);
for k = 1:D
    for l = 1:D
        gamma(:, :, k, l) = ...
            abs(img_conv(:, :, k, l) ./ ...
                sqrt(img_conv(:, :, k, k) .* img_conv(:, :, l, l)));
        gamma0(:, :, k, l) = ...
            abs(img(:, :, k, l) ./ ...
                sqrt(img(:, :, k, k) .* img(:, :, l, l)));
    end
end
img = fullmatrices(img, D, gamma ./ gamma0);

end

function x = fullmatrices(x, D, gamma)

if isscalar(gamma)
    gamma = (1-gamma) .* eye(D) + gamma;
end
x = bsxfun(@times, x, gamma);

end
