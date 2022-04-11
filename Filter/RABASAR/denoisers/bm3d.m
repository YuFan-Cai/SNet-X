function x = bm3d(y, sig, varargin)
%% Wraper to the BM3D implementation of the authors
%
% Input/Ouput:
%
%    Y          the observation (size MxNxK)
%
%    X          the solution (size MxNxK)
%
%    SIG        the standard deviation of the noise
%
%
% Reference
%
%    K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian,
%    Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering,
%    IEEE Transactions on Image Processing, vol. 16, no. 8, August, 2007.
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



options   = makeoptions(varargin{:});
cbwaitbar = getoptions(options, 'waitbar', @(dummy) []);

x = zeros(size(y));
m = min(y(:));
M = max(y(:));
K = size(y, 3);
if isscalar(sig)
    sig = sig * ones(K ,1);
end
parfor k = 1:K
    tau = 255 * sig(k) / (M - m);
    if tau > 40
        ratio = 40 / tau;
    else
        ratio = 1;
    end
    [~, x(:,:,k)] = BM3D(1, ratio * (y(:,:,k) - m) / (M - m), ...
                         ratio * tau, 'np', 0);
    x(:, :, k) = (M - m) * x(:, :, k) / ratio + m;
    cbwaitbar((K-k+1) / K);
end
x(isnan(x)) = 0;
cbwaitbar(1);
