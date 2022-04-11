function [u_hat_t, u_hat_si, L_hat_si, u_hat_dsi, L_hat_dsi] = rabasar(v, L, t, varargin)
%% Implements Rabasar as described in
%
%    Rabasar paper
%
% Input/Output
%
%    V          a M x N x T array
%
%    L          the number of looks: parameter of the Gamma
%               distribution linking V and U_HAT_T
%               For SLC images: L = 1
%               For MLC images: L = ENL
%
%    T          the index of the image to filter in the stack
%
%    U_HAT_T    the filtered image at time t
%
%    U_HAT_SI   unfiltered super image (with BWAM)
%
%    U_HAT_DSI  filtered super image
%
%    L_HAT_SI   estimated number of look for U_HAT_SI
%
%    L_HAT_DSI  estimated number of look for U_HAT_DSI
%
% Optional arguments
%
%    THRS       a M x N array of thresholds for patch similarity
%               to build the binary weighted super-image
%               default: inf (i.e., BWAM = AM)
%
%    ENL_ESTIMATE handle on a function ENL_ESTIMATE(x) estimating
%               the equivalent number of looks on the image x
%               default: @(x) enl_logmoment_sliding(x, 30)
%
%    DENOISER   handle on a function DENOISER(y, lambda, ...)
%               default: bm3d
%
%    CBWAITBAR  handle on a function CBWAITBAR(percentage) showing
%               a progress bar. Percentage lies in [0, 1].
%               default: @(p) []
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

options       = makeoptions(varargin{:});
thrs          = getoptions(options, 'thrs', inf * ones(size(v(:,:,1))));
cbwaitbar     = getoptions(options, 'waitbar', @(dummy) []);
enl_estimator = getoptions(options, 'enl_estimator', @(x) enl_logmoment_sliding(x, 30));
denoiser      = getoptions(options, 'denoiser', @bm3d);

% NB: Notations follow the paper notations
u_hat_si      = bwsi(v, t, 7, thrs);
L_hat_si      = enl_estimator(u_hat_si);
u_hat_dsi     = molog(u_hat_si, L_hat_si, denoiser, ...
                      'waitbar', @(p) cbwaitbar(p/2));
L_hat_dsi     = enl_estimator(u_hat_dsi);
r             = v(:,:,t) ./ u_hat_dsi;
rho           = rulog(r, L, L_hat_dsi, denoiser, ...
                      'waitbar', @(p) cbwaitbar(.5 + p/2));
u_hat_t       = u_hat_dsi .* rho;
