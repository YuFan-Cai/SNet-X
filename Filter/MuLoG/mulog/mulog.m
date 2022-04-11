function [Sigma, x, y] = mulog(C, L, denoiser, varargin)
%% Implements MuLoG as described in
%
%    Deledalle, C.A., Denis, L., Tabti, S. and Tupin, F., 2017.
%    MuLoG, or How to apply Gaussian denoisers to multi-channel SAR
%    speckle reduction?
%
% Input/Output
%
%    C          a M x N array, OR
%               a M x N field of D x D covariance matrices
%               size D x D x M x N
%
%    Sigma      estimated array, OR field of covariane matrices
%
%    x          estimated log-channels for output Sigma
%
%    y          extracted log-channels for input C
%
%    L          the number of looks: parameter of the Wishart
%               distribution linking C and Sigma
%               For SLC images: L = 1
%               For MLC images: L = ENL
%
%    DENOISER   handle on a function x = DENOISER(y, sig, ...)
%               removing noise for an image y damaged by Gaussian
%               noise with varaince sig^2
%
%               extra arguments of MULOG are passed to DENOISER
%
% Optional arguments
%
%    BETA       inner parameter of ADMM
%               default: 1 + 2 / L
%
%    LAMBDA     regularization parameter
%               default: 1
%
%    T          the number of iterations of ADMM
%               default: 6
%
%    K          the number of iterations for the Newton descent
%               default: 10
%
%    R          rank of input covariance matrix [1-D]
%               default: min(L, D)
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



options     = makeoptions(varargin{:});

%% Input data format conversion
if ismatrix(C)
    reshaped = true;
    D = 1;
    [M, N] = size(C);
    C = reshape(C, [M, N, 1, 1]);
elseif ndims(C) == 4
    reshaped = false;
    [M, N, ~, D] = size(C);
else
    error(['C must be a M x N field of D x D covariance matrices of ' ...
           'size DxDxMxN array']);
end
C = double(C);

%% Remove negative and zero entries from the diagonal (safeguard)
d = C(:, :, 1:D, 1:D);
d(abs(d) <= 0) = min(abs(d(abs(d) > 0)));
C(:, :, 1:D, 1:D) = d;

%% Avoid equal eigenvalues and other numerical issues that arise
%% when using spfunmatrices and eigmatrices
C = stabmatrices(C);

%% Diagonal loading
%   - make the energy closer to be strictly convex
condo = getoptions(options, 'condo', 1e-4);
C = condloading(C, condo);

%% Initialization is debiased in intensity and better conditionned
%   - only the initialization (helps to converge faster)
%   - the original C will be given for the data fidelity term
%   - get a more accurate matrix logarithm
C_init = C * L / exp(psi(L));
C_init = diagloading(C_init, L);

%% Log-channel decomposition: y = OmegaInv(log C)
[y, pca] = covariance_matrix_field_to_log_channels(C_init, L);

%% Define fidelity term
proxllk = @(x, C, lambda, varargin) ...
          proxfishtipp(x, C, lambda, L, D, pca, y, varargin{:});

%% Run Plug-and-play ADMM
beta = getoptions(options, 'beta', 1 + 2 / L);
x = admm(C, ...
         denoiser, ...
         'beta', beta, ...
         'sig', 1, ...
         'init', y, ...
         'proxllk', proxllk, ...
         varargin{:});

%% Return to the initial representation: Sigma = exp(Omega(x))
Sigma = log_channels_to_covariance_matrix_field(x, pca);

%% Output data format conversion
if reshaped
    Sigma = reshape(Sigma, [M, N]);
end

end

%%% Extract log channels from a given covariance matrix field
function [y, pca] = covariance_matrix_field_to_log_channels(C, L)

[M, N, ~, D] = size(C);

% 1. Find the PCA parameters (A, b) decorrelating the channels
alpha  = getchannels(logmatrices(C), D);
alpha  = reshape(alpha, [M*N, D^2])';
pca.b  = mean(alpha, 2);
tmp    = alpha - pca.b;
[pca.A, pca.S, ~] = svds(tmp / sqrt(M * N), D^2);
tmp    = pca.A' * tmp;

% 2. Estimate sigma on each channel (Phi)
pca.sigma = estimate_sigma(reshape(tmp, [D^2, M, N]), D, L);

% 3. Normalize to a noise variance of 1
y  = tmp ./ pca.sigma;
y  = reshape(y', [M, N, D^2]);
clear tmp;

end

%%% Reconstruct a covariance matrix field from the given log channels
function Sigma = log_channels_to_covariance_matrix_field(x, pca)

[M, N, D] = size(x);
D     = sqrt(D);
x     = reshape(x, [M * N, D^2])';
tmp   = x .* pca.sigma;
alpha = pca.A * tmp + pca.b;
clear tmp;
alpha = reshape(alpha', [M, N, D^2]);
Sigma = expmatrices(getmatrices(alpha, D));

end

%%% Apply a function f with proper reshaping
function x = applyr(x, f, M, N, D)

x = reshape(x, [M * N, D^2])';
x = f(x);
x = reshape(x', [M, N, size(x, 1)]);

end

%%% Prox of the Wishart Fisher-Tippet log likelihood

% Split the input onto the different cores
function z = proxfishtipp(x, C, lambda, L, D, pca, y, varargin)

options   = makeoptions(varargin{:});
cbwaitbar = getoptions(options, 'waitbar', @(dummy) []);
K         = getoptions(options, 'K', 10);
debug     = getoptions(options, 'debug', false);
parallel  = getoptions(options, 'parallel', true);

[M, N, P] = size(x);
if ~parallel
    z = subproxfishtipp(x, C, ...
                        lambda, L, D, pca, ...
                        y, K, ...
                        debug, ...
                        cbwaitbar);
else
    NC = feature('Numcores');
    q = ceil(M / NC);
    parfor c = 1:NC
        idx = floor((c-1)*q)+1:min(floor(c*q), M);
        zs{c} = subproxfishtipp(x(idx, :, :), C(idx, :, :, :), ...
                                lambda, L, D, pca, ...
                                y(idx, :, :), K, ...
                                debug, ...
                                @(p) cbwaitbar((NC - c + p) / NC));
        cbwaitbar((NC-c+1) / NC);
    end
    z = zeros(M, N, P);
    for c = 1:NC
        idx = floor((c-1)*q)+1:min(floor(c*q), M);
        z(idx, :, :) = zs{c};
    end
end

end

% Main function
function x = subproxfishtipp(z, C, lambda, L, ~, pca, y, K, debug, cbwaitbar)

[M, N, ~, D] = size(C);

Id = reshape(eye(D), [1, 1, D, D]);

Sigma   = diag(pca.sigma);
b       = pca.b;
A       = pca.A * Sigma;

x = (z ./lambda + y) ./ (1 + 1 ./ lambda);

if debug
    fprintf('%.2f %s\n', L, repmat('-', 1, 60));
end

Omega_aff = @(x) getmatrices(applyr(x, @(z) (A * z + b), M, N, D), D);
Omega_adj = @(x) applyr(getmatrices_adj(x, D), @(z) A' * z, M, N, D);
Omega_lin = @(x) getmatrices(applyr(x, @(z) A * z, M, N, D), D);
for k = 1:K
    % Compute gradient
    Omega_x = Omega_aff(x);
    [E, La, E_adj] = eigmatrices(-Omega_x);
    [G, J]  = expmatrices_getG(La);
    EA      = mulmatrices(E_adj, C, E, 'HNnhn');
    g       = x - z + lambda * L * Omega_adj(Id - mulmatrices(E, G .* EA, E_adj, 'HNnhn'));

    % Compute gradient direction
    d       = g ./ sqrt(sum(g.^2, 3));
    d(isnan(d) | isinf(d)) = 1;
    d       = d ./ sqrt(sum(d.^2, 3));

    % Compute second derivative in d
    Omega_d = Omega_lin(d);
    EB      = mulmatrices(E_adj, Omega_d, E, 'HNnhn');
    F       = expmatrices_getF(G, J, EA, EB);
    dotp    = real(sum(sum(EB .* conj(F), 3), 4));
    gamma   = 1 + lambda * L * abs(dotp);

    % Display loss, grad, hessian for debugging
    if debug
        show_debug_stats(x, z, C, g, d, 1 + lambda * L * dotp, lambda, L, Omega_aff);
    end

    % Quasi-Newton update
    x = x - g ./ gamma / 2;

    cbwaitbar(k / K);
end

end

function show_debug_stats(x, z, C, g, d, gamma, lambda, L, Omega_aff)

try
    inparfor = ~isempty(getCurrentTask());
catch
    inparfor = false;
end
if inparfor
    error('Cannot debug in Parallel loop');
end

[M, N, ~, D] = size(C);

L2loss = @(x) sum((x - z).^2, 3) / 2;
WFloss = @(S) L * tracematrices(S + mulmatrices(C, expmatrices(-S), 'Nhh'));
F = @(x) L2loss(x) + lambda * WFloss(Omega_aff(x));
F_x = F(x);
state = deterministic('on');
u = randn(M, N, D^2);
deterministic('off', state);
eps = 1e-8;
DF_u_CR = sum(g .* u, 3);
DF_u_FD = (F(x + eps * u) - F_x) / eps;
eps = 1e-4;
d_H_d_CR = gamma;
d_H_d_FD = (F(x + eps * d) - 2 * F_x + F(x - eps * d)) / eps^2;
fprintf('Loss: \t\t %f \t (|Grad|: %f)\n',     mean(F_x(:)),      mean(g(:).^2));
fprintf('- Grad accuracy:\t %f \t vs \t %f\n', mean(DF_u_CR(:)),  mean(DF_u_FD(:)));
fprintf('- Hess accuracy:\t %f \t vs \t %f\n', mean(d_H_d_CR(:)), mean(d_H_d_FD(:)));

end

%%% Extract the channels of a DxD hermitian matrix field
% Here the sqrt(2) makes getmatrices and getchannels
% adjoint and inverse of each others when ratio = 1.
% Moreover, it makes the standard deviation of the noise
% on off-diagonal elements similar to those of the diagonal.
function y = getchannels(x, ~)

ratio = 1;
[M, N, ~, D] = size(x);
y = zeros(M, N, D^2);
d = 1;
for k = 1:D
    y(:, :, d) = real(x(:, :, k, k));
    d = d + 1;
end
for k = 1:D
    for l = k+1:D
        y(:, :, d) = real(x(:, :, k, l)) * sqrt(2) / ratio;
        d = d + 1;
        y(:, :, d) = imag(x(:, :, k, l)) * sqrt(2) / ratio;
        d = d + 1;
    end
end

end

%%% Adjoint of getmatrices
function y = getmatrices_adj(x, ~)

ratio = 1;
[M, N, ~, D] = size(x);
y = zeros(M, N, D^2);
d = 1;
for k = 1:D
    y(:, :, d) = real(squeeze(x(:, :, k, k)));
    d = d + 1;
end
for k = 1:D
    for l = k+1:D
        y(:, :, d) = real(x(:, :, k, l)) * sqrt(2) * ratio;
        d = d + 1;
        y(:, :, d) = imag(x(:, :, k, l)) * sqrt(2) * ratio;
        d = d + 1;
    end
end

end

%%% Create a DxD hermitian matrix field form its channel decomposition
function y = getmatrices(x, D)

ratio = 1;
[M, N, ~] = size(x);
y = zeros(M, N, D, D);
d = 1;
for k = 1:D
    y(:, :, k, k) = x(:, :, d);
    d = d + 1;
end
for k = 1:D
    for l = (k+1):D
        y(:, :, k, l) = y(:, :, k, l) + ...
            x(:, :, d) / sqrt(2) * ratio;
        y(:, :, l, k) = y(:, :, l, k) + ...
            x(:, :, d) / sqrt(2) * ratio;
        d = d + 1;
        y(:, :, k, l) = y(:, :, k, l) + ...
            sqrt(-1) * x(:, :, d) / sqrt(2) * ratio;
        y(:, :, l, k) = y(:, :, l, k) + ...
            -sqrt(-1) * x(:, :, d) / sqrt(2) * ratio;
        d = d + 1;
    end
end

end

%%% Predict the noise standard deviation on each channel
function sigma = estimate_sigma(x, D, L)

var = 0;
for i = 0:D-1
    var = var + psi(1, max(D,L) - i);
end
if D == 1
    sigma = sqrt(var);
else
    sigma = vstdmad(x);
end

end

%%% Estimate the noise standard deviation on each channel
function s = vstdmad(x)

x = permute(x, [2 3 1]);
K = size(x, 3);
s = zeros(K, 1);
for k = 1:K
    s(k) = stdmad(x(:, :, k));
end

end
