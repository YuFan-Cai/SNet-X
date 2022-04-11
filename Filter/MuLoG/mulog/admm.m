function x = admm(y, denoiser, varargin)
%% Use Plug-and-Play ADMM to solve
%
%    x in argmin - log p(y | x) + lambda R(x)
%
% Input/Ouput:
%
%    Y          the observation (size MxNxK)
%
%    X          the solution (size MxNxK)
%
%    DENOISER   handle on a function x = DENOISER(y, lambda, ...)
%               solving
%
%                  x = argmin 1/2 ||y - x||^2 + lambda R(x)
%
%               extra arguments of ADMM are passed to DENOISER
%
% Optional arguments
%
%    INIT       initialization for x
%               default: y
%
%    SIG        noise in y should be homoscedastic with variance sig^2
%               default: 1
%
%    PROXLLK    prox of -log p(y | x) ie a function
%
%                  x = PROXLLK(z, y, lambda, ...)
%
%               solving
%
%                  x = argmin 1/2 ||z - x||^2 - lambda log p(y | x)
%
%               extra arguments of ADMM are passed to PROXLLK
%
%               default: -log p(y | x) = 1/2 ||y - x||^2 / sig^2
%
%    LAMBDA     regularization parameter
%               default: 1
%
%    BETA       inner parameter of ADMM
%               default: 1
%
%    T          the number of iterations
%               default: 6
%
%    CBWAITBAR  handle on a function CBWAITBAR(percentage) showing
%               a progress bar. Percentage lies in [0, 1].
%               default: @(p) []
%
%    CBWAITBAR  handle on a function CBWAITBAR(percentage) showing
%               a progress bar. Percentage lies in [0, 1].
%               default: @(p) []
%
% References
%
%    [1] A dual algorithm for the solution of nonlinear variational
%    problems via finite element approximation
%    Gabay, Daniel and Mercier, Bertrand
%    Computers & Mathematics with Applications, vol. 2, no. 1, pp; 17--40, 1976
%
%    [2] Multiplicative noise removal using variable splitting and constrained optimization
%    Bioucas-Dias, José M and Figueiredo, Mario AT
%    IEEE Transactions on Image Processing, vol. 19, no. 7, pp. 1720--1730, 2010
%
%    [3] Plug-and-play ADMM for image restoration: Fixed point convergence and applications,
%    Chan, Stanley H and Wang, Xiran and Elgendy, Omar A,
%    IEEE Transactions on Computational Imaging, 2016
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
x           = getoptions(options, 'init', y);          % Initialization
lambda      = getoptions(options, 'lambda', 1);        % Lambda
T           = getoptions(options, 'T', 6);             % Number of iterations

%% Callback towards the prox of the neg log likelihood
%   argmin 1/2 || x - y || - lambda log p(y | x)
if isfield(options, 'proxllk')
    proxllk = getoptions(options, 'proxllk', [], 1);
    sig     = getoptions(options, 'sig', 1);
else
    sig     = getoptions(options, 'sig', [], 1);
    proxllk = getoptions(options, ...
                         'proxllk', @(x, y, lambda, varargin) proxgauss(x, y, lambda, sig));
end

%% Default value of beta
beta        = getoptions(options, 'beta', 1) * lambda ./ sig.^2;

%% Callback for a waitbar
cbwaitbar   = getoptions(options, 'waitbar', @(dummy) []);

% Parameters that guarantee convergence accoding to [3]
eta = 0.95;
gamma = 1.05;

%% Quit if T=0
if T == 0
    return;
end

%% Core
z = denoiser(x, sig, ...
             'waitbar', @(p) cbwaitbar(p / (T+0.5)), ...
             varargin{:});
e = z - x;
x = z + e;
xold = x;
zold = z;
eold = e;
dold = inf;
for k = 1:T
    cbwaitbar(k / (T + 0.5));

    % Prox of log-likelihood
    x = proxllk(z + e, y, 1 ./ beta, ...
                'waitbar', @(p) cbwaitbar((k + p / 2) / (T+0.5)), ...
                varargin{:});

    % Quit if last iteration
    if k == T
        break;
    end

    % Denoiser
    z = denoiser(x - e, sqrt(lambda ./ beta), ...
                 'waitbar', @(p) cbwaitbar((k + 0.5 + p / 2) / (T+0.5)), ...
                 varargin{:});

    % Residual update
    e = e + z - x;

    % Block to guarantee convergence according to [3]
    d = norm(xold(:) - x(:)) + norm(zold(:) - z(:)) + norm(eold(:) - e(:));
    if d > eta * dold
        beta = gamma * beta;
    end
    xold = x;
    zold = z;
    eold = e;
    dold = d;

end
cbwaitbar(1);

end

%% Prox for Gaussian noise N(0, sigma^2)
function z = proxgauss(x, y, lambda, sig)

z = (x + lambda * y / sig^2) ./ (1 + lambda / sig^2);

end
