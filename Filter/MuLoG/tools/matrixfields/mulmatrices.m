function z = mulmatrices(varargin)
%%% Multiply matrix-wise two or more hermitian matrix fields
%
% R = mulmatrices(X, Y, 'Abc') : R = X * Y
%
%   X has property 'b', Y has property `c`
%   R is expected to have property `A`.
%
% R = mulmatrices(X, Y, Z, 'ABcde') : R = (X * Y) * Z
%
%   X has property 'c', Y has property `d`, Z has property `e`
%   (X * Y) is expected to have property `B`.
%   R is expected to have property `A`.
%
% R = mulmatrices(X, Y, Z, 'AbCde') : R = X * (Y * Z)
%
%   X has property 'b', Y has property `d`, Z has property `e`
%   (Y * Z) is expected to have property `C`.
%   R is expected to have property `A`.
%
% The available properties are
%
%   'n' general matrix
%   'h' Hermitian matrix
%   'd' diagonal matrix


if ischar(varargin{end})
    scheme = varargin{end};
    varargin = varargin(1:(end-1));
else
    scheme = [repmat('N', [1, length(varargin)-1]) ...
              repmat('n', [1, length(varargin)])];
end

[z, scheme, ~] = recmulmatrices(1, scheme, varargin{:});

if ~(length(scheme) == 1 &&  isstrprop(scheme(1), 'lower'))
   warning('Unexpected scheme format and/or number of arguments');
end

function [z, scheme, k] = recmulmatrices(k, scheme, varargin)

optr = lower(scheme(1));
opt1 = lower(scheme(2));
if length(scheme) >= 1 && isstrprop(scheme(2), 'upper')
    [x, schemes, k] = recmulmatrices(k, scheme(2:end), varargin{:});
    scheme = [scheme(1) schemes];
else
    x = varargin{k};
end
opt2 = lower(scheme(3));
if length(scheme) >= 1 && isstrprop(scheme(3), 'upper')
    [y, schemes, k] = recmulmatrices(k + 1, scheme(3:end), varargin{:});
    scheme = [scheme(1:2) schemes];
else
    y = varargin{k+1};
end
z = submulmatrices(x, y, opt1, opt2, optr);
k = k + 1;
scheme = [optr scheme(4:end)];

function z = submulmatrices(x, y, opt1, opt2, optr)

if ~(isnumeric(x) && isnumeric(y))
    error('Unexpected arguments');
end

if size(x, 4) ~= size(y, 3)
    error('Incompatible matrix sizes');
end
if size(x, 1) == size(y, 1) && size(x, 2) == size(y, 2) && ...
    size(x, 3) == size(y, 3) && size(x, 4) == size(y, 4) && ...
    size(x, 3) == size(x, 4)
    % Square matrices and same fields
    implem = 'matlab';
else
    % Different fields -> Need broadcasting
    implem = 'matlab';
end
%implem = 'mex';
switch implem
    case 'mex'
        z = submulmatrices_c(x, y, opt1, opt2, optr);
    case 'matlab'
        [Mx, Nx, Dx, D] = size(x);
        [My, Ny, ~, Dy] = size(y);
        M = max(Mx, My);
        N = max(Nx, Ny);
        z = zeros(M, N, Dx, Dy);
        switch [ optr ':' opt1 '-' opt2 ]
            case { 'n:n-n', 'n:h-h', 'n:n-h' }
                for k = 1:D
                    z = z + x(:, :, :, k) .* y(:, :, k, :);
                end
            case 'd:n-n'
                for i = 1:Dx % (Dx = Dy)
                    for k = 1:D
                        z(:,:,i,i) = z(:,:,i,i) + x(:,:,i,k) .* y(:,:,k,i);
                    end
                end
            case { 'h:n-n', 'h:n-h', 'h:h-h' }
                for i = 1:D
                    a = z(:,:,i,i:D);
                    for k = 1:D
                        a = a + x(:,:,i, k) .* y(:, :, k, i:D);
                    end
                    z(:, :, i,i:D) = a;
                    z(:, :, (i+1):D,i,:,:) = conj(a(:, :, 1,2:end));
                end
            case 'n:d-n'
                for k = 1:D
                    z(:,:,k,:) = x(:, :, k, k) .* y(:, :, k, :);
                end
            case 'd:d-n'
                for k = 1:D
                    z(k,k,:,:) = x(:, :, k, k) .* y(:, :, k, k);
                end
            case 'h:d-n'
                for k = 1:D
                    z(:,:,k,k:D) = x(:, :, k, k) .* y(:, :, k, k:D);
                    z(:,:,(k+1):D,k) = conj(z(:,:,k,(k+1):D));
                end
            case 'n:h-n'
                for i = 1:D
                    for j = 1:Dy
                        z(:,:,i,j) = z(:,:,i,j) + x(:, :, i, i) .* y(:, :, i, j);
                        for k = (i+1):D
                            a = x(:,:,i, k);
                            z(:,:,i,j) = z(:,:,i,j) + a .* y(:, :, k, j);
                            z(:,:,k,j) = z(:,:,k,j) + a .* y(:, :, i, j);
                        end
                    end
                end
            case 'n:n-d'
                for k = 1:D
                    z(:,:,:,k) = x(:, :, :, k) .* y(:, :, k, k);
                end
            case {'n:d-d', 'h:d-d', 'd:d-d'}
                for k = 1:D
                    z(:,:,k,k) = x(:, :, k, k) .* y(:, :, k, k);
                end
            otherwise
                error(['Unknown multiplication type: ' optr ':' opt1 '-' opt2 ]);
        end
end
