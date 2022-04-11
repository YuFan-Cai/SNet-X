function R = spfunscalars(C, func, stab)
%% This function applies a matrix-valued spectral function
%  to a 2d hermitian matrix field (an image of matrices).
%  Is is optimized for hermitian matrices of size up to 3x3.
%  It applies FUNC to the eigenvalues of the matrix field C
%  and reconstruct a matrix field with the same eigenvectors.
%  A detailed description is given in
%
%    Deledalle, C.A., Denis, L., Tabti, S. and Tupin, F., 2017.
%    Closed-form expression of the eigen decomposition
%    of 2 x 2 and 3 x 3 Hermitian matrices
%
%  Warning: if the matrices in input are not hermitian, the output
%  of this function will be nonsense.
%
%  Warning: if the matrices have zero entries or equal eigenvalue,
%  the function may crash or produce an unexpected result.
%  Please use stabmatrices on C, prior to calling this function,
%  in oreder to avoid such numerical issues.
%
% Input/Output
%
%    C          a M x N field of D x D hermitian matrices
%               size D x D x M x N
%               Input:  eigenvalues are lambda_i
%               Output: eigenvalues are func(lambda_i)
%                       eigenvectors are preserved
%
%    func       a scalar function
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
% Copyright 2017 Loïc Denis


[M, N, D, D] = size(C);

switch D
    case 1
        R = func(C);

    case 2 % apply formula for 2x2 matrices

        % extract channels
        a     = C(:, :, 1, 1);
        b     = C(:, :, 2, 2);
        c     = C(:, :, 2, 1);

        % compute relevant quantity
        delta = sqrt(4*abs(c).^2+(a-b).^2);

        % compute and update eigenvalues
        l1    = ((a+b+delta)/2);
        l2    = ((a+b-delta)/2);

        R  = func(cat(3, l1, l2));

    case 3 % apply formula for 3x3 matrices

                % extract channels
        a = C(:, :, 1, 1);
        b = C(:, :, 2, 2);
        c = C(:, :, 3, 3);
        d = C(:, :, 2, 1);
        e = C(:, :, 3, 2);
        f = C(:, :, 3, 1);

        % Avoid numerical errors (safeguard)
        f(abs(f(:)) == 0) = min(f(abs(f(:)) > 0));

        % compute relevant quantities
        x1 = a.^2 + b.^2 + c.^2 - a.*b - a.*c - b.*c +...
             3 *   ( real(d).^2 + imag(d).^2 + ...
                     real(e).^2 + imag(e).^2 + ...
                     real(f).^2 + imag(f).^2 );
        x2 = -       (2*a-b-c) .* (2*b-a-c) .* (2*c-a-b) ...
             +9  * ( (2*c-a-b) .* (real(d).^2 + imag(d).^2) + ...
                     (2*b-a-c) .* (real(f).^2 + imag(f).^2) + ...
                     (2*a-b-c) .* (real(e).^2 + imag(e).^2) )...
             -54 * ( (real(d).*real(e) - imag(d).*imag(e)) .* real(f) + ...
                     (real(d).*imag(e) + imag(d).*real(e)) .* imag(f) );
        phi     = atan(sqrt(4*x1.^3-x2.^2)./x2) + abs(sign(x2)).*(1-sign(x2))/2*pi;
        x1      = sqrt(x1);
        lambda1 = 1/3 * (a+b+c-2*x1.*cos(phi/3));
        lambda2 = 1/3 * (a+b+c+2*x1.*cos((phi-pi)/3));
        lambda3 = 1/3 * (a+b+c+2*x1.*cos((phi+pi)/3));
        clear x1;
        clear x2;

        R = func(cat(3, lambda1, lambda2, lambda3));

    otherwise % no formula... use Matlab function
        for i = 1:M
            for j = 1:N
                [E, L] = eig(squeeze(C(i, j, :, :)));
                R(i, j, :) = func(repmat(diag(L), [1, 1, D]));
            end
        end
end

end
