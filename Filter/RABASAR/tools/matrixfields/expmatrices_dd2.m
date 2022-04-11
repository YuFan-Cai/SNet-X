function J2CAB = expmatrices_dd2(C, A, B, JCA_)
%% This function applies a the matrix exponential
%  to a 2d hermitian matrix field (an image of matrices).
%  Is is optimized for hermitian matrices of size up to 3x3.
%
%    Deledalle, C.A., Denis, L., Tabti, S. and Tupin, F., 2017.
%    Closed-form expression of the eigen decomposition
%    of 2 x 2 and 3 x 3 Hermitian matrices
%
%  Warning: if the matrices in input are not hermitian, the output
%  of this function will be nonsense.
%
% Input/Output
%
%    C          a M x N field of D x D hermitian matrices
%               size D x D x M x N
%               Input:  eigenvalues are lambda_i
%               Output: eigenvalues are exp(lambda_i)
%                       eigenvectors are preserved
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


[~, ~, D, D] = size(B);
if exist('JCA_', 'var')
    E  = JCA_.E;
    E_adj = JCA_.E_adj;
    G  = JCA_.G;
    J  = JCA_.J;
    EA = JCA_.EA;
else
    [E, La] = eigmatrices(C);
    E_adj   = adjmatrices(E);
    [G, J]  = expmatrices_getG(La);
    EA      = mulmatrices(E_adj, A, E, 'HNnhn');
end

EB = mulmatrices(E_adj, B, E, 'HNnhn');
F = expmatrices_getF(G, J, EA, EB);
J2CAB = mulmatrices(E, F, E_adj, 'HNnhn');
