function [w, obj, exitflag, output, lambda] = WASP(colMeasure, distMat)

%% WASP function to calculate the Wasserstein Barycenter of a list of
% empirical measures and pairwise distance between atoms.
% Input:
% ======
% colMeasure:
% cell containing the wts of atoms of empirical measures.
%
% distMat:
% cell containing pairwise distance between atoms of empirical measures and
% the Wasserstein barycenter.
%
% Output:
% ======
% output of solving the LP (31) in the manuscript using ’linprog’

% each number of vars is nsample * nbar

nsubset = length(distMat);
vecColMeasure = cell2mat(cellfun(@(x) x', colMeasure, 'UniformOutput', false))';
nsample = size(vecColMeasure, 1);

vecDistMat = cell2mat(distMat);
vecDistMat = vecDistMat(:);
nbar = size(distMat{1}, 1);

fmatCell = cellfun(@(x) kron(ones(1, size(x, 1)), speye(nbar)), colMeasure, 'UniformOutput', false);
fmat = blkdiag(fmatCell{:}); % F

hmatCell = cellfun(@(x) kron(speye(size(x, 1)), ones(1, nbar)), colMeasure, 'UniformOutput', false);
hmat = blkdiag(hmatCell{:}); % H

gmat = kron(ones(nsubset, 1), speye(nbar)); % G

% Aeq in the matlab linprog function
aMat = sparse(1 + size(fmat,1) + size(hmat,1), nsample * nbar + nbar);
aMat(1, (nsample * nbar + 1):end) = 1;
aMat(2:(size(fmat,1)+1), 1:size(fmat,2)) = fmat;
aMat((size(fmat,1)+2):end, 1:size(hmat,2)) = hmat;
aMat(2:(size(fmat,1)+1), (size(fmat,2)+1):end) = -gmat;
% aMat = [zeros(1, nsample * nbar) ones(1, nbar);
%         fmat -gmat; 
%         hmat zeros(nsample, nbar)
%         ];
    
% beq in the matlab lingprog function
bVec = sparse(1 + nsubset * nbar + size(vecColMeasure,1),1);
bVec(1,1) = 1;
bVec((nsubset * nbar + 2):end,1) = vecColMeasure;
% bVec = [1;
%         zeros(nsubset * nbar, 1);
%         vecColMeasure
%         ];
    
% f in the matlab linprog function
costVec = sparse(size(vecDistMat,1) + nbar, 1);
costVec(1:size(vecDistMat,1),1) = vecDistMat;
% costVec = [vecDistMat;
%            zeros(nbar, 1)];

% upper bds = 1 and lower bds = 0
lbd = sparse(nsample*nbar + nbar, 1);
ubd = ones(nsample*nbar + nbar, 1);

[optSol, obj, exitflag, output, lambda] = linprog(costVec, [], [], aMat, bVec, lbd, ubd);
%[res] = msklpopt(costVec, aMat, bVec, bVec, lbd, ubd);
%sol = res.sol;

%optSol = sol.itr.xx';
w = optSol(end-nbar+1:end);
%obj = sol.itr.pobjval;