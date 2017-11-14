rand('seed',0);
N = 1000;

% Size of the grid
%grdsize = 100;
grdsize = sqrt(N);

% var-covar matrix for Multivariate Normal Distribution
sig = [1 1.5; 1.5 3];

% means for 3 subset posteriors
mu1 = [3 2];
mu2 = [2 3];
mu3 = [3 3];

% atoms for 3 subset posteriors; they only differ in their means.
spost{1} = mvnrnd(mu1, sig, N);%50);
spost{2} = mvnrnd(mu2, sig, N);%75);
spost{3} = mvnrnd(mu3, sig, N);%100);

% atoms for the WASP by forming a grid
lbd1 = min(cellfun(@(x) x(1), cellfun(@(x) min(x), spost,'UniformOutput', false)));
lbd2 = min(cellfun(@(x) x(2), cellfun(@(x) min(x), spost,'UniformOutput', false)));
ubd1 = max(cellfun(@(x) x(1), cellfun(@(x) max(x), spost,'UniformOutput', false)));
ubd2 = max(cellfun(@(x) x(2), cellfun(@(x) max(x), spost,'UniformOutput', false)));
[opostx, oposty] = meshgrid(linspace(lbd1, ubd1, grdsize), linspace(lbd2, ubd2, grdsize));
opost = [opostx(:) oposty(:)];

% calculate the pair-wise sq. euclidean distance between the atoms of subset
% posteriors and BarPost atoms
%m11 = diag(spost{1} * spost{1}');
%m22 = diag(spost{2} * spost{2}');
%m33 = diag(spost{3} * spost{3}');

%m00 = diag(opost * opost');
%m01 = opost * spost{1}';
%m02 = opost * spost{2}';
%m03 = opost * spost{3}';

% calculate distance between atoms
opost_extra = reshape(opost, size(opost,1), 1, size(opost,2));
spost1_extra = reshape(spost{1}, 1, size(spost{1},1), size(spost{1},2));
spost2_extra = reshape(spost{2}, 1, size(spost{2},1), size(spost{2},2));
spost3_extra = reshape(spost{3}, 1, size(spost{3},1), size(spost{3},2));

d01 = sum((opost_extra - spost1_extra).^2, 3);
d02 = sum((opost_extra - spost2_extra).^2, 3);
d03 = sum((opost_extra - spost3_extra).^2, 3);

%d01 = bsxfun(@plus, bsxfun(@plus, -2 * m01, m11'), m00);
%d02 = bsxfun(@plus, bsxfun(@plus, -2 * m02, m22'), m00);
%d03 = bsxfun(@plus, bsxfun(@plus, -2 * m03, m33'), m00);

% initialize the wts b_1 ... b_K for subset posteriors atoms; see Eq. 21
b = cellfun(@(x) ones(size(x, 1), 1) / size(x, 1), spost, 'UniformOutput', false);
colMeasure = b;
distMat{1} = d01;
distMat{2} = d02;
distMat{3} = d03;

tic
[optSol, optObj] = WASP(colMeasure, distMat); %, exitflag, output, lambda]
toc