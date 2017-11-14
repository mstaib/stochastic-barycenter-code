function [ val ] = compute_single_ot_distance_mosek( C, a, b )
%COMPUTE_SINGLE_OT_DISTANCE Summary of this function goes here
%   Detailed explanation goes here

n1 = length(a);
n2 = length(b);

i = zeros(n1*n2,1);
j = zeros(n1*n2,1);
s = ones(n1*n2,1);
for kk=1:n2
    i( (1:n1) + (kk-1)*n1 ) = (1:n1);
    j( (1:n1) + (kk-1)*n1 ) = (1:n1) + (kk-1)*n1;
end
A1 = sparse(i,j,s);

i = zeros(n1*n2,1);
j = zeros(n1*n2,1);
s = ones(n1*n2,1);
for kk=1:n2
    i( (1:n1) + (kk-1)*n1 ) = kk;
    j( (1:n1) + (kk-1)*n1 ) = (1:n1) + (kk-1)*n1;
end
A2 = sparse(i,j,s);

% param.MSK_IPAR_LOG = 0;
% param.MSK_IPAR_LOG_HEAD = 0;
% param.MSK_IPAR_NUM_THREADS = 1;
% 
% prob.c = [C(:)];
% prob.a = [A1; A2];
% prob.blc = [a(:); b(:)];
% prob.buc = [a(:); b(:)];
% prob.blx = zeros(n1*n2,1);
% prob.bux = ones(n1*n2,1);
% [r,res] = mosekopt('minimize echo(0)', prob, param);
% 
% % for simplex solver
% val = res.sol.bas.pobjval;
    
[~,val] = linprog(C(:), [], [], [A1; A2], [a(:); b(:)], zeros(n1*n2,1), ones(n1*n2,1));

end