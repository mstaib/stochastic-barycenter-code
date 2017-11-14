function [ val, val_hist ] = w2_distance(atoms1, w1, atoms2, w2)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

w1 = w1 / sum(w1);
w2 = w2 / sum(w2);

atoms1 = atoms1(w1 > 1e-7,:);
w1 = w1(w1 > 1e-7);
w1 = w1(:);

atoms2 = atoms2(w2 > 1e-7,:);
w2 = w2(w2 > 1e-7);
w2 = w2(:);

% we may have removed nonzero things
w1 = w1 / sum(w1);
w2 = w2 / sum(w2);

m = size(atoms1,1);
n = size(atoms2,1);

atoms1_extra = reshape(atoms1, size(atoms1,1), 1, size(atoms1,2));
atoms2_extra = reshape(atoms2, 1, size(atoms2,1), size(atoms2,2));

% m x n
D = sum((atoms1_extra - atoms2_extra).^2, 3);

if m*n < 200000% < 100 && n < 100
    val = sqrt(compute_single_ot_distance_mosek(D, w1, w2));
    val_hist = [val];
    return;
end

v = zeros(n,1);

val_hist = [];
stepsize_base = 50;
for iter=1:100000
    inx = randi(m);
    [val_min, i_min] = min(D(inx,:) - v);
    sub_inx = randi(length(val_min));
    val_min = val_min(sub_inx);
    i_min = i_min(sub_inx);
    
    stepsize = stepsize_base / sqrt(iter);
    val_inner = dot(v, w2);
    v = v + stepsize * w2 * w1(inx);
    v(i_min) = v(i_min) - stepsize * w1(inx);

    val_squared = val_min + val_inner;
    val_squared = val_squared;
    val = sqrt(val_squared);
    val_hist = [val_hist val];
end

val = mean(val_hist(end-5000:end));
end

