basedir = 'cpp/output-skin-noshuffle/';

% get barycenter_dists
barycenter_files = dir(strcat(basedir, 'barycenter_dist_*.h5'));
iters = zeros(length(barycenter_files),1);
barycenter_dists = cell(length(barycenter_files),1);
for ii=1:length(barycenter_files)
    file = barycenter_files(ii);
    s = regexp(file.name, ['(?<iter>\d+)'], 'names');
    iters(ii) = str2double(s(1).iter);
    barycenter_dists{ii} = h5read(strcat(file.folder, '/', file.name), strcat('/barycenter_dist_', num2str(iters(ii))));
end

[~, I] = sort(iters);
iters = iters(I);
barycenter_dists = barycenter_dists(I);

% get empirical_points_mat
empirical_points_mat = h5read(strcat(basedir, 'empirical_points_mat.h5'), '/empirical_points_mat');

all_w2_vals = zeros(length(barycenter_dists) - 1,1);
for ii=1:(length(barycenter_dists) - 1)
    [w2, w2_hist] = w2_distance(empirical_points_mat, barycenter_dists{ii}, empirical_points_mat, barycenter_dists{ii+1});
    all_w2_vals(ii) = w2;
    fprintf('Finished %d of %d\n', ii, length(barycenter_dists) - 1);
end

all_2norm_vals = zeros(length(barycenter_dists) - 1,1);
for ii=1:(length(barycenter_dists) - 1)
    all_2norm_vals(ii) = norm(barycenter_dists{ii} - barycenter_dists{ii+1});
end

figure;
plot(iters(1:end-1), all_w2_vals); hold on;
plot(iters(1:end-1), all_2norm_vals);
legend('w2 gaps', '2norm gaps');
xlabel('iteration');
ylabel('W2 distance between consecutive barycenter estimates');