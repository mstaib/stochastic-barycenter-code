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

% get sampler_dists
sampler_files = dir(strcat(basedir, 'sampler_*.h5'));
sampler_nums = zeros(length(sampler_files),1);
sampler_dists = cell(length(sampler_files),1);
for ii=1:length(sampler_files)
    file = sampler_files(ii);
    s = regexp(file.name, ['(?<iter>\d+)'], 'names');
    sampler_nums(ii) = str2double(s(1).iter);
    sampler_dists{ii} = h5read(strcat(file.folder, '/', file.name), strcat('/sampler_', num2str(sampler_nums(ii))));
end

[~, I] = sort(sampler_nums);
sampler_nums = sampler_nums(I);
sampler_dists = sampler_dists(I);

% get empirical_points_mat
empirical_points_mat = h5read(strcat(basedir, 'empirical_points_mat.h5'), '/empirical_points_mat');

% get value_history
value_history_files = dir(strcat(basedir, 'value_history_*.h5'));
value_history_iters = zeros(length(value_history_files),1);
value_histories = cell(length(value_history_files),1);
for ii=1:length(value_history_files)
    file = value_history_files(ii);
    s = regexp(file.name, ['(?<iter>\d+)'], 'names');
    value_history_iters(ii) = str2double(s(1).iter);
    value_history_this = h5read(strcat(file.folder, '/', file.name), strcat('/value_history_', s(1).iter));
    value_histories{ii} = value_history_this(:)';
end

[~, I] = sort(value_history_iters);
value_history_iters = value_history_iters(I);
value_histories = value_histories(I);

value_history = horzcat(value_histories{:});

% get machine_counts
machine_counts_files = dir(strcat(basedir, 'machine_counts_*.h5'));
machine_counts_iters = zeros(length(machine_counts_files),1);
machine_counts_cell = cell(length(machine_counts_files),1);
for ii=1:length(machine_counts_files)
    file = machine_counts_files(ii);
    s = regexp(file.name, ['(?<iter>\d+)'], 'names');
    machine_counts_iters(ii) = str2double(s(1).iter);
    machine_counts_this = h5read(strcat(file.folder, '/', file.name), strcat('/machine_counts_', s(1).iter));
    machine_counts_cell{ii} = machine_counts_this(:)';
end

[~, I] = sort(machine_counts_iters);
machine_counts_iters = machine_counts_iters(I);
machine_counts_cell = machine_counts_cell(I);

%% plot the barycenter estimates
for ii=length(barycenter_dists)
    figure;
    w = barycenter_dists{ii};
    X = empirical_points_mat(w > 0,:);
    scatter3(X(:,1),X(:,2),X(:,3), 20, w(w > 0));
end
xlim_barycenter = xlim;
ylim_barycenter = ylim;

%% plot the points sampled from the input distributions
for ii=1:length(sampler_dists)
    X = sampler_dists{ii};
    
    figure;
    scatter3(X(:,1), X(:,2), X(:,3));
    %xlim(xlim_barycenter); ylim(ylim_barycenter);
end

%% plot the true distribution
true_dist = h5read('cpp/sampler_full.h5', '/sampler_1000');

figure;
scatter3(true_dist(:,1), true_dist(:,2), true_dist(:,3));
%xlim(xlim_barycenter); ylim(ylim_barycenter);

%% plot dual convergence
default_color = [0 0.4470 0.7410];
background_color = 0.5*default_color + 0.5*[1 1 1];

figure;
plot(movmean(value_history, 5), 'Color', background_color); hold on;
plot(movmean(value_history, 500), 'Color', default_color);