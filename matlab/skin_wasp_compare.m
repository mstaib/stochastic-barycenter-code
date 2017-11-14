
addpath('WASP_fixed');

basedir = 'cpp/output-skin-noshuffle/';

% load in the full sampler
true_dist = h5read('cpp/sampler_full.h5', '/sampler_1000');

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

sample_nums_to_try = [20 30 50 100 200 300];%[10 20 30 40 50 100 200 300];
sample_nums_to_try = [500];

for kk=1:length(sample_nums_to_try)
    tic;

    n_samples = sample_nums_to_try(kk); % <= length(sampler_dists{1})
    %grid_N = length(barycenter_dists{1}); % N used by our algorithm
    grdsize = ceil(n_samples^(1/3));

    % decide on grid
    mins_cell = cellfun(@min, sampler_dists,'UniformOutput',false);
    mins = min(vertcat(mins_cell{:}));

    maxs_cell = cellfun(@max, sampler_dists,'UniformOutput',false);
    maxs = max(vertcat(maxs_cell{:}));

    % atoms for subset posteriors; they only differ in their means.
    for ii=1:length(sampler_dists)
        this_dist = sampler_dists{ii};
        spost{ii} = this_dist(1:n_samples,:);
    end

    widths = maxs - mins;
    delta = (prod(widths) / n_samples)^(1/3);
    len_x = round(widths(1) / delta);
    len_y = round(widths(2) / delta);
    len_z = round(widths(3) / delta);
    
    len_x = max(len_x, 2);
    len_y = max(len_y, 2);
    len_z = max(len_z, 2);

    [opostx, oposty, opostz] = meshgrid(linspace(mins(1), maxs(1), len_x), linspace(mins(2), maxs(2), len_y), linspace(mins(3), maxs(3), len_z));
    opost = [opostx(:) oposty(:) opostz(:)];

    actual_num_atoms(kk) = size(opost, 1);
    len_xs(kk) = len_x;
    len_ys(kk) = len_y;
    len_zs(kk) = len_z;

    fprintf('Trying with %d atoms total; %d x %d x %d\n', actual_num_atoms(kk), len_x, len_y, len_z);

    center = mean(true_dist);
    dist_to_mean(kk) = min(sqrt(sum((opost - center).^2, 2)));
    
    % calculate distance between atoms
    opost_extra = reshape(opost, size(opost,1), 1, size(opost,2));
    for ii=1:length(sampler_dists)
        spost_extra{ii} = reshape(spost{ii}, 1, size(spost{ii},1), size(spost{ii},2));
    end

    for ii=1:length(sampler_dists)
        distMat{ii} = sum((opost_extra - spost_extra{ii}).^2, 3);
    end

    % initialize the wts b_1 ... b_K for subset posteriors atoms; see Eq. 21
    b = cellfun(@(x) ones(size(x, 1), 1) / size(x, 1), spost, 'UniformOutput', false);
    colMeasure = b;

    
    [w, optObj] = WASP(colMeasure, distMat); %, exitflag, output, lambda]
    barycenter_weights{kk} = w;
    barycenter_support{kk} = opost;
    
    time(kk) = toc;

    %%
    [w2_WASP, w2_hist_WASP] = w2_distance(true_dist(1:10000,:), ones(10000,1), opost, w);

    
    w2_distances_WASP(kk) = w2_WASP;
end
