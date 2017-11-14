basedir = '/home/sebii/Dropbox (MIT)/output-vmf-drift/';

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

%% plot the barycenter estimates
% [azimuth,elevation,r] = cart2sph(empirical_points_mat(:,1),empirical_points_mat(:,2),empirical_points_mat(:,3));
% filename = 'polar-coords.gif';
% figure(1);
% for ii=1:length(barycenter_dists)
%     clf;
%     scatter(azimuth, elevation, 5, barycenter_dists{ii});
%     axis off;
%     axis equal;
%     set(gcf, 'color', 'w');
%     drawnow;
%     
%     frame = getframe(1);
%     im = frame2im(frame);
%     [imind, cm] = rgb2ind(im, 256);
%     if ii == 1
%         imwrite(imind, cm, filename, 'gif', 'Loopcount', inf);
%     else
%         imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
%     end
% end
xlim_barycenter = xlim;
ylim_barycenter = ylim;

%% plot the points sampled from the input distributions on the sphere
filename = 'sphere-scatter.gif';
fig = figure(2);
for ii=1:length(barycenter_dists)
    clf;
    id = barycenter_dists{ii} > 0.0002;
    C = barycenter_dists{ii}(id);
    
    scatter3(empirical_points_mat(:,1),empirical_points_mat(:,2),empirical_points_mat(:,3), 5, 0.8*[1 1 1], 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    scatter3(empirical_points_mat(id,1),empirical_points_mat(id,2),empirical_points_mat(id,3), 10, C, 'filled');
    
    xlim([-1 1]); ylim([-1 1]); zlim([-1, 1]);
    axis off;
    axis equal;
    set(gcf, 'color', 'w');
    view(120, 20);
    drawnow;    
  
%     frame = getframe(2);
%     im = frame2im(frame);
%     [imind, cm] = rgb2ind(im, 256);
%     if ii == 1
%         imwrite(imind, cm, filename, 'gif', 'Loopcount', inf);
%     else
%         imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
%     end
    
    if mod(ii-1, 5) == 0
        s = sprintf('figs/sphere-%d.pdf', ii-1);
        saveas(fig, s, 'pdf');
    end
end
%% plot the points sampled from the input distributions
% for ii=1:length(sampler_dists)
%     X = sampler_dists{ii};
%     [azimuth,elevation,r] = cart2sph(X(:,1),X(:,2),X(:,3));
%     
%     figure;
%     scatter(azimuth, elevation, 5);
%     xlim(xlim_barycenter); ylim(ylim_barycenter);
% end

%% plot dual convergence
default_color = [0 0.4470 0.7410];
background_color = 0.5*default_color + 0.5*[1 1 1];

figure;
plot(movmean(value_history, 5), 'Color', background_color); hold on;
plot(movmean(value_history, 500), 'Color', default_color);