ns = [10000 10000 10000 10000 10000]

stepsizes = {
'0.01'
'0.1'
'1'
'10'
'100'
};

times = [9	18	28	37	47	56	66	75	85	94	104	113
8	18	27	37	46	56	65	75	84	94	103	113
9	18	28	37	47	56	66	75	85	94	104	113
23	48	73	98	122	147	172	197	221	246	271	296
23	48	73	98	122	147	172	197	221	246	271	296
23	47	72	97	121	146	171	195	220	245	270	294
23	48	72	97	122	146	171	196	221	245	270	295
22	46	70	94	118	142	166	190	213	237	261	285
239	480	722	963	1204	1445	1687	1928	2170	2411	2652	2894
240	481	723	964	1205	1446	1688	1929	2170	2412	2653	2895
239	481	722	964	1205	1446	1688	1929	2170	2412	2653	2894
2426	4854	7283	9710	12139	14567	16995	19423	21851	24279	26707	29135
2409	4821	7233	9644	12057	14469	16881	19292	21704	24117	26530	28943
2411	4824	7235	9648	12060	14472	16885	19299	21711	24124	26536	28950
2426	4853	7283	9711	31748	31748	31748	31748	31748	31748	31748	31748];

%%

for kk=1:length(stepsizes)
    n = ns(kk);
    stepsize = stepsizes{kk};
    

    basedir = sprintf('cpp/output-skin-noshuffle-n-%d-stepsize-%s-fine/', n, stepsize);

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

    % load in the full sampler
    true_dist = h5read('cpp/sampler_full.h5', '/sampler_1000');

    all_w2_vals = zeros(length(barycenter_dists),1);
    parfor ii=1:length(barycenter_dists)
        [w2, w2_hist] = w2_distance(true_dist(1:5000,:), ones(5000,1), empirical_points_mat, barycenter_dists{ii});
        all_w2_vals(ii) = w2;
        fprintf('Finished %d of %d\n', ii, length(barycenter_dists));
    end

    w2_vals_each_run{kk} = all_w2_vals;

    %figure;
    %plot(iters, all_w2_vals); hold on;
    %plot(iters, all_w2_vals); hold on;
    %xlabel('seconds');
    %ylabel('W2 distance to true distribution');
    
end
% legend(stepsizes(4:8));
