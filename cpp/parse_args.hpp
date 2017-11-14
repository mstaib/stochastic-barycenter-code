#ifndef PARSE_ARGS_H
#define PARSE_ARGS_H

struct options_struct {
	int iters;
	std::string experiment;
	int subsets;
	int skip;
	int support;
	std::string outdir;
	int save_increment;
	double stepsize;
	int moving_window_width;
	double drift_rate;
	int burnin_iters;
	bool full_sampler;
	int num_datapoints;
};

std::shared_ptr<options_struct> parse_args(int argc, char* argv[]);

#endif