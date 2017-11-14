#include <thread>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "sampler.hpp"

// should be thread-safe, taken from:
// http://stackoverflow.com/questions/21237905/how-do-i-generate-thread-safe-uniform-random-numbers
double rand_normal() {
    static thread_local std::mt19937 generator;
    std::normal_distribution<> d(0,1);
    return d(generator);
}

int rand_int(int a, int b) {
    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<> d(a,b);
    return d(generator);
}

double rand_uniform(double a, double b) {
    static thread_local std::mt19937 generator;
    std::uniform_real_distribution<> d(a,b);
    return d(generator);
}

bool rand_bernoulli(double p) {
    static thread_local std::mt19937 generator;
    std::bernoulli_distribution d(p);
    return d(generator);
}

Sampler::Sampler() {
    std::random_device rd;
    generator = std::mt19937(rd());
}

double Sampler::rand_normal() {
    std::normal_distribution<> d(0,1);
    return d(generator);
}

int Sampler::rand_int(int a, int b) {
    std::uniform_int_distribution<> d(a,b);
    return d(generator);
}

double Sampler::rand_uniform(double a, double b) {
    std::uniform_real_distribution<> d(a,b);
    return d(generator);
}

bool Sampler::rand_bernoulli(double p) {
    std::bernoulli_distribution d(p);
    return d(generator);
}

// samples uniformly from the N-dimensional simplex
VectorXd simplex_sample(const int N) {
	std::vector<double> nums;
	nums.push_back(0.0);
	nums.push_back(1.0);
	for (int i = 0; i < N - 1; ++i) {
		nums.push_back(rand_uniform(0.0, 1.0));
	}

	std::sort(nums.begin(), nums.end());
	VectorXd weights = VectorXd::Zero(N);
	for (int i = 0; i < N; ++i) {
		weights(i) = nums[i+1] - nums[i];
	}

	return weights;
}

MatrixXd atoms(std::vector<std::shared_ptr<Sampler>> samplers, const int N) {
	MatrixXd samples = MatrixXd::Zero(samplers[0]->d, N);

	for (int i = 0; i < N; ++i) {
		VectorXd weights = simplex_sample(samplers.size());

		//#pragma omp parallel for
	    for (int j = 0; j < samplers.size(); ++j) {
	    	samples.block(0, i, samplers[0]->d, 1) += weights(j) * samplers[j]->Sample();
	    }
	}

	return samples;
}

MatrixXd get_sampler_bounding_box(std::shared_ptr<Sampler> sampler, int num_samples) {
    MatrixXd bounds = MatrixXd::Zero(sampler->d, 2); // 0th coordinate is min, 1st coordinate is max

    VectorXd sample = sampler->Sample();
    bounds.col(0) = sample;
    bounds.col(1) = sample;

    for (int i = 0; i < num_samples; ++i) {
        VectorXd sample = sampler->Sample();
        // now update bounds
        //samples.block(0,i,sampler->d,1) = sampler->Sample();

        bounds.col(0) = bounds.col(0).array().min(sample.array());
        bounds.col(1) = bounds.col(1).array().max(sample.array());
    }

    return bounds;
}
