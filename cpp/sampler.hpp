#ifndef SAMPLER_H
#define SAMPLER_H

#include <memory>
#include <random>

using namespace Eigen;

double rand_normal();

int rand_int(int a, int b);

double rand_uniform(double a, double b);

bool rand_bernoulli(double p);

// abstract class for sampling from distributions
// this will be useful later when we need to implement Samplers
// by way of MCMC
class Sampler {
protected:
    std::mt19937 generator;
    double rand_normal();
    int rand_int(int a, int b);
    double rand_uniform(double a, double b);
    bool rand_bernoulli(double p);
public:
    Sampler();
    virtual VectorXd Sample() = 0;
    int d; //dimension
};

VectorXd simplex_sample(const int N);

MatrixXd atoms(std::vector<std::shared_ptr<Sampler>> samplers, const int N);

MatrixXd get_sampler_bounding_box(std::shared_ptr<Sampler> sampler, int num_samples);

#endif
