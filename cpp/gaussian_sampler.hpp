#ifndef GAUSSIAN_SAMPLER_H
#define GAUSSIAN_SAMPLER_H

#include <memory>

using namespace Eigen;

class GaussianSampler : public Sampler {
    VectorXd center;
public:
    GaussianSampler(VectorXd center);
    VectorXd Sample();
};

std::vector<std::shared_ptr<Sampler>> gaussian_samplers();

MatrixXd gaussian_grid(const int dim, const int len_x, const int len_y);

#endif
