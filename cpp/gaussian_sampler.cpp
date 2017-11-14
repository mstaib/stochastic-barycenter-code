#include <Eigen/Dense>
#include <eigen3-hdf5.hpp>

#include "sampler.hpp"
#include "gaussian_sampler.hpp"

/// GaussianSampler
GaussianSampler::GaussianSampler(VectorXd center_vec) {
    center = center_vec;
    d = center.size();
}

VectorXd GaussianSampler::Sample() {
    return center.unaryExpr([&](double elem) {
        return elem + this->rand_normal();
    });
}

std::vector<std::shared_ptr<Sampler>> gaussian_samplers() {
    VectorXd c1(2);
    c1(0) = 0; c1(1) = 0;

    VectorXd c2(2);
    c2(0) = 4; c2(1) = 0;

    VectorXd c3(2);
    c3(0) = 2; c3(1) = 2;

    std::vector<std::shared_ptr<Sampler>> samplers;
    samplers.push_back(std::make_shared<GaussianSampler>(c1));
    samplers.push_back(std::make_shared<GaussianSampler>(c2));
    samplers.push_back(std::make_shared<GaussianSampler>(c3));

    return samplers;
}

MatrixXd gaussian_grid(const int dim, const int len_x, const int len_y) {
    int N = len_x * len_y;
    VectorXd x = VectorXd::LinSpaced(len_x, -3, 7);
    VectorXd y = VectorXd::LinSpaced(len_y, -3, 5);

    H5::H5File file_x("x.h5", H5F_ACC_TRUNC);
    EigenHDF5::save(file_x, "/x", x);

    H5::H5File file_y("y.h5", H5F_ACC_TRUNC);
    EigenHDF5::save(file_y, "/y", y);

    MatrixXd X = x.transpose().replicate(len_y, 1);
    MatrixXd Y = y.replicate(1, len_x);

    MatrixXd empirical_points_mat = MatrixXd::Zero(dim, N);
    empirical_points_mat.row(0) = Map<VectorXd>(X.data(), X.size());
    empirical_points_mat.row(1) = Map<VectorXd>(Y.data(), Y.size());

    return empirical_points_mat;
}
