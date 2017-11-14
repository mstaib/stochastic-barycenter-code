#include <Eigen/Dense>
#include <eigen3-hdf5.hpp>

#include "sampler.hpp"
#include "logit_sampler.hpp"

/// LogitSampler
LogitSampler::LogitSampler(MatrixXd x_mat, VectorXi y_vec, double gamma_scalar, int burnin_iters_scalar, int skip_scalar) {
    x = x_mat;
    d = x.rows();
    n = x.cols();

    y = y_vec;
    theta = VectorXd::Zero(d);

    gamma = gamma_scalar;
    burnin_iters = burnin_iters_scalar;
    skip = skip_scalar;

    DoBurnIn();
}

void LogitSampler::OneSampleIteration() {
    VectorXd theta_new = theta.unaryExpr([&](double elem) {
        return elem + 0.05 * this->rand_normal();
    });

    VectorXd p_bern = inverse(1 + (-x.transpose() * theta).array().exp());
    VectorXd p_bern_new = inverse(1 + (-x.transpose() * theta_new).array().exp());

    VectorXd p_y = VectorXd::Zero(n);
    VectorXd p_y_new = VectorXd::Zero(n);
    for (int i = 0; i < n; ++i) {
        if (y(i) == 1) {
            p_y(i) = p_bern(i);
            p_y_new(i) = p_bern_new(i);
        } else {
            p_y(i) = 1 - p_bern(i);
            p_y_new(i) = 1 - p_bern_new(i);
        }
    }

    double accept_prob = exp( gamma * (log(p_y_new.array()) - log(p_y.array())).sum() );
    accept_prob *= std::exp( 0.5 * (theta.squaredNorm() - theta_new.squaredNorm()) );
    accept_prob = std::min(1.0, accept_prob);

    if (this->rand_bernoulli(accept_prob)) {
        theta = theta_new;
    }
}

void LogitSampler::DoBurnIn() {
    for (int i = 0; i < burnin_iters; ++i) {
        OneSampleIteration();
    }
}

VectorXd LogitSampler::Sample() {
    for (int i = 0; i < skip; ++i) {
        OneSampleIteration();
    }
    return theta;
}

std::tuple<MatrixXd, VectorXi> logit_generate_data() {
    int n = 100000;
    int d = 2;
    VectorXd theta(2);
    theta(0) = -1; theta(1) = 1;

    MatrixXd x = MatrixXd::Zero(d, n);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < n; ++j) {
            x(i,j) = rand_int(0, 1) * 2 - 1;
        }
    }
    VectorXd p = inverse(1 + (-x.transpose() * theta).array().exp());
    VectorXi y = VectorXi::Zero(n);
    for (int i = 0; i < n; ++i) {
        if (rand_bernoulli(p(i))) {
            y(i) = 1;
        }
    }

    return std::make_tuple(x, y);
}

std::vector<std::shared_ptr<Sampler>> logit_samplers(const MatrixXd& x, const VectorXi& y, const int subsets, const double gamma) {
    int d = x.rows();
    int n = x.cols();

    std::vector<std::shared_ptr<Sampler>> samplers(subsets);
    int subset_size = n / subsets;

    //#pragma omp parallel for
    for (int i = 0; i < subsets; ++i) {
        int start_inx = i * subset_size;
        
        MatrixXd xi = x.block(0, start_inx, d, subset_size);
        VectorXi yi = y.segment(start_inx, subset_size);
        samplers[i] = std::make_shared<LogitSampler>(xi, yi, gamma);
    }

    return samplers;
}

MatrixXd logit_grid(const int dim, const int len_x, const int len_y, const bool write_to_file) {
    int N = len_x * len_y;
    VectorXd x = VectorXd::LinSpaced(len_x, -1.25, -0.75);
    VectorXd y = VectorXd::LinSpaced(len_y, 0.75, 1.25);

    if (write_to_file) {
        H5::H5File file_x("x.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file_x, "/x", x);

        H5::H5File file_y("y.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file_y, "/y", y);
    }

    MatrixXd X = x.transpose().replicate(len_y, 1);
    MatrixXd Y = y.replicate(1, len_x);

    MatrixXd empirical_points_mat = MatrixXd::Zero(dim, N);
    empirical_points_mat.row(0) = Map<VectorXd>(X.data(), X.size());
    empirical_points_mat.row(1) = Map<VectorXd>(Y.data(), Y.size());

    return empirical_points_mat;
}

MatrixXd logit_grid_3d(const int N, MatrixXd bounds) {
    Vector3d lower_bounds = bounds.col(0);
    Vector3d upper_bounds = bounds.col(1);

    // we want to have similar gap between points, in each dimensions
    Vector3d widths = upper_bounds - lower_bounds;
    double delta = std::pow(widths(0) * widths(1) * widths(2) / N, 1.0/3.0);

    int len_x = widths(0) / delta;
    int len_y = widths(1) / delta;
    int len_z = widths(2) / delta;
    
    VectorXd x = VectorXd::LinSpaced(len_x, lower_bounds(0), upper_bounds(0));
    VectorXd y = VectorXd::LinSpaced(len_y, lower_bounds(1), upper_bounds(1));
    VectorXd z = VectorXd::LinSpaced(len_z, lower_bounds(2), upper_bounds(2));

    const int true_N = len_x * len_y * len_z;
    
    MatrixXd empirical_points_mat = MatrixXd::Zero(3, true_N);
    int counter = 0;
    for (int i = 0; i < len_x; ++i) {
        for (int j = 0; j < len_y; ++j) {
            for (int k = 0; k < len_z; ++k) {
                Vector3d point;
                point << x(i), y(j), z(k);
                empirical_points_mat.col(counter) = point;
                counter++;
            }
        }
    }

    return empirical_points_mat;
}
