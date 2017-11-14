#ifndef LOGIT_SAMPLER_H
#define LOGIT_SAMPLER_H

#include <memory>

using namespace Eigen;

class LogitSampler : public Sampler {
    // samples x, f(x) -- should have same dimension
    MatrixXd x;
    VectorXi y;
    //int d;
    int n;
    double gamma; //correction factor
    VectorXd theta;
    int burnin_iters; // = 1000;
    int skip; // = 100;
public:
    LogitSampler(MatrixXd x_mat, VectorXi y_vec, double gamma_scalar = 1, int burnin_iters = 1000, int skip = 100);
    VectorXd Sample();
private:
    void DoBurnIn();
    void OneSampleIteration();
};

std::tuple<MatrixXd, VectorXi> logit_generate_data();

std::vector<std::shared_ptr<Sampler>> logit_samplers(const MatrixXd& x, const VectorXi& y, const int subsets, const double gamma);

MatrixXd logit_grid(const int dim, const int len_x, const int len_y, const bool write_to_file);

MatrixXd logit_grid_3d(const int N, MatrixXd bounds);

#endif
