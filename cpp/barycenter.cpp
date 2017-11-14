#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <random>
#include <algorithm>
#include <queue>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <eigen3-hdf5.hpp>
#include <cxxopts.hpp>

#include "sampler.hpp"
#include "logit_sampler.hpp"
#include "gaussian_sampler.hpp"
#include "parse_args.hpp"
#include "vmf_sampler.hpp"
#include "utils.hpp"

using namespace Eigen;

// made this global since so many components have to access it
std::string outdir("");

static std::mutex barrier;
static std::mutex io_barrier;
const int num_dists = 3;

VectorXd sums;
VectorXi counts;
VectorXi machine_counts; // = VectorXi::Zero(num_dists);
std::queue<int> machine_counts_window;

VectorXd value_history; // = VectorXd::Zero(save_increment);
VectorXd diff_history; // = VectorXd::Zero(save_increment);
int iter = 0; //global iteration counter

// 0: use Euclidean 2-norm, 1: use geodesic distance on sphere
int use_sphere_distance = 0;

void save_samples(const std::string name, std::shared_ptr<Sampler> sampler) {
    const int num_samples = 1000;

    MatrixXd samples(sampler->d,num_samples);
    for (int i = 0; i < num_samples; ++i) {
        samples.block(0,i,sampler->d,1) = sampler->Sample();
    }

    std::string filename = name + ".h5";
    std::string recordname = "/" + name;

    std::unique_lock<std::mutex> lock(io_barrier, std::defer_lock);
    lock.lock();
    H5::H5File file(filename, H5F_ACC_TRUNC);
    EigenHDF5::save(file, recordname, samples);
    lock.unlock();
}

void save_samples_thread(const int index, std::shared_ptr<Sampler> sampler) {
    const std::string name = "sampler_" + std::to_string(index);
    save_samples(name, sampler);
}

void worker_thread(const int num_dists,
                   const int iters, 
                   const double stepsize,
                   const int save_increment,
                   std::shared_ptr<Sampler> sampler,
                   const MatrixXd& empirical_points_mat) {

    const int N = sums.size();
    VectorXd v = VectorXd::Zero(N);

    std::unique_lock<std::mutex> lock(barrier, std::defer_lock);

    for (int i = 0; i < iters; ++i) {
        VectorXd x = sampler->Sample();
        
        VectorXd diff;
        if (use_sphere_distance) {
            VectorXd sqnorms = (x.transpose() * empirical_points_mat).array().acos().square();
            diff = sqnorms - v;
        } else {
            auto sqnorms = (empirical_points_mat.colwise() - x).colwise().squaredNorm();
            diff = sqnorms.transpose() - v;
        }

        unsigned long long index;
        diff.minCoeff(&index);



        VectorXd::Index sums_index;
        lock.lock();
        sums.minCoeff(&sums_index);

        // are we grabbing sums at the right time?
        value_history(iter % save_increment) = diff(index) + sums(sums_index) / num_dists;

        counts(sums_index) += 1;
        sums(sums_index) += stepsize / num_dists;
        sums(index) -= stepsize * 1.0;
        
        if (iter % save_increment == save_increment - 1 || iter == iters * num_dists) {
            // std::cout << "Iteration " << iter << std::endl;
            iteration_report(iter, outdir, counts, value_history, diff_history, machine_counts);
        }
        iter++;

        lock.unlock();

        v(sums_index) += stepsize / num_dists;
        v(index) -= stepsize * 1.0;
    }
}

void stochastic_barycenter_dual(std::vector<std::shared_ptr<Sampler>> samplers, 
                                MatrixXd empirical_points_mat,
                                const int iters,
                                const double stepsize,
                                const int save_increment) {
    
    const int N = empirical_points_mat.cols();
    sums = VectorXd::Zero(N);
    counts = VectorXi::Zero(N);
    machine_counts = VectorXi::Zero(samplers.size());
    value_history = VectorXd::Zero(save_increment);
    diff_history = VectorXd::Zero(save_increment);

    std::vector<std::thread> worker_threads;
    for (auto &sampler : samplers) {
        worker_threads.push_back(std::thread(worker_thread,
                                             samplers.size(),
                                             iters, 
                                             stepsize,
                                             save_increment,
                                             sampler, 
                                             empirical_points_mat));
    }

    for (auto &t : worker_threads) {
        t.join();
    }

    std::cout << "total examples seen: " << counts.sum() << std::endl;
}

void draw_samples(std::vector<std::shared_ptr<Sampler>> samplers) {
    std::vector<std::thread> worker_threads;
    for (int i = 0; i < samplers.size(); ++i) {//auto &sampler : samplers) {
        worker_threads.push_back(std::thread(save_samples_thread,
                                             i,
                                             samplers[i]));
    }

    for (auto &t : worker_threads) {
        t.join();
    }
}

std::vector<std::shared_ptr<Sampler>> vmf_samplers(const int num_samplers,
                                                   const int concentration,
                                                   const double drift_rate) {
    // need to coordinate the choice of centers else they will all be the same
    // because RNG seeds may be the same
    std::vector<std::shared_ptr<Sampler>> samplers(num_samplers);

    //#pragma omp parallel for
    for (int i = 0; i < num_samplers; ++i) {
        Vector3d center;
        center << rand_normal(), rand_normal(), rand_normal();
        center = center / center.sum();

        samplers[i] = std::make_shared<VMFSampler>(center, concentration, drift_rate);
    }

    return samplers;
}

int main(int argc, char* argv[]) {
    auto opt_struct = parse_args(argc, argv);
    const int iters = opt_struct->iters;
    std::string experiment = opt_struct->experiment;
    const int subsets = opt_struct->subsets;
    const bool full = opt_struct->full_sampler;
    const int N_requested = opt_struct->support;
    outdir = opt_struct->outdir;

    // int N;

    std::vector<std::shared_ptr<Sampler>> samplers;
    MatrixXd empirical_points_mat;

    if (!experiment.compare("gaussian")) {
        samplers = gaussian_samplers();
        int len_x = 40; int len_y = 40;
        empirical_points_mat = gaussian_grid(samplers[0]->d, len_x, len_y);

        int N = len_x * len_y;
        VectorXd sums = VectorXd::Zero(N);
        VectorXd counts = VectorXd::Zero(N);
    } else if (!experiment.compare("logit")) {
        auto data_tuple = logit_generate_data();
        auto x = std::get<0>(data_tuple);
        auto y = std::get<1>(data_tuple);

        samplers = logit_samplers(x, y, subsets, (double)subsets);
        draw_samples(samplers);

        if (full) {
            std::cout << "Producing sample from the full chain (may take some time)" << std::endl;
            auto full_sampler = std::make_shared<LogitSampler>(x, y, 1.0);
            save_samples("full_mcmc_samples", full_sampler);
        }

        int N = 1000;
        empirical_points_mat = atoms(samplers, N);

        H5::H5File file("empirical_points_mat.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "/empirical_points_mat", empirical_points_mat);
    } else if (!experiment.compare("vmf")) {
        use_sphere_distance = 1;
        
        const double concentration = 20;
        samplers = vmf_samplers(3, concentration, opt_struct->drift_rate);
        empirical_points_mat = uniform_sphere_points(N_requested);

        // N = empirical_points_mat.cols();

        H5::H5File file("empirical_points_mat.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "/empirical_points_mat", empirical_points_mat);
    } else {
        std::cout << "not implemented" << std::endl;
        exit(0);
    }

    stochastic_barycenter_dual(samplers, 
                               empirical_points_mat, 
                               iters,
                               opt_struct->stepsize,
                               opt_struct->save_increment);

    VectorXd barycenter_dist = counts.cast<double>();
    barycenter_dist = barycenter_dist / counts.sum();

    H5::H5File file("barycenter_dist.h5", H5F_ACC_TRUNC);
    EigenHDF5::save(file, "/barycenter_dist", barycenter_dist);

    return 0;
}

/*

List of options that we'll want for the CLI:
- iters [int]
- experiment [str]
  - "gaussian" just yields the current experiment
  - "logit" yields the Srivastava Logit experiment, etc.

*/
