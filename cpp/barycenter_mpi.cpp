#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <mpi.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <eigen3-hdf5.hpp>
#include <cxxopts.hpp>

#include "sampler.hpp"
#include "vmf_sampler.hpp"
#include "logit_sampler.hpp"
#include "gaussian_sampler.hpp"
#include "parse_args.hpp"
#include "utils.hpp"

using namespace Eigen;

// made this global since so many components have to access it
std::string outdir("");

// 0: use Euclidean 2-norm, 1: use geodesic distance on sphere
int use_sphere_distance = 0;

// type for communicating min index and objective value estimate from worker to master
typedef struct {
    unsigned long long index;
    double value;
} WorkerInfo;
MPI_Datatype worker_info_type;

struct mpi_thread_info {
    int world_rank;
    int world_size;
    MPI_Comm slave_comm;
    int is_slave;
    int slave_rank;
    int slave_size;
};

mpi_thread_info get_mpi_info_this_thread() {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm slave_comm;
    int is_slave = (world_rank != 0);
    MPI_Comm_split(MPI_COMM_WORLD, is_slave, world_rank, &slave_comm);

    int slave_rank;
    MPI_Comm_rank(slave_comm, &slave_rank);
    int slave_size;
    MPI_Comm_size(slave_comm, &slave_size);

    mpi_thread_info mpi_info = {};
    mpi_info.world_rank = world_rank;
    mpi_info.world_size = world_size;
    mpi_info.slave_comm = slave_comm;
    mpi_info.is_slave = is_slave;
    mpi_info.slave_rank = slave_rank;
    mpi_info.slave_size = slave_size;

    return mpi_info;
}

void save_samples_thread(const int index, std::shared_ptr<Sampler> sampler) {
    const int num_samples = 10000;

    MatrixXd samples(sampler->d,num_samples);
    for (int i = 0; i < num_samples; ++i) {
        samples.block(0,i,sampler->d,1) = sampler->Sample();
    }

    std::string filename = "sampler_" + std::to_string(index) + ".h5";
    std::string recordname = "/sampler_" + std::to_string(index);

    H5::H5File file(outdir + filename, H5F_ACC_TRUNC);
    EigenHDF5::save(file, recordname, samples);
}

void worker_thread(const int world_rank,
                   const int num_dists,
                   const double stepsize,
                   const int iters, 
                   std::shared_ptr<Sampler> sampler,
                   const MatrixXd& empirical_points_mat) {

    std::cout << print_time() << " starting worker " << world_rank << std::endl;

    const int N = empirical_points_mat.cols();
    VectorXd v = VectorXd::Zero(N);

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

        WorkerInfo worker_info;
        worker_info.index = index;
        worker_info.value = diff(index);

        unsigned long long sums_index;
        MPI_Send(&worker_info, 1, worker_info_type, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&sums_index, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        v(index) -= stepsize;

        // gradient step
        v(index) += stepsize / 2;
        v(sums_index) += stepsize / (2 * num_dists);
    }
}

void master_thread(const int N, 
                   const int num_dists, 
                   const double stepsize, 
                   const int total_iters,
                   const int save_increment,
                   const int moving_window_width) {

    std::cout << print_time() << " starting master" << std::endl;

    VectorXi counts = VectorXi::Zero(N);
    std::queue<int> counts_window;

    // keep track of how fast the machines are over a sliding window
    VectorXi machine_counts = VectorXi::Zero(num_dists);
    std::queue<int> machine_counts_window;

    VectorXd sums = VectorXd::Zero(N);
    unsigned long long sums_index = 0;

    VectorXd value_history = VectorXd::Zero(save_increment);
    VectorXd diff_history = VectorXd::Zero(save_increment);

    for (int i = 0; i < total_iters; ++i) {
        WorkerInfo worker_info;
        MPI_Status status;
        MPI_Recv(&worker_info, 1, worker_info_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Send(&sums_index, 1, MPI_UNSIGNED_LONG_LONG, status.MPI_SOURCE, 0, MPI_COMM_WORLD);

        // increment the counter for this worker
        machine_counts(status.MPI_SOURCE - 1) += 1;
        
        // update the sums and the min coefficient
        value_history(i % save_increment) = worker_info.value + sums(sums_index) / num_dists;
        if (false) {
            VectorXd barycenter_dist = counts.cast<double>();
            barycenter_dist = barycenter_dist / counts.sum();
            counts(sums_index) += 1;
            VectorXd barycenter_dist_new = counts.cast<double>();
            barycenter_dist_new = barycenter_dist_new / counts.sum();

            diff_history(i % save_increment) = (barycenter_dist - barycenter_dist_new).lpNorm<1>();
        } else {
            counts(sums_index) += 1;
        }
        sums(sums_index) += stepsize / num_dists;

        // gradient step
        sums(worker_info.index) -= stepsize / 2;
        sums(sums_index) -= stepsize / (2 * num_dists);

        // update the queue (and possibly update counts) if we are storing
        // only the moving average
        if (moving_window_width > 0) {
            counts_window.push(sums_index);
            if (counts_window.size() > moving_window_width) {
                int decrement_index = counts_window.front();
                counts_window.pop();
                counts(decrement_index) -= 1;
            }

            machine_counts_window.push(status.MPI_SOURCE - 1);
            if (machine_counts_window.size() > moving_window_width) {
                int decrement_index = machine_counts_window.front();
                machine_counts_window.pop();
                machine_counts(decrement_index) -= 1;
            }
        }

        // get sums_index ready for the next iteration
        sums.minCoeff(&sums_index);

        if (i % save_increment == save_increment - 1 || i == total_iters) {
            iteration_report(i, outdir, counts, value_history, diff_history, machine_counts);
        }
    }
}

std::shared_ptr<Sampler> logit_sampler_this_process(mpi_thread_info mpi_info, 
                                                    int &d, 
                                                    const int burnin_iters,
                                                    const int skip) {
    int rank = mpi_info.slave_rank;
    int size = mpi_info.slave_size;
    MPI_Comm comm = mpi_info.slave_comm;

    MatrixXd x;
    VectorXi y;
    int dims[2] = {0};
    if (rank == 0) {
        auto data_tuple = logit_generate_data();
        x = std::get<0>(data_tuple);
        y = std::get<1>(data_tuple);
        dims[0] = x.rows();
        dims[1] = x.cols();
    }

    MPI_Bcast(dims, 2, MPI_INT, 0, comm);
    d = dims[0];
    int n = dims[1];

    int subset_size = n / size;
    MatrixXd x_subset = MatrixXd::Zero(d, subset_size);
    MPI_Scatter(x.data(), d * subset_size, MPI_DOUBLE, 
            x_subset.data(), d * subset_size, MPI_DOUBLE, 0, comm);

    VectorXi y_subset = VectorXi::Zero(subset_size);
    MPI_Scatter(y.data(), subset_size, MPI_INT, 
            y_subset.data(), subset_size, MPI_INT, 0, comm);

    return std::make_shared<LogitSampler>(x_subset, y_subset, (double)size, burnin_iters, skip);
}

std::shared_ptr<Sampler> skin_logit_sampler_this_process(mpi_thread_info mpi_info, 
                                                         int &d, 
                                                         const int burnin_iters,
                                                         const int skip,
                                                         const int num_datapoints,
                                                         const bool full_sampler) {

    int rank = mpi_info.slave_rank;
    int size = mpi_info.slave_size;
    MPI_Comm comm = mpi_info.slave_comm;


    MatrixXd x;
    VectorXi y;
    int dims[2] = {0};
    if (rank == 0) {
        std::vector<int> label_vec;
        std::vector<int> a_vec;
        std::vector<int> b_vec;
        std::vector<int> c_vec;

        // here is where we have to read in the file
        std::string filepath("../input_data/skin_nonskin.txt");
        
        std::ifstream infile(filepath);
        std::string line;
        while (std::getline(infile, line)) {
            if (num_datapoints != 0 && label_vec.size() >= num_datapoints) {
                break;
            }

            std::istringstream iss(line);

            int label; 
            std::string a_str, b_str, c_str; // a, b, c are the features for this example
            if (!(iss >> label >> a_str >> b_str >> c_str)) { break; }

            int a, b, c;
            a = std::stoi(a_str.substr(2));
            b = std::stoi(b_str.substr(2));
            c = std::stoi(c_str.substr(2));

            // y originally is either 1 or 2, we want -1 or +1
            label_vec.push_back(label == 1 ? -1 : 1);
            a_vec.push_back(a);
            b_vec.push_back(b);
            c_vec.push_back(c);
        }

        Map<ArrayXi> y_Xi(label_vec.data(), label_vec.size());
        Map<ArrayXi> a_Xi(a_vec.data(), a_vec.size());
        Map<ArrayXi> b_Xi(b_vec.data(), b_vec.size());
        Map<ArrayXi> c_Xi(c_vec.data(), c_vec.size());
        
        x = MatrixXd::Zero(3, a_vec.size());
        x.block(0, 0, 1, a_vec.size()) = a_Xi.cast<double>().transpose();
        x.block(1, 0, 1, b_vec.size()) = b_Xi.cast<double>().transpose();
        x.block(2, 0, 1, c_vec.size()) = c_Xi.cast<double>().transpose();

        y = y_Xi;

        dims[0] = x.rows();
        dims[1] = x.cols();

        VectorXd max_x = x.rowwise().maxCoeff();
        for (int i = 0; i < x.rows(); ++i) {
            x.row(i) /= max_x(i);
        }
        VectorXd mean_x = x.rowwise().mean();
        for (int i = 0; i < x.rows(); ++i) {
            for (int j = 0; j < x.cols(); ++j) {
                x(i, j) -= mean_x(i);
            }
        }

        // this is where the code from the server starts
        // we are making the distribution of data points to servers non-iid
        int first_pos_index = 0;
        for (int i = 0; i < dims[1]; ++i) {
            if (y(i) > 0) {
                first_pos_index = i;
                break;
            }
        }

        int num_negative_examples = first_pos_index - 1;
        int num_positive_examples = dims[1] - num_negative_examples;

        int negative_examples_per_node = num_negative_examples / size;
        int positive_examples_per_node = num_positive_examples / size;
        int examples_per_node = negative_examples_per_node + positive_examples_per_node;

        int new_num_examples = examples_per_node*size;
        MatrixXd x_shuffled(dims[0], new_num_examples);
        VectorXi y_shuffled(new_num_examples);

        int curr_inx = 0;
        int curr_negative_inx = 0;
        int curr_positive_inx = negative_examples_per_node*size;
        for (int node = 0; node < size; ++node) {
            for (int i = 0; i < examples_per_node; ++i) {
                if (i < negative_examples_per_node) {
                    x_shuffled.col(curr_inx) = x.col(curr_negative_inx);
                    y_shuffled(curr_inx) = y(curr_negative_inx);
                    curr_inx++; curr_negative_inx++;
                } else {
                    x_shuffled.col(curr_inx) = x.col(curr_positive_inx);
                    y_shuffled(curr_inx) = y(curr_positive_inx);
                    curr_inx++; curr_positive_inx++;
                }
            }
        }
        x = x_shuffled; y = y_shuffled;

        // this is where the code from the server ends
        if (full_sampler) {
            std::cout << "about to make full sampler" << std::endl;
            auto sampler = std::make_shared<LogitSampler>(x, y, 1, burnin_iters, skip);
            std::cout << "about to start full save samples thread" << std::endl;
            save_samples_thread(1000, sampler); //means full
            std::cout << "done, exiting" << std::endl;
            exit(0);
        }
    }

    MPI_Bcast(dims, 2, MPI_INT, 0, comm);
    d = dims[0];
    int n = dims[1];

    int subset_size = n / size;
    MatrixXd x_subset = MatrixXd::Zero(d, subset_size);
    MPI_Scatter(x.data(), d * subset_size, MPI_DOUBLE, 
            x_subset.data(), d * subset_size, MPI_DOUBLE, 0, comm);

    VectorXi y_subset = VectorXi::Zero(subset_size);
    MPI_Scatter(y.data(), subset_size, MPI_INT, 
            y_subset.data(), subset_size, MPI_INT, 0, comm);

    return std::make_shared<LogitSampler>(x_subset, y_subset, (double)size, burnin_iters, skip);
}

std::shared_ptr<Sampler> vmf_sampler_this_process(mpi_thread_info mpi_info,
                                                  const int concentration,
                                                  const double drift_rate) {
    // need to coordinate the choice of centers else they will all be the same
    // because RNG seeds may be the same
    int rank = mpi_info.slave_rank;
    int size = mpi_info.slave_size;
    MPI_Comm comm = mpi_info.slave_comm;

    MatrixXd centers(3, size);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            Vector3d center;
            center << rand_normal(), rand_normal(), rand_normal();
            center = center / center.sum();

            centers.block(0, i, 3, 1) = center;
        }
    }

    Vector3d my_center;
    MPI_Scatter(centers.data(), 3, MPI_DOUBLE, 
            my_center.data(), 3, MPI_DOUBLE, 0, comm);

    return std::make_shared<VMFSampler>(my_center, concentration, drift_rate);
}

MatrixXd atoms_this_process(int rank, 
                            int size, 
                            MPI_Comm comm,
                            int d, 
                            int N, 
                            std::shared_ptr<Sampler> sampler) {

    MatrixXd weights(N, size);
    if (rank == 0) {
        weights = MatrixXd::Zero(N, size);
        for (int i = 0; i < N; ++i) {
            VectorXd sample = simplex_sample(size);
            weights.block(i, 0, 1, size) = sample.transpose();
        }
    }

    VectorXd local_weights(N);
    MPI_Scatter(weights.data(), N, MPI_DOUBLE, 
            local_weights.data(), N, MPI_DOUBLE, 0, comm);

    MatrixXd local_samples(N, d);
    for (int i = 0; i < N; ++i) {
        local_samples.block(i, 0, 1, d) = local_weights(i) * sampler->Sample().transpose();
    }

    double* gathered_samples;
    if (rank == 0) {
        gathered_samples = new double[N * d * size];
    }
    MPI_Gather(local_samples.data(), N * d, MPI_DOUBLE, 
            gathered_samples, N * d, MPI_DOUBLE, 0, comm);

    MatrixXd empirical_points_mat(d, N);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            Map<MatrixXd> gathered_samples_mat(gathered_samples + i * N * d, N, d);
            empirical_points_mat += gathered_samples_mat.transpose();
        }
    }

    MPI_Bcast(empirical_points_mat.data(), d * N, MPI_DOUBLE, 0, comm);
    return empirical_points_mat;
}

void build_worker_info_type() {
    // set up WorkerInfo MPI Type
    // MPI_Datatype worker_info_type already a global variable
    MPI_Datatype oldtypes[2];
    int blockcounts[2];

    MPI_Aint offsets[2], lb, extent;

    offsets[0] = 0;
    oldtypes[0] = MPI_UNSIGNED_LONG_LONG;
    blockcounts[0] = 1;

    MPI_Type_get_extent(MPI_UNSIGNED_LONG_LONG, &lb, &extent);
    offsets[1] = 1 * extent;
    oldtypes[1] = MPI_DOUBLE;
    blockcounts[1] = 1;

    MPI_Type_create_struct(2, blockcounts, offsets, oldtypes, &worker_info_type);
    MPI_Type_commit(&worker_info_type);
}


MatrixXd aggregate_bounding_boxes(mpi_thread_info mpi_info, std::shared_ptr<Sampler> sampler) {
    int rank = mpi_info.slave_rank;
    int size = mpi_info.slave_size;
    MPI_Comm comm = mpi_info.slave_comm;

    MatrixXd local_bounds = get_sampler_bounding_box(sampler, 10000);

    VectorXd local_bounds_vector = VectorXd::Zero(sampler->d * 2);
    local_bounds_vector.head(sampler->d) = local_bounds.col(0);
    local_bounds_vector.tail(sampler->d) = local_bounds.col(1);

    MatrixXd gathered_bounds(sampler->d * 2, size);
    MPI_Allgather(local_bounds_vector.data(), sampler->d * 2, MPI_DOUBLE,
               gathered_bounds.data(), sampler->d * 2, MPI_DOUBLE, comm);

    MatrixXd final_bounds(sampler->d, 2);
    final_bounds.col(0) = gathered_bounds.block(0, 0, sampler->d, 1);
    final_bounds.col(1) = gathered_bounds.block(sampler->d, 0, sampler->d, 1);
    for (int i = 0; i < size; ++i) {
        final_bounds.col(0) = final_bounds.col(0).array().min(gathered_bounds.block(0, i, sampler->d, 1).array());
        final_bounds.col(1) = final_bounds.col(1).array().max(gathered_bounds.block(sampler->d, i, sampler->d, 1).array());
    }

    if (rank == 0) {
        std::cout << "final bounds: " << final_bounds << std::endl;
    }

    return final_bounds;
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    build_worker_info_type();

    auto opt_struct = parse_args(argc, argv);
    const int iters = opt_struct->iters;
    std::string experiment = opt_struct->experiment;
    const int skip = opt_struct->skip;
    const int N_requested = opt_struct->support; //support
    outdir = opt_struct->outdir;
    const int save_increment = opt_struct->save_increment; //how often to save dumps
    const double stepsize = opt_struct->stepsize;
    const int moving_window_width = opt_struct->moving_window_width;
    const double drift_rate = opt_struct->drift_rate;
    const int burnin_iters = opt_struct->burnin_iters;
    const bool full_sampler = opt_struct->full_sampler;
    int N = N_requested; //may update this depending on our empirical support points

    std::shared_ptr<Sampler> sampler;
    int d;
    MatrixXd empirical_points_mat;

    mpi_thread_info mpi_info = get_mpi_info_this_thread();

    if (!experiment.compare("logit")) {
        if (mpi_info.is_slave) {
            sampler = logit_sampler_this_process(mpi_info, d, burnin_iters, skip);
            if (N <= 1000) {
                empirical_points_mat = atoms_this_process(mpi_info.slave_rank, mpi_info.slave_size, mpi_info.slave_comm, d, N, sampler);
            } else {
                const int len_x = (int)std::round(std::sqrt(N));
                const int len_y = (int)std::round(std::sqrt(N));
                empirical_points_mat = logit_grid(d, len_x, len_y, mpi_info.slave_rank == 0);
            }

            N = empirical_points_mat.cols();
            if (mpi_info.slave_rank == 0) {
                MPI_Send(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        } else {
            MPI_Recv(&N, 1, MPI_INT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    } else if (!experiment.compare("skin")) {
        if (mpi_info.is_slave) {
            sampler = skin_logit_sampler_this_process(mpi_info, d, burnin_iters, skip, opt_struct->num_datapoints, full_sampler);
            std::cout << "done building samplers " << mpi_info.slave_rank << std::endl;
            //empirical_points_mat = atoms_this_process(slave_rank, slave_size, slave_comm, d, N, sampler);
            MatrixXd global_bounds = aggregate_bounding_boxes(mpi_info, sampler);
            empirical_points_mat = logit_grid_3d(N_requested, global_bounds);
            std::cout << "done deciding atoms " << mpi_info.slave_rank << std::endl;

            N = empirical_points_mat.cols();
            if (mpi_info.slave_rank == 0) {
                MPI_Send(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        } else {
            MPI_Recv(&N, 1, MPI_INT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    } else if (!experiment.compare("vmf")) {
        use_sphere_distance = 1;
        if (mpi_info.is_slave) {
            const double concentration = 20;
            sampler = vmf_sampler_this_process(mpi_info, concentration, drift_rate);
            empirical_points_mat = uniform_sphere_points(N_requested);

            N = empirical_points_mat.cols();
            if (mpi_info.slave_rank == 0) {
                MPI_Send(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        } else {
            MPI_Recv(&N, 1, MPI_INT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        std::cout << "_really_ not implemented" << std::endl;
        exit(0);
    }

    if (mpi_info.is_slave && mpi_info.slave_rank == 0) {
        H5::H5File file(outdir + "empirical_points_mat.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "/empirical_points_mat", empirical_points_mat);
    }

    if (mpi_info.is_slave) {
        std::cout << "starting save samples thread: " << mpi_info.slave_rank << std::endl;
        save_samples_thread(mpi_info.slave_rank, sampler);
    }

    std::cout << "starting main loop " << mpi_info.slave_rank << std::endl;
    const int num_dists = mpi_info.world_size - 1; // = slave_size
    if (mpi_info.is_slave) {
        worker_thread(mpi_info.world_rank, num_dists, stepsize, iters, sampler, empirical_points_mat);
    } else {
        master_thread(N, num_dists, stepsize, iters * num_dists, save_increment, moving_window_width);
    }

    MPI_Finalize();
    return 0;
}

/*

List of options that we'll want for the CLI:
- iters [int]
- experiment [str]
  - "gaussian" just yields the current experiment
  - "logit" yields the Srivastava Logit experiment, etc.
  - "skin" yields logistic regression on the UCI skin/no skin dataset
  - "vmf" yields Von Mises-Fisher experiment

*/
