#include "utils.hpp"
#include <eigen3-hdf5.hpp>
#include <ctime>
#include <iostream>

void iteration_report(const int i, 
                      const std::string outdir,
                      VectorXi& counts,
                      VectorXd& value_history,
                      VectorXd& diff_history,
                      VectorXi& machine_counts) {
    std::cout << print_time() << " Iteration " << i << std::endl;

    VectorXd barycenter_dist = counts.cast<double>();
    barycenter_dist = barycenter_dist / counts.sum();

    std::string dist_name = "barycenter_dist_" + std::to_string(i + 1);
    H5::H5File dist_file(outdir + dist_name + ".h5", H5F_ACC_TRUNC);
    EigenHDF5::save(dist_file, "/" + dist_name, barycenter_dist);

    std::string hist_name = "value_history_" + std::to_string(i + 1);
    H5::H5File hist_file(outdir + hist_name + ".h5", H5F_ACC_TRUNC);
    EigenHDF5::save(hist_file, "/" + hist_name, value_history);
    
    std::string diff_name = "diff_history_" + std::to_string(i + 1);
    H5::H5File diff_file(outdir + diff_name + ".h5", H5F_ACC_TRUNC);
    EigenHDF5::save(diff_file, "/" + diff_name, diff_history);

    std::string machine_name = "machine_counts_" + std::to_string(i + 1);
    H5::H5File machine_file(outdir + machine_name + ".h5", H5F_ACC_TRUNC);
    EigenHDF5::save(machine_file, "/" + machine_name, machine_counts);
}

std::string print_time() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, sizeof(buffer), "%d-%m-%Y %I:%M:%S", timeinfo);
    std::string str(buffer);
    return str;
}