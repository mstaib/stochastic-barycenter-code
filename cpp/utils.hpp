#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <Eigen/Core>

using namespace Eigen;

void iteration_report(const int iter, 
					  const std::string outdir,
                      VectorXi& counts,
                      VectorXd& value_history,
                      VectorXd& diff_history,
                      VectorXi& machine_counts);

std::string print_time();

#endif