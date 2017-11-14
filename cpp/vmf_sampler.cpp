#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <eigen3-hdf5.hpp>
#include <iostream>

#include "sampler.hpp"
#include "vmf_sampler.hpp"

constexpr double const_pi() { return std::atan(1)*4; }

/// Von Mises Fisher Sampler
VMFSampler::VMFSampler(Vector3d center_vec, double concentration_scalar, double drift_rate_scalar) {
    center = center_vec;
    concentration = concentration_scalar;
    drift_rate = drift_rate_scalar;
    d = 3;
    
    // http://stackoverflow.com/questions/15043130/rotation-matrix-in-eigen
    e3 << 0, 0, 1;
    rotation = Quaterniond().setFromTwoVectors(e3, center).toRotationMatrix();
}   

void VMFSampler::update_center_location() {
    double r = std::sqrt(center(0) * center(0) + center(1) * center(1));
    double phi = std::atan2(center(1), center(0));
    phi += drift_rate;

    center(0) = r * std::cos(phi);
    center(1) = r * std::sin(phi);
    rotation = Quaterniond().setFromTwoVectors(e3, center).toRotationMatrix();
}

Vector2d VMFSampler::SampleUnitCircle() {
    double theta = this->rand_uniform(0, 2*const_pi());
    Vector2d out;
    out << std::cos(theta), std::sin(theta);
    return out;
}

// http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
VectorXd VMFSampler::Sample() {
    Vector2d v = SampleUnitCircle();

    double u = this->rand_uniform(0, 1);
    double w = 1 + std::log(u + (1-u)* std::exp(-2*concentration)) / concentration;
    double scale = std::sqrt(1 - w*w);

    Vector3d out_not_rotated;
    out_not_rotated << scale * v(0), scale * v(1), w;
    
    update_center_location();

    return rotation * out_not_rotated;
}

// https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
MatrixXd uniform_sphere_points(const int N) {
    std::vector<Vector3d> points;

    double a = 4 * const_pi() / N;
    double d = std::sqrt(a);
    int M_theta = (int)std::round(const_pi() / d);
    double d_theta = const_pi() / M_theta;
    double d_phi = a / d_theta;

    for (int m = 0; m < M_theta; ++m) {
        double theta = const_pi() * (m + 0.5) / M_theta;
        int M_phi = (int)std::round(2 * const_pi() * std::sin(theta) / d_phi);

        for (int n = 0; n < M_phi; ++n) {
            double phi = 2 * const_pi() * n / M_phi;

            Vector3d x;
            x << std::sin(theta) * std::cos(phi),
                 std::sin(theta) * std::sin(phi),
                 std::cos(theta);

            points.push_back(x);
        }
    }

    MatrixXd grid(3, points.size());
    for (int i = 0; i < points.size(); ++i) {
        grid.col(i) = points[i];
    }

    return grid;
}
