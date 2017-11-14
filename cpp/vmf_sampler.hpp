#ifndef VMF_SAMPLER_H
#define VMF_SAMPLER_H

#include <memory>

using namespace Eigen;

class VMFSampler : public Sampler {
    Vector3d e3;
    Vector3d center;
    double concentration;
    double drift_rate;
    Matrix3d rotation;
public:
    VMFSampler(Vector3d center, double concentration, double drift_rate);
    VectorXd Sample();
private:
    Vector2d SampleUnitCircle();
    void update_center_location();
};

MatrixXd uniform_sphere_points(const int N);

#endif
