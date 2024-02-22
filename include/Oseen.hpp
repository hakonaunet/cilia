// Oseen.hpp

#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <complex>
#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <omp.h>
#include <cuda_runtime.h>

#include "SharedDataOseen.hpp"
#include "Oseen.cuh"
#include "OseenParameters.hpp"

class Oseen {
public:
    Oseen(SharedDataOseen& data);

    ~Oseen();

    void iteration();
    void plotOrderParameter(std::string filename) const;

    double getCiliaRadius() const { return params.cilia_radius; }
    double getGridSpacing() const { return params.grid_spacing; }
    unsigned int getWidth() const { return params.width; }
    unsigned int getHeight() const { return params.height; }
    const Eigen::MatrixX3d& getPositions() const { return positions_; }
    const Eigen::MatrixXd& getAngles() const { return angles_; }
    double getVelocityMagnitudeAtPoint(Eigen::Vector3d point, Eigen::MatrixXd& angles);
    Eigen::Vector3d getVelocityAtPoint(Eigen::Vector3d point, Eigen::MatrixXd& angles);
    Eigen::Vector3d getVelocityAtPoint(Eigen::Vector3d point);
    Eigen::Vector3d getForce(int x, int y, Eigen::MatrixXd& angles);
    double getForceMagnitude(int x, int y, Eigen::MatrixXd& angles);
    Eigen::Vector3d getR(int x, int y, Eigen::MatrixXd& angles);
    Eigen::Vector3d getTangent(int x, int y, Eigen::MatrixXd& angles);

private:
    SharedDataOseen& sharedData;

    OseenParameters params;
    
    Eigen::MatrixX3d positions_;
    Eigen::MatrixX3d velocities_;
    Eigen::MatrixXd angles_, tempAngles_;
    Eigen::MatrixXd k1_, k2_, k3_, k4_;
    
    bool findOrderParameter_;
    std::vector<double> simulationTimes_;
    std::vector<std::complex<double>> orderParameters_;
    
    Eigen::Vector3d initializePosition(int x, int y);
    double initializeAngle();
    void updateVelocities(Eigen::MatrixXd& angles);
    void calculateVelocity(size_t x, size_t y, Eigen::MatrixXd& angles);
    void updateAngles();
    void normalizeAngles(Eigen::MatrixXd& angles);
    void rungeKutta4();
    void calculateStep(Eigen::MatrixXd& angles, Eigen::MatrixXd& result, double dt);
    double f(double x, double y, Eigen::MatrixXd& angles);
    Eigen::Vector3d stokeslet(int x_1, int y_1, int x_2, int y_2, Eigen::MatrixXd& angles);
    Eigen::Vector3d stokeslet(Eigen::Vector3d point, int x, int y, Eigen::MatrixXd& angles);
    void calcOmega();
    std::complex<double> calculateOrderParameter();
    
    // CUDA
    double* d_angles, *d_pos_x, *d_pos_y, *d_velocities_x, *d_velocities_y;
    OseenParameters* d_params;

    void checkCudaError(cudaError_t err, const char* operation);
    void initializeCUDA();
    void updateCUDA();
    void freeCUDA();
    void CUDArungeKutta4();
    void CUDAcalculateStep(Eigen::MatrixXd& angles, Eigen::MatrixXd& result, double dt);
    void CUDAupdateVelocities(Eigen::MatrixXd& angles);
};