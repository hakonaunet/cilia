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

    float getCiliaRadius() const { return params.cilia_radius; }
    float getGridSpacing() const { return params.grid_spacing; }
    unsigned int getWidth() const { return params.width; }
    unsigned int getHeight() const { return params.height; }
    const Eigen::MatrixX3f& getPositions() const { return positions_; }
    const Eigen::MatrixXf& getAngles() const { return angles_; }
    float getVelocityMagnitudeAtPoint(Eigen::Vector3f point, Eigen::MatrixXf& angles);
    Eigen::Vector3f getVelocityAtPoint(Eigen::Vector3f point, Eigen::MatrixXf& angles);
    Eigen::Vector3f getVelocityAtPoint(Eigen::Vector3f point);
    Eigen::Vector3f getForce(int x, int y, Eigen::MatrixXf& angles);
    float getForceMagnitude(int x, int y, Eigen::MatrixXf& angles);
    Eigen::Vector3f getR(int x, int y, Eigen::MatrixXf& angles);
    Eigen::Vector3f getTangent(int x, int y, Eigen::MatrixXf& angles);

private:
    SharedDataOseen& sharedData;

    OseenParameters params;
    
    Eigen::MatrixX3f positions_;
    Eigen::MatrixX3f velocities_;
    Eigen::MatrixXf angles_, tempAngles_;
    Eigen::MatrixXf k1_, k2_, k3_, k4_;
    
    bool findOrderParameter_;
    std::vector<float> simulationTimes_;
    std::vector<std::complex<float>> orderParameters_;
    
    Eigen::Vector3f initializePosition(int x, int y);
    float initializeAngle();
    void updateVelocities(Eigen::MatrixXf& angles);
    void calculateVelocity(size_t x, size_t y, Eigen::MatrixXf& angles);
    void updateAngles();
    void normalizeAngles(Eigen::MatrixXf& angles);
    void rungeKutta4();
    void calculateStep(Eigen::MatrixXf& angles, Eigen::MatrixXf& result, float dt);
    float f(float x, float y, Eigen::MatrixXf& angles);
    Eigen::Vector3f stokeslet(int x_1, int y_1, int x_2, int y_2, Eigen::MatrixXf& angles);
    Eigen::Vector3f stokeslet(Eigen::Vector3f point, int x, int y, Eigen::MatrixXf& angles);
    void calcOmega();
    std::complex<float> calculateOrderParameter();
    
    // CUDA
    float* d_angles, *d_pos_x, *d_pos_y, *d_velocities_x, *d_velocities_y;
    OseenParameters* d_params;

    void checkCudaError(cudaError_t err, const char* operation);
    void initializeCUDA();
    void updateCUDA(Eigen::MatrixXf& angles);
    void freeCUDA();
    void CUDArungeKutta4();
    void CUDAcalculateStep(Eigen::MatrixXf& angles, Eigen::MatrixXf& result, float dt);
    void CUDAupdateVelocities(Eigen::MatrixXf& angles);
};