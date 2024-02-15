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

#include "SharedDataOseen.hpp"

class Oseen {
public:
    Oseen(SharedDataOseen& data);

    void iteration();
    void plotOrderParameter(std::string filename) const;

    double getCiliaRadius() const { return cilia_radius_; }
    double getGridSpacing() const { return grid_spacing_; }
    unsigned int getWidth() const { return width_; }
    unsigned int getHeight() const { return height_; }
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

    unsigned int width_, height_, N_;
    double z_, force_, mu_, cilia_radius_, fluid_viscosity_, drag_coefficient_, grid_spacing_;
    
    Eigen::MatrixX3d positions_;
    Eigen::MatrixX3d velocities_;
    Eigen::MatrixXd intrinsicFrequencies_;
    Eigen::MatrixXd angles_, tempAngles_;
    Eigen::MatrixXd k1_, k2_, k3_, k4_;
    
    bool findOrderParameter_;
    std::vector<double> simulationTimes_;
    std::vector<std::complex<double>> orderParameters_;
    
    Eigen::Vector3d initializePosition(int x, int y);
    double initializeFrequency();
    double initializeAngle();
    void updateVelocities(Eigen::MatrixXd& angles);
    void calculateVelocity(size_t x, size_t y, Eigen::MatrixXd& angles);
    void updateAngles();
    void rungeKutta4();
    void calculateStep(Eigen::MatrixXd& angles, Eigen::MatrixXd& result, double dt);
    double f(double x, double y, Eigen::MatrixXd& angles);
    Eigen::Vector3d stokeslet(int x_1, int y_1, int x_2, int y_2, Eigen::MatrixXd& angles);
    Eigen::Vector3d stokeslet(Eigen::Vector3d point, int x, int y, Eigen::MatrixXd& angles);
    void calcOmega();
    std::complex<double> calculateOrderParameter();
};