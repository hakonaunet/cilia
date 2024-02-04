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

    double getCiliaRadius() const { return cilia_radius; }
    double getGridSpacing() const { return grid_spacing; }
    unsigned int getWidth() const { return width; }
    unsigned int getHeight() const { return height; }
    const std::vector<std::vector<Eigen::Vector3d>>& getPositions() const { return positions; }
    const std::vector<std::vector<double>>& getAngles() const { return angles; }
    double getVelocityMagnitudeAtPoint(Eigen::Vector3d point);
    Eigen::Vector3d getVelocityAtPoint(Eigen::Vector3d point);
    Eigen::Vector3d getForce(int x, int y);
    double getForceMagnitude(int x, int y);
    Eigen::Vector3d getR(int x, int y);
    Eigen::Vector3d getTangent(int x, int y);

private:
    SharedDataOseen& sharedData;

    unsigned int width, height, N;
    double z, force, mu, cilia_radius, fluid_viscosity, drag_coefficient, grid_spacing;
    
    std::vector<std::vector<Eigen::Vector3d>> positions;
    std::vector<std::vector<Eigen::Vector3d>> velocities;
    std::vector<std::vector<double>> intrinsicFrequencies;
    std::vector<std::vector<double>> angles;

    bool findOrderParameter;
    std::vector<double> simulationTimes;
    std::vector<std::complex<double>> orderParameters;
    
    Eigen::Vector3d initializePosition(int x, int y);
    double initializeFrequency();
    double initializeAngle();
    void calculateVelocity(size_t x, size_t y);
    void updateAngles();
    Eigen::Vector3d stokeslet(int x_1, int y_1, int x_2, int y_2);
    Eigen::Vector3d stokeslet(Eigen::Vector3d point, int x, int y);
    void calcOmega();
    std::complex<double> calculateOrderParameter();
};