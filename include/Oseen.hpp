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

    void plotOrderParameter(std::string filename) const;

private:
    SharedDataOseen& sharedData;

    unsigned int width, height, N;
    double z, force, mu, cilia_radius, fluid_viscosity, drag_coefficient;
    
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
    void iteration();
    void calculateVelocity(int x, int y, int break_point);
    void updateAngles();
    double getForce(int x, int y);
    std::complex<double> calculateOrderParameter();
};