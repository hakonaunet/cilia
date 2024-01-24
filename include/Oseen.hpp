// Oseen.hpp

#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <complex>
#include <iostream>
#include <fstream>

#include <Eigen/Dense>

#include "SharedDataOseen.hpp"

class Oseen {
public:
    Oseen(SharedDataOseen& data);

    void plotOrderParameter(std::string filename) const;

private:
    SharedDataOseen& sharedData;

    unsigned int width, height;
    double dt;
    
    std::vector<std::vector<Eigen::Vector3d>> positions;
    std::vector<std::vector<double>> intrinsicFrequencies;
    std::vector<std::vector<double>> angles;

    bool findOrderParameter;
    std::vector<double> simulationTimes;
    std::vector<std::complex<double>> orderParameters;
    
    Eigen::Vector3d Oseen::initializePosition(int x, int y);
    double initializeFrequency();
    double initializeAngle();
    void iteration();
    std::complex<double> calculateOrderParameter();
};