// Grid.hpp
#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <complex>
#include <iostream>
#include <fstream>

#include "Oscillator.hpp"
#include "SharedDataKuramoto.hpp"
#include "Coupling.hpp"

#include <omp.h>
#include <pybind11/embed.h>

namespace py = pybind11;

class Grid {
public:
    Grid(SharedDataKuramoto& data);

    void updateGrid();
    
    std::vector<std::vector<Oscillator>>& getGrid(); // Non-const version
    const std::vector<std::vector<Oscillator>>& getGrid() const; // Const version

    void plotOrderParameter(std::string filename) const;

private:
    SharedDataKuramoto& sharedData;
    
    std::vector<std::vector<Oscillator>> grid;

    bool findOrderParameter;
    std::vector<double> simulationTimes;
    std::vector<std::complex<double>> orderParameters;
    std::complex<double> calculateOrderParameter();

    std::vector<Oscillator*> getNeighbors4(size_t x, size_t y);
    std::vector<Oscillator*> getNeighbors8(size_t x, size_t y);
    
    // Updated signature to use grid positions for distance calculation
    double computeCouplingStrength(int x1, int y1, int x2, int y2, Coupling method);

    double kuramotoSum(int x, int y, Coupling couplingMethod);
};