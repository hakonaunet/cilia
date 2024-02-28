// SharedDataOseen.hpp  

#pragma once
#include <cmath>

#include "SharedData.hpp"
#include "OseenEnums.hpp"

struct SharedDataOseen : public SharedData {
    int width, height;
    float  gridSpacing, frequencyWidth, noiseWidth, frequencyDeviation, angleDeviation, angleWidth, cilia_radius, force, force_amplitude,
            alfa1, beta1, alfa2, beta2;
    const float noiseWidthMax, frequencyWidthMax, frequencyDeviationMax, angleWidthMax, angleDeviationMax;
    NoiseMode noiseMode;
    FrequencyDistribution frequencyDistribution;
    AngleDistribution angleDistribution;

    SharedDataOseen() : SharedData(), width(100), height(1), gridSpacing(1), frequencyWidth(0.1), noiseWidth(0), frequencyDeviation(0.1), 
                    angleDeviation(0.1), angleWidth(M_PI), cilia_radius(0.2), force(1.5), force_amplitude(1), 
                    alfa1(0.5), beta1(0.5), alfa2(0.5), beta2(0.5),
                    noiseWidthMax(2), frequencyWidthMax(1), frequencyDeviationMax(1), angleWidthMax(M_PI), angleDeviationMax(1), 
                    noiseMode(NoiseMode::None), frequencyDistribution(FrequencyDistribution::None), angleDistribution(AngleDistribution::None) {
        deltaTime = 0.005;
        reset();
    }
    void reset() override {
        SharedData::reset();
    }
};