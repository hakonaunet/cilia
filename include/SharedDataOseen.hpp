// SharedDataOseen.hpp  

#pragma once
#include <cmath>

#include "SharedData.hpp"
#include "OseenEnums.hpp"

struct SharedDataOseen : public SharedData {
    int width, height;
    float  gridSpacing, frequencyWidth, noiseWidth, frequencyDeviation, angleDeviation, angleWidth;
    const double noiseWidthMax, frequencyWidthMax, frequencyDeviationMax, angleWidthMax, angleDeviationMax;
    NoiseMode noiseMode;
    FrequencyDistribution frequencyDistribution;
    AngleDistribution angleDistribution;

    SharedDataOseen() : SharedData(), width(2), height(1), frequencyWidth(0.1), gridSpacing(4), noiseWidth(0), noiseWidthMax(2),
                        frequencyDeviation(0.1), frequencyWidthMax(1), frequencyDeviationMax(1), 
                        angleWidth(M_PI), angleWidthMax(M_PI), angleDeviation(0.1), angleDeviationMax(1),
                        noiseMode(NoiseMode::None), frequencyDistribution(FrequencyDistribution::None), angleDistribution(AngleDistribution::Uniform) {
        reset();
    }

    void reset() override {
        SharedData::reset();
    }
};