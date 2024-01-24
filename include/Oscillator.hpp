// Oscillator.hpp
#pragma once
#include <cmath>

class Oscillator {
private:
    double angle;
    double intrinsicFrequency;

public:
    Oscillator(float epsilon);

    static constexpr double TWO_PI = 2 * M_PI;

    void update(const double& dtheta);
    double getAngle() const;
    double getIntrinsicFrequency() const;
    void setAngle(double newAngle);
};