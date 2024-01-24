// Oscillator.cpp
#include "Oscillator.hpp"
#include <iostream>

Oscillator::Oscillator(float epsilon) {
    angle = ((double)rand() / RAND_MAX) * 2 * M_PI;

    const double lowestFrequency = 1-epsilon; // Lowest possible intrinsic frequency
    const double highestFrequency = 1+epsilon; // Highest possible intrinsic frequency
    const double scale = 1; // Scale factor for intrinsic frequency

    intrinsicFrequency = (lowestFrequency + ((double)rand() / RAND_MAX) * (highestFrequency - lowestFrequency))*scale;
}

void Oscillator::update(const double& dtheta) {
    angle += dtheta;
    // Ensure the angle is between [0, 2*PI]
    while(angle > TWO_PI) angle -= TWO_PI;
    while(angle < 0) angle += TWO_PI;
}

double Oscillator::getAngle() const { return angle; }

double Oscillator::getIntrinsicFrequency() const { return intrinsicFrequency; }

void Oscillator::setAngle(double newAngle) {
    angle = newAngle;
    // Ensure the angle is between [0, 2*PI]
    while(angle > TWO_PI) angle -= TWO_PI;
    while(angle < 0) angle += TWO_PI;
}