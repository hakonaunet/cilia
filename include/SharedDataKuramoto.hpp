// SharedDataKuramoto.hpp  

#pragma once
#include "SharedData.hpp"
#include "Coupling.hpp"

struct SharedDataKuramoto : public SharedData {
    int N;
    Coupling coupling;
    float epsilon;

    SharedDataKuramoto() : SharedData(), N(20), coupling(Coupling::Uniform), epsilon(0.1) {
        reset();
    }

    void reset() override {
        SharedData::reset();
    }
};