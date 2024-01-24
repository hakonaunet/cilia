// SharedData.hpp

#pragma once
#include <chrono>

struct SharedData {
    float deltaTime;
    int iteration;
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::duration totalPauseTime;        
    double simulationTime;
    double updateTime;
    bool isSimulationRunning;
    bool isPaused;
    bool startSimulation;
    bool trackKuramotoOrderParameter;
    bool shouldPlotOrderParameter;
    bool gpuParallelization;

    SharedData() : deltaTime(0.1), updateTime(0), isSimulationRunning(false), 
                    isPaused(false), startSimulation(false),
                    trackKuramotoOrderParameter(false), shouldPlotOrderParameter(false), 
                    gpuParallelization(false) {
        reset();
    }

    virtual void reset() {
        startTime = std::chrono::high_resolution_clock::now();
        totalPauseTime = std::chrono::high_resolution_clock::duration::zero();
        simulationTime = 0;
        iteration = 0;
    }
};