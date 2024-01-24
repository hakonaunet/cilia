// UI.hpp

#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>

#include "imgui.h"
#include "imgui-SFML.h"
#include "SFML/Graphics.hpp"

#include "SharedDataKuramoto.hpp"
#include "SharedDataOseen.hpp"
#include "Coupling.hpp"
#include "Grid.hpp"

class UI {
public:
    UI(SharedData& data) 
        : sharedData(&data), 
            lastUpdateTime(std::chrono::high_resolution_clock::now()) {}

    void render(); // Implement the UI rendering logic for the simulations

    std::string formatTime(int timeInSeconds) const; // Format the time as a string
    std::string getRuntime() const; // Return the simulation time as a string
    std::string getSimulationTime() const; // Return the simulation time as a string

    std::chrono::high_resolution_clock::time_point pauseStartTime;
    void pause();
    void resume();

private:
    SharedData* sharedData = nullptr;
    // Member variables to handle UI update frequency
    const std::chrono::milliseconds minUpdateInterval{200};
    std::chrono::high_resolution_clock::time_point lastUpdateTime;
};