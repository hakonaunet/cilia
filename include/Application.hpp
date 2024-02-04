// Application.hpp

#pragma once

#include <SFML/Graphics.hpp>
#include "imgui.h"
#include "imgui-SFML.h"

#include <iostream>

#include "UI.hpp"
#include "SharedData.hpp"
#include "SharedDataKuramoto.hpp"
#include "SharedDataOseen.hpp"
#include "OscillatorView.hpp"
#include "OseenView.hpp"
#include "Grid.hpp"
#include "PlotWindow.hpp"
#include "Oseen.hpp"
#include "Tests.hpp"


enum class SimulationMode {
    Kuramoto,
    Oseen,
    Test1
};

class Application {
public:
    Application();

    void run();
    void init();

private:
    void promptForSimulationMode();

    SimulationMode mode;
    sf::Clock deltaClock;
    std::unique_ptr<SharedData> sharedData;
    std::unique_ptr<Grid> grid;
    std::unique_ptr<Oseen> oseen;
    std::unique_ptr<UI> ui;
};