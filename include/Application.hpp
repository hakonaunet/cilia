// Application.hpp

#pragma once

#include <SFML/Graphics.hpp>
#include "imgui.h"
#include "imgui-SFML.h"

#include <iostream>

#include "UI.hpp"
#include "SharedDataKuramoto.hpp"
#include "OscillatorView.hpp"
#include "Grid.hpp"
#include "PlotWindow.hpp"

class Application {
public:
    Application();

    void run();

private:
    SharedDataKuramoto SharedDataKuramoto;
    Grid grid;
    UI ui;
};