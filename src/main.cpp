#include "Grid.hpp"
#include "OscillatorView.hpp"
#include "Application.hpp"

// #include <SFML/Graphics.hpp>
// #include <SFML/System/Clock.hpp>
// #include <SFML/System/Sleep.hpp>
// #include <../imgui/imgui.h>
// #include <../imgui-sfml/imgui-SFML.h>

#include <ctime>

int main() {
    py::scoped_interpreter guard{}; // Start the Python interpreter
    Application app;
    app.run();
    return 0;
}