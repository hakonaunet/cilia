// main.cpp

#include "Application.hpp"

#include <ctime>

int main() {
    py::scoped_interpreter guard{}; // Start the Python interpreter
    Application app;
    app.run();
    return 0;
}