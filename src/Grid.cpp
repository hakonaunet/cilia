// Grid.cpp
#include "Grid.hpp"

Grid::Grid(SharedDataKuramoto& data) : sharedData(data) {
    try {
        for (int i = 0; i < sharedData.N; i++) {
            std::vector<Oscillator> row;
            for (int j = 0; j < sharedData.N; j++) {
                row.push_back(Oscillator(sharedData.epsilon));
            }
            grid.push_back(row);
        }
        sharedData.startTime = std::chrono::high_resolution_clock::now();

        // Initialize the order parameter calculation if flag is set
        if (findOrderParameter) {
            std::complex<double> orderParameter = calculateOrderParameter(); // Replace with your actual function
            simulationTimes.push_back(sharedData.simulationTime);
            orderParameters.push_back(orderParameter);
        }
        else {
            simulationTimes.clear();
            orderParameters.clear();
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        // Handle the error (e.g., by cleaning up and terminating the program)
    }
}

std::vector<Oscillator*> Grid::getNeighbors4(size_t x, size_t y) {
    std::vector<Oscillator*> neighbors;
    if (x > 0) neighbors.push_back(&grid[x - 1][y]);
    if (x < grid.size() - 1) neighbors.push_back(&grid[x + 1][y]);
    if (y > 0) neighbors.push_back(&grid[x][y - 1]);
    if (y < grid[0].size() - 1) neighbors.push_back(&grid[x][y + 1]);
    return neighbors;
}

std::vector<Oscillator*> Grid::getNeighbors8(size_t x, size_t y) {
    std::vector<Oscillator*> neighbors = getNeighbors4(x, y);
    if (x > 0 && y > 0) neighbors.push_back(&grid[x - 1][y - 1]);
    if (x > 0 && y < grid[0].size() - 1) neighbors.push_back(&grid[x - 1][y + 1]);
    if (x < grid.size() - 1 && y > 0) neighbors.push_back(&grid[x + 1][y - 1]);
    if (x < grid.size() - 1 && y < grid[0].size() - 1) neighbors.push_back(&grid[x + 1][y + 1]);
    return neighbors;
}

double Grid::computeCouplingStrength(int x1, int y1, int x2, int y2, Coupling method) {
    double distance = std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));

    switch (method) {
        case Coupling::DistanceR:
            return (distance == 0) ? 0 : 1.0 / distance;

        case Coupling::DistanceR2:
            return (distance == 0) ? 0 : 1.0 / (distance * distance);

        case Coupling::None:
        case Coupling::Nearest4:
        case Coupling::Nearest8:
        case Coupling::Uniform:
            return 1.0; // Adjust as needed

        default:
            return 1.0;
    }
}

// Function to compute the Kuramoto model's sum term for different coupling methods
double Grid::kuramotoSum(int x, int y, Coupling couplingMethod) {
    double sum = 0.0;
    int N = grid.size() * grid[0].size(); // Total number of oscillators

    switch (couplingMethod) {
        case Coupling::None:
            // No coupling
            break;

        case Coupling::Nearest4: {
            auto neighbors = getNeighbors4(x, y);
            for (Oscillator* neighbor : neighbors) {
                sum += sin(neighbor->getAngle() - grid[x][y].getAngle());
            }
            break;
        }

        case Coupling::Nearest8: {
            auto neighbors = getNeighbors8(x, y);
            for (Oscillator* neighbor : neighbors) {
                sum += sin(neighbor->getAngle() - grid[x][y].getAngle());
            }
            break;
        }

        case Coupling::DistanceR:
        case Coupling::DistanceR2:
            // Implement distance-based coupling logic
            // ...

        case Coupling::Uniform:
            for (const auto& row : grid) {
                for (const Oscillator& osc : row) {
                    sum += sin(osc.getAngle() - grid[x][y].getAngle());
                }
            }
            break;

        default:
            // Default behavior or error handling
            break;
    }

    return sum / (couplingMethod == Coupling::Uniform ? N : 1);
}

// Function to update the grid using RK4
void Grid::updateGrid() {
    double K = 1.0; // Set the coupling strength
    double dt = sharedData.deltaTime; // Time step
    sharedData.simulationTime += dt; // Update simulation time
    sharedData.iteration++; // Update frame count

    auto startTime = std::chrono::high_resolution_clock::now(); // Start measuring time

    if (sharedData.gpuParallelization) {
        // Parallelize the grid update using CUDA
        
    }
    else {
        #pragma omp parallel for collapse(2) // Parallelize both x and y loops
        for (size_t x = 0; x < grid.size(); x++) {
            for (size_t y = 0; y < grid[0].size(); y++) {
                Oscillator& osc = grid[x][y];
                double theta = osc.getAngle();
                double omega = osc.getIntrinsicFrequency();

                // RK4 integration
                double k1 = dt * (omega + K * kuramotoSum(x, y, sharedData.coupling));
                double k2 = dt * (omega + K * kuramotoSum(x, y, sharedData.coupling) + 0.5 * k1);
                double k3 = dt * (omega + K * kuramotoSum(x, y, sharedData.coupling) + 0.5 * k2);
                double k4 = dt * (omega + K * kuramotoSum(x, y, sharedData.coupling) + k3);

                osc.setAngle(theta + (k1 + 2 * k2 + 2 * k3 + k4) / 6);
            }
        }
    }

    // Calculate the Kuramoto order parameter
    if (findOrderParameter) {
        std::complex<double> orderParameter = calculateOrderParameter(); // Replace with your actual function
        simulationTimes.push_back(sharedData.simulationTime);
        orderParameters.push_back(orderParameter);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    sharedData.updateTime = static_cast<double>(duration.count());
}

// Function to calculate the Kuramoto order parameter
std::complex<double> Grid::calculateOrderParameter() {
    double realPart = 0.0;
    double imagPart = 0.0;
    int N = grid.size() * grid[0].size(); // Total number of oscillators

    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < grid.size(); x++) {
        for (size_t y = 0; y < grid[0].size(); y++) {
            Oscillator& osc = grid[x][y];
            double theta = osc.getAngle();
            double cos_theta = std::cos(theta);
            double sin_theta = std::sin(theta);
            #pragma omp atomic
            realPart += cos_theta;
            #pragma omp atomic
            imagPart += sin_theta;
        }
    }

    std::complex<double> orderParameter(realPart, imagPart);
    return orderParameter / static_cast<double>(N);
}

std::vector<std::vector<Oscillator>>& Grid::getGrid() {
    return grid;
}

const std::vector<std::vector<Oscillator>>& Grid::getGrid() const {
    return grid;
}

void Grid::plotOrderParameter(std::string filename) const {

    std::cout << "Length of simulationTimes: " << simulationTimes.size() << std::endl;
    std::cout << "Length of orderParameters: " << orderParameters.size() << std::endl;

    // Convert the data to Python lists
    py::list pySimulationTimes;
    for (double t : simulationTimes) {
        pySimulationTimes.append(t);
    }

    py::list pyOrderParameters;
    for (const auto& op : orderParameters) {
        pyOrderParameters.append(std::abs(op)); // Use the absolute value of the order parameter
    }

    // Get the path to the current source file
    std::string currentFile = __FILE__;

    // Remove the filename to get the directory
    std::string currentDir = currentFile.substr(0, currentFile.rfind('/'));

    // Construct the path to the Python script
    std::string scriptPath = currentDir + "/../pyplot/PlotOrderParameter.py";

    // Read the Python script
    std::ifstream file(scriptPath);
    std::string script((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Execute the Python script
    py::exec(script);

    // Call the plot_data function
    py::object plot_data = py::globals()["plot_data"];
    plot_data(pySimulationTimes, pyOrderParameters, filename);
}