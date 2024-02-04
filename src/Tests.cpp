// Tests.cpp

#include "Tests.hpp"

void test1() {
    SharedDataOseen sharedData_;
    Oseen oseen_(sharedData_);
    std::vector<double> dataPoints;
    for (int i = 0; i < 100; i++) {
        oseen_.iteration();
    }
    unsigned int test_points = 100;
    double radius = 10.0; // Replace with your desired radius
    double distance_between_points = 2.0 * radius / (test_points - 1);

    std::vector<Eigen::Vector3d> data_points(test_points);
    std::vector<Eigen::Vector3d> velocities(test_points);
    for (unsigned int i = 0; i < test_points; ++i) {
        double z = -radius + i * distance_between_points;
        data_points[i] = Eigen::Vector3d(-3, 0, z);
    }
    for (unsigned int i = 0; i < test_points; ++i) {
        velocities[i] = oseen_.getVelocityAtPoint(data_points[i]);
    }
    plotTest1(data_points, velocities);
}

void plotTest1(std::vector<Eigen::Vector3d> test_points, std::vector<Eigen::Vector3d> velocities) {
    // Convert the data to arrays that ImPlot can use
    std::vector<double> measurement_points(test_points.size());
    std::vector<double> velocity_x(velocities.size());
    std::vector<double> velocity_y(velocities.size());
    std::vector<double> velocity_z(velocities.size());

    for (size_t i = 0; i < test_points.size(); ++i) {
        measurement_points[i] = test_points[i].z();
        velocity_x[i] = velocities[i].x();
        velocity_y[i] = velocities[i].y();
        velocity_z[i] = velocities[i].z();
    }

    // Plot the data using ImPlot
    if (ImPlot::BeginPlot("Test 1: symmetric velocity field around cilia plane", 
                          "Measurement points along the z-axis", 
                          "Velocity field")) {
        ImPlot::PlotLine("x-component", measurement_points.data(), velocity_x.data(), measurement_points.size());
        ImPlot::PlotLine("y-component", measurement_points.data(), velocity_y.data(), measurement_points.size());
        ImPlot::PlotLine("z-component", measurement_points.data(), velocity_z.data(), measurement_points.size());
        ImPlot::EndPlot();
    }

    // Save the plot to a file
    // Note: ImPlot does not support saving to a file directly. You need to use a library like stb_image_write to save the ImGui render target to a file.
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

#ifdef RUN_TESTS

int main() {
    py::scoped_interpreter guard{}; // Start the Python interpreter
    test1();
}

#endif