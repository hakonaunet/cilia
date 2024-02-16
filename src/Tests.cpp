// Tests.cpp

#include "Tests.hpp"

void test1() {
    SharedDataOseen sharedData_;
    sharedData_.width = 3;
    sharedData_.height = 4;
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

    // Convert the data to Python lists
    py::list zCoordinates, xVelocities, yVelocities, zVelocities;
    
    for (size_t i = 0; i < test_points.size(); ++i) {
        zCoordinates.append(test_points[i].z());
        xVelocities.append(velocities[i].x());
        yVelocities.append(velocities[i].y());
        zVelocities.append(velocities[i].z());
    }

    // Get the path to the current source file
    std::string currentFile = __FILE__;

    // Remove the filename to get the directory
    std::string currentDir = currentFile.substr(0, currentFile.rfind('/'));

    // Construct the path to the Python script
    std::string scriptPath = currentDir + "/../pyplot/Plots.py";

    // Read the Python script
    std::ifstream file(scriptPath);
    std::string script((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Execute the Python script
    py::exec(script);

    // Call the plot_data function
    py::object plot_test_1 = py::globals()["plot_test_1"];
    plot_test_1(xVelocities, yVelocities, zVelocities, zCoordinates, "Test1.png");

}

#ifdef RUN_TESTS

int main() {
    py::scoped_interpreter guard{}; // Start the Python interpreter
    test1();
}

#endif