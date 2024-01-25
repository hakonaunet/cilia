// Oseen.cpp

#include "Oseen.hpp"

// Initialize the system
// Initialize the system
Oseen::Oseen(SharedDataOseen& data) : sharedData(data) { // Changed variable name to 'data'
        
    data.startTime = std::chrono::high_resolution_clock::now();

    width = data.width;
    height = data.height;
    N = width * height;
    z = 0;
    force = 1;
    mu = 1;

    positions = std::vector<std::vector<Eigen::Vector3d>>(data.width, std::vector<Eigen::Vector3d>(data.height, Eigen::Vector3d::Zero()));
    intrinsicFrequencies = std::vector<std::vector<double>>(data.width, std::vector<double>(data.height, 0.0));
    angles = std::vector<std::vector<double>>(data.width, std::vector<double>(data.height, 0.0));
    velocities = std::vector<std::vector<Eigen::Vector3d>>(data.width, std::vector<Eigen::Vector3d>(data.height, Eigen::Vector3d::Zero()));

    // Parse through the positions and set the initial values
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j < height; ++j) {
            // Set the initial values using initializePosition
            positions[i][j] = initializePosition(i, j);
            // Set the intrinsic frequencies using initializeFrequencoes
            intrinsicFrequencies[i][j] = initializeFrequency();
        }
    }

    // Initialize the order parameter calculation if flag is set
    if (findOrderParameter) {
        std::complex<double> orderParameter = calculateOrderParameter(); // Added template argument 'double'
        simulationTimes.push_back(data.simulationTime); // Changed 'SharedDataKuramoto' to 'data'
        orderParameters.push_back(orderParameter);
    }
    else {
        simulationTimes.clear();
        orderParameters.clear();
    }
}

Eigen::Vector3d Oseen::initializePosition(int x, int y) {
    // Calculate the grid point
    double gridX = x * sharedData.gridSpacing;
    double gridY = y * sharedData.gridSpacing;

    switch (sharedData.noiseMode) {
        case NoiseMode::None: {
            // Return the grid position without any noise
            return Eigen::Vector3d(gridX, gridY, 0);
        }
        case NoiseMode::Square: {
            // Generate a random position within a square centered at the grid point
            double noiseX = (std::rand() / (double)RAND_MAX - 0.5) * sharedData.noiseWidthMax * 2;
            double noiseY = (std::rand() / (double)RAND_MAX - 0.5) * sharedData.noiseWidthMax * 2;
            return Eigen::Vector3d(gridX + noiseX, gridY + noiseY, 0);
        }
        case NoiseMode::Circular: {
            // Generate a random angle and a random radius
            double angle = (std::rand() / (double)RAND_MAX) * 2 * M_PI;
            double radius = std::sqrt(std::rand() / (double)RAND_MAX) * sharedData.noiseWidthMax;

            // Convert polar coordinates to Cartesian coordinates
            double noiseX = radius * std::cos(angle);
            double noiseY = radius * std::sin(angle);

            // Return the final position
            return Eigen::Vector3d(gridX + noiseX, gridY + noiseY, 0);
        }
        default: {
            // Handle other cases if necessary
            return Eigen::Vector3d(gridX, gridY, 0);
        }
    }
}

double Oseen::initializeFrequency() {
    double defaultFrequency = 1.0;
    switch (sharedData.frequencyDistribution) {
        case FrequencyDistribution::None: {
            // Return the default frequency
            return defaultFrequency;
        }
        case FrequencyDistribution::Uniform: {
            // Generate a random frequency between defaultFrequency - frequencyWidth and defaultFrequency + frequencyWidth
            return defaultFrequency - sharedData.frequencyWidth + (std::rand() / (double)RAND_MAX) * 2 * sharedData.frequencyWidth;
        }
        case FrequencyDistribution::Gaussian: {
            // Generate a random frequency using a Gaussian distribution
            std::default_random_engine generator;
        std::normal_distribution<double> distribution(defaultFrequency, sharedData.frequencyDeviation);
            return distribution(generator);
        }
        default: {
            // Handle other cases if necessary
            return defaultFrequency;
        }
    }
}

double Oseen::initializeAngle() {
    double defaultAngle = M_PI; // Default angle is pi
    switch (sharedData.angleDistribution) {
        case AngleDistribution::None: {
            // Return the default angle
            return defaultAngle;
        }
        case AngleDistribution::Uniform: {
            // Generate a random angle in the range [defaultAngle - angleWidth, defaultAngle + angleWidth]
            double lowerBound = defaultAngle - sharedData.angleWidth;
            double upperBound = defaultAngle + sharedData.angleWidth;
            double randomValue = (std::rand() / (double)RAND_MAX);
            double angle = lowerBound + randomValue * (upperBound - lowerBound);

            // Ensure the angle is within [0, 2pi)
            angle = fmod(angle, 2 * M_PI);
            if (angle < 0) {
                angle += 2 * M_PI;
            }
            return angle;
        }
        case AngleDistribution::Gaussian: {
            // Generate a random angle using a Gaussian distribution
            std::default_random_engine generator;
            std::normal_distribution<double> distribution(defaultAngle, sharedData.angleDeviation);
            double angle = distribution(generator);
            // Ensure the angle is within [0, 2pi)
            angle = fmod(angle, 2 * M_PI);
            if (angle < 0) {
                angle += 2 * M_PI;
            }
            return angle;
        }
        default: {
            // Handle other cases if necessary
            return defaultAngle;
        }
    }
}

void Oseen::iteration() {
    unsigned int cilia_left_to_consider = N+1;
    for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < height; ++y) {
            calculateVelocity(x, y, cilia_left_to_consider);
            cilia_left_to_consider--;
        }
    }

    // Update the positions
    // ...
}

// Get velocity of cilia at position (x, y)
void Oseen::calculateVelocity(int x, int y, int break_point) {
    int counter = 0;
    Eigen::Vector3d r_i = positions[x][y] + Eigen::Vector3d(cos(angles[x][y]), sin(angles[x][y]), z);
    Eigen::Vector3d f_i = Eigen::Vector3d(-force * sin(angles[x][y]), force * cos(angles[x][y]), z);
    for (int i = width-1; i >= 0; i--) {
        for (int j = height-1; j >= 0; j--) {
            if (i == x && j == y) {
                continue;
            }
            if (counter >= break_point) {
                return;               
            }
            Eigen::Vector3d r_j = positions[i][j] + Eigen::Vector3d(cos(angles[i][j]), sin(angles[i][j]), z);
            Eigen::Vector3d r = r_i - r_j;
            double r_length = r.norm(); // Length of r
            Eigen::Vector3d f_j = Eigen::Vector3d(-force * sin(angles[i][j]), force * cos(angles[i][j]), z);
            velocities[x][y] += (1/(8*M_PI*mu*r_length)) * (f_j + (r.dot(f_j)/r_length) * r);
            velocities[i][j] += (1/(8*M_PI*mu*r_length)) * (f_i + (r.dot(f_i)/r_length) * r);
            counter++;
        }
    }
}

// Function to calculate the Kuramoto order parameter
std::complex<double> Oseen::calculateOrderParameter() {
    double realPart = 0.0;
    double imagPart = 0.0;
    int N = angles.size() * angles[0].size(); // Total number of oscillators

    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < angles.size(); x++) {
        for (size_t y = 0; y < angles[0].size(); y++) {
            double theta = angles[x][y];
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