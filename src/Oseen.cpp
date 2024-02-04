// Oseen.cpp

#include "Oseen.hpp"

// Initialize the system
Oseen::Oseen(SharedDataOseen& data) : sharedData(data) { // Changed variable name to 'data'
        
    data.startTime = std::chrono::high_resolution_clock::now();

    width = data.width;
    height = data.height;
    N = width * height;
    z = 0;
    force = data.force;
    mu = 1;
    cilia_radius = data.cilia_radius;
    fluid_viscosity = 1;
    drag_coefficient = 6*M_PI*fluid_viscosity*cilia_radius;
    grid_spacing = data.gridSpacing;

    positions = std::vector<std::vector<Eigen::Vector3d>>(data.width, std::vector<Eigen::Vector3d>(data.height, Eigen::Vector3d::Zero()));
    intrinsicFrequencies = std::vector<std::vector<double>>(data.width, std::vector<double>(data.height, 0.0));
    angles = std::vector<std::vector<double>>(data.width, std::vector<double>(data.height, 0.0));
    velocities = std::vector<std::vector<Eigen::Vector3d>>(data.width, std::vector<Eigen::Vector3d>(data.height, Eigen::Vector3d::Zero()));

    // Parse through the positions and set the initial values
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j < height; ++j) {

            positions[i][j] = initializePosition(i, j);
            intrinsicFrequencies[i][j] = initializeFrequency();
            angles[i][j] = initializeAngle();
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
    double gridX = x * grid_spacing;
    double gridY = y * grid_spacing;

    switch (sharedData.noiseMode) {
        case NoiseMode::None: {
            // Return the grid position without any noise
            return Eigen::Vector3d(gridX, gridY, z);
        }
        case NoiseMode::Square: {
            // Generate a random position within a square centered at the grid point
            double noiseX = (std::rand() / (double)RAND_MAX - 0.5) * sharedData.noiseWidthMax * 2;
            double noiseY = (std::rand() / (double)RAND_MAX - 0.5) * sharedData.noiseWidthMax * 2;
            return Eigen::Vector3d(gridX + noiseX, gridY + noiseY, z);
        }
        case NoiseMode::Circular: {
            // Generate a random angle and a random radius
            double angle = (std::rand() / (double)RAND_MAX) * 2 * M_PI;
            double radius = std::sqrt(std::rand() / (double)RAND_MAX) * sharedData.noiseWidthMax;

            // Convert polar coordinates to Cartesian coordinates
            double noiseX = radius * std::cos(angle);
            double noiseY = radius * std::sin(angle);

            // Return the final position
            return Eigen::Vector3d(gridX + noiseX, gridY + noiseY, z);
        }
        default: {
            // Handle other cases if necessary
            return Eigen::Vector3d(gridX, gridY, z);
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
    sharedData.simulationTime += sharedData.deltaTime; // Update simulation time
    sharedData.iteration++; // Update frame count
    auto startTime = std::chrono::high_resolution_clock::now(); // Start measuring time

    // Reset velocities
    velocities = std::vector<std::vector<Eigen::Vector3d>>(width, std::vector<Eigen::Vector3d>(height, Eigen::Vector3d::Zero()));

    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < height; ++y) {
            calculateVelocity(x, y);
        }
    }
    updateAngles();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    sharedData.updateTime = static_cast<double>(duration.count());
}

// Get velocity of cilia at position (x, y)
void Oseen::calculateVelocity(size_t x, size_t y) {
    Eigen::Vector3d r_i = getR(x, y);
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j < height; ++j) {
            if (i == x && j == y) {
                continue;
            }
            velocities[x][y] += stokeslet(r_i, i, j);
        }
    }
}

void Oseen::updateAngles() {
    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < height; ++y) {
            double f_i = getForceMagnitude(x, y);
            Eigen::Vector3d t_i = Eigen::Vector3d(-sin(angles[x][y]), cos(angles[x][y]), 0);
            double omega_i = f_i / (drag_coefficient * cilia_radius);
            double dot_product = t_i.dot(velocities[x][y]);

            // Runge-Kutta 4 method
            double k1 = sharedData.deltaTime * (omega_i + dot_product / cilia_radius);
            double k2 = sharedData.deltaTime * (omega_i + (dot_product + 0.5 * sharedData.deltaTime * k1) / cilia_radius);
            double k3 = sharedData.deltaTime * (omega_i + (dot_product + 0.5 * sharedData.deltaTime * k2) / cilia_radius);
            double k4 = sharedData.deltaTime * (omega_i + (dot_product + sharedData.deltaTime * k3) / cilia_radius);

            angles[x][y] += (k1 + 2*k2 + 2*k3 + k4) / 6;

            // Ensure the angle is between 0 and 2π
            if (angles[x][y] < 0) {
                angles[x][y] += 2 * M_PI;
            } else if (angles[x][y] >= 2 * M_PI) {
                angles[x][y] -= 2 * M_PI;
            }
        }
    }
}

Eigen::Vector3d Oseen::stokeslet(int x_1, int y_1, int x_2, int y_2) {
    Eigen::Vector3d f_2 = getForce(x_2, y_2);
    Eigen::Vector3d r = getR(x_1, y_1) - getR(x_2, y_2);
    double r_length = r.norm();
    return (1/(8*M_PI*mu*r_length)) * (f_2 + (r.dot(f_2)/(r_length*r_length)) * r);
}

Eigen::Vector3d Oseen::stokeslet(Eigen::Vector3d point, int x, int y){
    Eigen::Vector3d f_2 = getForce(x, y);
    Eigen::Vector3d r = point - getR(x, y);
    double r_length = r.norm();
    return (1/(8*M_PI*mu*r_length)) * (f_2 + (r.dot(f_2)/(r_length*r_length)) * r);
}

Eigen::Vector3d Oseen::getForce(int x, int y) {
    return getForceMagnitude(x, y) * getTangent(x, y);
}

double Oseen::getForceMagnitude(int x, int y) {
    return sharedData.force + sharedData.force_amplitude*sin(angles[x][y]);
}

Eigen::Vector3d Oseen::getR(int x, int y) {
    return positions[x][y] + cilia_radius * Eigen::Vector3d(cos(angles[x][y]), sin(angles[x][y]), 0);
}

Eigen::Vector3d Oseen::getTangent(int x, int y) {
    return Eigen::Vector3d(-sin(angles[x][y]), cos(angles[x][y]), 0);
}

double Oseen::getVelocityMagnitudeAtPoint(Eigen::Vector3d point) {
    return getVelocityAtPoint(point).norm();
}

Eigen::Vector3d Oseen::getVelocityAtPoint(Eigen::Vector3d point) {
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < height; ++y) {
            velocity += stokeslet(point, x, y);
        }
    }
    return velocity;
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

void Oseen::calcOmega() {
    
}