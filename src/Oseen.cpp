// Oseen.cpp

#include "Oseen.hpp"

// Initialize the system
Oseen::Oseen(SharedDataOseen& data) : sharedData(data), positions(data.width * data.height, 3), velocities(data.width * data.height, 3),
    intrinsicFrequencies(data.width, data.height), angles(data.width, data.height) {
        
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

    // Parse through the positions and set the initial values
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j < height; ++j) {
            size_t index = i * height + j;
            positions.row(index) = initializePosition(i, j);
            velocities.row(index).setZero(); // Assuming you start with zero velocity
            intrinsicFrequencies(i, j) = initializeFrequency();
            angles(i, j) = initializeAngle();
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

    updateVelocities();
    updateAngles();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    sharedData.updateTime = static_cast<double>(duration.count());
}

void Oseen::updateVelocities() {
    velocities.setZero();
    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < height; ++y) {
            calculateVelocity(x, y);
        }
    }
}

// Get velocity of cilia at position (x, y)
void Oseen::calculateVelocity(size_t x, size_t y) {
    Eigen::Vector3d r_i = getR(x, y);
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j < height; ++j) {
            if (i == x && j == y) continue;
            // Directly add the result of stokeslet to the corresponding row
            velocities.block<1, 3>(x * height + y, 0) += stokeslet(r_i, i, j);
        }
    }
}

void Oseen::updateAngles() {
    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < height; ++y) {
            double f_i = getForceMagnitude(x, y);
            Eigen::Vector3d t_i = Eigen::Vector3d(-sin(angles(x, y)), cos(angles(x, y)), 0);
            double omega_i = f_i / (drag_coefficient * cilia_radius);
            Eigen::Vector3d velocity_at_xy = velocities.row(x * height + y); // Assuming linear indexing
            double dot_product = t_i.dot(velocity_at_xy);

            // Runge-Kutta 4 method
            double k1 = sharedData.deltaTime * (omega_i + dot_product / cilia_radius);

            angles(x, y) += k1;

            // Ensure the angle is between 0 and 2π
            while (angles(x, y) < 0) angles(x, y) += 2 * M_PI;
            while (angles(x, y) >= 2 * M_PI) angles(x, y) -= 2 * M_PI;
        }
    }
}

Eigen::Vector3d Oseen::stokeslet(Eigen::Vector3d point, int x, int y){
    Eigen::Vector3d f_2 = getForce(x, y);
    Eigen::Vector3d r = point - getR(x, y);
    double r_length = r.norm();
    return (1/(8*M_PI*mu*r_length)) * (f_2 + (r.dot(f_2)/(r_length*r_length)) * r);
}

Eigen::Vector3d Oseen::stokeslet(int x_1, int y_1, int x_2, int y_2) {
    Eigen::Vector3d f_2 = getForce(x_2, y_2);
    Eigen::Vector3d r = getR(x_1, y_1) - getR(x_2, y_2);
    double r_length = r.norm();
    return (1/(8*M_PI*mu*r_length)) * (f_2 + (r.dot(f_2)/(r_length*r_length)) * r);
}

Eigen::Vector3d Oseen::getForce(int x, int y) {
    return getForceMagnitude(x, y) * getTangent(x, y);
}

double Oseen::getForceMagnitude(int x, int y) {
    return sharedData.force + sharedData.force_amplitude*sin(angles(x,y));
}

Eigen::Vector3d Oseen::getR(int x, int y) {
    Eigen::Vector3d pos_at_xy = positions.row(x * height + y); // Assuming linear indexing for positions
    return pos_at_xy + cilia_radius * Eigen::Vector3d(cos(angles(x, y)), sin(angles(x, y)), 0);
}

Eigen::Vector3d Oseen::getTangent(int x, int y) {
    return Eigen::Vector3d(-sin(angles(x, y)), cos(angles(x, y)), 0);
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

    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < width; x++) {
        for (size_t y = 0; y < height; y++) {
            double theta = angles(x,y);
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