// Oseen.cpp

#include "Oseen.hpp"

// Initialize the system
Oseen::Oseen(SharedDataOseen& data) : sharedData(data), positions_(data.width * data.height, 3), velocities_(data.width * data.height, 3),
    intrinsicFrequencies_(data.width, data.height), angles_(data.width, data.height), tempAngles_(data.width, data.height), 
    k1_(data.width, data.height), k2_(data.width, data.height), k3_(data.width, data.height), k4_(data.width, data.height) {
        
    data.startTime = std::chrono::high_resolution_clock::now();

    width_ = data.width;
    height_ = data.height;
    N_ = width_ * height_;
    z_ = 0;
    force_ = data.force;
    mu_ = 1;
    cilia_radius_ = data.cilia_radius;
    fluid_viscosity_ = 1;
    drag_coefficient_ = 6*M_PI*fluid_viscosity_*cilia_radius_;
    grid_spacing_ = data.gridSpacing;

    // Parse through the positions_ and set the initial values
    for (size_t i = 0; i < width_; ++i) {
        for (size_t j = 0; j < height_; ++j) {
            size_t index = i * height_ + j;
            positions_.row(index) = initializePosition(i, j);
            velocities_.row(index).setZero(); // Assuming you start with zero velocity
            intrinsicFrequencies_(i, j) = initializeFrequency();
            angles_(i, j) = initializeAngle();
        }
    }

    // Initialize the order parameter calculation if flag is set
    if (findOrderParameter_) {
        std::complex<double> orderParameter = calculateOrderParameter(); // Added template argument 'double'
        simulationTimes_.push_back(data.simulationTime); // Changed 'SharedDataKuramoto' to 'data'
        orderParameters_.push_back(orderParameter);
    }
    else {
        simulationTimes_.clear();
        orderParameters_.clear();
    }
}

Eigen::Vector3d Oseen::initializePosition(int x, int y) {
    // Calculate the grid point
    double gridX = x * grid_spacing_;
    double gridY = y * grid_spacing_;

    switch (sharedData.noiseMode) {
        case NoiseMode::None: {
            // Return the grid position without any noise
            return Eigen::Vector3d(gridX, gridY, z_);
        }
        case NoiseMode::Square: {
            // Generate a random position within a square centered at the grid point
            double noiseX = (std::rand() / (double)RAND_MAX - 0.5) * sharedData.noiseWidthMax * 2;
            double noiseY = (std::rand() / (double)RAND_MAX - 0.5) * sharedData.noiseWidthMax * 2;
            return Eigen::Vector3d(gridX + noiseX, gridY + noiseY, z_);
        }
        case NoiseMode::Circular: {
            // Generate a random angle and a random radius
            double angle = (std::rand() / (double)RAND_MAX) * 2 * M_PI;
            double radius = std::sqrt(std::rand() / (double)RAND_MAX) * sharedData.noiseWidthMax;

            // Convert polar coordinates to Cartesian coordinates
            double noiseX = radius * std::cos(angle);
            double noiseY = radius * std::sin(angle);

            // Return the final position
            return Eigen::Vector3d(gridX + noiseX, gridY + noiseY, z_);
        }
        default: {
            // Handle other cases if necessary
            return Eigen::Vector3d(gridX, gridY, z_);
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
            // Generate a random frequency between defaultFrequency - frequencywidth_ and defaultFrequency + frequencywidth_
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
            // Generate a random angle in the range [defaultAngle - anglewidth_, defaultAngle + anglewidth_]
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

    rungeKutta4();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    sharedData.updateTime = static_cast<double>(duration.count());
}

void Oseen::updateVelocities(Eigen::MatrixXd& angles) {
    velocities_.setZero();
    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < width_; ++x) {
        for (size_t y = 0; y < height_; ++y) {
            calculateVelocity(x, y, angles);
        }
    }
}

// Get velocity of cilia at position (x, y)
void Oseen::calculateVelocity(size_t x, size_t y, Eigen::MatrixXd& angles) {
    Eigen::Vector3d r_i = getR(x, y, angles);
    for (size_t i = 0; i < width_; ++i) {
        for (size_t j = 0; j < height_; ++j) {
            if (i == x && j == y) continue;
            // Directly add the result of stokeslet to the corresponding row
            velocities_.block<1, 3>(x * height_ + y, 0) += stokeslet(r_i, i, j, angles);
        }
    }
}

void Oseen::updateAngles() {
    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < width_; ++x) {
        for (size_t y = 0; y < height_; ++y) {
            double f_i = getForceMagnitude(x, y, angles_);
            Eigen::Vector3d t_i = Eigen::Vector3d(-sin(angles_(x, y)), cos(angles_(x, y)), 0);
            double omega_i = f_i / (drag_coefficient_ * cilia_radius_);
            Eigen::Vector3d velocity_at_xy = velocities_.row(x * height_ + y); // Assuming linear indexing
            double dot_product = t_i.dot(velocity_at_xy);

            // Euler's method
            angles_(x, y) += sharedData.deltaTime * (omega_i + dot_product / cilia_radius_);

            // Ensure the angle is between 0 and 2π
            while (angles_(x, y) < 0) angles_(x, y) += 2 * M_PI;
            while (angles_(x, y) >= 2 * M_PI) angles_(x, y) -= 2 * M_PI;
        }
    }
}

void Oseen::rungeKutta4() {
    calculateStep(angles_, k1_, sharedData.deltaTime);
    tempAngles_ = angles_ + 0.5 * k1_;
    calculateStep(tempAngles_, k2_, sharedData.deltaTime);
    tempAngles_ = angles_ + 0.5 * k2_;
    calculateStep(tempAngles_, k3_, sharedData.deltaTime);
    tempAngles_ = angles_ + k3_;
    calculateStep(tempAngles_, k4_, sharedData.deltaTime);
    angles_ += (k1_ + 2 * k2_ + 2 * k3_ + k4_) / 6;
}

void Oseen::calculateStep(Eigen::MatrixXd& angles, Eigen::MatrixXd& result, double dt) {
    updateVelocities(angles);
    // Iterate across every angle
    for (size_t x = 0; x < width_; ++x) {
        for (size_t y = 0; y < height_; ++y) {
            result(x,y) = dt*f(x, y, angles);
        }
    }
}



double Oseen::f(double x, double y, Eigen::MatrixXd& angles) {
    double f_i = getForceMagnitude(x, y, angles);                       // Get the force magnitude fpr cilium at (x, y)
    double omega_i = f_i / (drag_coefficient_ * cilia_radius_);         // 
    Eigen::Vector3d t_i = getTangent(x, y, angles);
    Eigen::Vector3d velocity_at_xy = velocities_.row(x * height_ + y);
    double dot_product = t_i.dot(velocity_at_xy);
    return omega_i + dot_product / cilia_radius_;
}

Eigen::Vector3d Oseen::stokeslet(Eigen::Vector3d point, int x, int y, Eigen::MatrixXd& angles){
    Eigen::Vector3d f_2 = getForce(x, y, angles);
    Eigen::Vector3d r = point - getR(x, y, angles);
    double r_length = r.norm();
    return (1/(8*M_PI*mu_*r_length)) * (f_2 + (r.dot(f_2)/(r_length*r_length)) * r);
}

Eigen::Vector3d Oseen::stokeslet(int x_1, int y_1, int x_2, int y_2, Eigen::MatrixXd& angles) {
    Eigen::Vector3d f_2 = getForce(x_2, y_2, angles);
    Eigen::Vector3d r = getR(x_1, y_1, angles) - getR(x_2, y_2, angles);
    double r_length = r.norm();
    return (1/(8*M_PI*mu_*r_length)) * (f_2 + (r.dot(f_2)/(r_length*r_length)) * r);
}

Eigen::Vector3d Oseen::getForce(int x, int y, Eigen::MatrixXd& angles) {
    return getForceMagnitude(x, y, angles) * getTangent(x, y, angles);
}

double Oseen::getForceMagnitude(int x, int y, Eigen::MatrixXd& angles) {
    return sharedData.force + sharedData.force_amplitude*sin(angles(x,y));
}

Eigen::Vector3d Oseen::getR(int x, int y, Eigen::MatrixXd& angles) {
    Eigen::Vector3d pos_at_xy = positions_.row(x * height_ + y); // Assuming linear indexing for positions_
    return pos_at_xy + cilia_radius_ * Eigen::Vector3d(cos(angles(x, y)), sin(angles(x, y)), 0);
}

Eigen::Vector3d Oseen::getTangent(int x, int y, Eigen::MatrixXd& angles) {
    return Eigen::Vector3d(-sin(angles(x, y)), cos(angles(x, y)), 0);
}

double Oseen::getVelocityMagnitudeAtPoint(Eigen::Vector3d point, Eigen::MatrixXd& angles) {
    return getVelocityAtPoint(point, angles).norm();
}

Eigen::Vector3d Oseen::getVelocityAtPoint(Eigen::Vector3d point, Eigen::MatrixXd& angles) {
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    for (size_t x = 0; x < width_; ++x) {
        for (size_t y = 0; y < height_; ++y) {
            velocity += stokeslet(point, x, y, angles);
        }
    }
    return velocity;
}

Eigen::Vector3d Oseen::getVelocityAtPoint(Eigen::Vector3d point) {
    return getVelocityAtPoint(point, angles_);
}

// Function to calculate the Kuramoto order parameter
std::complex<double> Oseen::calculateOrderParameter() {
    double realPart = 0.0;
    double imagPart = 0.0;

    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < width_; x++) {
        for (size_t y = 0; y < height_; y++) {
            double theta = angles_(x,y);
            double cos_theta = std::cos(theta);
            double sin_theta = std::sin(theta);
            #pragma omp atomic
            realPart += cos_theta;
            #pragma omp atomic
            imagPart += sin_theta;
        }
    }

    std::complex<double> orderParameter(realPart, imagPart);
    return orderParameter / static_cast<double>(N_);
}

void Oseen::calcOmega() {
    
}