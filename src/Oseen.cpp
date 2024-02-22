// Oseen.cpp

#include "Oseen.hpp"

// Initialize the system
Oseen::Oseen(SharedDataOseen& data) : sharedData(data), positions_(data.width * data.height, 3), velocities_(data.width * data.height, 3),
    intrinsicFrequencies_(data.width, data.height), angles_(data.width, data.height), tempAngles_(data.width, data.height), 
    k1_(data.width, data.height), k2_(data.width, data.height), k3_(data.width, data.height), k4_(data.width, data.height) {
        
    data.startTime = std::chrono::high_resolution_clock::now();

    params.width = data.width;
    params.height = data.height;
    params.N = params.width * params.height;
    params.z = 0;
    params.force = data.force;
    params.mu = 1;
    params.cilia_radius = data.cilia_radius;
    params.fluid_viscosity = 1;
    params.drag_coefficient = 6*M_PI*params.fluid_viscosity*0.05; // Change later!
    params.grid_spacing = data.gridSpacing;

    // Parse through the positions_ and set the initial values
    for (size_t i = 0; i < params.width; ++i) {
        for (size_t j = 0; j < params.height; ++j) {
            size_t index = i * params.height + j;
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
    initializeCUDA();
}

Oseen::~Oseen() {
    //freeCUDA();
}

Eigen::Vector3d Oseen::initializePosition(int x, int y) {
    // Calculate the grid point
    double gridX = x * params.grid_spacing;
    double gridY = y * params.grid_spacing;

    switch (sharedData.noiseMode) {
        case NoiseMode::None: {
            // Return the grid position without any noise
            return Eigen::Vector3d(gridX, gridY, params.z);
        }
        case NoiseMode::Square: {
            // Generate a random position within a square centered at the grid point
            double noiseX = (std::rand() / (double)RAND_MAX - 0.5) * sharedData.noiseWidth * 2;
            double noiseY = (std::rand() / (double)RAND_MAX - 0.5) * sharedData.noiseWidth * 2;
            return Eigen::Vector3d(gridX + noiseX, gridY + noiseY, params.z);
        }
        case NoiseMode::Circular: {
            // Generate a random angle and a random radius
            double angle = (std::rand() / (double)RAND_MAX) * 2 * M_PI;
            double radius = std::sqrt(std::rand() / (double)RAND_MAX) * sharedData.noiseWidth;

            // Convert polar coordinates to Cartesian coordinates
            double noiseX = radius * std::cos(angle);
            double noiseY = radius * std::sin(angle);

            // Return the final position
            return Eigen::Vector3d(gridX + noiseX, gridY + noiseY, params.z);
        }
        default: {
            // Handle other cases if necessary
            return Eigen::Vector3d(gridX, gridY, params.z);
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


void Oseen::checkCudaError(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to perform %s (error code %s)!\n", operation, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void Oseen::initializeCUDA() {
    // Allocate memory on the GPU
    checkCudaError(cudaMalloc(&d_angles, params.N * sizeof(double)), "cudaMalloc d_angles");
    checkCudaError(cudaMalloc(&d_pos_x, params.N * sizeof(double)), "cudaMalloc d_pos_x");
    checkCudaError(cudaMalloc(&d_pos_y, params.N * sizeof(double)), "cudaMalloc d_pos_y");
    checkCudaError(cudaMalloc(&d_velocities_x, params.N * sizeof(double)), "cudaMalloc d_velocities_x");
    checkCudaError(cudaMalloc(&d_velocities_y, params.N * sizeof(double)), "cudaMalloc d_velocities_y");
    std::cout << "Allocating d_params" << std::endl;
    checkCudaError(cudaMalloc(&d_params, sizeof(OseenParameters)), "cudaMalloc d_params");


    // Create temporary vectors to hold the positions
    std::vector<double> pos_x(params.N);
    std::vector<double> pos_y(params.N);

    // Copy data from positions_ to the temporary vectors
    for (unsigned int i = 0; i < params.N; ++i) {
        pos_x[i] = positions_(i, 0);
        pos_y[i] = positions_(i, 1);
    }

    // Copy initial data to the GPU
    checkCudaError(cudaMemcpy(d_angles, angles_.data(), params.N * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_angles");
    checkCudaError(cudaMemcpy(d_pos_x, pos_x.data(), params.N * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_pos_x");
    checkCudaError(cudaMemcpy(d_pos_y, pos_y.data(), params.N * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_pos_y");
    checkCudaError(cudaMemcpy(d_velocities_x, velocities_.data(), params.N * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_velocities_x");
    checkCudaError(cudaMemcpy(d_velocities_y, velocities_.data(), params.N * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_velocities_y");
    std::cout << "Copying d_params to device" << std::endl;
    checkCudaError(cudaMemcpy(d_params, &params, sizeof(OseenParameters), cudaMemcpyHostToDevice), "cudaMemcpy d_params");
}

void Oseen::updateCUDA() {
    // Update the parameters from sharedData
    params.deltaTime = sharedData.deltaTime;
    params.force = sharedData.force;
    params.force_amplitude = sharedData.force_amplitude;

    // Copy the updated parameters to the device
    checkCudaError(cudaMemcpy(d_params, &params, sizeof(OseenParameters), cudaMemcpyHostToDevice), "cudaMemcpy d_params");
}

void Oseen::freeCUDA() {
    // Free memory on the GPU
    checkCudaError(cudaFree(d_angles), "cudaFree d_angles");
    checkCudaError(cudaFree(d_pos_x), "cudaFree d_pos_x");
    checkCudaError(cudaFree(d_pos_y), "cudaFree d_pos_y");
    checkCudaError(cudaFree(d_velocities_x), "cudaFree d_velocities_x");
    checkCudaError(cudaFree(d_velocities_y), "cudaFree d_velocities_y");
    checkCudaError(cudaFree(d_params), "cudaFree d_params");
}

void Oseen::CUDAupdateVelocities(Eigen::MatrixXd& angles) {
    // Copy angles to the GPU
    checkCudaError(cudaMemcpy(d_angles, angles.data(), params.N * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy angles");

    // Launch the kernel
    launchCalculateVelocityKernel(d_pos_x, d_pos_y, d_angles, d_velocities_x, d_velocities_y, d_params, params.N);
    
    // Check for errors in the kernel launch
    checkCudaError(cudaGetLastError(), "Launching calculateVelocityKernel");

    // Wait for the GPU to finish and check for errors
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing after kernel launch");

    // Create temporary arrays to hold the velocities
    std::vector<double> velocities_x(params.N);
    std::vector<double> velocities_y(params.N);

    // Copy velocities back to the CPU
    checkCudaError(cudaMemcpy(velocities_x.data(), d_velocities_x, params.N * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy d_velocities_x");
    checkCudaError(cudaMemcpy(velocities_y.data(), d_velocities_y, params.N * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy d_velocities_y");

    // Update the 'velocities_' member variable
    for (size_t i = 0; i < params.N; ++i) {
        velocities_(i, 0) = velocities_x[i];
        velocities_(i, 1) = velocities_y[i];
        velocities_(i, 2) = 0.0;  // Assuming the z component is zero
    }
}

void Oseen::CUDArungeKutta4() {
    CUDAcalculateStep(angles_, k1_, sharedData.deltaTime);
    tempAngles_ = angles_ + 0.5 * k1_;
    normalizeAngles(tempAngles_);
    CUDAcalculateStep(tempAngles_, k2_, sharedData.deltaTime);
    tempAngles_ = angles_ + 0.5 * k2_;
    normalizeAngles(tempAngles_);
    CUDAcalculateStep(tempAngles_, k3_, sharedData.deltaTime);
    tempAngles_ = angles_ + k3_;
    normalizeAngles(tempAngles_);
    CUDAcalculateStep(tempAngles_, k4_, sharedData.deltaTime);
    angles_ += (k1_ + 2*k2_ + 2*k3_ + k4_) / 6;
    normalizeAngles(angles_);
    std::cout << "Velocities (5,5): " << velocities_.row(5 * params.height + 5) << std::endl;
}

void Oseen::CUDAcalculateStep(Eigen::MatrixXd& angles, Eigen::MatrixXd& result, double dt) {
    CUDAupdateVelocities(angles);
    // Iterate across every cilium
    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < params.width; ++x) {
        for (size_t y = 0; y < params.height; ++y) {
            result(x,y) = dt*f(x, y, angles);
        }
    }
}

void Oseen::iteration() {
    sharedData.simulationTime += sharedData.deltaTime; // Update simulation time
    sharedData.iteration++; // Update frame count
    auto startTime = std::chrono::high_resolution_clock::now(); // Start measuring time

    CUDArungeKutta4();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    sharedData.updateTime = static_cast<double>(duration.count());
}

void Oseen::updateVelocities(Eigen::MatrixXd& angles) {
    velocities_.setZero();
    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < params.width; ++x) {
        for (size_t y = 0; y < params.height; ++y) {
            calculateVelocity(x, y, angles);
        }
    }
}

// Get velocity of cilia at position (x, y)
void Oseen::calculateVelocity(size_t x, size_t y, Eigen::MatrixXd& angles) {
    Eigen::Vector3d r_i = getR(x, y, angles);
    for (size_t i = 0; i < params.width; ++i) {
        for (size_t j = 0; j < params.height; ++j) {
            if (i == x && j == y) continue;
            // Directly add the result of stokeslet to the corresponding row
            velocities_.block<1, 3>(x * params.height + y, 0) += stokeslet(r_i, i, j, angles);
        }
    }
}

void Oseen::updateAngles() {
    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < params.width; ++x) {
        for (size_t y = 0; y < params.height; ++y) {
            double f_i = getForceMagnitude(x, y, angles_);
            Eigen::Vector3d t_i = Eigen::Vector3d(-sin(angles_(x, y)), cos(angles_(x, y)), 0);
            double omega_i = f_i / (params.drag_coefficient * params.cilia_radius);
            Eigen::Vector3d velocity_at_xy = velocities_.row(x * params.height + y); // Assuming linear indexing
            double dot_product = t_i.dot(velocity_at_xy);

            // Euler's method
            angles_(x, y) += sharedData.deltaTime * (omega_i + dot_product / params.cilia_radius);

            // Ensure the angle is between 0 and 2π
            while (angles_(x, y) < 0) angles_(x, y) += 2 * M_PI;
            while (angles_(x, y) >= 2 * M_PI) angles_(x, y) -= 2 * M_PI;
        }
    }
}

void Oseen::normalizeAngles(Eigen::MatrixXd& angles) {
    angles.array() = angles.array().unaryExpr([](double val) { 
        val = std::fmod(val, 2 * M_PI);
        return val < 0 ? val + 2 * M_PI : val; 
    });
}

void Oseen::rungeKutta4() {
    calculateStep(angles_, k1_, sharedData.deltaTime);
    tempAngles_ = angles_ + 0.5 * k1_;
    normalizeAngles(tempAngles_);
    calculateStep(tempAngles_, k2_, sharedData.deltaTime);
    tempAngles_ = angles_ + 0.5 * k2_;
    normalizeAngles(tempAngles_);
    calculateStep(tempAngles_, k3_, sharedData.deltaTime);
    tempAngles_ = angles_ + k3_;
    normalizeAngles(tempAngles_);
    calculateStep(tempAngles_, k4_, sharedData.deltaTime);
    angles_ += (k1_ + 2*k2_ + 2*k3_ + k4_) / 6;
    normalizeAngles(angles_);
}

void Oseen::calculateStep(Eigen::MatrixXd& angles, Eigen::MatrixXd& result, double dt) {
    updateVelocities(angles);
    // Iterate across every cilium
    #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < params.width; ++x) {
        for (size_t y = 0; y < params.height; ++y) {
            result(x,y) = dt*f(x, y, angles);
        }
    }
}

double Oseen::f(double x, double y, Eigen::MatrixXd& angles) {
    double f_i = getForceMagnitude(x, y, angles);                       // Get the force magnitude fpr cilium at (x, y)
    double omega_i = f_i / (params.drag_coefficient * params.cilia_radius);         // 
    Eigen::Vector3d t_i = getTangent(x, y, angles);
    Eigen::Vector3d velocity_at_xy = velocities_.row(x * params.height + y);
    double dot_product = t_i.dot(velocity_at_xy);
    return omega_i + dot_product / params.cilia_radius;
}

Eigen::Vector3d Oseen::stokeslet(Eigen::Vector3d point, int x, int y, Eigen::MatrixXd& angles){
    Eigen::Vector3d f_2 = getForce(x, y, angles);
    Eigen::Vector3d r = point - getR(x, y, angles);
    double r_length = r.norm();
    return (1/(8*M_PI*params.mu*r_length)) * (f_2 + (r.dot(f_2)/(r_length*r_length)) * r);
}

Eigen::Vector3d Oseen::stokeslet(int x_1, int y_1, int x_2, int y_2, Eigen::MatrixXd& angles) {
    Eigen::Vector3d f_2 = getForce(x_2, y_2, angles);
    Eigen::Vector3d r = getR(x_1, y_1, angles) - getR(x_2, y_2, angles);
    double r_length = r.norm();
    return (1/(8*M_PI*params.mu*r_length)) * (f_2 + (r.dot(f_2)/(r_length*r_length)) * r);
}

Eigen::Vector3d Oseen::getForce(int x, int y, Eigen::MatrixXd& angles) {
    return getForceMagnitude(x, y, angles) * getTangent(x, y, angles);
}

double Oseen::getForceMagnitude(int x, int y, Eigen::MatrixXd& angles) {
    return sharedData.force + sharedData.force_amplitude*sin(angles(x,y))+sharedData.force_amplitude*cos(angles(x,y))+sharedData.force_amplitude*sin(2*angles(x,y))+sharedData.force_amplitude*cos(2*angles(x,y));
}

Eigen::Vector3d Oseen::getR(int x, int y, Eigen::MatrixXd& angles) {
    Eigen::Vector3d pos_at_xy = positions_.row(x * params.height + y); // Assuming linear indexing for positions_
    return pos_at_xy + params.cilia_radius * Eigen::Vector3d(cos(angles(x, y)), sin(angles(x, y)), 0);
}

Eigen::Vector3d Oseen::getTangent(int x, int y, Eigen::MatrixXd& angles) {
    return Eigen::Vector3d(-sin(angles(x, y)), cos(angles(x, y)), 0);
}

double Oseen::getVelocityMagnitudeAtPoint(Eigen::Vector3d point, Eigen::MatrixXd& angles) {
    return getVelocityAtPoint(point, angles).norm();
}

Eigen::Vector3d Oseen::getVelocityAtPoint(Eigen::Vector3d point, Eigen::MatrixXd& angles) {
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    for (size_t x = 0; x < params.width; ++x) {
        for (size_t y = 0; y < params.height; ++y) {
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
    for (size_t x = 0; x < params.width; x++) {
        for (size_t y = 0; y < params.height; y++) {
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
    return orderParameter / static_cast<double>(params.N);
}

void Oseen::calcOmega() {
    
}