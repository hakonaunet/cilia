/* void Oseen::iteration() {
    unsigned int cilia_left_to_consider = N+1;
    // Reset velocities
    velocities = std::vector<std::vector<Eigen::Vector3d>>(width, std::vector<Eigen::Vector3d>(height, Eigen::Vector3d::Zero()));
    // #pragma omp parallel for collapse(2) // Parallelize both x and y loops
    for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < height; ++y) {
            cilia_left_to_consider--;
            calculateVelocity(x, y, cilia_left_to_consider);
        }
    }
    updateAngles();
}

// Get velocity of cilia at position (x, y)
void Oseen::calculateVelocity(int x, int y, int break_point) {
    int counter = 0;
    Eigen::Vector3d r_i = positions[x][y] + cilia_radius * Eigen::Vector3d(cos(angles[x][y]), sin(angles[x][y]), 0);
    double f_i_magnitude = getForce(x, y);
    Eigen::Vector3d f_i = Eigen::Vector3d(-f_i_magnitude * sin(angles[x][y]), f_i_magnitude * cos(angles[x][y]), 0);
    for (int i = width-1; i >= 0; i--) {
        for (int j = height-1; j >= 0; j--) {
            if (i == x && j == y) {
                continue;
            }
            if (counter >= break_point) {
                return;               
            }
            Eigen::Vector3d r_j = positions[i][j] + cilia_radius * Eigen::Vector3d(cos(angles[i][j]), sin(angles[i][j]), 0);
            Eigen::Vector3d r = r_i - r_j;
            double r_length = r.norm(); // Length of r
            double f_j_magnitude = getForce(i, j);
            Eigen::Vector3d f_j = Eigen::Vector3d(-f_j_magnitude * sin(angles[i][j]), f_j_magnitude * cos(angles[i][j]), 0);
            velocities[x][y] += (1/(8*M_PI*mu*r_length)) * (f_j + (r.dot(f_j)/r_length) * r);
            velocities[i][j] += (1/(8*M_PI*mu*r_length)) * (f_i + (r.dot(f_i)/r_length) * r);
            counter++;
        }
    }
} */