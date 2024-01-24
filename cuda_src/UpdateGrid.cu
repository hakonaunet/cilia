#include "UpdateGrid.cuh"

void cudaUpdateGrid(Oscillator* grid, size_t gridWidth, size_t gridHeight, SharedData sharedData, double K, double dt) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((gridWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (gridHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);
    updateGridKernel<<<numBlocks, threadsPerBlock>>>(grid, gridWidth, gridHeight, sharedData, K, dt);
}

__global__ void updateGridKernel(Oscillator* grid, size_t gridWidth, size_t gridHeight, SharedData sharedData, double K, double dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < gridWidth && y < gridHeight) {
        Oscillator& osc = grid[x * gridHeight + y];
        double theta = getAngle(&osc);
        double omega = getIntrinsicFrequency(&osc);

        // RK4 integration
        double k1 = dt * (omega + K * kuramotoSumCUDA(x, y, grid, gridWidth, gridHeight));
        double k2 = dt * (omega + K * kuramotoSumCUDA(x, y, grid, gridWidth, gridHeight) + 0.5 * k1);
        double k3 = dt * (omega + K * kuramotoSumCUDA(x, y, grid, gridWidth, gridHeight) + 0.5 * k2);
        double k4 = dt * (omega + K * kuramotoSumCUDA(x, y, grid, gridWidth, gridHeight) + k3);

        setAngle(&osc, theta + (k1 + 2 * k2 + 2 * k3 + k4) / 6);
    }
}

__device__ double kuramotoSumCUDA(int x, int y, Oscillator* grid, size_t gridWidth, size_t gridHeight) {
    double sum = 0.0;
    int N = gridWidth * gridHeight; // Total number of oscillators

    for (int i = 0; i < gridWidth; ++i) {
        for (int j = 0; j < gridHeight; ++j) {
            Oscillator& osc = grid[i * gridHeight + j];
            sum += sin(getAngle(&osc) - getAngle(&grid[x * gridHeight + y]));
        }
    }

    return sum / N;
}

__device__ double getAngle(Oscillator* osc) {
    return osc->angle;
}

__device__ double getIntrinsicFrequency(Oscillator* osc) {
    return osc->intrinsicFrequency;
}

__device__ void setAngle(Oscillator* osc, double angle) {
    osc->angle = angle;
    // Ensure the angle is between [0, 2*PI]
    while(osc->angle > TWO_PI) osc->angle -= TWO_PI;
    while(osc->angle < 0) osc->angle += TWO_PI;
}