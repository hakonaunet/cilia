// updateGrid.hpp

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Oscillator.hpp>
#include <SharedDataKuramoto.hpp>
#include <Grid.hpp>

void cudaUpdateGrid(Oscillator* grid, size_t gridWidth, size_t gridHeight, SharedDataKuramoto SharedDataKuramoto, double K, double dt);
__global__ void updateGridKernel(Oscillator* grid, size_t gridWidth, size_t gridHeight, SharedDataKuramoto SharedDataKuramoto, double K, double dt);
__device__ double kuramotoSumCUDA(int x, int y, Oscillator* grid, size_t gridWidth, size_t gridHeight);

__device__ double getAngle(Oscillator* osc);
__device__ double getIntrinsicFrequency(Oscillator* osc);
__device__ void setAngle(Oscillator* osc, double angle);