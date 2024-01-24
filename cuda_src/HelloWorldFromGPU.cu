// cudaParallelization.cu

#include <stdio.h>
#include <iostream>

__global__ void helloFromGPU (void) {
    printf("Hello World from block %d, thread %d!\n", blockIdx.x, threadIdx.x);
}

int runHelloFromGPU(void) {
    helloFromGPU<<<2, 2>>>();
    cudaDeviceSynchronize();

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}

#ifdef RUN_HELLO_FROM_GPU
int main() {
    printf("Hello World from CPU!\n");
    runHelloFromGPU();
    return 0;
}
#endif