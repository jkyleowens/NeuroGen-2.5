#include <engine/RandomStateInit.cuh>
#include <engine/GridBlockUtils.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void initializeRandomStates(curandState* states, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Each thread gets different seed, same sequence
        curand_init(seed + idx, 0, 0, &states[idx]);
    }
}

void launchRandomStateInit(curandState* d_states, int num_states, unsigned long seed) {
    dim3 block = makeBlock();
    dim3 grid = makeGrid(num_states);
    initializeRandomStates<<<grid, block>>>(d_states, num_states, seed);
    cudaDeviceSynchronize();
}