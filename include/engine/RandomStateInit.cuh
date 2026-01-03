#ifndef RANDOM_STATE_INIT_CUH
#define RANDOM_STATE_INIT_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * CUDA kernel to initialize random states for CURAND
 * @param states Array of curandState to initialize
 * @param n Number of states to initialize
 * @param seed Random seed value
 */
__global__ void initializeRandomStates(curandState* states, int n, unsigned long seed);

/**
 * Host function to launch the random state initialization kernel
 * @param d_states Device pointer to curandState array
 * @param num_states Number of states to initialize
 * @param seed Random seed value
 */
void launchRandomStateInit(curandState* d_states, int num_states, unsigned long seed);

#endif // RANDOM_STATE_INIT_CUH