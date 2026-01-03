#ifndef GRID_BLOCK_UTILS_CUH
#define GRID_BLOCK_UTILS_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>

/**
 * Creates a standard CUDA block dimension for kernel launches
 * @return dim3 block dimensions (typically 256 threads per block)
 */
inline dim3 makeBlock() {
    return dim3(256);
}

/**
 * Creates a CUDA grid dimension based on the number of elements to process
 * @param n Number of elements to process
 * @return dim3 grid dimensions with enough blocks to cover all elements
 */
inline dim3 makeGrid(int n) {
    return dim3((n + 255) / 256);
}
#endif // __CUDACC__

#endif // GRID_BLOCK_UTILS_CUH