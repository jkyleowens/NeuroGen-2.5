#include "engine/AgenticNeuron.h"
#include "engine/AgenticNeuronKernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>

namespace neurogen {
namespace engine {

// ============================================================================
// KERNEL DEFINITIONS
// ============================================================================

__global__ void update_agentic_neurons_kernel(
    int num_neurons,
    float t,
    float dt,
    NeuronParams params,
    NeuronState state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_neurons) {
        AgenticNeuron::update_tick(idx, t, dt, params, state);
    }
}

__global__ void init_agentic_neurons_kernel(
    int num_neurons,
    NeuronParams params,
    NeuronState state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_neurons) {
        state.v[idx] = params.v_rest;
        state.a[idx] = 0.0f;
        state.v_thresh[idx] = params.v_thresh_base;
        state.last_spike[idx] = -1000.0f; // Long time ago
        state.output[idx] = 0.0f;
        state.trace[idx] = 0.0f;
        state.i_syn[idx] = 0.0f;
        state.i_stim[idx] = 0.0f;
    }
}

// ============================================================================
// HOST WRAPPERS
// ============================================================================

void launch_agentic_neuron_update(
    int num_neurons,
    float t,
    float dt,
    const NeuronParams& params,
    NeuronState& state,
    cudaStream_t stream
) {
    int blockSize = 256;
    int numBlocks = (num_neurons + blockSize - 1) / blockSize;

    update_agentic_neurons_kernel<<<numBlocks, blockSize, 0, stream>>>(
        num_neurons, t, dt, params, state
    );
    
    // Check for launch errors (optional, can be disabled in release)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void launch_agentic_neuron_init(
    int num_neurons,
    const NeuronParams& params,
    NeuronState& state,
    cudaStream_t stream
) {
    int blockSize = 256;
    int numBlocks = (num_neurons + blockSize - 1) / blockSize;

    init_agentic_neurons_kernel<<<numBlocks, blockSize, 0, stream>>>(
        num_neurons, params, state
    );
}

} // namespace engine
} // namespace neurogen
