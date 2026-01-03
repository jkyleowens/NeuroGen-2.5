#ifndef AGENTIC_NEURON_KERNELS_CUH
#define AGENTIC_NEURON_KERNELS_CUH

#include "AgenticNeuron.h"

namespace neurogen {
namespace engine {

// ============================================================================
// KERNEL DECLARATIONS
// ============================================================================

/**
 * @brief Global kernel to update all agentic neurons
 */
__global__ void update_agentic_neurons_kernel(
    int num_neurons,
    float t,
    float dt,
    NeuronParams params,
    NeuronState state
);

/**
 * @brief Kernel to initialize neuron state to resting values
 */
__global__ void init_agentic_neurons_kernel(
    int num_neurons,
    NeuronParams params,
    NeuronState state
);

// ============================================================================
// HOST LAUNCH WRAPPERS
// ============================================================================

/**
 * @brief Launch the neuron update kernel
 */
void launch_agentic_neuron_update(
    int num_neurons,
    float t,
    float dt,
    const NeuronParams& params,
    NeuronState& state,
    cudaStream_t stream = 0
);

/**
 * @brief Launch the neuron initialization kernel
 */
void launch_agentic_neuron_init(
    int num_neurons,
    const NeuronParams& params,
    NeuronState& state,
    cudaStream_t stream = 0
);

} // namespace engine
} // namespace neurogen

#endif // AGENTIC_NEURON_KERNELS_CUH
