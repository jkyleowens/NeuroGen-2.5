#ifndef ENHANCED_STDP_KERNEL_CUH
#define ENHANCED_STDP_KERNEL_CUH

// This header now correctly includes the necessary data structures.
#include "GPUNeuralStructures.h"

/**
 * @brief Main kernel for multi-factor, biologically-inspired synaptic plasticity.
 *
 * This kernel calculates the potential for synaptic change (LTP or LTD) based on
 * several factors: precise spike timing, local calcium concentration, and the
 * current state of the synapse.
 */
__global__ void enhancedSTDPKernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses
);

/**
 * @brief Overloaded STDP kernel for backward compatibility
 *
 * This overload provides compatibility with existing code that expects
 * a different parameter set including learning_rate.
 */
__global__ void enhancedSTDPKernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float learning_rate,
    float dt,
    int num_synapses
);

#endif // ENHANCED_STDP_KERNEL_CUH