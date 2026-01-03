#ifndef FUSED_KERNELS_CUH
#define FUSED_KERNELS_CUH

#include <engine/GPUNeuralStructures.h>

/**
 * @file FusedKernels.cuh
 * @brief Fused CUDA kernels for maximum performance
 * 
 * These kernels combine multiple operations that were previously separate
 * kernel launches, reducing overhead and improving data locality.
 * 
 * Performance Benefits:
 * - Single global memory load per neuron/synapse
 * - Eliminates kernel launch overhead (~10-20μs per launch)
 * - Better register usage and occupancy
 * - Improved cache utilization
 */

// ============================================================================
// FUSED NEURON UPDATE KERNEL
// ============================================================================

/**
 * @brief Fused kernel combining neuron update, calcium dynamics, and neuromodulation
 * 
 * Combines:
 * 1. Neuron state update (Izhikevich dynamics)
 * 2. Calcium diffusion
 * 3. Neuromodulator application
 * 
 * @param arrays Neuron data in SoA layout
 * @param current_time Current simulation time (ms)
 * @param dt Time step (ms)
 * @param dopamine_level Global dopamine concentration
 * @param serotonin_level Global serotonin concentration
 * @param num_neurons Total number of neurons
 */
__global__ void fusedNeuronUpdateKernel(
    NeuronArrays arrays,
    float current_time,
    float dt,
    float dopamine_level,
    float serotonin_level,
    int num_neurons
);

// ============================================================================
// FUSED PLASTICITY KERNEL
// ============================================================================

/**
 * @brief Fused kernel combining STDP, eligibility traces, and reward modulation
 * 
 * Combines:
 * 1. STDP weight updates (with sign-preserving eligibility)
 * 2. Eligibility trace updates
 * 3. Reward-modulated learning
 * 
 * @param synapse_arrays Synapse data in SoA layout
 * @param neuron_arrays Neuron data in SoA layout
 * @param reward_signal Reward prediction error
 * @param current_time Current simulation time (ms)
 * @param dt Time step (ms)
 * @param num_synapses Total number of synapses
 */
__global__ void fusedPlasticityKernel(
    SynapseArrays synapse_arrays,
    NeuronArrays neuron_arrays,
    float reward_signal,
    float current_time,
    float dt,
    int num_synapses
);

#endif // FUSED_KERNELS_CUH

