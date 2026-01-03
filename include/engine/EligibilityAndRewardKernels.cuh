#ifndef ELIGIBILITY_AND_REWARD_KERNELS_CUH
#define ELIGIBILITY_AND_REWARD_KERNELS_CUH

// >>> FIX: Added the missing include for the GPU data structures.
// This defines GPUSynapse and GPUNeuronState, resolving the "undefined identifier" errors.
#include "GPUNeuralStructures.h"
// <<< END FIX

/**
 * @brief Applies reward (dopamine) signal to consolidate synaptic changes.
 */
__global__ void applyRewardKernel(
    GPUSynapse* synapses,
    float reward,
    float dt,
    int num_synapses
);

/**
 * @brief Adapts the sensitivity of synapses to neuromodulators like dopamine.
 */
__global__ void adaptNeuromodulationKernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_dopamine,
    int num_synapses,
    float current_time // Pass current_time as a parameter
);

/**
 * @brief Update eligibility traces for reinforcement learning
 */
__global__ void eligibilityTraceUpdateKernel(GPUSynapse* synapses,
                                            const GPUNeuronState* neurons,
                                            float current_time,
                                            float dt,
                                            int num_synapses);

/**
 * @brief Reset eligibility traces
 */
__global__ void eligibilityTraceResetKernel(GPUSynapse* synapses,
                                           int num_synapses,
                                           bool reset_fast,
                                           bool reset_slow,
                                           bool reset_ultraslow);

/**
 * @brief Monitor trace statistics
 */
__global__ void traceMonitoringKernel(const GPUSynapse* synapses,
                                     int num_synapses,
                                     float* trace_stats);

/**
 * @brief Update metaplasticity mechanisms
 */
__global__ void updateMetaplasticityKernel(GPUSynapse* synapses,
                                          const GPUNeuronState* neurons,
                                          float current_time,
                                          float dt,
                                          int num_synapses);

/**
 * @brief Late-phase plasticity kernel
 */
__global__ void latePhaseePlasticityKernel(GPUSynapse* synapses,
                                          const GPUNeuronState* neurons,
                                          float protein_synthesis_signal,
                                          float current_time,
                                          float dt,
                                          int num_synapses);

/**
 * @brief Emergency network stabilization
 */
__global__ void emergencyStabilizationKernel(GPUSynapse* synapses,
                                            GPUNeuronState* neurons,
                                            float network_activity,
                                            float emergency_threshold,
                                            int num_synapses,
                                            int num_neurons);

#endif // ELIGIBILITY_AND_REWARD_KERNELS_CUH