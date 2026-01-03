#ifndef REWARD_MODULATION_KERNEL_CUH
#define REWARD_MODULATION_KERNEL_CUH

#include "GPUNeuralStructures.h"

// ============================================================================
// REWARD SYSTEM CONSTANTS
// ============================================================================
constexpr float BASELINE_DOPAMINE = 0.3f;

// ============================================================================
// REWARD MODULATION KERNEL DECLARATIONS
// ============================================================================

/**
 * Compute reward prediction error and dopamine release
 */
__global__ void rewardPredictionErrorKernel(GPUNeuronState* neurons,
                                           float external_reward,
                                           float* predicted_reward,
                                           float* prediction_error,
                                           float* dopamine_level,
                                           float current_time,
                                           float dt,
                                           int num_neurons);

/**
 * Apply reward modulation to synaptic plasticity
 */
__global__ void rewardModulationKernel(GPUSynapse* synapses,
                                      GPUNeuronState* neurons,
                                      float external_reward,
                                      float dopamine_level,
                                      float prediction_error,
                                      float current_time,
                                      float dt,
                                      int num_synapses);

/**
 * Adapt dopamine sensitivity based on recent reward history
 */
__global__ void dopamineSensitivityAdaptationKernel(GPUSynapse* synapses,
                                                   GPUNeuronState* neurons,
                                                   float recent_reward_average,
                                                   float adaptation_rate,
                                                   float dt,
                                                   int num_synapses);

/**
 * Update reward trace for temporal credit assignment
 */
__global__ void rewardTraceUpdateKernel(float* reward_trace, 
                                       float decay_factor,
                                       float dt);

#endif // REWARD_MODULATION_KERNEL_CUH
