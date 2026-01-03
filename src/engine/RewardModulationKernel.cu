#include <engine/RewardModulationKernel.cuh>
#include <engine/GPUNeuralStructures.h>

// ============================================================================
// REWARD MODULATION KERNEL IMPLEMENTATIONS
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
                                           int num_neurons) {
    // Simple implementation for compatibility
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *predicted_reward = 0.0f;
        *prediction_error = external_reward - *predicted_reward;
        *dopamine_level = BASELINE_DOPAMINE + *prediction_error * 0.5f;
        
        // Clamp dopamine level
        if (*dopamine_level > 1.0f) *dopamine_level = 1.0f;
        if (*dopamine_level < 0.0f) *dopamine_level = 0.0f;
    }
}

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
                                      int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Modulate synaptic strength based on dopamine
    float dopamine_factor = dopamine_level / BASELINE_DOPAMINE;
    float modulation = (dopamine_factor - 1.0f) * 0.1f;
    
    synapse.weight += modulation * synapse.eligibility_trace * dt;
    
    // Bound weight
    if (synapse.weight > 1.0f) synapse.weight = 1.0f;
    if (synapse.weight < 0.01f) synapse.weight = 0.01f;
}

/**
 * Adapt dopamine sensitivity based on recent reward history
 */
__global__ void dopamineSensitivityAdaptationKernel(GPUSynapse* synapses,
                                                   GPUNeuronState* neurons,
                                                   float recent_reward_average,
                                                   float adaptation_rate,
                                                   float dt,
                                                   int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Adapt sensitivity to avoid saturation
    float target_sensitivity = 1.0f / (1.0f + fabsf(recent_reward_average));
    synapse.dopamine_sensitivity += (target_sensitivity - synapse.dopamine_sensitivity) * adaptation_rate * dt;
    
    // Bound sensitivity
    if (synapse.dopamine_sensitivity > 2.0f) synapse.dopamine_sensitivity = 2.0f;
    if (synapse.dopamine_sensitivity < 0.1f) synapse.dopamine_sensitivity = 0.1f;
}

/**
 * Update reward trace for temporal credit assignment
 */
__global__ void rewardTraceUpdateKernel(float* reward_trace, 
                                       float decay_factor,
                                       float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        *reward_trace *= expf(-decay_factor * dt);
    }
}
