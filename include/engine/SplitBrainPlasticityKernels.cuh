#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "engine/GPUNeuralStructures.h"

namespace neurogen {
namespace kernels {

/**
 * @brief Learning mode enum (device-side copy)
 */
enum class DeviceLearningMode : int {
    PURE_STDP = 0,
    REWARD_MODULATED_STDP = 1,
    MIXED_STDP = 2
};

/**
 * @brief Parameters for split-brain plasticity
 */
struct SplitBrainPlasticityParams {
    DeviceLearningMode mode;
    float stdp_learning_rate;
    float stdp_tau_positive;
    float stdp_tau_negative;
    float stdp_a_positive;
    float stdp_a_negative;
    float reward_sensitivity;
    float eligibility_decay;
    float unsupervised_weight;
    float supervised_weight;
};

/**
 * @brief Unified plasticity kernel supporting all learning modes
 * 
 * This kernel implements the "split-brain" learning strategy:
 * - Pure STDP: Only spike timing matters (sensory/memory regions)
 * - R-STDP: Eligibility × reward (action/control regions)
 * - Mixed: Blend of both (executive regions)
 */
__global__ void splitBrainPlasticityKernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float* eligibility_traces,
    const SplitBrainPlasticityParams params,
    float reward_signal,
    float current_time,
    float dt,
    int num_synapses
);

/**
 * @brief Update eligibility traces (for reward-modulated regions)
 */
__global__ void updateEligibilityTracesKernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float* eligibility_traces,
    const SplitBrainPlasticityParams params,
    float current_time,
    float dt,
    int num_synapses
);

/**
 * @brief Apply pure STDP update (for sensory/memory regions)
 * 
 * This kernel COMPLETELY IGNORES the reward signal.
 * Weight changes are based ONLY on spike timing correlations.
 */
__global__ void pureSTDPKernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    const SplitBrainPlasticityParams params,
    float current_time,
    float dt,
    int num_synapses
);

/**
 * @brief Apply reward-modulated plasticity (for action/control regions)
 * 
 * Implements the three-factor rule: ΔW = η × eligibility × (reward - baseline)
 */
__global__ void rewardModulatedPlasticityKernel(
    GPUSynapse* synapses,
    float* eligibility_traces,
    const SplitBrainPlasticityParams params,
    float reward_prediction_error,
    float dt,
    int num_synapses
);

// ============================================================================
// KERNEL IMPLEMENTATIONS
// ============================================================================

__global__ void pureSTDPKernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    const SplitBrainPlasticityParams params,
    float current_time,
    float dt,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    if (!synapse.is_active) return;
    
    // Get pre and post neurons
    const GPUNeuronState& pre = neurons[synapse.pre_neuron_id];
    const GPUNeuronState& post = neurons[synapse.post_neuron_id];
    
    // Calculate spike time difference
    float delta_t = post.last_spike_time - pre.last_spike_time;
    
    // STDP window (typically 20-50ms)
    if (fabsf(delta_t) > 50.0f) {
        return;  // Outside STDP window
    }
    
    float weight_change = 0.0f;
    
    if (delta_t > 0.0f) {
        // Pre before post → LTP (potentiation)
        weight_change = params.stdp_a_positive * expf(-delta_t / params.stdp_tau_positive);
    } else {
        // Post before pre → LTD (depression)
        weight_change = -params.stdp_a_negative * expf(delta_t / params.stdp_tau_negative);
    }
    
    // Apply weight change (IGNORING REWARD)
    synapse.weight += params.stdp_learning_rate * weight_change * dt;
    
    // Clamp weights
    synapse.weight = fmaxf(0.0f, fminf(2.0f, synapse.weight));
}

__global__ void updateEligibilityTracesKernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float* eligibility_traces,
    const SplitBrainPlasticityParams params,
    float current_time,
    float dt,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    if (!synapse.is_active) return;
    
    // Decay existing trace
    eligibility_traces[idx] *= expf(-dt / (params.eligibility_decay * 1000.0f));
    
    // Get pre and post neurons
    const GPUNeuronState& pre = neurons[synapse.pre_neuron_id];
    const GPUNeuronState& post = neurons[synapse.post_neuron_id];
    
    // Calculate STDP contribution
    float delta_t = post.last_spike_time - pre.last_spike_time;
    
    if (fabsf(delta_t) < 50.0f) {  // Within STDP window
        float stdp_contribution = 0.0f;
        
        if (delta_t > 0.0f) {
            stdp_contribution = params.stdp_a_positive * expf(-delta_t / params.stdp_tau_positive);
        } else {
            stdp_contribution = -params.stdp_a_negative * expf(delta_t / params.stdp_tau_negative);
        }
        
        // Add to eligibility trace
        eligibility_traces[idx] += stdp_contribution;
    }
    
    // Clamp eligibility
    eligibility_traces[idx] = fmaxf(-1.0f, fminf(1.0f, eligibility_traces[idx]));
}

__global__ void rewardModulatedPlasticityKernel(
    GPUSynapse* synapses,
    float* eligibility_traces,
    const SplitBrainPlasticityParams params,
    float reward_prediction_error,
    float dt,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    if (!synapse.is_active) return;
    
    // Three-factor learning rule: ΔW = η × eligibility × RPE
    float weight_change = params.stdp_learning_rate * 
                         eligibility_traces[idx] * 
                         reward_prediction_error * 
                         params.reward_sensitivity;
    
    // Apply weight update
    synapse.weight += weight_change * dt;
    
    // Clamp weights
    synapse.weight = fmaxf(0.0f, fminf(2.0f, synapse.weight));
    
    // Decay eligibility after learning
    eligibility_traces[idx] *= 0.9f;
}

__global__ void splitBrainPlasticityKernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float* eligibility_traces,
    const SplitBrainPlasticityParams params,
    float reward_signal,
    float current_time,
    float dt,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    if (!synapse.is_active) return;
    
    // Get pre and post neurons
    const GPUNeuronState& pre = neurons[synapse.pre_neuron_id];
    const GPUNeuronState& post = neurons[synapse.post_neuron_id];
    
    // Calculate spike time difference
    float delta_t = post.last_spike_time - pre.last_spike_time;
    
    // Calculate STDP contribution
    float stdp_contribution = 0.0f;
    if (fabsf(delta_t) < 50.0f) {
        if (delta_t > 0.0f) {
            stdp_contribution = params.stdp_a_positive * expf(-delta_t / params.stdp_tau_positive);
        } else {
            stdp_contribution = -params.stdp_a_negative * expf(delta_t / params.stdp_tau_negative);
        }
    }
    
    float weight_change = 0.0f;
    
    // Switch based on learning mode
    switch (params.mode) {
        case DeviceLearningMode::PURE_STDP:
            // Pure STDP: IGNORE reward completely
            weight_change = params.stdp_learning_rate * stdp_contribution;
            break;
            
        case DeviceLearningMode::REWARD_MODULATED_STDP:
            // Update eligibility trace
            eligibility_traces[idx] *= expf(-dt / (params.eligibility_decay * 1000.0f));
            eligibility_traces[idx] += stdp_contribution;
            eligibility_traces[idx] = fmaxf(-1.0f, fminf(1.0f, eligibility_traces[idx]));
            
            // Three-factor rule: η × eligibility × reward
            weight_change = params.stdp_learning_rate * 
                           eligibility_traces[idx] * 
                           reward_signal * 
                           params.reward_sensitivity;
            break;
            
        case DeviceLearningMode::MIXED_STDP:
            // Mixed: blend unsupervised and supervised
            float unsupervised_term = params.unsupervised_weight * stdp_contribution;
            
            // Update eligibility for supervised component
            eligibility_traces[idx] *= expf(-dt / (params.eligibility_decay * 1000.0f));
            eligibility_traces[idx] += stdp_contribution;
            eligibility_traces[idx] = fmaxf(-1.0f, fminf(1.0f, eligibility_traces[idx]));
            
            float supervised_term = params.supervised_weight * 
                                   eligibility_traces[idx] * 
                                   reward_signal * 
                                   params.reward_sensitivity;
            
            weight_change = params.stdp_learning_rate * (unsupervised_term + supervised_term);
            break;
    }
    
    // Apply weight update
    synapse.weight += weight_change * dt;
    
    // Clamp weights
    synapse.weight = fmaxf(0.0f, fminf(2.0f, synapse.weight));
}

} // namespace kernels
} // namespace neurogen
