// ============================================================================
// COMPLETE CUDA KERNEL IMPLEMENTATIONS
// File: src/cuda/EnhancedSTDPKernels.cu
// ============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cstdio>

// Include our GPU data structures
#include <engine/GPUNeuralStructures.h>

// ============================================================================
// ADDITIONAL STRUCTURES FOR ADVANCED LEARNING
// ============================================================================

/**
 * @brief Actor-Critic learning state
 */
struct ActorCriticState {
    float policy_parameters[16];        // Policy network weights
    float value_parameters[16];         // Value network weights
    float action_probabilities[16];     // Action probability distribution
    float state_value;                  // Current state value estimate
    float baseline_estimate;            // Value baseline estimate
    float advantage_estimate;           // Advantage estimate
    float policy_gradient[16];          // Policy gradient
    float value_gradient[16];           // Value gradient
    float learning_rate;                // Learning rate
    bool is_active;                     // State active flag
};

/**
 * @brief Curiosity-driven exploration state
 */
struct CuriosityState {
    float novelty_detector[32];         // Novelty detection features
    float surprise_level;               // Current surprise level
    float mastery_level;                // Skill mastery level
    float familiarity_level;            // Environment familiarity
    float random_exploration;           // Random exploration factor
    float directed_exploration;         // Directed exploration factor
    float prediction_error;             // Forward model prediction error
    bool exploration_active;            // Exploration enabled flag
};

// ============================================================================
// ENHANCED STDP KERNEL WITH MULTI-FACTOR PLASTICITY
// ============================================================================

/**
 * @brief Enhanced STDP kernel with biological realism and multi-factor learning
 */
__global__ void enhancedSTDPKernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    GPUPlasticityState* plasticity_states,
    GPUNetworkConfig* config,
    GPUNeuromodulatorState* neuromodulators,
    float current_time,
    float dt,
    int num_synapses
) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[synapse_idx];
    
    // Check if synapse is active and plastic
    if (synapse.active == 0 || !synapse.is_plastic) return;
    
    // Get pre and post neurons
    if (synapse.pre_neuron_idx < 0 || synapse.post_neuron_idx < 0) return;
    
    const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_idx];
    const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];
    
    // Multi-compartment STDP with dendritic integration
    int target_compartment = synapse.post_compartment;
    if (target_compartment >= MAX_COMPARTMENTS) target_compartment = 0;
    
    // Enhanced spike timing calculation
    float pre_spike_time = pre_neuron.last_spike_time;
    float post_spike_time = post_neuron.last_spike_time;
    float delta_t = post_spike_time - pre_spike_time;
    
    // Only process recent spike pairs (within 100ms window)
    if (fabsf(delta_t) > 100.0f) return;
    
    // Multi-factor plasticity computation
    float plasticity_magnitude = 0.0f;
    
    // 1. Classical STDP component
    float stdp_window = 20.0f; // 20ms STDP window
    if (fabsf(delta_t) < stdp_window) {
        if (delta_t > 0) {
            // LTP: Post after pre
            plasticity_magnitude = __expf(-delta_t / 10.0f) * 0.01f;
        } else {
            // LTD: Pre after post
            plasticity_magnitude = -__expf(delta_t / 10.0f) * 0.008f;
        }
    }
    
    // 2. Calcium-dependent modulation
    float ca_concentration = post_neuron.ca_conc[target_compartment];
    float ca_factor = 1.0f + (ca_concentration - 1.0f) * 0.5f; // Baseline ca = 1.0
    plasticity_magnitude *= ca_factor;
    
    // 3. Neuromodulation (dopamine, acetylcholine)
    float dopamine_factor = 1.0f + neuromodulators->dopamine_concentration * 
                           synapse.dopamine_sensitivity * 0.3f;
    float ach_factor = 1.0f + neuromodulators->acetylcholine_concentration * 
                      synapse.acetylcholine_sensitivity * 0.2f;
    plasticity_magnitude *= dopamine_factor * ach_factor;
    
    // 4. Metaplasticity: history-dependent scaling
    plasticity_magnitude *= synapse.metaplasticity_factor;
    
    // 5. BCM-like threshold modulation
    float post_activity = post_neuron.average_firing_rate;
    float bcm_threshold = post_neuron.bcm_threshold;
    float bcm_factor = post_activity * (post_activity - bcm_threshold);
    if (bcm_factor > 0) {
        plasticity_magnitude *= 1.2f; // Enhance LTP above threshold
    } else {
        plasticity_magnitude *= 0.8f; // Reduce below threshold
    }
    
    // Apply learning rate and time step
    float weight_change = plasticity_magnitude * synapse.learning_rate * dt;
    
    // Update synaptic weight with bounds
    synapse.weight += weight_change;
    synapse.weight = fmaxf(synapse.min_weight, fminf(synapse.weight, synapse.max_weight));
    
    // Update eligibility trace
    synapse.eligibility_trace += fabsf(weight_change);
    synapse.eligibility_trace *= __expf(-dt / 1000.0f); // 1s decay
    
    // Update metaplasticity factor
    synapse.metaplasticity_factor += weight_change * 0.001f;
    synapse.metaplasticity_factor = fmaxf(0.5f, fminf(synapse.metaplasticity_factor, 2.0f));
    
    // Update activity metrics
    synapse.activity_metric = synapse.activity_metric * 0.99f + fabsf(weight_change) * 0.01f;
    synapse.last_active_time = current_time;
}

/**
 * @brief BCM learning rule with adaptive threshold
 */
__global__ void bcmLearningKernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    GPUPlasticityState* plasticity_states,
    float current_time,
    float dt,
    int num_synapses
) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[synapse_idx];
    if (synapse.active == 0 || !synapse.is_plastic) return;
    
    const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_idx];
    const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];
    
    // BCM rule: weight change depends on pre and post activity
    float pre_rate = pre_neuron.average_firing_rate;
    float post_rate = post_neuron.average_firing_rate;
    float threshold = post_neuron.bcm_threshold;
    
    // BCM learning rule
    float weight_change = plasticity_states->bcm_learning_rate * pre_rate * post_rate * 
                         (post_rate - threshold) * dt;
    
    // Apply change with bounds
    synapse.weight += weight_change;
    synapse.weight = fmaxf(synapse.min_weight, fminf(synapse.weight, synapse.max_weight));
}

/**
 * @brief Homeostatic regulation kernel
 */
__global__ void homeostaticRegulationKernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float target_activity,
    float regulation_strength,
    float dt,
    int num_neurons,
    int num_synapses
) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[neuron_idx];
    if (neuron.active == 0) return;
    
    // Synaptic scaling based on activity deviation
    float activity_deviation = neuron.average_firing_rate - target_activity;
    float scaling_factor = 1.0f - regulation_strength * activity_deviation * dt;
    scaling_factor = fmaxf(0.5f, fminf(scaling_factor, 2.0f));
    
    // Apply scaling to all synapses of this neuron
    neuron.synaptic_scaling_factor *= scaling_factor;
    neuron.synaptic_scaling_factor = fmaxf(0.1f, fminf(neuron.synaptic_scaling_factor, 5.0f));
    
    // Intrinsic excitability adjustment
    float excitability_change = -regulation_strength * activity_deviation * dt * 0.1f;
    neuron.excitability += excitability_change;
    neuron.excitability = fmaxf(0.5f, fminf(neuron.excitability, 2.0f));
    
    // Update BCM threshold
    float threshold_change = regulation_strength * activity_deviation * dt * 0.01f;
    neuron.bcm_threshold += threshold_change;
    neuron.bcm_threshold = fmaxf(0.1f, fminf(neuron.bcm_threshold, 10.0f));
}

// ============================================================================
// REINFORCEMENT LEARNING KERNELS
// File: src/cuda/ReinforcementLearningKernels.cu
// ============================================================================

/**
 * @brief Dopaminergic system update kernel
 */
__global__ void dopamineUpdateKernel(
    GPUNeuronState* da_neurons,
    GPUNeuronState* network_neurons,
    float reward_signal,
    float predicted_reward,
    float current_time,
    float dt,
    int num_da_neurons,
    int num_network_neurons
) {
    int da_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (da_idx >= num_da_neurons) return;
    
    GPUNeuronState& da_neuron = da_neurons[da_idx];
    if (da_neuron.active == 0) return;
    
    // Compute reward prediction error
    float prediction_error = reward_signal - predicted_reward;
    
    // Update dopamine concentration based on prediction error
    float dopamine_change = prediction_error * 0.1f * dt;
    da_neuron.dopamine_concentration += dopamine_change;
    da_neuron.dopamine_concentration = fmaxf(0.0f, fminf(da_neuron.dopamine_concentration, 2.0f));
    
    // Update firing rate based on dopamine level
    da_neuron.average_firing_rate = da_neuron.dopamine_concentration * 10.0f; // 10Hz per unit dopamine
    
    // Diffuse dopamine to nearby network neurons
    for (int i = 0; i < fminf(64, num_network_neurons); i++) {
        int target_idx = (da_idx * 64 + i) % num_network_neurons;
        GPUNeuronState& neuron = network_neurons[target_idx];
        
        if (neuron.active) {
            // Simple spatial diffusion model
            float distance = fabsf(float(target_idx - da_idx * 64));
            float diffusion_factor = __expf(-distance / 20.0f); // 20-neuron diffusion radius
            
            float dopamine_transfer = da_neuron.dopamine_concentration * diffusion_factor * dt * 0.01f;
            neuron.dopamine_concentration += dopamine_transfer;
            neuron.dopamine_concentration = fmaxf(0.0f, fminf(neuron.dopamine_concentration, 1.0f));
        }
    }
    
    // Decay dopamine
    da_neuron.dopamine_concentration *= __expf(-dt / 100.0f); // 100ms decay
}

/**
 * @brief Actor-critic learning kernel
 */
__global__ void actorCriticUpdateKernel(
    ActorCriticState* actor_critic_states,
    const float* state_features,
    int action_taken,
    float reward_received,
    float dt,
    int num_features,
    int num_actions
) {
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (state_idx >= 1) return; // Single actor-critic state for now
    
    ActorCriticState& ac = actor_critic_states[state_idx];
    
    // Update state value estimate
    float td_error = reward_received - ac.state_value;
    ac.state_value += 0.01f * td_error * dt; // Critic learning rate
    
    // Update action preferences (policy gradient)
    if (action_taken >= 0 && action_taken < num_actions) {
        float advantage = td_error; // Simplified advantage estimate
        
        // Update action preferences
        for (int a = 0; a < num_actions; a++) {
            if (a == action_taken) {
                ac.policy_parameters[a] += 0.001f * advantage * dt; // Actor learning rate
            } else {
                ac.policy_parameters[a] -= 0.001f * advantage * dt * 0.1f; // Negative update for other actions
            }
        }
        
        // Compute action probabilities using softmax
        float max_param = ac.policy_parameters[0];
        for (int a = 1; a < num_actions; a++) {
            max_param = fmaxf(max_param, ac.policy_parameters[a]);
        }
        
        float sum_exp = 0.0f;
        for (int a = 0; a < num_actions; a++) {
            ac.action_probabilities[a] = __expf(ac.policy_parameters[a] - max_param);
            sum_exp += ac.action_probabilities[a];
        }
        
        // Normalize probabilities
        for (int a = 0; a < num_actions; a++) {
            ac.action_probabilities[a] /= (sum_exp + 1e-8f);
        }
    }
    
    // Update baseline estimate
    ac.baseline_estimate = ac.baseline_estimate * 0.99f + reward_received * 0.01f;
    
    // Update advantage estimate
    ac.advantage_estimate = reward_received - ac.baseline_estimate;
}

/**
 * @brief Curiosity-driven exploration kernel
 */
__global__ void curiosityUpdateKernel(
    CuriosityState* curiosity_states,
    const float* environmental_features,
    float prediction_error,
    float dt,
    int num_features
) {
    int curiosity_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (curiosity_idx >= 1) return; // Single curiosity state
    
    CuriosityState& curiosity = curiosity_states[curiosity_idx];
    
    // Update surprise level based on prediction error
    curiosity.surprise_level = curiosity.surprise_level * 0.9f + prediction_error * 0.1f;
    
    // Update novelty detector
    float state_novelty = 0.0f;
    for (int i = 0; i < fminf(32, num_features); i++) {
        float feature = (i < num_features) ? environmental_features[i] : 0.0f;
        float distance = fabsf(feature - curiosity.novelty_detector[i]);
        state_novelty += distance;
        
        // Update novelty detector with exponential moving average
        curiosity.novelty_detector[i] = curiosity.novelty_detector[i] * 0.99f + feature * 0.01f;
    }
    
    curiosity.familiarity_level = 1.0f / (1.0f + state_novelty);
    
    // Update exploration drive
    curiosity.random_exploration = 0.1f * (1.0f - curiosity.mastery_level);
    curiosity.directed_exploration = state_novelty * (1.0f - curiosity.familiarity_level) * 0.1f;
    
    // Update mastery level based on prediction accuracy
    float prediction_accuracy = 1.0f / (1.0f + curiosity.surprise_level);
    curiosity.mastery_level = curiosity.mastery_level * 0.999f + prediction_accuracy * 0.001f;
}

// ============================================================================
// SIMPLIFIED WRAPPER FUNCTION FOR BACKWARD COMPATIBILITY
// ============================================================================

/**
 * @brief Simplified enhancedSTDPKernel wrapper with basic signature
 * This provides backward compatibility for existing code
 */
__global__ void enhancedSTDPKernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses
) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[synapse_idx];
    
    // Check if synapse is active and plastic
    if (synapse.active == 0 || !synapse.is_plastic) return;
    
    // Get pre and post neurons
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    if (pre_idx < 0 || post_idx < 0) return;
    
    const GPUNeuronState& pre_neuron = neurons[pre_idx];
    const GPUNeuronState& post_neuron = neurons[post_idx];
    
    // Simple STDP rule based on spike timing
    float time_diff = post_neuron.last_spike_time - pre_neuron.last_spike_time;
    
    // STDP window parameters
    const float tau_plus = 20.0f;  // LTP time constant (ms)
    const float tau_minus = 20.0f; // LTD time constant (ms)
    const float A_plus = 0.01f;    // LTP amplitude
    const float A_minus = 0.01f;   // LTD amplitude
    
    float weight_change = 0.0f;
    
    // Apply STDP rule
    if (time_diff > 0.0f && time_diff < 50.0f) {
        // LTP: post fires after pre
        weight_change = A_plus * expf(-time_diff / tau_plus);
    } else if (time_diff < 0.0f && time_diff > -50.0f) {
        // LTD: pre fires after post  
        weight_change = -A_minus * expf(time_diff / tau_minus);
    }
    
    // Update synaptic weight
    synapse.weight += weight_change * dt;
    
    // Bound weight within valid range
    synapse.weight = fmaxf(0.0f, fminf(synapse.weight, 1.0f));
}

// ============================================================================
// DEVICE HELPER FUNCTION FOR SIMPLIFIED STDP
// ============================================================================

/**
 * Device function for simple STDP update
 */
__device__ void enhancedSTDPKernelSimple(GPUSynapse* synapses, 
                                          const GPUNeuronState* neurons, 
                                          float dt, 
                                          int synapse_idx) {
    
    // Simple STDP implementation for backward compatibility
    GPUSynapse& synapse = synapses[synapse_idx];
    
    int pre_idx = synapse.pre_neuron_idx;
    int post_idx = synapse.post_neuron_idx;
    
    if (pre_idx < 0 || post_idx < 0) return;
    
    const GPUNeuronState& pre_neuron = neurons[pre_idx];
    const GPUNeuronState& post_neuron = neurons[post_idx];
    
    // Simple STDP rule
    float time_diff = post_neuron.last_spike_time - pre_neuron.last_spike_time;
    float weight_change = 0.0f;
    
    const float learning_rate = 0.01f; // Default learning rate
    
    if (time_diff > 0.0f && time_diff < 50.0f) {
        // LTP
        weight_change = learning_rate * expf(-time_diff / 20.0f);
    } else if (time_diff < 0.0f && time_diff > -50.0f) {
        // LTD
        weight_change = -learning_rate * expf(time_diff / 20.0f);
    }
    
    // Update weight
    synapse.weight += weight_change * dt;
    synapse.weight = fmaxf(0.0f, fminf(synapse.weight, 1.0f));
}