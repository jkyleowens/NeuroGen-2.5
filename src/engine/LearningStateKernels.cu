// File: src/cuda/LearningStateKernels.cu (NEW FILE)

#include <engine/LearningStateKernels.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

// ============================================================================
// CUDA KERNEL IMPLEMENTATIONS
// ============================================================================

__global__ void kernel_update_eligibility_traces(
    GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    GPUSynapse* synapses,
    float reward_signal,
    float dt,
    int num_synapses
) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_idx >= num_synapses) return;
    
    // Get current eligibility trace
    float current_trace = learning_state->eligibility_traces[synapse_idx];
    
    // Decay existing trace
    current_trace *= 0.95f; // Decay rate
    
    // Get connected neurons
    int pre_neuron = synapses[synapse_idx].pre_neuron_idx;
    int post_neuron = synapses[synapse_idx].post_neuron_idx;
    
    if (pre_neuron < 0 || post_neuron < 0) return;
    
    // Add new trace based on neuron activity
    float pre_activity = neurons[pre_neuron].V; // Use membrane potential as activity proxy
    float post_activity = neurons[post_neuron].V;
    
    if (pre_activity > -50.0f && post_activity > -50.0f) { // Both neurons active
        current_trace += pre_activity * post_activity * dt * 0.001f; // Scale factor
    }
    
    // Apply reward modulation if significant reward
    // Note: fabsf checks magnitude, but reward_signal itself preserves sign
    if (fabsf(reward_signal) > 0.01f) {
        float weight_change = learning_state->learning_rates[post_neuron] * reward_signal * current_trace;
        synapses[synapse_idx].weight += weight_change;
        
        // Clamp weight
        synapses[synapse_idx].weight = fmaxf(-1.0f, fminf(1.0f, synapses[synapse_idx].weight));
    }
    
    // Update eligibility trace - PRESERVE SIGN for proper credit assignment
    // Allow both positive and negative traces for LTP/LTD distinction
    learning_state->eligibility_traces[synapse_idx] = fmaxf(-1.0f, fminf(1.0f, current_trace));
}

__global__ void kernel_apply_synaptic_tagging(
    GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    float novelty_signal,
    float dt,
    int num_neurons,
    int num_synapses
) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;
    
    // Get neuron activity
    float activity = neurons[neuron_idx].V + 70.0f; // Normalize membrane potential
    activity = fmaxf(0.0f, activity / 50.0f); // Scale to 0-1 range
    
    // Get plasticity threshold for this neuron
    float threshold = learning_state->plasticity_thresholds[neuron_idx];
    
    // Apply tagging if activity exceeds threshold and there's novelty
    if (activity > threshold && fabsf(novelty_signal) > 0.01f) {
        // Find synapses connected to this neuron and apply tags
        // This is simplified - in practice you'd need synapse lookup tables
        int start_synapse = neuron_idx * 100; // Assume ~100 synapses per neuron
        int end_synapse = min(start_synapse + 100, num_synapses);
        
        for (int syn_idx = start_synapse; syn_idx < end_synapse; syn_idx++) {
            float tag_strength = activity * fabsf(novelty_signal);
            learning_state->synaptic_tags[syn_idx] = fmaxf(
                learning_state->synaptic_tags[syn_idx],
                tag_strength
            );
        }
    }
}

__global__ void kernel_update_neuromodulators(
    GPULearningState* learning_state,
    GPUNeuronState* neurons,
    float dopamine_level,
    float acetylcholine_level,
    float norepinephrine_level,
    float dt,
    int num_neurons
) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;
    
    // Get current neuromodulator levels (stored as 3 consecutive floats per neuron)
    int base_idx = neuron_idx * 3;
    float& current_dopamine = learning_state->neuromodulator_levels[base_idx];
    float& current_acetylcholine = learning_state->neuromodulator_levels[base_idx + 1];
    float& current_norepinephrine = learning_state->neuromodulator_levels[base_idx + 2];
    
    // Update with decay and new input
    current_dopamine = 0.9f * current_dopamine + 0.1f * dopamine_level;
    current_acetylcholine = 0.95f * current_acetylcholine + 0.05f * acetylcholine_level;
    current_norepinephrine = 0.8f * current_norepinephrine + 0.2f * norepinephrine_level;
    
    // Clamp to physiological ranges
    current_dopamine = fmaxf(0.0f, fminf(2.0f, current_dopamine));
    current_acetylcholine = fmaxf(0.0f, fminf(1.0f, current_acetylcholine));
    current_norepinephrine = fmaxf(0.0f, fminf(1.5f, current_norepinephrine));
    
    // Modulate neuron parameters based on neuromodulators
    // Dopamine affects learning rate
    learning_state->learning_rates[neuron_idx] *= (1.0f + 0.5f * current_dopamine);
    
    // Acetylcholine affects attention/plasticity threshold
    learning_state->plasticity_thresholds[neuron_idx] *= (1.0f - 0.3f * current_acetylcholine);
    
    // Norepinephrine affects overall excitability
    neurons[neuron_idx].I_ext += current_norepinephrine * 2.0f; // Add external current
}

__global__ void kernel_perform_memory_consolidation(
    GPULearningState* learning_state,
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    float consolidation_strength,
    int* consolidated_count,
    int num_synapses
) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_idx >= num_synapses) return;
    
    // Calculate consolidation score
    float eligibility = learning_state->eligibility_traces[synapse_idx];
    float tag = learning_state->synaptic_tags[synapse_idx];
    float consolidation_score = eligibility + tag;
    
    // Consolidate if score exceeds threshold
    if (consolidation_score > 0.5f) {
        // Strengthen the synapse
        float strengthening = consolidation_strength * consolidation_score;
        synapses[synapse_idx].weight += strengthening * synapses[synapse_idx].weight;
        
        // Clamp weight
        synapses[synapse_idx].weight = fmaxf(-1.0f, fminf(1.0f, synapses[synapse_idx].weight));
        
        // Decay eligibility and tags after consolidation
        learning_state->eligibility_traces[synapse_idx] *= 0.8f;
        learning_state->synaptic_tags[synapse_idx] *= 0.8f;
        
        // Increment consolidation counter
        atomicAdd(consolidated_count, 1);
        
        // Update consolidation weight
        learning_state->consolidation_weights[synapse_idx] += strengthening;
    }
}

__global__ void kernel_update_inter_module_connections(
    GPUInterModuleState* inter_module_state,
    const GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    float learning_rate_multiplier,
    float dt,
    int num_neurons
) {
    // This kernel is complex and requires a proper implementation
    // Placeholder implementation
}

namespace LearningStateKernels {

void update_eligibility_traces(
    GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    GPUSynapse* synapses,
    float reward_signal,
    float dt,
    int num_neurons,
    int num_synapses
) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_synapses + threads_per_block - 1) / threads_per_block;
    kernel_update_eligibility_traces<<<blocks_per_grid, threads_per_block>>>(
        learning_state, neurons, synapses, reward_signal, dt, num_synapses
    );
}

void apply_synaptic_tagging(
    GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    float novelty_signal,
    float dt,
    int num_neurons,
    int num_synapses
) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_neurons + threads_per_block - 1) / threads_per_block;
    kernel_apply_synaptic_tagging<<<blocks_per_grid, threads_per_block>>>(
        learning_state, neurons, novelty_signal, dt, num_neurons, num_synapses
    );
}

void update_neuromodulators(
    GPULearningState* learning_state,
    GPUNeuronState* neurons,
    float dopamine_level,
    float acetylcholine_level,
    float norepinephrine_level,
    float dt,
    int num_neurons
) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_neurons + threads_per_block - 1) / threads_per_block;
    kernel_update_neuromodulators<<<blocks_per_grid, threads_per_block>>>(
        learning_state, neurons, dopamine_level, acetylcholine_level, norepinephrine_level, dt, num_neurons
    );
}

void perform_memory_consolidation(
    GPULearningState* learning_state,
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    float consolidation_strength,
    int* consolidated_count,
    int num_neurons,
    int num_synapses
) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_synapses + threads_per_block - 1) / threads_per_block;
    kernel_perform_memory_consolidation<<<blocks_per_grid, threads_per_block>>>(
        learning_state, neurons, synapses, consolidation_strength, consolidated_count, num_synapses
    );
}

void consolidate_module(
    GPULearningState* learning_state,
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int module_id,
    float consolidation_strength,
    int num_neurons,
    int num_synapses
) {
    // Placeholder
}

void update_inter_module_connections(
    GPUInterModuleState* inter_module_state,
    const GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    float learning_rate_multiplier,
    float dt,
    int num_neurons
) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_neurons + threads_per_block - 1) / threads_per_block;
    kernel_update_inter_module_connections<<<blocks_per_grid, threads_per_block>>>(
        inter_module_state, learning_state, neurons, learning_rate_multiplier, dt, num_neurons
    );
}

void apply_hebbian_learning(
    GPUInterModuleState* inter_module_state,
    const float* source_activities,
    const float* target_activities,
    float learning_rate,
    int num_connections
) {
    // Placeholder
}

void save_module_learning_state(
    const GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    const GPUSynapse* synapses,
    int module_id,
    float* output_buffer,
    size_t buffer_size,
    int num_neurons,
    int num_synapses
) {
    // Placeholder
}

void load_module_learning_state(
    GPULearningState* learning_state,
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int module_id,
    const float* input_buffer,
    size_t buffer_size,
    int num_neurons,
    int num_synapses
) {
    // Placeholder
}

void save_complete_learning_state(
    const GPULearningState* learning_state,
    const GPUInterModuleState* inter_module_state,
    uint8_t* host_buffer,
    size_t buffer_size,
    int num_neurons,
    int num_synapses
) {
    // Placeholder
}

void load_complete_learning_state(
    GPULearningState* learning_state,
    GPUInterModuleState* inter_module_state,
    const uint8_t* host_buffer,
    size_t buffer_size,
    int num_neurons,
    int num_synapses
) {
    // Placeholder
}

void calculate_module_performance(
    const GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    float* module_metrics,
    int num_modules,
    int num_neurons
) {
    // Placeholder
}

void update_learning_statistics(
    GPULearningState* learning_state,
    const GPUNeuronState* neurons,
    float reward_signal,
    float prediction_error,
    float dt,
    int num_neurons
) {
    // Placeholder
}

} // namespace LearningStateKernels
