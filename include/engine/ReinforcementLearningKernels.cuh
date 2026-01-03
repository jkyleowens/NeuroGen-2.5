// cuda/ReinforcementLearningKernels.cu
// CUDA kernels for reward-modulated STDP and eligibility traces

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

// Enhanced synapse structure with eligibility trace
struct EnhancedSynapse {
    float weight;
    float eligibility_trace;
    float last_pre_spike_time;
    float last_post_spike_time;
    int pre_neuron_id;
    int post_neuron_id;
    bool is_active;
};

// Neuron structure with spike timing information
struct EnhancedNeuron {
    float voltage;
    float last_spike_time;
    float spike_count;
    bool is_spiking;
    float activity_trace;
};

// Global reward signal and learning parameters
__device__ float d_global_reward = 0.0f;
__device__ float d_learning_rate = 0.01f;
__device__ float d_eligibility_decay = 0.95f;
__device__ float d_current_time = 0.0f;

// Kernel to update eligibility traces based on spike timing
__global__ void updateEligibilityTracesKernel(
    EnhancedSynapse* synapses,
    EnhancedNeuron* neurons,
    int num_synapses,
    float dt,
    float tau_eligibility
) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (synapse_idx >= num_synapses) return;
    
    EnhancedSynapse& synapse = synapses[synapse_idx];
    
    if (!synapse.is_active) return;
    
    // Get pre and post neurons
    EnhancedNeuron& pre_neuron = neurons[synapse.pre_neuron_id];
    EnhancedNeuron& post_neuron = neurons[synapse.post_neuron_id];
    
    // Decay existing eligibility trace
    synapse.eligibility_trace *= expf(-dt / tau_eligibility);
    
    // Check for recent spike combinations
    float time_window = 50.0f; // 50ms window for STDP
    float pre_post_delta = post_neuron.last_spike_time - pre_neuron.last_spike_time;
    
    if (fabsf(pre_post_delta) < time_window) {
        // Calculate STDP contribution to eligibility trace
        float stdp_window = 20.0f; // 20ms STDP window
        float eligibility_increment = 0.0f;
        
        if (pre_post_delta > 0 && pre_post_delta < stdp_window) {
            // Pre-before-post: potentiation
            eligibility_increment = expf(-pre_post_delta / 10.0f) * 0.5f;
        } else if (pre_post_delta < 0 && pre_post_delta > -stdp_window) {
            // Post-before-pre: depression
            eligibility_increment = -expf(pre_post_delta / 10.0f) * 0.3f;
        }
        
        synapse.eligibility_trace += eligibility_increment;
        
        // Update spike timing records
        synapse.last_pre_spike_time = pre_neuron.last_spike_time;
        synapse.last_post_spike_time = post_neuron.last_spike_time;
    }
    
    // Clamp eligibility trace
    synapse.eligibility_trace = fmaxf(-1.0f, fminf(1.0f, synapse.eligibility_trace));
}

// Kernel to apply reward-modulated weight updates
__global__ void applyRewardModulatedLearningKernel(
    EnhancedSynapse* synapses,
    int num_synapses,
    float reward_signal,
    float learning_rate,
    float weight_decay
) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (synapse_idx >= num_synapses) return;
    
    EnhancedSynapse& synapse = synapses[synapse_idx];
    
    if (!synapse.is_active) return;
    
    // Three-factor learning rule: STDP × Reward × Learning Rate
    float weight_change = learning_rate * reward_signal * synapse.eligibility_trace;
    
    // Apply weight update
    synapse.weight += weight_change;
    
    // Apply weight decay
    synapse.weight *= (1.0f - weight_decay);
    
    // Clamp weights to reasonable bounds
    synapse.weight = fmaxf(0.0f, fminf(2.0f, synapse.weight));
    
    // Decay eligibility trace after learning
    synapse.eligibility_trace *= 0.9f;
}

// Kernel for neural activity tracking (for neurogenesis decisions)
__global__ void updateNeuralActivityKernel(
    EnhancedNeuron* neurons,
    int num_neurons,
    float dt,
    float activity_decay
) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_idx >= num_neurons) return;
    
    EnhancedNeuron& neuron = neurons[neuron_idx];
    
    // Decay activity trace
    neuron.activity_trace *= expf(-dt / activity_decay);
    
    // Add current spike contribution
    if (neuron.is_spiking) {
        neuron.activity_trace += 1.0f;
        neuron.spike_count += 1.0f;
    }
    
    // Update timing
    if (neuron.is_spiking) {
        neuron.last_spike_time = d_current_time;
    }
}

// Kernel for structural plasticity - identify synapses for pruning
__global__ void identifyPruningSynapses(
    EnhancedSynapse* synapses,
    int* prune_candidates,
    int num_synapses,
    float pruning_threshold,
    float time_window
) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (synapse_idx >= num_synapses) return;
    
    EnhancedSynapse& synapse = synapses[synapse_idx];
    prune_candidates[synapse_idx] = 0;
    
    if (!synapse.is_active) return;
    
    // Check if synapse is underutilized
    float time_since_last_activity = d_current_time - fmaxf(synapse.last_pre_spike_time, synapse.last_post_spike_time);
    
    bool is_weak = synapse.weight < pruning_threshold;
    bool is_inactive = time_since_last_activity > time_window;
    bool has_low_eligibility = fabsf(synapse.eligibility_trace) < 0.01f;
    
    if (is_weak && is_inactive && has_low_eligibility) {
        prune_candidates[synapse_idx] = 1;
    }
}

// Kernel for creating new synapses (neurogenesis support)
__global__ void createNewSynapsesKernel(
    EnhancedSynapse* synapses,
    EnhancedNeuron* neurons,
    int* new_synapse_indices,
    int num_new_synapses,
    int total_synapses,
    float initial_weight,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_new_synapses) return;
    
    int synapse_idx = new_synapse_indices[idx];
    if (synapse_idx >= total_synapses) return;
    
    // Initialize new synapse with random connections
    // This is a simplified version - in practice, you'd want more sophisticated connection rules
    EnhancedSynapse& synapse = synapses[synapse_idx];
    
    // Use simple linear congruential generator for randomness
    unsigned int random_state = seed + idx;
    random_state = random_state * 1664525u + 1013904223u;
    
    int total_neurons = 1024; // Should be passed as parameter
    synapse.pre_neuron_id = random_state % total_neurons;
    random_state = random_state * 1664525u + 1013904223u;
    synapse.post_neuron_id = random_state % total_neurons;
    
    // Avoid self-connections
    if (synapse.pre_neuron_id == synapse.post_neuron_id) {
        synapse.post_neuron_id = (synapse.post_neuron_id + 1) % total_neurons;
    }
    
    // Initialize synapse properties
    synapse.weight = initial_weight;
    synapse.eligibility_trace = 0.0f;
    synapse.last_pre_spike_time = 0.0f;
    synapse.last_post_spike_time = 0.0f;
    synapse.is_active = true;
}

// Host function to update global reward signal
extern "C" void setGlobalReward(float reward) {
    cudaMemcpyToSymbol(d_global_reward, &reward, sizeof(float));
}

// Host function to update learning parameters
extern "C" void setLearningParameters(float learning_rate, float eligibility_decay, float current_time) {
    cudaMemcpyToSymbol(d_learning_rate, &learning_rate, sizeof(float));
    cudaMemcpyToSymbol(d_eligibility_decay, &eligibility_decay, sizeof(float));
    cudaMemcpyToSymbol(d_current_time, &current_time, sizeof(float));
}

// Host function to launch eligibility trace update
extern "C" void launchEligibilityTraceUpdate(
    EnhancedSynapse* d_synapses,
    EnhancedNeuron* d_neurons,
    int num_synapses,
    float dt,
    float tau_eligibility
) {
    int threads_per_block = 256;
    int num_blocks = (num_synapses + threads_per_block - 1) / threads_per_block;
    
    updateEligibilityTracesKernel<<<num_blocks, threads_per_block>>>(
        d_synapses, d_neurons, num_synapses, dt, tau_eligibility
    );
    
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] Eligibility trace update failed: %s\n", cudaGetErrorString(error));
    }
}

// Host function to launch reward-modulated learning
extern "C" void launchRewardModulatedLearning(
    EnhancedSynapse* d_synapses,
    int num_synapses,
    float reward_signal,
    float learning_rate,
    float weight_decay
) {
    int threads_per_block = 256;
    int num_blocks = (num_synapses + threads_per_block - 1) / threads_per_block;
    
    applyRewardModulatedLearningKernel<<<num_blocks, threads_per_block>>>(
        d_synapses, num_synapses, reward_signal, learning_rate, weight_decay
    );
    
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] Reward-modulated learning failed: %s\n", cudaGetErrorString(error));
    }
}

// Host function for structural plasticity operations
extern "C" void launchStructuralPlasticity(
    EnhancedSynapse* d_synapses,
    EnhancedNeuron* d_neurons,
    int num_synapses,
    int num_neurons,
    float pruning_threshold,
    float time_window,
    bool enable_neurogenesis
) {
    int threads_per_block = 256;
    int num_blocks;
    
    // Update neural activity traces
    num_blocks = (num_neurons + threads_per_block - 1) / threads_per_block;
    updateNeuralActivityKernel<<<num_blocks, threads_per_block>>>(
        d_neurons, num_neurons, 1.0f, 100.0f
    );
    
    // Identify synapses for pruning
    int* d_prune_candidates;
    cudaMalloc(&d_prune_candidates, num_synapses * sizeof(int));
    
    num_blocks = (num_synapses + threads_per_block - 1) / threads_per_block;
    identifyPruningSynapses<<<num_blocks, threads_per_block>>>(
        d_synapses, d_prune_candidates, num_synapses, pruning_threshold, time_window
    );
    
    if (enable_neurogenesis) {
        // Create new synapses to replace pruned ones (simplified implementation)
        int max_new_synapses = num_synapses / 100; // Create up to 1% new synapses
        int* d_new_indices;
        cudaMalloc(&d_new_indices, max_new_synapses * sizeof(int));
        
        // This would need a more sophisticated implementation to properly
        // select indices and manage synapse creation
        // For now, this is a placeholder
        
        cudaFree(d_new_indices);
    }
    
    cudaFree(d_prune_candidates);
    
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] Structural plasticity failed: %s\n", cudaGetErrorString(error));
    }
}