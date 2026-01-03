#include <engine/HebbianLearningKernel.cuh>
#include <engine/GPUNeuralStructures.h>

// ============================================================================
// HEBBIAN LEARNING KERNEL IMPLEMENTATIONS
// ============================================================================

/**
 * Standard Hebbian learning rule: "cells that fire together, wire together"
 */
__global__ void hebbianLearningKernel(GPUSynapse* synapses,
                                     const GPUNeuronState* neurons,
                                     float learning_rate,
                                     float dt,
                                     int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Get pre and post synaptic neurons
    const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_idx];
    const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];
    
    // Hebbian learning: strengthen when both neurons are active
    float pre_activity = pre_neuron.average_firing_rate;
    float post_activity = post_neuron.average_firing_rate;
    
    float hebbian_factor = pre_activity * post_activity;
    synapse.weight += learning_rate * hebbian_factor * dt;
    
    // Bound weight
    if (synapse.weight > 1.0f) synapse.weight = 1.0f;
    if (synapse.weight < 0.01f) synapse.weight = 0.01f;
}

/**
 * Oja's learning rule: Hebbian learning with weight normalization
 */
__global__ void ojasLearningKernel(GPUSynapse* synapses,
                                  const GPUNeuronState* neurons,
                                  float learning_rate,
                                  float dt,
                                  int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Get pre and post synaptic neurons
    const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_idx];
    const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];
    
    float pre_activity = pre_neuron.average_firing_rate;
    float post_activity = post_neuron.average_firing_rate;
    
    // Oja's rule: Hebbian term minus weight decay term
    float hebbian_term = pre_activity * post_activity;
    float decay_term = post_activity * post_activity * synapse.weight;
    
    synapse.weight += learning_rate * (hebbian_term - decay_term) * dt;
    
    // Bound weight
    if (synapse.weight > 1.0f) synapse.weight = 1.0f;
    if (synapse.weight < 0.01f) synapse.weight = 0.01f;
}

/**
 * BCM (Bienenstock-Cooper-Munro) learning rule with sliding threshold
 */
__global__ void bcmLearningKernel(GPUSynapse* synapses,
                                 GPUNeuronState* neurons,
                                 float learning_rate,
                                 float dt,
                                 int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Get pre and post synaptic neurons
    const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_idx];
    GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];
    
    float pre_activity = pre_neuron.average_firing_rate;
    float post_activity = post_neuron.average_firing_rate;
    
    // BCM threshold (sliding based on average activity)
    float threshold = post_neuron.average_activity * post_neuron.average_activity;
    
    // BCM plasticity rule
    float plasticity_factor = post_activity * (post_activity - threshold);
    synapse.weight += learning_rate * pre_activity * plasticity_factor * dt;
    
    // Bound weight
    if (synapse.weight > 1.0f) synapse.weight = 1.0f;
    if (synapse.weight < 0.01f) synapse.weight = 0.01f;
}

/**
 * Correlation-based learning with temporal window
 */
__global__ void correlationLearningKernel(GPUSynapse* synapses,
                                         const GPUNeuronState* neurons,
                                         float* correlation_buffer,
                                         float learning_rate,
                                         float dt,
                                         int num_synapses,
                                         int correlation_window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Get pre and post synaptic neurons
    const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_idx];
    const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];
    
    // Compute correlation over time window
    float correlation = 0.0f;
    if (correlation_buffer && correlation_window > 0) {
        int buffer_idx = idx % correlation_window;
        correlation = correlation_buffer[buffer_idx];
        
        // Update correlation buffer
        correlation_buffer[buffer_idx] = pre_neuron.average_firing_rate * post_neuron.average_firing_rate;
    } else {
        // Simple instantaneous correlation
        correlation = pre_neuron.average_firing_rate * post_neuron.average_firing_rate;
    }
    
    synapse.weight += learning_rate * correlation * dt;
    
    // Bound weight
    if (synapse.weight > 1.0f) synapse.weight = 1.0f;
    if (synapse.weight < 0.01f) synapse.weight = 0.01f;
}
