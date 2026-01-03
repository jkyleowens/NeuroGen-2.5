#include <engine/HomeostaticMechanismsKernel.cuh>
#include <engine/NeuronModelConstants.h>

// --- Kernel for Synaptic Scaling ---
__global__ void synapticScalingKernel(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int num_neurons,
    int total_synapses,
    float current_time)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[neuron_idx];

    // --- 1. Update long-term average firing rate ---
    float decay = expf(-0.1f / NeuronModelConstants::HOMEOSTATIC_TIMESCALE);
    bool just_spiked = (neuron.last_spike_time >= current_time - 0.1f);
    float instantaneous_rate = just_spiked ? 10.0f : 0.0f; // Simplified rate in Hz
    neuron.average_firing_rate = neuron.average_firing_rate * decay + instantaneous_rate * (1.0f - decay);

    // --- 2. Calculate Scaling Adjustment ---
    float rate_error = neuron.average_firing_rate - NeuronModelConstants::TARGET_FIRING_RATE;
    float scaling_adjustment = -rate_error * 0.0001f; // Slow adjustment

    neuron.synaptic_scaling_factor = fmaxf(0.5f, fminf(1.5f, neuron.synaptic_scaling_factor + scaling_adjustment));
}

// --- Kernel for Intrinsic Plasticity ---
__global__ void intrinsicPlasticityKernel(GPUNeuronState* neurons, int num_neurons) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[neuron_idx];

    // --- 1. Update long-term average activity (membrane potential) ---
    float decay = expf(-0.1f / NeuronModelConstants::HOMEOSTATIC_TIMESCALE);
    neuron.average_activity = neuron.average_activity * decay + neuron.V * (1.0f - decay);

    // --- 2. Adjust Intrinsic Excitability ---
    float activity_error = neuron.average_activity - NeuronModelConstants::RESET_POTENTIAL;
    float excitability_change = -activity_error * 0.00005f; // Very slow adjustment

    neuron.excitability = fmaxf(0.8f, fminf(1.2f, neuron.excitability + excitability_change));
}

// ============================================================================

/**
 * @brief Kernel to apply synaptic scaling to maintain network stability.
 * This kernel is currently disabled to simplify the initial model.
 * To enable, uncomment the code below and ensure it's called from a wrapper.
 */
__global__ void applySynapticScalingKernel(GPUSynapse* d_synapses, const GPUNeuronState* d_neurons, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    int neuron_idx = d_synapses[idx].target_neuron_idx;
    float scaling_factor = d_neurons[neuron_idx].synaptic_scaling_factor;

    // Apply scaling factor to the synaptic weight
    d_synapses[idx].weight *= scaling_factor;

    // Clamp weight to a reasonable range
    d_synapses[idx].weight = fmaxf(0.001f, fminf(d_synapses[idx].weight, 1.0f));
}

/**
 * @brief Kernel for weight normalization to prevent runaway synaptic strengths.
 * This kernel is currently disabled.
 * To enable, uncomment the code below.
 */
__global__ void weightNormalizationKernel(GPUSynapse* d_synapses, int* d_neuron_synapse_indices, int num_neurons, int synapses_per_neuron) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    float sum_squared_weights = 0.0f;
    int start_idx = neuron_idx * synapses_per_neuron;

    // Calculate sum of squared weights for the neuron's synapses
    for (int i = 0; i < synapses_per_neuron; ++i) {
        int synapse_idx = d_neuron_synapse_indices[start_idx + i];
        sum_squared_weights += d_synapses[synapse_idx].weight * d_synapses[synapse_idx].weight;
    }

    // Normalize if the sum is greater than 1
    if (sum_squared_weights > 1.0f) {
        float norm_factor = 1.0f / sqrtf(sum_squared_weights);
        for (int i = 0; i < synapses_per_neuron; ++i) {
            int synapse_idx = d_neuron_synapse_indices[start_idx + i];
            d_synapses[synapse_idx].weight *= norm_factor;
        }
    }
}

/**
 * @brief Kernel to regulate neuron activity levels.
 * This kernel is currently disabled.
 * To enable, uncomment the code below.
 */
__global__ void activityRegulationKernel(GPUNeuronState* d_neurons, float target_activity, float regulation_strength, int num_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    float current_activity = d_neurons[idx].firing_rate;
    float activity_error = target_activity - current_activity;

    // Adjust neuron's excitability based on the error
    d_neurons[idx].threshold += regulation_strength * activity_error;

    // Clamp threshold to a reasonable range
    d_neurons[idx].threshold = fmaxf(0.1f, fminf(d_neurons[idx].threshold, 2.0f));
}