#include <engine/CalciumDiffusionKernel.cuh>
#include <engine/NeuronModelConstants.h>

// Helper function to model the non-linear, voltage-dependent gating of NMDA receptors.
__device__ inline float nmda_gating_factor(float V) {
    return 1.0f / (1.0f + expf(-(V + 30.0f) * 0.15f));
}

// --- Kernel to update calcium concentration with realistic dynamics ---
__global__ void calciumDiffusionKernel(GPUNeuronState* neurons, float current_time, float dt, int num_neurons) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[neuron_idx];

    // Determine if the neuron spiked in the last time step.
    bool just_spiked = (neuron.last_spike_time >= (current_time - dt));

    // Loop through each dendritic compartment to update its calcium level
    for (int i = 0; i < 4; ++i) {
        float current_calcium = neuron.ca_conc[i];
        float influx = 0.0f;

        // 1. Voltage-Dependent Influx from Synaptic Activity (NMDA)
        float gating = nmda_gating_factor(neuron.V);
        influx += neuron.I_syn[i] * gating * 0.2f;

        // 2. Influx from Back-Propagating Action Potential (BAP)
        if (just_spiked) {
            influx += 0.8f;
        }

        // 3. Natural Decay of Calcium Concentration
        // This line now compiles correctly with the added constant.
        float decayed_calcium = current_calcium * NeuronModelConstants::CALCIUM_DECAY;

        // 4. Update and Clamp
        float new_calcium = decayed_calcium + influx * dt;
        neuron.ca_conc[i] = fmaxf(0.0f, new_calcium);
    }
}