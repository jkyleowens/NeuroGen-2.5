#include <engine/NeuronUpdateKernel.cuh>
#include <engine/NeuronModelConstants.h>

// --- Main kernel to update neuron states using the Izhikevich model ---
__global__ void neuronUpdateKernel(GPUNeuronState* neurons, float current_time, float dt, int num_neurons) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[neuron_idx];

    // --- 1. Check if neuron is in a refractory period ---
    if (current_time < neuron.last_spike_time + NeuronModelConstants::ABSOLUTE_REFRACTORY_PERIOD) {
        return; // Neuron is refractory, do not update voltage
    }

    // --- 2. Gather total synaptic input current ---
    // This includes external inputs and is modulated by intrinsic excitability
    float total_current = (neuron.I_syn[0] + neuron.I_syn[1] + neuron.I_syn[2] + neuron.I_syn[3]) * neuron.excitability;

    // --- 3. Update membrane potential and recovery variable (Izhikevich model) ---
    float v = neuron.V;
    float u = neuron.u;

    // Izhikevich model dynamics (solved with forward Euler method)
    neuron.V += dt * (0.04f * v * v + 5.0f * v + 140.0f - u + total_current);
    neuron.u += dt * (0.02f * (0.2f * v - u)); // Using typical 'a' and 'b' parameters

    // --- 4. Check for spike firing ---
    if (neuron.V >= NeuronModelConstants::SPIKE_THRESHOLD) {
        neuron.V = NeuronModelConstants::RESET_POTENTIAL; // Reset potential
        neuron.u += 8.0f; // Reset recovery variable (using typical 'd' parameter)
        neuron.last_spike_time = current_time; // Record the time of the spike
    }
}