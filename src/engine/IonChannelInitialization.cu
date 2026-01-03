#include <engine/IonChannelInitialization.cuh>
#include <engine/NeuronModelConstants.h>

// --- Kernel to initialize ion channel states and membrane potentials ---
__global__ void ionChannelInitializationKernel(GPUNeuronState* neurons, int num_neurons) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[neuron_idx];

    // This kernel correctly accesses the members of the definitive GPUNeuronState struct.

    // 1. Set Membrane Potential to Resting Potential
    // This is the baseline voltage of the neuron when it is not active.
    neuron.V = NeuronModelConstants::RESET_POTENTIAL;

    // 2. Initialize the Recovery Variable
    // For Izhikevich neurons, u = b * V. This sets its initial value consistent with the resting potential.
    neuron.u = 0.2f * neuron.V; // Using a typical 'b' value for Izhikevich model

    // 3. Clear Initial Synaptic Currents and Calcium Concentrations
    // Ensures there is no residual current or calcium from previous simulation runs.
    for (int i = 0; i < 4; ++i) { // Assuming 4 compartments
        neuron.I_syn[i] = 0.0f;
        neuron.ca_conc[i] = 0.0f;
    }

    // 4. Reset Spike Time
    // A large negative value indicates the neuron has not spiked for a very long time.
    neuron.last_spike_time = -1e9f;

    // 5. Initialize Homeostatic Variables to a neutral state
    // These values represent the neuron's baseline long-term activity.
    neuron.average_firing_rate = 0.0f;
    neuron.average_activity = NeuronModelConstants::RESET_POTENTIAL;
    neuron.excitability = 1.0f; // Start with no modulation
    neuron.synaptic_scaling_factor = 1.0f; // Start with no scaling
}