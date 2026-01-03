#include <engine/NeuronInitialization.cuh>
#include <engine/NeuronModelConstants.h>

// --- Kernel to initialize all state variables for every neuron ---
__global__ void neuronInitializationKernel(GPUNeuronState* neurons, int num_neurons) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[neuron_idx];

    // This kernel correctly interfaces with the definitive GPUNeuronState struct.

    // 1. Set Membrane Potential to Resting Potential
    // This establishes the baseline voltage of the neuron when it is not active.
    neuron.V = NeuronModelConstants::RESET_POTENTIAL;

    // 2. Initialize the Izhikevich model Recovery Variable 'u'
    // This value counteracts the membrane potential's rise, contributing to spike adaptation.
    // Initializing it relative to the resting potential ensures a stable start.
    neuron.u = 0.2f * neuron.V; // Using a typical 'b' parameter value for stability

    // 3. Clear Initial Synaptic Currents and Calcium Concentrations
    // Ensures that there is no residual activity from previous simulation runs.
    for (int i = 0; i < 4; ++i) { // Assuming 4 dendritic compartments/receptor types
        neuron.I_syn[i] = 0.0f;
        neuron.ca_conc[i] = 0.0f;
    }

    // 4. Reset Spike Time
    // A large negative value indicates the neuron has not spiked for a very long time.
    neuron.last_spike_time = -1e9f;

    // 5. Initialize Homeostatic Variables to a neutral, baseline state
    // These variables track long-term activity and should start at zero or a neutral value.
    neuron.average_firing_rate = 0.0f;
    neuron.average_activity = NeuronModelConstants::RESET_POTENTIAL;
    neuron.excitability = 1.0f; // Start with no excitability modulation
    neuron.synaptic_scaling_factor = 1.0f; // Start with no synaptic scaling
}