#include <engine/NeuromodulationKernels.cuh>
#include <engine/NeuronModelConstants.h>

// --- Kernel to modulate intrinsic neuron excitability ---
__global__ void applyIntrinsicNeuromodulationKernel(
    GPUNeuronState* neurons,
    float ACh_level,
    float SER_level,
    int num_neurons)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[neuron_idx];

    // Calculate the effect of each neuromodulator.
    float ach_effect = ACh_level * NeuronModelConstants::ACETYLCHOLINE_EXCITABILITY_FACTOR;
    float ser_effect = SER_level * NeuronModelConstants::SEROTONIN_INHIBITORY_FACTOR;

    // Update the existing 'excitability' member. This allows neuromodulation
    // to work in concert with homeostatic intrinsic plasticity.
    neuron.excitability += (ach_effect - ser_effect);
    neuron.excitability = fmaxf(0.5f, fminf(1.5f, neuron.excitability)); // Clamp to a reasonable range
}

// --- Kernel to modulate synaptic plasticity ---
__global__ void applySynapticNeuromodulationKernel(
    GPUSynapse* synapses,
    float ACh_level,
    int num_synapses)
{
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[synapse_idx];
    if (synapse.active == 0) return;

    // Acetylcholine is known to enhance the potentiation of synapses (LTP).
    float ach_plasticity_bonus = ACh_level * synapse.acetylcholine_sensitivity * NeuronModelConstants::ACETYLCHOLINE_PLASTICITY_FACTOR;

    // This bonus is applied to the plasticity_modulation factor, which can then be
    // used by other kernels (like STDP) to scale the learning rate.
    synapse.plasticity_modulation += ach_plasticity_bonus;
    synapse.plasticity_modulation = fmaxf(0.1f, fminf(2.0f, synapse.plasticity_modulation));
}