#ifndef NEUROMODULATION_KERNELS_CUH
#define NEUROMODULATION_KERNELS_CUH

#include "GPUNeuralStructures.h"

/**
 * @brief Modulates the intrinsic excitability of neurons based on neuromodulator levels.
 */
__global__ void applyIntrinsicNeuromodulationKernel(
    GPUNeuronState* neurons,
    float ACh_level, // Acetylcholine level
    float SER_level, // Serotonin level
    int num_neurons
);

/**
 * @brief Modulates the plasticity of synapses based on neuromodulator levels.
 */
__global__ void applySynapticNeuromodulationKernel(
    GPUSynapse* synapses,
    float ACh_level,
    int num_synapses
);

#endif // NEUROMODULATION_KERNELS_CUH