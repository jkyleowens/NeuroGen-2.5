#ifndef STRUCTURAL_PLASTICITY_KERNELS_CUH
#define STRUCTURAL_PLASTICITY_KERNELS_CUH

#include "GPUNeuralStructures.h"
#include <curand_kernel.h>

/**
 * @brief Marks weak or underutilized synapses for pruning.
 *
 * This kernel identifies synapses whose weights have fallen below a certain
 * threshold and have low activity, marking them as inactive.
 */
__global__ void markPrunableSynapsesKernel(GPUSynapse* synapses, int num_synapses);

/**
 * @brief Drives neurogenesis by identifying highly active network regions
 * and activating new neurons to increase computational capacity.
 *
 * This kernel simulates the process where new neurons are created and integrated
 * into circuits that are under high load or exhibit high plasticity.
 */
__global__ void neurogenesisKernel(GPUNeuronState* neurons, const GPUSynapse* synapses,
                                 curandState* rng_states, int num_neurons);

#endif // STRUCTURAL_PLASTICITY_KERNELS_CUH